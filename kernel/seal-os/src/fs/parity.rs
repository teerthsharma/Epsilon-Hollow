// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Cross-filesystem parity harness for FAT16 and ext2.
//!
//! Two fixture images are formatted in-tree by [`format_fat16`] and
//! [`format_ext2`] (no external `mkfs`, fully deterministic), loaded into
//! RAM-backed block devices, and mounted through the same [`FileSystem`]
//! trait the VFS routes through. One identical mutation sequence is applied to
//! each image and every observable result is compared:
//!
//! * directory listings — normalised by dropping `.`/`..` and sorting by name
//! * file contents — byte-for-byte plus an FNV-1a-64 digest over the whole tree
//! * stat metadata — the fields both formats can express
//! * error codes — the same illegal operation must fail the same way
//!
//! A negative control deliberately flips one byte of one file on the ext2
//! image and requires the comparator to report the mismatch, so the
//! comparison is proven capable of failing.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use spin::Mutex;

use crate::drivers::block::{register_block_device, BlockDevice, BlockError};
use crate::fs::ext2::{
    BlockGroupDescriptor, DirEntryHeader, Ext2Fs, Inode, Superblock, EXT2_MAGIC, EXT2_S_IFDIR,
};
use crate::fs::fat::{BiosParameterBlock, FatFs};
use crate::fs::vfs::{FileSystem, VfsError, VfsNodeType};

/// Block device number for the FAT16 fixture image.
pub const FAT_DEV: u32 = 0x900;
/// Block device number for the ext2 fixture image.
pub const EXT2_DEV: u32 = 0x901;

// ---------------------------------------------------------------- FAT16 image

const F_SECTOR: usize = 512;
const F_TOTAL_SECTORS: u16 = 5120;
const F_RESERVED: u16 = 1;
const F_NUM_FATS: u8 = 2;
const F_ROOT_ENTRIES: u16 = 512;
const F_FAT_SECTORS: u16 = 21;

/// Exact byte length of the FAT16 fixture image: 5120 sectors x 512 B.
pub const FAT_IMAGE_BYTES: usize = F_TOTAL_SECTORS as usize * F_SECTOR;

// ----------------------------------------------------------------- ext2 image

const E_BS: usize = 1024;
const E_BLOCKS: u32 = 1024;
const E_INODES: u32 = 128;
const E_INODE_SIZE: u32 = 128;
const E_BLOCKS_PER_GROUP: u32 = 8192;
const E_FIRST_DATA_BLOCK: u32 = 1;
const E_BLK_BGD: u32 = 2;
const E_BLK_BLOCK_BITMAP: u32 = 3;
const E_BLK_INODE_BITMAP: u32 = 4;
const E_BLK_INODE_TABLE: u32 = 5;
const E_ITABLE_BLOCKS: u32 = 16; // 128 inodes x 128 B / 1024 B
const E_BLK_ROOT_DIR: u32 = 21;
const E_FIRST_FREE_BLOCK: u32 = 22;
const E_RESERVED_INODES: u32 = 11;
const E_ROOT_INO: u32 = 2;

/// Exact byte length of the ext2 fixture image: 1024 blocks x 1024 B.
pub const EXT2_IMAGE_BYTES: usize = E_BLOCKS as usize * E_BS;

// On-disk layout the formatters below depend on. If any driver struct drifts,
// the fixture images silently stop being valid — fail the build instead.
const _: () = {
    assert!(size_of::<BiosParameterBlock>() == 512);
    assert!(size_of::<BlockGroupDescriptor>() == 32);
    assert!(size_of::<Inode>() == E_INODE_SIZE as usize);
    assert!(size_of::<DirEntryHeader>() == 8);
    // FAT16 requires 4085..65525 clusters; the driver infers the type from this.
    assert!(F_TOTAL_SECTORS as usize > 4085);
    // Every FAT entry must fit in the FAT sectors reserved for it.
    assert!(
        (F_TOTAL_SECTORS as usize
            - (F_RESERVED as usize + F_NUM_FATS as usize * F_FAT_SECTORS as usize + 32)
            + 2)
            * 2
            <= F_FAT_SECTORS as usize * F_SECTOR
    );
    // The inode table must end exactly where the root directory block begins.
    assert!(E_BLK_INODE_TABLE + E_ITABLE_BLOCKS == E_BLK_ROOT_DIR);
    assert!(E_INODES * E_INODE_SIZE == E_ITABLE_BLOCKS * E_BS as u32);
};

// -------------------------------------------------------------------- fixture

/// One fixture file: deterministic pseudo-random contents from `seed`.
pub struct FixtureFile {
    pub path: &'static str,
    pub seed: u64,
    pub len: usize,
}

/// Files created by the mutation sequence, in order.
pub const CREATED_FILES: [FixtureFile; 4] = [
    FixtureFile {
        path: "/README.TXT",
        seed: 0x5EA1_0001,
        len: 640,
    },
    FixtureFile {
        path: "/DATA/ALPHA.BIN",
        seed: 0x5EA1_0002,
        len: 1536,
    },
    FixtureFile {
        path: "/DATA/BETA.BIN",
        seed: 0x5EA1_0003,
        len: 100,
    },
    FixtureFile {
        path: "/LOGS/BOOT.LOG",
        seed: 0x5EA1_0004,
        len: 2048,
    },
];

/// Paths that must exist, with identical bytes, after the sequence completes.
pub const FINAL_PATHS: [&str; 4] = [
    "/DATA/ALPHA.BIN",
    "/DATA/GAMMA.BIN",
    "/LOGS/BOOT.LOG",
    "/README.TXT",
];

/// Directories whose listings are compared.
pub const COMPARED_DIRS: [&str; 3] = ["/", "/DATA", "/LOGS"];

const APPEND_SEED: u64 = 0x5EA1_00FF;
const APPEND_LEN: usize = 64;
const APPEND_OFFSET: u64 = 640;

// ----------------------------------------------------------------- RAM device

/// Memory-backed block device: 512-byte sectors over a heap-resident image.
pub struct MemDisk {
    data: Mutex<Vec<u8>>,
}

impl Default for MemDisk {
    fn default() -> Self {
        Self::new()
    }
}

impl MemDisk {
    /// Create a device with no image; call [`MemDisk::load`] to populate it.
    pub const fn new() -> Self {
        Self {
            data: Mutex::new(Vec::new()),
        }
    }

    /// Replace the whole image (moves, does not copy).
    pub fn load(&self, image: Vec<u8>) {
        *self.data.lock() = image;
    }

    /// FNV-1a-64 over the entire image.
    pub fn digest(&self) -> u64 {
        fnv(FNV_OFFSET, &self.data.lock())
    }

    /// Image length in bytes.
    pub fn len(&self) -> usize {
        self.data.lock().len()
    }

    /// Returns `true` when no image has been loaded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl BlockDevice for MemDisk {
    fn sector_size(&self) -> u64 {
        F_SECTOR as u64
    }

    fn num_sectors(&self) -> u64 {
        (self.data.lock().len() / F_SECTOR) as u64
    }

    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let data = self.data.lock();
        let start = (lba as usize).saturating_mul(F_SECTOR);
        let end = start.saturating_add(buf.len());
        if end > data.len() {
            return Err(BlockError::InvalidLba);
        }
        buf.copy_from_slice(&data[start..end]);
        Ok(())
    }

    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        let mut data = self.data.lock();
        let start = (lba as usize).saturating_mul(F_SECTOR);
        let end = start.saturating_add(buf.len());
        if end > data.len() {
            return Err(BlockError::InvalidLba);
        }
        data[start..end].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

static FAT_DISK: Mutex<Option<&'static MemDisk>> = Mutex::new(None);
static EXT2_DISK: Mutex<Option<&'static MemDisk>> = Mutex::new(None);

fn attach(slot: &Mutex<Option<&'static MemDisk>>, dev: u32, image: Vec<u8>) -> &'static MemDisk {
    let existing = *slot.lock();
    let disk = match existing {
        Some(d) => d,
        None => {
            let leaked: &'static MemDisk = Box::leak(Box::new(MemDisk::new()));
            register_block_device(dev, leaked);
            *slot.lock() = Some(leaked);
            leaked
        }
    };
    disk.load(image);
    disk
}

// -------------------------------------------------------------------- digests

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Deterministic fixture contents: a 64-bit LCG folded down to bytes.
pub fn pattern(seed: u64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    for _ in 0..len {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((s >> 33) as u8);
    }
    out
}

fn put<T>(img: &mut [u8], off: usize, val: &T) {
    let bytes =
        unsafe { core::slice::from_raw_parts(val as *const T as *const u8, size_of::<T>()) };
    img[off..off + bytes.len()].copy_from_slice(bytes);
}

fn set_bit(bitmap: &mut [u8], bit: usize) {
    bitmap[bit / 8] |= 1 << (bit % 8);
}

// ------------------------------------------------------------------ formatters

/// Build a blank FAT16 image. Byte-for-byte reproducible: no timestamps, no
/// volume serial, no randomness.
pub fn format_fat16() -> Vec<u8> {
    let mut img = alloc::vec![0u8; FAT_IMAGE_BYTES];

    let bpb = BiosParameterBlock {
        boot_jump: [0xEB, 0x3C, 0x90],
        oem_name: *b"SEALPRTY",
        bytes_per_sector: F_SECTOR as u16,
        sectors_per_cluster: 1,
        reserved_sectors: F_RESERVED,
        num_fats: F_NUM_FATS,
        root_entries: F_ROOT_ENTRIES,
        total_sectors_16: F_TOTAL_SECTORS,
        media: 0xF8,
        fat_size_16: F_FAT_SECTORS,
        sectors_per_track: 32,
        num_heads: 8,
        hidden_sectors: 0,
        total_sectors_32: 0,
        fat_size_32: 0,
        root_cluster: 0,
        _padding: [0; 468],
    };
    put(&mut img, 0, &bpb);
    img[510] = 0x55;
    img[511] = 0xAA;

    // Both FAT copies get the reserved cluster-0/1 entries; everything else free.
    for copy in 0..F_NUM_FATS as usize {
        let base = (F_RESERVED as usize + copy * F_FAT_SECTORS as usize) * F_SECTOR;
        img[base..base + 4].copy_from_slice(&[0xF8, 0xFF, 0xFF, 0xFF]);
    }

    img
}

/// Build a blank ext2 image (1 KiB blocks, one block group, rev 1, 128-byte
/// inodes). Byte-for-byte reproducible: every timestamp field is zero.
pub fn format_ext2() -> Vec<u8> {
    let mut img = alloc::vec![0u8; EXT2_IMAGE_BYTES];

    // Blocks 1..=21 are metadata; the bitmap addresses blocks 1..=1023.
    let meta_blocks = E_FIRST_FREE_BLOCK - E_FIRST_DATA_BLOCK;
    let free_blocks = (E_BLOCKS - E_FIRST_DATA_BLOCK) - meta_blocks;
    let free_inodes = E_INODES - E_RESERVED_INODES;

    let sb = Superblock {
        s_inodes_count: E_INODES,
        s_blocks_count: E_BLOCKS,
        s_r_blocks_count: 0,
        s_free_blocks_count: free_blocks,
        s_free_inodes_count: free_inodes,
        s_first_data_block: E_FIRST_DATA_BLOCK,
        s_log_block_size: 0,
        s_log_frag_size: 0,
        s_blocks_per_group: E_BLOCKS_PER_GROUP,
        s_frags_per_group: E_BLOCKS_PER_GROUP,
        s_inodes_per_group: E_INODES,
        s_mtime: 0,
        s_wtime: 0,
        s_mnt_count: 0,
        s_max_mnt_count: 0xFFFF,
        s_magic: EXT2_MAGIC,
        s_state: 1,
        s_errors: 1,
        s_minor_rev_level: 0,
        s_lastcheck: 0,
        s_checkinterval: 0,
        s_creator_os: 0,
        s_rev_level: 1,
        s_def_resuid: 0,
        s_def_resgid: 0,
        s_first_ino: E_RESERVED_INODES,
        s_inode_size: E_INODE_SIZE as u16,
        s_block_group_nr: 0,
        s_feature_compat: 0,
        s_feature_incompat: 0,
        s_feature_ro_compat: 0,
        s_uuid: *b"SEAL-PARITY-EXT2",
        s_volume_name: *b"sealparity\0\0\0\0\0\0",
        s_last_mounted: [0; 64],
        s_algo_bitmap: 0,
        padding: [0; 788],
    };
    put(&mut img, E_BS, &sb);

    let bgd = BlockGroupDescriptor {
        bg_block_bitmap: E_BLK_BLOCK_BITMAP,
        bg_inode_bitmap: E_BLK_INODE_BITMAP,
        bg_inode_table: E_BLK_INODE_TABLE,
        bg_free_blocks_count: free_blocks as u16,
        bg_free_inodes_count: free_inodes as u16,
        bg_used_dirs_count: 1,
        bg_pad: 0,
        bg_reserved: [0; 3],
    };
    put(&mut img, E_BLK_BGD as usize * E_BS, &bgd);

    // Block bitmap: bit i addresses block i + s_first_data_block.
    {
        let base = E_BLK_BLOCK_BITMAP as usize * E_BS;
        let bitmap = &mut img[base..base + E_BS];
        for block in E_FIRST_DATA_BLOCK..E_FIRST_FREE_BLOCK {
            set_bit(bitmap, (block - E_FIRST_DATA_BLOCK) as usize);
        }
        // Past-the-end padding bits must read as allocated.
        for bit in (E_BLOCKS - E_FIRST_DATA_BLOCK) as usize..E_BS * 8 {
            set_bit(bitmap, bit);
        }
    }

    // Inode bitmap: bit i addresses inode i + 1.
    {
        let base = E_BLK_INODE_BITMAP as usize * E_BS;
        let bitmap = &mut img[base..base + E_BS];
        for bit in 0..E_RESERVED_INODES as usize {
            set_bit(bitmap, bit);
        }
        for bit in E_INODES as usize..E_BS * 8 {
            set_bit(bitmap, bit);
        }
    }

    // Root inode (2) points at the single root directory block.
    let mut root = Inode {
        i_mode: EXT2_S_IFDIR | 0o755,
        i_uid: 0,
        i_size: E_BS as u32,
        i_atime: 0,
        i_ctime: 0,
        i_mtime: 0,
        i_dtime: 0,
        i_gid: 0,
        i_links_count: 2,
        i_blocks: (E_BS / 512) as u32,
        i_flags: 0,
        i_osd1: 0,
        i_block: [0; 15],
        i_generation: 0,
        i_file_acl: 0,
        i_dir_acl: 0,
        i_faddr: 0,
        i_osd2: [0; 3],
    };
    root.i_block[0] = E_BLK_ROOT_DIR;
    let root_off =
        E_BLK_INODE_TABLE as usize * E_BS + (E_ROOT_INO as usize - 1) * E_INODE_SIZE as usize;
    put(&mut img, root_off, &root);

    // Root directory block: "." then ".." filling the rest of the block.
    let dir = E_BLK_ROOT_DIR as usize * E_BS;
    let hdr = size_of::<DirEntryHeader>();
    let dot_len = (hdr + 1 + 3) & !3;
    put(
        &mut img,
        dir,
        &DirEntryHeader {
            inode: E_ROOT_INO,
            rec_len: dot_len as u16,
            name_len: 1,
            file_type: 2,
        },
    );
    img[dir + hdr] = b'.';
    put(
        &mut img,
        dir + dot_len,
        &DirEntryHeader {
            inode: E_ROOT_INO,
            rec_len: (E_BS - dot_len) as u16,
            name_len: 2,
            file_type: 2,
        },
    );
    img[dir + dot_len + hdr] = b'.';
    img[dir + dot_len + hdr + 1] = b'.';

    img
}

// ------------------------------------------------------------ operation matrix

/// Apply the identical mutation sequence, returning the number of filesystem
/// operations executed.
fn build_tree(fs: &mut dyn FileSystem) -> Result<usize, VfsError> {
    let mut ops = 0usize;

    fs.mkdir("/DATA")?;
    ops += 1;
    fs.mkdir("/LOGS")?;
    ops += 1;

    for f in CREATED_FILES.iter() {
        let handle = fs.create(f.path)?;
        ops += 1;
        fs.write(handle, &pattern(f.seed, f.len), 0)?;
        ops += 1;
    }

    // create + write + unlink round trip.
    let tmp = fs.create("/TEMP.TMP")?;
    ops += 1;
    fs.write(tmp, b"transient", 0)?;
    ops += 1;
    fs.unlink("/TEMP.TMP")?;
    ops += 1;

    // mkdir + rmdir round trip.
    fs.mkdir("/SCRATCH")?;
    ops += 1;
    fs.rmdir("/SCRATCH")?;
    ops += 1;

    fs.rename("/DATA/BETA.BIN", "/DATA/GAMMA.BIN")?;
    ops += 1;

    // Append past the first cluster/block boundary.
    let readme = fs.lookup("/README.TXT")?;
    ops += 1;
    fs.write(readme, &pattern(APPEND_SEED, APPEND_LEN), APPEND_OFFSET)?;
    ops += 1;

    fs.sync()?;
    ops += 1;

    Ok(ops)
}

/// Directory listing with `.`/`..` removed and entries sorted by name.
fn normalised_dir(fs: &dyn FileSystem, path: &str) -> Result<Vec<(String, VfsNodeType)>, VfsError> {
    let handle = fs.lookup(path)?;
    let mut entries: Vec<(String, VfsNodeType)> = fs
        .readdir(handle)?
        .into_iter()
        .filter(|e| e.name != "." && e.name != "..")
        .map(|e| (e.name, e.node_type))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

fn read_all(fs: &dyn FileSystem, path: &str) -> Result<Vec<u8>, VfsError> {
    let handle = fs.lookup(path)?;
    let node = fs.stat(handle)?;
    let mut data = alloc::vec![0u8; node.size as usize];
    let mut got = 0usize;
    while got < data.len() {
        let n = fs.read(handle, &mut data[got..], got as u64)?;
        if n == 0 {
            break;
        }
        got += n;
    }
    data.truncate(got);
    Ok(data)
}

/// FNV-1a-64 over `(path, length, bytes)` for every surviving file, plus the
/// total byte count compared.
fn tree_digest(fs: &dyn FileSystem) -> Result<(u64, usize), VfsError> {
    let mut h = FNV_OFFSET;
    let mut total = 0usize;
    for path in FINAL_PATHS.iter() {
        let data = read_all(fs, path)?;
        h = fnv(h, path.as_bytes());
        h = fnv(h, &[0]);
        h = fnv(h, &(data.len() as u64).to_le_bytes());
        h = fnv(h, &data);
        total += data.len();
    }
    Ok((h, total))
}

fn same_error(a: &Result<(), VfsError>, b: &Result<(), VfsError>) -> bool {
    match (a, b) {
        (Err(x), Err(y)) => core::mem::discriminant(x) == core::mem::discriminant(y),
        (Ok(_), Ok(_)) => true,
        _ => false,
    }
}

/// Run every illegal operation against `fs` and collect the outcomes in a
/// fixed order so the two filesystems can be compared element-wise.
fn error_probes(fs: &mut dyn FileSystem) -> Vec<Result<(), VfsError>> {
    alloc::vec![
        fs.lookup("/NOSUCH.TXT").map(|_| ()),
        fs.create("/README.TXT").map(|_| ()),
        fs.mkdir("/DATA").map(|_| ()),
        fs.lookup("/DATA")
            .and_then(|h| fs.read(h, &mut [0u8; 4], 0))
            .map(|_| ()),
        fs.rmdir("/DATA"),
        fs.rmdir("/README.TXT"),
        fs.unlink("/DATA"),
        // Same-directory rename onto an existing name. A cross-directory
        // rename would not be comparable here: the FAT driver rejects the move
        // itself before it ever reaches the name collision, so the two
        // filesystems would be failing for two different reasons. That
        // difference is measured separately as the `crossdir` divergence.
        fs.rename("/DATA/ALPHA.BIN", "/DATA/GAMMA.BIN"),
    ]
}

// ---------------------------------------------------------------------- report

/// Every value in this struct is measured at runtime by [`measure`].
#[derive(Debug, Clone, Copy)]
pub struct ParityReport {
    pub fat_mounted: bool,
    pub ext2_mounted: bool,
    pub fat_image_bytes: usize,
    pub ext2_image_bytes: usize,
    pub fat_blank_digest: u64,
    pub ext2_blank_digest: u64,
    pub fat_ops: usize,
    pub ext2_ops: usize,
    pub files_compared: usize,
    pub bytes_compared: usize,
    pub fat_digest: u64,
    pub ext2_digest: u64,
    pub dirs_compared: usize,
    pub dirs_equal: usize,
    pub stat_fields_compared: usize,
    pub stat_fields_equal: usize,
    pub error_cases: usize,
    pub error_matches: usize,
    pub div_mode: usize,
    pub div_mtime: usize,
    pub div_dirsize: usize,
    pub div_case: usize,
    pub div_name83: usize,
    pub div_crossdir: usize,
    pub div_error: usize,
    pub negative_control_detected: bool,
    pub negative_control_digest: u64,
    pub negative_control_restored: bool,
    pub ok: bool,
}

impl ParityReport {
    /// Total divergences observed, inherent and defect alike.
    pub fn divergences(&self) -> usize {
        self.div_mode
            + self.div_mtime
            + self.div_dirsize
            + self.div_case
            + self.div_name83
            + self.div_crossdir
            + self.div_error
    }

    fn kinds(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.div_mode > 0 {
            parts.push("mode");
        }
        if self.div_mtime > 0 {
            parts.push("mtime");
        }
        if self.div_dirsize > 0 {
            parts.push("dirsize");
        }
        if self.div_case > 0 {
            parts.push("case");
        }
        if self.div_name83 > 0 {
            parts.push("name8_3");
        }
        if self.div_crossdir > 0 {
            parts.push("crossdir");
        }
        if self.div_error > 0 {
            parts.push("error");
        }
        if parts.is_empty() {
            return "none".to_string();
        }
        parts.join("+")
    }
}

fn blank_report() -> ParityReport {
    ParityReport {
        fat_mounted: false,
        ext2_mounted: false,
        fat_image_bytes: 0,
        ext2_image_bytes: 0,
        fat_blank_digest: 0,
        ext2_blank_digest: 0,
        fat_ops: 0,
        ext2_ops: 0,
        files_compared: 0,
        bytes_compared: 0,
        fat_digest: 0,
        ext2_digest: 0,
        dirs_compared: 0,
        dirs_equal: 0,
        stat_fields_compared: 0,
        stat_fields_equal: 0,
        error_cases: 0,
        error_matches: 0,
        div_mode: 0,
        div_mtime: 0,
        div_dirsize: 0,
        div_case: 0,
        div_name83: 0,
        div_crossdir: 0,
        div_error: 0,
        negative_control_detected: false,
        negative_control_digest: 0,
        negative_control_restored: false,
        ok: false,
    }
}

/// Format both fixture images, mount them, run the full operation matrix on
/// each, and compare every observable result.
pub fn measure() -> ParityReport {
    let mut r = blank_report();

    let fat_image = format_fat16();
    let ext2_image = format_ext2();
    r.fat_image_bytes = fat_image.len();
    r.ext2_image_bytes = ext2_image.len();
    r.fat_blank_digest = fnv(FNV_OFFSET, &fat_image);
    r.ext2_blank_digest = fnv(FNV_OFFSET, &ext2_image);

    attach(&FAT_DISK, FAT_DEV, fat_image);
    attach(&EXT2_DISK, EXT2_DEV, ext2_image);

    let mut fat = FatFs::new(FAT_DEV);
    r.fat_mounted = fat.mount().is_ok();
    let mut ext2 = Ext2Fs::new(EXT2_DEV);
    r.ext2_mounted = ext2.mount().is_ok();
    if !r.fat_mounted || !r.ext2_mounted {
        return r;
    }

    r.fat_ops = match build_tree(&mut fat) {
        Ok(n) => n,
        Err(_) => return r,
    };
    r.ext2_ops = match build_tree(&mut ext2) {
        Ok(n) => n,
        Err(_) => return r,
    };

    // --- content parity -------------------------------------------------
    let (fat_digest, fat_bytes) = match tree_digest(&fat) {
        Ok(v) => v,
        Err(_) => return r,
    };
    let (ext2_digest, ext2_bytes) = match tree_digest(&ext2) {
        Ok(v) => v,
        Err(_) => return r,
    };
    r.fat_digest = fat_digest;
    r.ext2_digest = ext2_digest;
    r.bytes_compared = fat_bytes;
    let mut contents_equal = fat_digest == ext2_digest && fat_bytes == ext2_bytes;
    for path in FINAL_PATHS.iter() {
        match (read_all(&fat, path), read_all(&ext2, path)) {
            (Ok(a), Ok(b)) => {
                if a != b {
                    contents_equal = false;
                }
                r.files_compared += 1;
            }
            _ => contents_equal = false,
        }
    }

    // --- directory parity -----------------------------------------------
    for dir in COMPARED_DIRS.iter() {
        r.dirs_compared += 1;
        if let (Ok(a), Ok(b)) = (normalised_dir(&fat, dir), normalised_dir(&ext2, dir)) {
            if a == b {
                r.dirs_equal += 1;
            }
        }
    }

    // --- stat parity ------------------------------------------------------
    for path in FINAL_PATHS.iter().chain(["/DATA", "/LOGS"].iter()) {
        let a = fat.lookup(path).and_then(|h| fat.stat(h));
        let b = ext2.lookup(path).and_then(|h| ext2.stat(h));
        if let (Ok(a), Ok(b)) = (a, b) {
            let is_dir = a.node_type == VfsNodeType::Directory;
            // Fields both formats can express. A subdirectory's size is not
            // among them: FAT stores 0 in the directory entry by
            // specification, ext2 stores the directory's byte length.
            for equal in [
                Some(a.node_type == b.node_type),
                Some(a.permissions == b.permissions),
                Some(a.uid == b.uid),
                Some(a.gid == b.gid),
                if is_dir { None } else { Some(a.size == b.size) },
            ]
            .into_iter()
            .flatten()
            {
                r.stat_fields_compared += 1;
                if equal {
                    r.stat_fields_equal += 1;
                }
            }
            if is_dir && a.size != b.size {
                r.div_dirsize += 1;
            }
            // Fields that cannot agree: FAT has no format bits in `mode` and
            // stores no timestamps in this driver.
            if a.mode != b.mode {
                r.div_mode += 1;
            }
            if a.mtime != b.mtime {
                r.div_mtime += 1;
            }
        } else {
            r.stat_fields_compared += 4;
        }
    }

    // --- error parity -----------------------------------------------------
    let fat_errors = error_probes(&mut fat);
    let ext2_errors = error_probes(&mut ext2);
    r.error_cases = fat_errors.len();
    for (a, b) in fat_errors.iter().zip(ext2_errors.iter()) {
        if same_error(a, b) {
            r.error_matches += 1;
        } else {
            r.div_error += 1;
        }
    }

    // --- negative control -------------------------------------------------
    // Flip one byte of one file on the ext2 image only; the comparator must
    // notice, and must go quiet again once the byte is restored.
    let victim = "/LOGS/BOOT.LOG";
    let victim_offset = 17u64;
    if let Ok(handle) = ext2.lookup(victim) {
        let mut orig = [0u8; 1];
        if ext2.read(handle, &mut orig, victim_offset).is_ok()
            && ext2.write(handle, &[orig[0] ^ 0xFF], victim_offset).is_ok()
        {
            if let Ok((corrupt, _)) = tree_digest(&ext2) {
                r.negative_control_digest = corrupt;
                r.negative_control_detected = corrupt != fat_digest;
            }
            let _ = ext2.write(handle, &orig, victim_offset);
            if let Ok((restored, _)) = tree_digest(&ext2) {
                r.negative_control_restored = restored == fat_digest;
            }
        }
    }

    // --- divergence probes (run last: they mutate the tree) ---------------
    // Case folding: FAT upper-cases 8.3 names, ext2 preserves byte-exact names.
    if fat.lookup("/readme.txt").is_ok() != ext2.lookup("/readme.txt").is_ok() {
        r.div_case += 1;
    }
    // Long names: FAT truncates to 8.3 without an LFN chain, ext2 keeps 255.
    let long = "/LONGNAMEFILE.DATA";
    let _ = fat.create(long);
    let _ = ext2.create(long);
    if fat.lookup(long).is_ok() != ext2.lookup(long).is_ok() {
        r.div_name83 += 1;
    }
    // Cross-directory rename: the FAT driver refuses to move an entry between
    // directories at all; ext2 rewrites the entry and the moved "..".
    let moved = "/MOVED.LOG";
    if fat.rename("/LOGS/BOOT.LOG", moved).is_ok() != ext2.rename("/LOGS/BOOT.LOG", moved).is_ok() {
        r.div_crossdir += 1;
    }

    r.ok = r.fat_mounted
        && r.ext2_mounted
        && contents_equal
        && r.files_compared == FINAL_PATHS.len()
        && r.fat_ops == r.ext2_ops
        && r.dirs_compared == r.dirs_equal
        && r.stat_fields_compared == r.stat_fields_equal
        && r.error_cases == r.error_matches
        && r.div_error == 0
        && r.negative_control_detected
        && r.negative_control_restored;
    r
}

/// `[FSPARITY] proof version=1 ...` — every field measured by [`measure`].
pub fn fs_parity_proof_line() -> String {
    let r = measure();
    format!(
        "[FSPARITY] proof version=1 fat_image=fat16_fixture fat_mounted={} fat_image_bytes={} fat_blank_digest={:#018x} ext2_image=ext2_rev1_1k_fixture ext2_mounted={} ext2_image_bytes={} ext2_blank_digest={:#018x} ops_fat={} ops_ext2={} files_compared={} bytes_compared={} content_digest_fat={:#018x} content_digest_ext2={:#018x} content_parity={} dirs_compared={} dirs_equal={} stat_fields_compared={} stat_fields_equal={} error_cases={} error_matches={} divergences={} divergence_kinds={} negative_control_digest={:#018x} negative_control={} negative_control_restored={} result={}",
        if r.fat_mounted { "ok" } else { "fail" },
        r.fat_image_bytes,
        r.fat_blank_digest,
        if r.ext2_mounted { "ok" } else { "fail" },
        r.ext2_image_bytes,
        r.ext2_blank_digest,
        r.fat_ops,
        r.ext2_ops,
        r.files_compared,
        r.bytes_compared,
        r.fat_digest,
        r.ext2_digest,
        if r.fat_digest == r.ext2_digest {
            "byte_for_byte"
        } else {
            "mismatch"
        },
        r.dirs_compared,
        r.dirs_equal,
        r.stat_fields_compared,
        r.stat_fields_equal,
        r.error_cases,
        r.error_matches,
        r.divergences(),
        r.kinds(),
        r.negative_control_digest,
        if r.negative_control_detected {
            "detected"
        } else {
            "missed"
        },
        if r.negative_control_restored {
            "ok"
        } else {
            "fail"
        },
        if r.ok { "pass" } else { "fail" }
    )
}

/// Emit the parity proof line to the serial console.
pub fn emit_boot_proof() {
    crate::serial_println!("{}", fs_parity_proof_line());
}

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// `measure()` reformats both images, so cache one run for all assertions.
    static CACHED: Mutex<Option<ParityReport>> = Mutex::new(None);

    fn report() -> ParityReport {
        let mut guard = CACHED.lock();
        if guard.is_none() {
            *guard = Some(measure());
        }
        guard.unwrap()
    }

    fn test_fixture_images_are_deterministic() -> TestResult {
        test_assert_eq!(format_fat16().len(), FAT_IMAGE_BYTES);
        test_assert_eq!(format_ext2().len(), EXT2_IMAGE_BYTES);
        test_assert_eq!(
            fnv(FNV_OFFSET, &format_fat16()),
            fnv(FNV_OFFSET, &format_fat16())
        );
        test_assert_eq!(
            fnv(FNV_OFFSET, &format_ext2()),
            fnv(FNV_OFFSET, &format_ext2())
        );
        TestResult::Pass
    }

    fn test_both_images_mount() -> TestResult {
        let r = report();
        test_assert!(r.fat_mounted, "FAT16 fixture image failed to mount");
        test_assert!(r.ext2_mounted, "ext2 fixture image failed to mount");
        TestResult::Pass
    }

    fn test_operation_matrix_runs_on_both() -> TestResult {
        let r = report();
        test_assert!(r.fat_ops > 0, "no operations executed on FAT");
        test_assert_eq!(r.fat_ops, r.ext2_ops);
        TestResult::Pass
    }

    fn test_content_parity_byte_for_byte() -> TestResult {
        let r = report();
        test_assert_eq!(r.files_compared, FINAL_PATHS.len());
        test_assert!(r.bytes_compared > 0, "no file bytes compared");
        test_assert_eq!(r.fat_digest, r.ext2_digest);
        TestResult::Pass
    }

    fn test_readdir_parity() -> TestResult {
        let r = report();
        test_assert_eq!(r.dirs_compared, COMPARED_DIRS.len());
        test_assert_eq!(r.dirs_equal, r.dirs_compared);
        TestResult::Pass
    }

    fn test_stat_parity_on_shared_fields() -> TestResult {
        let r = report();
        test_assert!(r.stat_fields_compared > 0, "no stat fields compared");
        test_assert_eq!(r.stat_fields_equal, r.stat_fields_compared);
        TestResult::Pass
    }

    fn test_error_parity() -> TestResult {
        let r = report();
        test_assert!(r.error_cases > 0, "no error probes executed");
        test_assert_eq!(r.error_matches, r.error_cases);
        test_assert_eq!(r.div_error, 0);
        TestResult::Pass
    }

    fn test_negative_control_detects_corruption() -> TestResult {
        let r = report();
        test_assert!(
            r.negative_control_detected,
            "comparator missed a one-byte corruption"
        );
        test_assert!(
            r.negative_control_digest != r.fat_digest,
            "corrupted digest matched the FAT digest"
        );
        test_assert!(
            r.negative_control_restored,
            "digest did not return to parity after restore"
        );
        TestResult::Pass
    }

    fn test_divergences_are_enumerated() -> TestResult {
        let r = report();
        // Each of these is inherent to the two formats and must be visible.
        test_assert!(r.div_mode > 0, "expected mode divergence was not observed");
        test_assert!(
            r.div_mtime > 0,
            "expected mtime divergence was not observed"
        );
        test_assert!(
            r.div_dirsize > 0,
            "expected directory-size divergence was not observed"
        );
        test_assert!(r.div_name83 > 0, "expected 8.3 divergence was not observed");
        test_assert!(
            r.div_crossdir > 0,
            "expected cross-directory rename divergence was not observed"
        );
        TestResult::Pass
    }

    fn test_proof_line_reports_pass() -> TestResult {
        let r = report();
        test_assert!(r.ok, "parity proof did not reach result=pass");
        TestResult::Pass
    }

    /// FAT `stat` on a cluster that is not referenced by any directory entry
    /// must terminate with NotFound rather than walking `.` forever.
    fn test_fat_stat_unknown_cluster_terminates() -> TestResult {
        let r = report();
        test_assert!(r.fat_mounted, "FAT16 fixture image failed to mount");
        let mut fat = FatFs::new(FAT_DEV);
        test_assert!(fat.mount().is_ok(), "remount failed");
        let bogus = crate::fs::vfs::VfsHandle {
            fs_idx: 0,
            inode: 4000,
        };
        test_assert!(fat.stat(bogus).is_err(), "bogus cluster should not resolve");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "filesystem::parity_fixture_images_deterministic",
            test_fixture_images_are_deterministic,
        );
        crate::testing::register_test(
            "filesystem::parity_both_images_mount",
            test_both_images_mount,
        );
        crate::testing::register_test(
            "filesystem::parity_operation_matrix",
            test_operation_matrix_runs_on_both,
        );
        crate::testing::register_test(
            "filesystem::parity_content_byte_for_byte",
            test_content_parity_byte_for_byte,
        );
        crate::testing::register_test("filesystem::parity_readdir", test_readdir_parity);
        crate::testing::register_test("filesystem::parity_stat", test_stat_parity_on_shared_fields);
        crate::testing::register_test("filesystem::parity_errors", test_error_parity);
        crate::testing::register_test(
            "filesystem::parity_negative_control",
            test_negative_control_detects_corruption,
        );
        crate::testing::register_test(
            "filesystem::parity_divergences_enumerated",
            test_divergences_are_enumerated,
        );
        crate::testing::register_test(
            "filesystem::parity_proof_pass",
            test_proof_line_reports_pass,
        );
        crate::testing::register_test(
            "filesystem::parity_fat_unknown_cluster_terminates",
            test_fat_stat_unknown_cluster_terminates,
        );
    }
}
