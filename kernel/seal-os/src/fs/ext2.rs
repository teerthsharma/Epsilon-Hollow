// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Ext2 filesystem driver.
//!
//! Provides parsing for Ext2 superblocks, block group descriptors, and inodes.
//! Compliant with Phase 2.4 of the Epsilon-Hollow roadmap.
//!
//! **Topology integration:**
//! * Indirect block writes use T1 (Voronoi cell of logical block) to group
//!   allocations into the same block-group where possible.
//! * Buffer-cache dirty-batch flushing is governed by T4 ε (low ε = large
//!   stable batches; high ε = frequent small flushes).

use alloc::vec::Vec;
use spin::Mutex;

use crate::fs::buffer_cache::BufferCache;
use crate::fs::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};

pub const EXT2_MAGIC: u16 = 0xEF53;

pub const EXT2_S_IFMT: u16 = 0xF000;
pub const EXT2_S_IFSOCK: u16 = 0xC000;
pub const EXT2_S_IFLNK: u16 = 0xA000;
pub const EXT2_S_IFREG: u16 = 0x8000;
pub const EXT2_S_IFBLK: u16 = 0x6000;
pub const EXT2_S_IFDIR: u16 = 0x4000;
pub const EXT2_S_IFCHR: u16 = 0x2000;
pub const EXT2_S_IFIFO: u16 = 0x1000;

/// Ext2 Superblock (on-disk structure).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Superblock {
    pub s_inodes_count: u32,
    pub s_blocks_count: u32,
    pub s_r_blocks_count: u32,
    pub s_free_blocks_count: u32,
    pub s_free_inodes_count: u32,
    pub s_first_data_block: u32,
    pub s_log_block_size: u32,
    pub s_log_frag_size: u32,
    pub s_blocks_per_group: u32,
    pub s_frags_per_group: u32,
    pub s_inodes_per_group: u32,
    pub s_mtime: u32,
    pub s_wtime: u32,
    pub s_mnt_count: u16,
    pub s_max_mnt_count: u16,
    pub s_magic: u16,
    pub s_state: u16,
    pub s_errors: u16,
    pub s_minor_rev_level: u16,
    pub s_lastcheck: u32,
    pub s_checkinterval: u32,
    pub s_creator_os: u32,
    pub s_rev_level: u32,
    pub s_def_resuid: u16,
    pub s_def_resgid: u16,
    // Extended fields begin here (if s_rev_level >= 1)
    pub s_first_ino: u32,
    pub s_inode_size: u16,
    pub s_block_group_nr: u16,
    pub s_feature_compat: u32,
    pub s_feature_incompat: u32,
    pub s_feature_ro_compat: u32,
    pub s_uuid: [u8; 16],
    pub s_volume_name: [u8; 16],
    pub s_last_mounted: [u8; 64],
    pub s_algo_bitmap: u32,
    // Preallocate for padding to 1024 bytes (1024 - 236 = 788)
    pub padding: [u8; 788],
}

/// Ext2 Block Group Descriptor (on-disk structure).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockGroupDescriptor {
    pub bg_block_bitmap: u32,
    pub bg_inode_bitmap: u32,
    pub bg_inode_table: u32,
    pub bg_free_blocks_count: u16,
    pub bg_free_inodes_count: u16,
    pub bg_used_dirs_count: u16,
    pub bg_pad: u16,
    pub bg_reserved: [u32; 3],
}

/// Ext2 Inode (on-disk structure).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Inode {
    pub i_mode: u16,
    pub i_uid: u16,
    pub i_size: u32,
    pub i_atime: u32,
    pub i_ctime: u32,
    pub i_mtime: u32,
    pub i_dtime: u32,
    pub i_gid: u16,
    pub i_links_count: u16,
    pub i_blocks: u32,
    pub i_flags: u32,
    pub i_osd1: u32,
    pub i_block: [u32; 15],
    pub i_generation: u32,
    pub i_file_acl: u32,
    pub i_dir_acl: u32,
    pub i_faddr: u32,
    pub i_osd2: [u32; 3],
}

/// Ext2 Directory Entry (on-disk structure).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DirEntryHeader {
    pub inode: u32,
    pub rec_len: u16,
    pub name_len: u8,
    pub file_type: u8,
}

pub struct Ext2Fs {
    dev_num: u32,
    superblock: Option<Superblock>,
    block_size: u32,
    bgd_block: u32,
    inodes_per_group: u32,
    inode_size: u32,
    buffer_cache: Mutex<BufferCache>,
}

impl Ext2Fs {
    /// Create a new, unmounted Ext2 filesystem driver for the specified block device.
    pub fn new(dev_num: u32) -> Self {
        Self {
            dev_num,
            superblock: None,
            block_size: 0,
            bgd_block: 0,
            inodes_per_group: 0,
            inode_size: 0,
            buffer_cache: Mutex::new(BufferCache::new(64, 1024)), // default 1 KiB blocks, resized on mount
        }
    }

    /// Read a block through the buffer cache.
    fn read_disk_block(&self, block: u32, buf: &mut [u8]) -> Result<(), VfsError> {
        let mut cache = self.buffer_cache.lock();
        let cached = cache
            .get_block(self.dev_num, block as u64)
            .ok_or(VfsError::IoError)?;
        let len = buf.len().min(cached.data.len());
        buf[..len].copy_from_slice(&cached.data()[..len]);
        Ok(())
    }

    /// Write a block through the buffer cache.
    fn write_disk_block(&self, block: u32, buf: &[u8]) -> Result<(), VfsError> {
        let mut cache = self.buffer_cache.lock();
        cache.write_block(self.dev_num, block as u64, buf);
        Ok(())
    }

    /// Zero a newly allocated block in the cache.
    fn zero_block_in_cache(&self, block: u32) -> Result<(), VfsError> {
        let mut cache = self.buffer_cache.lock();
        let cached = cache
            .get_block(self.dev_num, block as u64)
            .ok_or(VfsError::IoError)?;
        cached.data_mut().fill(0);
        cached.mark_dirty();
        Ok(())
    }

    /// Validate on-disk superblock fields that later arithmetic in this file
    /// treats as trusted, and compute the derived `block_size`.
    ///
    /// `s_log_frag_size` and `s_frags_per_group` were checked and cleared:
    /// neither is read anywhere in this file.
    ///
    /// The remaining fields are bounded here because each one sizes an
    /// allocation, bounds a loop, or computes an index:
    ///
    /// * `s_blocks_count` is the sole multiplicand of `checked_size`'s
    ///   ceiling, which is the entire basis of the directory-`i_size` and
    ///   regular-file-`total_size` bounds this file already carries. Left
    ///   unbounded it re-opens both: `s_blocks_count = u32::MAX` puts that
    ///   ceiling at 4 TiB even on a 1 MiB device, so a directory inode
    ///   claiming `i_size = 0xFFFF_FFF0` passes `checked_dir_size` and
    ///   reaches `alloc::vec![0u8; 4 GiB]` — and this `no_std` kernel
    ///   registers no `#[alloc_error_handler]`, so the allocation failure
    ///   aborts. It also bounds `allocate_block`'s group scan. `device_sectors`
    ///   is the extent of the device the superblock was read from, so the
    ///   ceiling is a physical fact rather than a disk-controlled claim.
    /// * `s_inodes_count` bounds `allocate_inode`'s group scan the same way
    ///   `s_blocks_count` bounds `allocate_block`'s. It is tied to the block
    ///   count here through the group count they must agree on, which is
    ///   ext2's own invariant, so the now device-bounded block count bounds
    ///   the inode scan too.
    /// * `s_blocks_per_group` and `s_inodes_per_group` bound `local` in
    ///   `free_block`/`free_inode`, whose `local / 8` then indexes a `Vec`
    ///   exactly one block long. A group's block bitmap and inode bitmap are
    ///   one block each by construction, so `block_size * 8` is both the real
    ///   ext2 ceiling and precisely the bound that keeps `byte_idx` inside
    ///   that `Vec`: at the maximum, `((block_size * 8) - 1) / 8 ==
    ///   block_size - 1`. Without it, `s_blocks_per_group = 0x8000_0000` and
    ///   a corrupt `i_block` entry index `buf[268435455]` of a 1024-byte
    ///   buffer, which panics, and this kernel's profiles set
    ///   `panic = "abort"`.
    /// * `s_inode_size` (only read when `s_rev_level >= 1`) divides
    ///   `block_size` into `inodes_per_block` and then scales the offset that
    ///   `read_inode`/`write_inode` hand to `read_unaligned::<Inode>`. Zero
    ///   divides directly and a value above `block_size` floors
    ///   `inodes_per_block` to zero, a second divide by zero — but anything
    ///   below `size_of::<Inode>()` is worse than either: `s_inode_size = 1`
    ///   puts the last inode of a block at offset 1023 and reads 128 bytes
    ///   from there, 127 of them past the end of the buffer, through a raw
    ///   pointer that will not trap.
    /// * `s_log_block_size` is a raw `u32`, and `1024u32 << n` is
    ///   `2^(10+n)`: for `n` in `22..=31` that overflows `u32`, and with
    ///   neither Cargo profile in this crate setting `overflow-checks`,
    ///   release silently wraps it to exactly `0`; for `n >= 32` the shift
    ///   amount is masked to `n % 32` (defined behaviour for `<<`, not UB),
    ///   which merely yields a wrong block size rather than a zero one —
    ///   still wrong, rejected below for the same reason. `self.block_size`
    ///   then feeds roughly a dozen division sites in this file (`read_bgd`,
    ///   `read_inode`, `read_inode_data`, `write_inode_data`, …); dividing by
    ///   zero panics unconditionally in Rust regardless of overflow-checks.
    ///   The real ext2 spec permits block sizes 1024..=65536
    ///   (`s_log_block_size` 0..=6), but this file's own `rec_len:
    ///   self.block_size as u16` in `add_dir_entry` (new-block path) can only
    ///   hold values up to `65535` — a `block_size` of exactly `65536`
    ///   truncates to `0` there, which every directory-entry walker in this
    ///   file (`find_dir_entry`, `readdir`, `lookup`, …) reads as "end of
    ///   directory", silently corrupting the entry. So the range accepted
    ///   here is narrowed to `0..=5` (max block size 32768), not the spec's
    ///   `0..=6`.
    /// * `s_first_data_block` is never a divisor, but it is added un-checked
    ///   into `bgd_block` below (`+ 1`), so it is bounded to inside the
    ///   volume to keep that addition from wrapping.
    ///
    /// Every rejection is an error, never a clamp: a filesystem whose own
    /// superblock does not describe it is refused rather than mounted with
    /// values this driver invented.
    fn validate_superblock(sb: &Superblock, device_sectors: u64) -> Result<u32, VfsError> {
        if sb.s_log_block_size > 5 {
            return Err(VfsError::IoError);
        }
        let block_size = 1024u32 << sb.s_log_block_size;

        if sb.s_inodes_per_group == 0 || sb.s_blocks_per_group == 0 {
            return Err(VfsError::IoError);
        }

        // One bitmap block per group covers `block_size * 8` entries.
        let bits_per_bitmap_block = block_size * 8;
        if sb.s_blocks_per_group > bits_per_bitmap_block
            || sb.s_inodes_per_group > bits_per_bitmap_block
        {
            return Err(VfsError::IoError);
        }

        if sb.s_rev_level >= 1
            && ((sb.s_inode_size as u32) < core::mem::size_of::<Inode>() as u32
                || (sb.s_inode_size as u32) > block_size)
        {
            return Err(VfsError::IoError);
        }

        if sb.s_first_data_block >= sb.s_blocks_count {
            return Err(VfsError::IoError);
        }

        // `block_size` is at least 1024, so the sector count per block is at
        // least 2 and the product cannot exceed `u32::MAX * 64`.
        if (sb.s_blocks_count as u64) * (block_size as u64 / 512) > device_sectors {
            return Err(VfsError::IoError);
        }

        // `allocate_block` derives its group count from the block side and
        // `allocate_inode` from the inode side; ext2 requires them to agree,
        // and only the block side is bounded by the device.
        let groups = sb.s_blocks_count.div_ceil(sb.s_blocks_per_group) as u64;
        if sb.s_inodes_count as u64 > groups * sb.s_inodes_per_group as u64 {
            return Err(VfsError::IoError);
        }

        Ok(block_size)
    }

    /// Read and verify the Ext2 superblock.
    ///
    /// Mounts once per instance. The last thing this function does is replace
    /// the buffer cache, and `BufferCache` has no `Drop` impl
    /// (`fs/buffer_cache.rs`), so on a second call every dirty buffer the
    /// filesystem has accumulated — inode table, bitmaps, directory blocks,
    /// file data — is dropped without being written back, silently discarding
    /// writes that already reported success to their caller. No caller in the
    /// tree can reach that today: `fs::init_vfs`, `apps::installer`, and
    /// `fs::parity::measure` each build a fresh `Ext2Fs::new` and mount it
    /// exactly once. `mount` is `pub` on a `pub` struct, so the guard is here
    /// rather than at those three call sites, and it fails closed: a remount
    /// is refused, not silently turned into a flush-and-remount, because
    /// nothing needs remount semantics and inventing them would be inventing
    /// a crash-consistency story to go with them.
    pub fn mount(&mut self) -> Result<(), VfsError> {
        if self.superblock.is_some() {
            return Err(VfsError::InvalidOperation);
        }

        let mut buf = [0u8; 1024];

        // Superblock is always located at offset 1024 of the device.
        // Assuming 512-byte sectors, it begins at LBA 2.
        crate::drivers::block::read_block(self.dev_num, 2, &mut buf)
            .map_err(|_| VfsError::IoError)?;

        let sb = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const Superblock) };
        if sb.s_magic != EXT2_MAGIC {
            return Err(VfsError::IoError);
        }

        // The extent of the device this superblock claims to describe. The
        // read above already resolved `dev_num` through the same registry, so
        // a device missing here means it was unregistered mid-mount.
        let device_sectors = crate::drivers::block::list_devices()
            .into_iter()
            .find(|(dev, _)| *dev == self.dev_num)
            .map(|(_, sectors)| sectors)
            .ok_or(VfsError::IoError)?;

        let block_size = Self::validate_superblock(&sb, device_sectors)?;
        self.superblock = Some(sb);
        self.block_size = block_size;
        self.inodes_per_group = sb.s_inodes_per_group;
        self.inode_size = if sb.s_rev_level >= 1 {
            sb.s_inode_size as u32
        } else {
            128
        };
        self.bgd_block = sb.s_first_data_block + 1;

        // Resize buffer cache to match the filesystem block size.
        *self.buffer_cache.lock() = BufferCache::new(64, self.block_size as usize);
        Ok(())
    }

    fn read_bgd(&self, group: u32) -> Result<BlockGroupDescriptor, VfsError> {
        let bgds_per_block = self.block_size / core::mem::size_of::<BlockGroupDescriptor>() as u32;
        let block_offset = group / bgds_per_block;
        let bgd_idx_in_block = group % bgds_per_block;

        let mut cache = self.buffer_cache.lock();
        let cached = cache
            .get_block(self.dev_num, (self.bgd_block + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        let bgd = unsafe {
            let ptr = cached
                .data()
                .as_ptr()
                .add((bgd_idx_in_block as usize) * core::mem::size_of::<BlockGroupDescriptor>());
            core::ptr::read_unaligned(ptr as *const BlockGroupDescriptor)
        };
        Ok(bgd)
    }

    fn write_bgd(&self, group: u32, bgd: &BlockGroupDescriptor) -> Result<(), VfsError> {
        let bgds_per_block = self.block_size / core::mem::size_of::<BlockGroupDescriptor>() as u32;
        let block_offset = group / bgds_per_block;
        let bgd_idx_in_block = group % bgds_per_block;

        let mut cache = self.buffer_cache.lock();
        let cached = cache
            .get_block(self.dev_num, (self.bgd_block + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        unsafe {
            let ptr = cached
                .data_mut()
                .as_mut_ptr()
                .add((bgd_idx_in_block as usize) * core::mem::size_of::<BlockGroupDescriptor>());
            core::ptr::copy_nonoverlapping(
                bgd as *const BlockGroupDescriptor as *const u8,
                ptr,
                core::mem::size_of::<BlockGroupDescriptor>(),
            );
        }
        cached.mark_dirty();
        Ok(())
    }

    fn read_inode(&self, ino: u32) -> Result<Inode, VfsError> {
        if ino == 0 {
            return Err(VfsError::NotFound);
        }
        let group = (ino - 1) / self.inodes_per_group;
        let index = (ino - 1) % self.inodes_per_group;

        let bgd = self.read_bgd(group)?;

        let inodes_per_block = self.block_size / self.inode_size;
        let block_offset = index / inodes_per_block;
        let inode_idx_in_block = index % inodes_per_block;

        let mut cache = self.buffer_cache.lock();
        let cached = cache
            .get_block(self.dev_num, (bgd.bg_inode_table + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        let inode = unsafe {
            let ptr = cached
                .data()
                .as_ptr()
                .add((inode_idx_in_block as usize) * (self.inode_size as usize));
            core::ptr::read_unaligned(ptr as *const Inode)
        };
        Ok(inode)
    }

    fn write_inode(&self, ino: u32, inode: &Inode) -> Result<(), VfsError> {
        if ino == 0 {
            return Err(VfsError::NotFound);
        }
        let group = (ino - 1) / self.inodes_per_group;
        let index = (ino - 1) % self.inodes_per_group;

        let bgd = self.read_bgd(group)?;

        let inodes_per_block = self.block_size / self.inode_size;
        let block_offset = index / inodes_per_block;
        let inode_idx_in_block = index % inodes_per_block;

        let mut cache = self.buffer_cache.lock();
        let cached = cache
            .get_block(self.dev_num, (bgd.bg_inode_table + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        unsafe {
            let ptr = cached
                .data_mut()
                .as_mut_ptr()
                .add((inode_idx_in_block as usize) * (self.inode_size as usize));
            core::ptr::copy_nonoverlapping(
                inode as *const Inode as *const u8,
                ptr,
                core::mem::size_of::<Inode>(),
            );
        }
        cached.mark_dirty();
        Ok(())
    }

    /// File blocks reachable through one triple-indirect pointer:
    /// `ptrs_per_block^3`.
    ///
    /// `ptrs_per_block` is `block_size / 4`, so for every block size this file
    /// accepts with `s_log_block_size >= 3` (8192, 16384, 32768) it is a power
    /// of two >= 2048 and its cube is an exact multiple of `2^32`. Computed in
    /// `u32` — with neither Cargo profile in this crate setting
    /// `overflow-checks` — that wraps to exactly `0` in release, which turns
    /// any bound built on it into "reject everything" or "bound nothing"
    /// depending on which side of the comparison it lands. A `u64` cannot wrap
    /// for any `ptrs_per_block` this file can produce (max `8192`, cube
    /// `~5.5e11`).
    ///
    /// Both bounds that need this number — the read bound in
    /// `get_disk_block_from_inode` and the write bound in
    /// `get_or_allocate_data_block` — come from here, so a change to one
    /// cannot leave the other computing a different capacity.
    fn triple_indirect_capacity(ptrs_per_block: u32) -> u64 {
        (ptrs_per_block as u64).pow(3)
    }

    fn get_disk_block_from_inode(&self, inode: &Inode, file_block: u32) -> Result<u32, VfsError> {
        let ptrs_per_block = self.block_size / 4;

        if file_block < 12 {
            return Ok(inode.i_block[file_block as usize]);
        }

        let mut file_block = file_block - 12;
        if file_block < ptrs_per_block {
            let ind_block = inode.i_block[12];
            if ind_block == 0 {
                return Ok(0);
            }
            let mut buf = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(ind_block, &mut buf)?;
            let off = file_block as usize * 4;
            let block = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            return Ok(block);
        }

        file_block -= ptrs_per_block;
        if file_block < ptrs_per_block * ptrs_per_block {
            let dind_block = inode.i_block[13];
            if dind_block == 0 {
                return Ok(0);
            }
            let mut buf = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(dind_block, &mut buf)?;
            let ind_idx = file_block / ptrs_per_block;
            let off = ind_idx as usize * 4;
            let ind_block =
                u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            if ind_block == 0 {
                return Ok(0);
            }

            self.read_disk_block(ind_block, &mut buf)?;
            let block_idx = file_block % ptrs_per_block;
            let off2 = block_idx as usize * 4;
            let block =
                u32::from_le_bytes([buf[off2], buf[off2 + 1], buf[off2 + 2], buf[off2 + 3]]);
            return Ok(block);
        }

        file_block -= ptrs_per_block * ptrs_per_block;

        // Triple indirect. The two branches above are guarded by the `if`
        // that selects them; this one is the fall-through, so nothing has
        // bounded `file_block` against what the block-pointer arrays can
        // actually address. `read_inode_data` bounds the *size* against the
        // volume's capacity (`checked_size`), which is a different and larger
        // ceiling: a volume with more blocks than `12 + ptrs_per_block +
        // ptrs_per_block^2 + ptrs_per_block^3` — 16,843,020 blocks, about
        // 16 GiB, at the 1024-byte block size — can hold a size that clears
        // `checked_size` and still names a file block no `i_block` entry
        // reaches. `dind_idx` below then exceeds `ptrs_per_block`, and
        // `buf[off]` indexes past the end of the block just read, which
        // panics; both Cargo profiles here set `panic = "abort"`.
        //
        // Refused rather than reported as a hole (`Ok(0)`): a block the inode
        // cannot address is corruption, and returning zeros would hand the
        // caller fabricated file data. Checked before `i_block[14]` is read so
        // the refusal does not depend on that pointer being populated, and
        // before the block-sized buffer below is allocated.
        if (file_block as u64) >= Self::triple_indirect_capacity(ptrs_per_block) {
            return Err(VfsError::IoError);
        }

        let tind_block = inode.i_block[14];
        if tind_block == 0 {
            return Ok(0);
        }
        let mut buf = alloc::vec![0u8; self.block_size as usize];
        self.read_disk_block(tind_block, &mut buf)?;

        let dind_idx = file_block / (ptrs_per_block * ptrs_per_block);
        let off = dind_idx as usize * 4;
        let dind_block = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        if dind_block == 0 {
            return Ok(0);
        }

        self.read_disk_block(dind_block, &mut buf)?;
        let ind_idx = (file_block / ptrs_per_block) % ptrs_per_block;
        let off2 = ind_idx as usize * 4;
        let ind_block =
            u32::from_le_bytes([buf[off2], buf[off2 + 1], buf[off2 + 2], buf[off2 + 3]]);
        if ind_block == 0 {
            return Ok(0);
        }

        self.read_disk_block(ind_block, &mut buf)?;
        let block_idx = file_block % ptrs_per_block;
        let off3 = block_idx as usize * 4;
        let block = u32::from_le_bytes([buf[off3], buf[off3 + 1], buf[off3 + 2], buf[off3 + 3]]);
        Ok(block)
    }

    fn read_inode_data(
        &self,
        inode: &Inode,
        mut offset: u64,
        mut buf: &mut [u8],
    ) -> Result<usize, VfsError> {
        let size = inode.i_size as u64;
        let total_size = if (inode.i_mode & EXT2_S_IFMT) == EXT2_S_IFREG
            && self
                .superblock
                .as_ref()
                .map(|sb| sb.s_rev_level >= 1)
                .unwrap_or(false)
        {
            size | ((inode.i_dir_acl as u64) << 32)
        } else {
            size
        };

        // `total_size` is disk-controlled (directly for `i_size`, and via
        // the `i_dir_acl` high-bits extension for large regular files) and
        // is not otherwise bounded before it drives `file_block` below.
        // A corrupted, oversized `total_size` combined with a caller offset
        // still inside it pushes `file_block` past the real capacity of
        // `get_disk_block_from_inode`'s triple-indirect block-pointer
        // arrays, producing an out-of-bounds `buf[off]` index there —
        // `panic = "abort"` in this kernel's profiles turns that into a
        // halt. Reject it here, before any block lookup, using the same
        // capacity ceiling `checked_dir_size` already applies to directory
        // sizes.
        self.checked_size(total_size)?;

        if offset >= total_size {
            return Ok(0);
        }
        let mut read_bytes = 0;

        let mut temp_buf = alloc::vec![0u8; self.block_size as usize];

        while !buf.is_empty() && offset < total_size {
            let file_block = (offset / self.block_size as u64) as u32;
            let block_offset = (offset % self.block_size as u64) as usize;

            let mut read_len = self.block_size as usize - block_offset;
            if read_len > buf.len() {
                read_len = buf.len();
            }
            if offset + read_len as u64 > total_size {
                read_len = (total_size - offset) as usize;
            }

            let disk_block = self.get_disk_block_from_inode(inode, file_block)?;
            if disk_block == 0 {
                buf[..read_len].fill(0);
            } else {
                self.read_disk_block(disk_block, &mut temp_buf)?;
                buf[..read_len].copy_from_slice(&temp_buf[block_offset..block_offset + read_len]);
            }

            let temp = buf;
            buf = &mut temp[read_len..];
            offset += read_len as u64;
            read_bytes += read_len;
        }

        Ok(read_bytes)
    }

    /// Bound a raw byte size against what the mounted filesystem can
    /// physically hold, before that size is used to size an allocation or
    /// drive index arithmetic over on-disk block-pointer arrays.
    ///
    /// Shared by `checked_dir_size` (directory `i_size`, straight off disk
    /// via `read_inode`) and `read_inode_data` (regular-file `total_size`,
    /// including the `i_dir_acl`-extended high bits) so the ceiling is
    /// defined and can be adjusted in exactly one place. A prior fix to
    /// this file added this bound for directories only; a second,
    /// independently-derived copy for regular files is the failure mode
    /// this branch has hit before — the two would drift the moment one of
    /// them changed and the other did not.
    ///
    /// The ceiling is the total byte capacity of the mounted volume:
    /// `block_size` (derived from superblock field `s_log_block_size`,
    /// `mount()` above) times `s_blocks_count` (total block count, also
    /// read from the superblock). No single inode can legitimately hold
    /// more data than the whole filesystem, so a size beyond that is
    /// definitionally corrupt. Fails closed: out-of-range returns an error
    /// instead of allocating, and instead of silently truncating the read
    /// (which would turn corruption into wrong-but-plausible data).
    fn checked_size(&self, size: u64) -> Result<usize, VfsError> {
        let sb = self.superblock.as_ref().ok_or(VfsError::IoError)?;
        let ceiling = (self.block_size as u64) * (sb.s_blocks_count as u64);
        if size > ceiling {
            return Err(VfsError::IoError);
        }
        Ok(size as usize)
    }

    /// Bound a directory inode's `i_size` against filesystem capacity — see
    /// `checked_size` for the ceiling and the rationale. `i_size` is read
    /// straight off disk (`read_inode`) and is therefore untrusted: a
    /// corrupted or crafted image can set a directory inode's `i_size` to
    /// e.g. `0xFFFF_FFF0` (~4 GiB). Every directory-entry parser used to do
    /// `alloc::vec![0u8; inode.i_size as usize]` with no upper bound, which
    /// reaches the allocator before a single byte is read from disk.
    /// `no_std` seal-os has no `#[alloc_error_handler]`, so an allocation
    /// failure aborts the kernel outright rather than returning an error —
    /// a single `ls` of a bit-rotted directory halts the machine.
    fn checked_dir_size(&self, inode: &Inode) -> Result<usize, VfsError> {
        self.checked_size(inode.i_size as u64)
    }

    /// Resolve or allocate the indirect block referenced by `inode.i_block[idx]`.
    fn get_or_allocate_indirect_block(
        &mut self,
        inode: &mut Inode,
        idx: usize,
        block_incr: u32,
    ) -> Result<u32, VfsError> {
        if inode.i_block[idx] == 0 {
            let new_block = self.allocate_block()?;
            inode.i_block[idx] = new_block;
            inode.i_blocks += block_incr;
            self.zero_block_in_cache(new_block)?;
        }
        Ok(inode.i_block[idx])
    }

    /// Resolve or allocate the disk block for `file_block` within `inode`.
    ///
    /// * Direct blocks (0–11): i_block[n]
    /// * Single indirect (12): i_block[12] → array of data pointers
    /// * Double indirect (13): i_block[13] → array of indirect blocks → data pointers
    fn get_or_allocate_data_block(
        &mut self,
        inode: &mut Inode,
        file_block: u32,
        block_incr: u32,
    ) -> Result<u32, VfsError> {
        let ptrs_per_block = self.block_size / 4;

        // Direct blocks.
        if file_block < 12 {
            if inode.i_block[file_block as usize] == 0 {
                let new_block = self.allocate_block()?;
                inode.i_block[file_block as usize] = new_block;
                inode.i_blocks += block_incr;
                self.zero_block_in_cache(new_block)?;
            }
            return Ok(inode.i_block[file_block as usize]);
        }

        // Single indirect.
        let mut rel = file_block - 12;
        if rel < ptrs_per_block {
            let ind_block = self.get_or_allocate_indirect_block(inode, 12, block_incr)?;
            let mut cache = self.buffer_cache.lock();
            let cached = cache
                .get_block(self.dev_num, ind_block as u64)
                .ok_or(VfsError::IoError)?;
            let off = rel as usize * 4;
            let mut data_block = u32::from_le_bytes([
                cached.data()[off],
                cached.data()[off + 1],
                cached.data()[off + 2],
                cached.data()[off + 3],
            ]);
            drop(cache);
            if data_block == 0 {
                data_block = self.allocate_block()?;
                self.zero_block_in_cache(data_block)?;
                let mut cache = self.buffer_cache.lock();
                let cached = cache
                    .get_block(self.dev_num, ind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off..off + 4].copy_from_slice(&data_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
                inode.i_blocks += block_incr;
            }
            return Ok(data_block);
        }

        // Double indirect.
        rel -= ptrs_per_block;
        if rel < ptrs_per_block * ptrs_per_block {
            let dind_block = self.get_or_allocate_indirect_block(inode, 13, block_incr)?;

            // Read double-indirect block to find the indirect block index.
            let mut cache = self.buffer_cache.lock();
            let cached = cache
                .get_block(self.dev_num, dind_block as u64)
                .ok_or(VfsError::IoError)?;
            let ind_idx = rel / ptrs_per_block;
            let off = ind_idx as usize * 4;
            let mut ind_block = u32::from_le_bytes([
                cached.data()[off],
                cached.data()[off + 1],
                cached.data()[off + 2],
                cached.data()[off + 3],
            ]);
            drop(cache);

            if ind_block == 0 {
                ind_block = self.allocate_block()?;
                self.zero_block_in_cache(ind_block)?;
                let mut cache = self.buffer_cache.lock();
                let cached = cache
                    .get_block(self.dev_num, dind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off..off + 4].copy_from_slice(&ind_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
            }

            let block_idx = rel % ptrs_per_block;
            let mut cache = self.buffer_cache.lock();
            let cached = cache
                .get_block(self.dev_num, ind_block as u64)
                .ok_or(VfsError::IoError)?;
            let off2 = block_idx as usize * 4;
            let mut data_block = u32::from_le_bytes([
                cached.data()[off2],
                cached.data()[off2 + 1],
                cached.data()[off2 + 2],
                cached.data()[off2 + 3],
            ]);
            drop(cache);

            if data_block == 0 {
                data_block = self.allocate_block()?;
                self.zero_block_in_cache(data_block)?;
                let mut cache = self.buffer_cache.lock();
                let cached = cache
                    .get_block(self.dev_num, ind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off2..off2 + 4].copy_from_slice(&data_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
                inode.i_blocks += block_incr;
            }
            return Ok(data_block);
        }

        // Triple indirect.
        rel -= ptrs_per_block * ptrs_per_block;
        // A `u32` cube wraps to `0` on every block size >= 8192 and
        // permanently disables this branch; `triple_indirect_capacity`
        // computes it in `u64`, and is shared with the matching read bound in
        // `get_disk_block_from_inode` so the two cannot disagree.
        if (rel as u64) < Self::triple_indirect_capacity(ptrs_per_block) {
            let tind_block = self.get_or_allocate_indirect_block(inode, 14, block_incr)?;

            let mut cache = self.buffer_cache.lock();
            let cached = cache
                .get_block(self.dev_num, tind_block as u64)
                .ok_or(VfsError::IoError)?;
            let dind_idx = rel / (ptrs_per_block * ptrs_per_block);
            let off = dind_idx as usize * 4;
            let mut dind_block = u32::from_le_bytes([
                cached.data()[off],
                cached.data()[off + 1],
                cached.data()[off + 2],
                cached.data()[off + 3],
            ]);
            drop(cache);

            if dind_block == 0 {
                dind_block = self.allocate_block()?;
                self.zero_block_in_cache(dind_block)?;
                let mut cache = self.buffer_cache.lock();
                let cached = cache
                    .get_block(self.dev_num, tind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off..off + 4].copy_from_slice(&dind_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
            }

            let mut cache = self.buffer_cache.lock();
            let cached = cache
                .get_block(self.dev_num, dind_block as u64)
                .ok_or(VfsError::IoError)?;
            let ind_idx = (rel % (ptrs_per_block * ptrs_per_block)) / ptrs_per_block;
            let off2 = ind_idx as usize * 4;
            let mut ind_block = u32::from_le_bytes([
                cached.data()[off2],
                cached.data()[off2 + 1],
                cached.data()[off2 + 2],
                cached.data()[off2 + 3],
            ]);
            drop(cache);

            if ind_block == 0 {
                ind_block = self.allocate_block()?;
                self.zero_block_in_cache(ind_block)?;
                let mut cache = self.buffer_cache.lock();
                let cached = cache
                    .get_block(self.dev_num, dind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off2..off2 + 4].copy_from_slice(&ind_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
            }

            let block_idx = rel % ptrs_per_block;
            let mut cache = self.buffer_cache.lock();
            let cached = cache
                .get_block(self.dev_num, ind_block as u64)
                .ok_or(VfsError::IoError)?;
            let off3 = block_idx as usize * 4;
            let mut data_block = u32::from_le_bytes([
                cached.data()[off3],
                cached.data()[off3 + 1],
                cached.data()[off3 + 2],
                cached.data()[off3 + 3],
            ]);
            drop(cache);

            if data_block == 0 {
                data_block = self.allocate_block()?;
                self.zero_block_in_cache(data_block)?;
                let mut cache = self.buffer_cache.lock();
                let cached = cache
                    .get_block(self.dev_num, ind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off3..off3 + 4].copy_from_slice(&data_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
                inode.i_blocks += block_incr;
            }
            return Ok(data_block);
        }

        Err(VfsError::IoError)
    }

    fn write_inode_data(
        &mut self,
        inode: &mut Inode,
        mut offset: u64,
        buf: &[u8],
    ) -> Result<usize, VfsError> {
        let mut written = 0;
        let block_incr = self.block_size / 512;

        while written < buf.len() {
            let file_block = (offset / self.block_size as u64) as u32;
            let block_offset = (offset % self.block_size as u64) as usize;
            let write_len =
                core::cmp::min(buf.len() - written, self.block_size as usize - block_offset);

            let disk_block = self.get_or_allocate_data_block(inode, file_block, block_incr)?;

            {
                let mut cache = self.buffer_cache.lock();
                let cached = cache
                    .get_block(self.dev_num, disk_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[block_offset..block_offset + write_len]
                    .copy_from_slice(&buf[written..written + write_len]);
                cached.mark_dirty();
            }

            offset += write_len as u64;
            written += write_len;
        }

        Ok(written)
    }

    /// Scan block group descriptor bitmaps for a free block.
    pub fn allocate_block(&mut self) -> Result<u32, VfsError> {
        let sb = self.superblock.ok_or(VfsError::IoError)?;
        let total_groups = sb.s_blocks_count.div_ceil(sb.s_blocks_per_group);

        for group in 0..total_groups {
            let bgd = self.read_bgd(group)?;
            if bgd.bg_free_blocks_count == 0 {
                continue;
            }

            let mut buf = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(bgd.bg_block_bitmap, &mut buf)?;

            for byte_idx in 0..buf.len() {
                if buf[byte_idx] == 0xFF {
                    continue;
                }
                for bit in 0..8 {
                    if (buf[byte_idx] & (1 << bit)) == 0 {
                        buf[byte_idx] |= 1 << bit;
                        self.write_disk_block(bgd.bg_block_bitmap, &buf)?;

                        let mut new_bgd = bgd;
                        new_bgd.bg_free_blocks_count -= 1;
                        self.write_bgd(group, &new_bgd)?;

                        let block_num = group * sb.s_blocks_per_group
                            + byte_idx as u32 * 8
                            + bit
                            + sb.s_first_data_block;
                        return Ok(block_num);
                    }
                }
            }
        }
        Err(VfsError::IoError)
    }

    /// Clear a bit in the block bitmap.
    pub fn free_block(&mut self, block: u32) -> Result<(), VfsError> {
        let sb = self.superblock.ok_or(VfsError::IoError)?;
        let group = (block - sb.s_first_data_block) / sb.s_blocks_per_group;
        let local = (block - sb.s_first_data_block) % sb.s_blocks_per_group;
        let byte_idx = (local / 8) as usize;
        let bit = (local % 8) as usize;

        let bgd = self.read_bgd(group)?;
        let mut buf = alloc::vec![0u8; self.block_size as usize];
        self.read_disk_block(bgd.bg_block_bitmap, &mut buf)?;
        buf[byte_idx] &= !(1 << bit);
        self.write_disk_block(bgd.bg_block_bitmap, &buf)?;

        let mut new_bgd = bgd;
        new_bgd.bg_free_blocks_count += 1;
        self.write_bgd(group, &new_bgd)
    }

    /// Scan inode bitmaps for a free inode.
    pub fn allocate_inode(&mut self) -> Result<u32, VfsError> {
        let sb = self.superblock.ok_or(VfsError::IoError)?;
        let total_groups = sb.s_inodes_count.div_ceil(sb.s_inodes_per_group);

        for group in 0..total_groups {
            let bgd = self.read_bgd(group)?;
            if bgd.bg_free_inodes_count == 0 {
                continue;
            }

            let mut buf = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(bgd.bg_inode_bitmap, &mut buf)?;

            for byte_idx in 0..buf.len() {
                if buf[byte_idx] == 0xFF {
                    continue;
                }
                for bit in 0..8 {
                    if (buf[byte_idx] & (1 << bit)) == 0 {
                        buf[byte_idx] |= 1 << bit;
                        self.write_disk_block(bgd.bg_inode_bitmap, &buf)?;

                        let mut new_bgd = bgd;
                        new_bgd.bg_free_inodes_count -= 1;
                        self.write_bgd(group, &new_bgd)?;

                        let ino = group * sb.s_inodes_per_group + byte_idx as u32 * 8 + bit + 1;
                        return Ok(ino);
                    }
                }
            }
        }
        Err(VfsError::IoError)
    }

    /// Clear a bit in the inode bitmap.
    pub fn free_inode(&mut self, ino: u32) -> Result<(), VfsError> {
        let sb = self.superblock.ok_or(VfsError::IoError)?;
        let group = (ino - 1) / sb.s_inodes_per_group;
        let local = (ino - 1) % sb.s_inodes_per_group;
        let byte_idx = (local / 8) as usize;
        let bit = (local % 8) as usize;

        let bgd = self.read_bgd(group)?;
        let mut buf = alloc::vec![0u8; self.block_size as usize];
        self.read_disk_block(bgd.bg_inode_bitmap, &mut buf)?;
        buf[byte_idx] &= !(1 << bit);
        self.write_disk_block(bgd.bg_inode_bitmap, &buf)?;

        let mut new_bgd = bgd;
        new_bgd.bg_free_inodes_count += 1;
        self.write_bgd(group, &new_bgd)
    }

    /// Recursively free all data blocks attached to an inode, including indirect
    /// and double-indirect blocks.
    fn free_inode_blocks(&mut self, inode: &mut Inode) -> Result<(), VfsError> {
        let ptrs_per_block = self.block_size / 4;

        // Direct blocks.
        for i in 0..12 {
            if inode.i_block[i] != 0 {
                self.free_block(inode.i_block[i])?;
                inode.i_block[i] = 0;
            }
        }

        // Single indirect.
        if inode.i_block[12] != 0 {
            let mut indirect = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(inode.i_block[12], &mut indirect)?;
            for j in 0..ptrs_per_block {
                let off = j as usize * 4;
                let block = u32::from_le_bytes([
                    indirect[off],
                    indirect[off + 1],
                    indirect[off + 2],
                    indirect[off + 3],
                ]);
                if block != 0 {
                    self.free_block(block)?;
                }
            }
            self.free_block(inode.i_block[12])?;
            inode.i_block[12] = 0;
        }

        // Double indirect.
        if inode.i_block[13] != 0 {
            let mut dindirect = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(inode.i_block[13], &mut dindirect)?;
            for j in 0..ptrs_per_block {
                let off = j as usize * 4;
                let ind_block = u32::from_le_bytes([
                    dindirect[off],
                    dindirect[off + 1],
                    dindirect[off + 2],
                    dindirect[off + 3],
                ]);
                if ind_block != 0 {
                    let mut indirect = alloc::vec![0u8; self.block_size as usize];
                    self.read_disk_block(ind_block, &mut indirect)?;
                    for k in 0..ptrs_per_block {
                        let off2 = k as usize * 4;
                        let block = u32::from_le_bytes([
                            indirect[off2],
                            indirect[off2 + 1],
                            indirect[off2 + 2],
                            indirect[off2 + 3],
                        ]);
                        if block != 0 {
                            self.free_block(block)?;
                        }
                    }
                    self.free_block(ind_block)?;
                }
            }
            self.free_block(inode.i_block[13])?;
            inode.i_block[13] = 0;
        }

        inode.i_blocks = 0;
        Ok(())
    }

    fn dir_entry_size(name_len: usize) -> usize {
        let size = core::mem::size_of::<DirEntryHeader>() + name_len;
        (size + 3) & !3
    }

    fn split_last(path: &str) -> (&str, &str) {
        let path = path.trim_matches('/');
        match path.rfind('/') {
            Some(idx) => {
                let parent = &path[..idx];
                let name = &path[idx + 1..];
                if parent.is_empty() {
                    ("", name)
                } else {
                    (parent, name)
                }
            }
            None => ("", path),
        }
    }

    fn find_dir_entry(&self, dir_ino: u32, name: &str) -> Result<Option<u32>, VfsError> {
        let inode = self.read_inode(dir_ino)?;
        let size = self.checked_dir_size(&inode)?;
        if size == 0 {
            return Ok(None);
        }
        let mut dir_data = alloc::vec![0u8; size];
        self.read_inode_data(&inode, 0, &mut dir_data)?;

        let mut offset = 0;
        while offset + core::mem::size_of::<DirEntryHeader>() <= dir_data.len() {
            let header = unsafe {
                core::ptr::read_unaligned(dir_data.as_ptr().add(offset) as *const DirEntryHeader)
            };
            if header.inode != 0 {
                let name_start = offset + core::mem::size_of::<DirEntryHeader>();
                let name_end = name_start + header.name_len as usize;
                if name_end <= dir_data.len() {
                    let entry_name =
                        core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
                    if entry_name == name {
                        return Ok(Some(header.inode));
                    }
                }
            }
            if header.rec_len == 0 {
                break;
            }
            offset += header.rec_len as usize;
        }
        Ok(None)
    }

    fn add_dir_entry(
        &mut self,
        dir_ino: u32,
        name: &str,
        ino: u32,
        file_type: u8,
    ) -> Result<(), VfsError> {
        if name.len() > 255 {
            return Err(VfsError::InvalidPath);
        }

        let mut dir_inode = self.read_inode(dir_ino)?;
        let needed = Self::dir_entry_size(name.len());
        let dir_size = self.checked_dir_size(&dir_inode)?;

        let mut dir_data = alloc::vec![0u8; dir_size];
        if dir_size > 0 {
            self.read_inode_data(&dir_inode, 0, &mut dir_data)?;
        }

        // Reuse a deleted entry if it has enough space.
        let mut offset = 0;
        while offset + core::mem::size_of::<DirEntryHeader>() <= dir_data.len() {
            let header = unsafe {
                core::ptr::read_unaligned(dir_data.as_ptr().add(offset) as *const DirEntryHeader)
            };
            if header.inode == 0 && header.rec_len as usize >= needed {
                let new_header = DirEntryHeader {
                    inode: ino,
                    rec_len: header.rec_len,
                    name_len: name.len() as u8,
                    file_type,
                };
                unsafe {
                    core::ptr::write_unaligned(
                        dir_data.as_mut_ptr().add(offset) as *mut DirEntryHeader,
                        new_header,
                    );
                }
                let name_start = offset + core::mem::size_of::<DirEntryHeader>();
                dir_data[name_start..name_start + name.len()].copy_from_slice(name.as_bytes());
                self.write_inode_data(&mut dir_inode, 0, &dir_data)?;
                self.write_inode(dir_ino, &dir_inode)?;
                return Ok(());
            }
            if header.rec_len == 0 {
                break;
            }
            offset += header.rec_len as usize;
        }

        // Try to split the last entry in the last block.
        if dir_size > 0 {
            let last_block_start =
                ((dir_size - 1) / self.block_size as usize) * self.block_size as usize;
            let mut last_offset = last_block_start;
            let mut last_header = DirEntryHeader {
                inode: 0,
                rec_len: 0,
                name_len: 0,
                file_type: 0,
            };

            while last_offset + core::mem::size_of::<DirEntryHeader>() <= dir_data.len()
                && last_offset < last_block_start + self.block_size as usize
            {
                let h = unsafe {
                    core::ptr::read_unaligned(
                        dir_data.as_ptr().add(last_offset) as *const DirEntryHeader
                    )
                };
                if h.rec_len == 0 {
                    break;
                }
                let next_offset = last_offset + h.rec_len as usize;
                if next_offset >= last_block_start + self.block_size as usize
                    || next_offset >= dir_data.len()
                {
                    last_header = h;
                    break;
                }
                last_offset = next_offset;
            }

            if last_header.inode != 0 {
                let actual = Self::dir_entry_size(last_header.name_len as usize);
                if last_header.rec_len as usize >= actual + needed {
                    let new_rec_len = last_header.rec_len - actual as u16;
                    let mut updated_header = last_header;
                    updated_header.rec_len = actual as u16;
                    unsafe {
                        core::ptr::write_unaligned(
                            dir_data.as_mut_ptr().add(last_offset) as *mut DirEntryHeader,
                            updated_header,
                        );
                    }

                    let new_offset = last_offset + actual;
                    let new_header = DirEntryHeader {
                        inode: ino,
                        rec_len: new_rec_len,
                        name_len: name.len() as u8,
                        file_type,
                    };
                    unsafe {
                        core::ptr::write_unaligned(
                            dir_data.as_mut_ptr().add(new_offset) as *mut DirEntryHeader,
                            new_header,
                        );
                    }
                    let name_start = new_offset + core::mem::size_of::<DirEntryHeader>();
                    dir_data[name_start..name_start + name.len()].copy_from_slice(name.as_bytes());
                    self.write_inode_data(&mut dir_inode, 0, &dir_data)?;
                    self.write_inode(dir_ino, &dir_inode)?;
                    return Ok(());
                }
            }
        }

        // Allocate a new block via the general path (supports indirect).
        let block_idx = dir_size as u32 / self.block_size;
        let phys_block =
            self.get_or_allocate_data_block(&mut dir_inode, block_idx, self.block_size / 512)?;
        dir_inode.i_size += self.block_size;
        dir_inode.i_blocks += self.block_size / 512;

        let mut block_buf = alloc::vec![0u8; self.block_size as usize];
        let header = DirEntryHeader {
            inode: ino,
            rec_len: self.block_size as u16,
            name_len: name.len() as u8,
            file_type,
        };
        unsafe {
            core::ptr::write_unaligned(block_buf.as_mut_ptr() as *mut DirEntryHeader, header);
        }
        let name_start = core::mem::size_of::<DirEntryHeader>();
        block_buf[name_start..name_start + name.len()].copy_from_slice(name.as_bytes());
        self.write_disk_block(phys_block, &block_buf)?;
        self.write_inode(dir_ino, &dir_inode)
    }

    fn remove_dir_entry(&mut self, dir_ino: u32, name: &str) -> Result<(), VfsError> {
        let mut dir_inode = self.read_inode(dir_ino)?;
        let size = self.checked_dir_size(&dir_inode)?;
        let mut dir_data = alloc::vec![0u8; size];
        self.read_inode_data(&dir_inode, 0, &mut dir_data)?;

        let mut offset = 0;
        while offset + core::mem::size_of::<DirEntryHeader>() <= dir_data.len() {
            let header = unsafe {
                core::ptr::read_unaligned(dir_data.as_ptr().add(offset) as *const DirEntryHeader)
            };
            if header.inode != 0 {
                let name_start = offset + core::mem::size_of::<DirEntryHeader>();
                let name_end = name_start + header.name_len as usize;
                if name_end <= dir_data.len() {
                    let entry_name =
                        core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
                    if entry_name == name {
                        let mut new_header = header;
                        new_header.inode = 0;
                        unsafe {
                            core::ptr::write_unaligned(
                                dir_data.as_mut_ptr().add(offset) as *mut DirEntryHeader,
                                new_header,
                            );
                        }
                        self.write_inode_data(&mut dir_inode, 0, &dir_data)?;
                        self.write_inode(dir_ino, &dir_inode)?;
                        return Ok(());
                    }
                }
            }
            if header.rec_len == 0 {
                break;
            }
            offset += header.rec_len as usize;
        }
        Err(VfsError::NotFound)
    }
}

impl FileSystem for Ext2Fs {
    fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let mut current_ino = 2; // ROOT_INO is 2
        let path = path.trim_matches('/');
        if path.is_empty() {
            return Ok(VfsHandle {
                fs_idx: 0,
                inode: current_ino as u64,
            });
        }

        for component in path.split('/') {
            if component.is_empty() {
                continue;
            }

            let inode = self.read_inode(current_ino)?;
            if (inode.i_mode & EXT2_S_IFMT) != EXT2_S_IFDIR {
                return Err(VfsError::NotADirectory);
            }

            let dir_size = self.checked_dir_size(&inode)?;
            let mut dir_data = alloc::vec![0u8; dir_size];
            let read_len = self.read_inode_data(&inode, 0, &mut dir_data)?;
            if read_len != dir_size {
                return Err(VfsError::IoError);
            }

            let mut offset = 0;
            let mut found = false;
            while offset < dir_data.len() {
                if offset + core::mem::size_of::<DirEntryHeader>() > dir_data.len() {
                    break;
                }

                let header = unsafe {
                    core::ptr::read_unaligned(dir_data.as_ptr().add(offset) as *const DirEntryHeader)
                };

                if header.inode != 0 {
                    let name_start = offset + core::mem::size_of::<DirEntryHeader>();
                    let name_end = name_start + header.name_len as usize;
                    if name_end <= dir_data.len() {
                        let name =
                            core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
                        if name == component {
                            current_ino = header.inode;
                            found = true;
                            break;
                        }
                    }
                }

                if header.rec_len == 0 {
                    break;
                }
                offset += header.rec_len as usize;
            }

            if !found {
                return Err(VfsError::NotFound);
            }
        }

        Ok(VfsHandle {
            fs_idx: 0,
            inode: current_ino as u64,
        })
    }

    fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> {
        let inode = self.read_inode(handle.inode as u32)?;
        if (inode.i_mode & EXT2_S_IFMT) == EXT2_S_IFDIR {
            return Err(VfsError::InvalidOperation);
        }
        self.read_inode_data(&inode, offset, buf)
    }

    fn write(&mut self, handle: VfsHandle, buf: &[u8], offset: u64) -> Result<usize, VfsError> {
        let mut inode = self.read_inode(handle.inode as u32)?;
        if (inode.i_mode & EXT2_S_IFMT) == EXT2_S_IFDIR {
            return Err(VfsError::InvalidOperation);
        }

        let end = offset + buf.len() as u64;
        // 48 KiB limit REMOVED — indirect blocks are now supported.

        let written = self.write_inode_data(&mut inode, offset, buf)?;

        if end > inode.i_size as u64 {
            inode.i_size = end as u32;
        }
        inode.i_mtime = crate::drivers::interrupts::ticks() as u32;
        self.write_inode(handle.inode as u32, &inode)?;
        Ok(written)
    }

    fn create(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let (parent_path, name) = Self::split_last(path);
        if name.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let parent_ino = self.lookup(parent_path)?.inode as u32;

        if self.find_dir_entry(parent_ino, name)?.is_some() {
            return Err(VfsError::AlreadyExists);
        }

        let ino = self.allocate_inode()?;
        let ticks = crate::drivers::interrupts::ticks() as u32;
        let inode = Inode {
            i_mode: 0o100644 | EXT2_S_IFREG,
            i_uid: 0,
            i_size: 0,
            i_atime: ticks,
            i_ctime: ticks,
            i_mtime: ticks,
            i_dtime: 0,
            i_gid: 0,
            i_links_count: 1,
            i_blocks: 0,
            i_flags: 0,
            i_osd1: 0,
            i_block: [0; 15],
            i_generation: 0,
            i_file_acl: 0,
            i_dir_acl: 0,
            i_faddr: 0,
            i_osd2: [0; 3],
        };
        self.write_inode(ino, &inode)?;
        self.add_dir_entry(parent_ino, name, ino, 1)?;
        Ok(VfsHandle {
            fs_idx: 0,
            inode: ino as u64,
        })
    }

    fn mkdir(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let (parent_path, name) = Self::split_last(path);
        if name.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let parent_ino = self.lookup(parent_path)?.inode as u32;

        if self.find_dir_entry(parent_ino, name)?.is_some() {
            return Err(VfsError::AlreadyExists);
        }

        let ino = self.allocate_inode()?;
        let block = self.allocate_block()?;
        let ticks = crate::drivers::interrupts::ticks() as u32;

        let mut inode = Inode {
            i_mode: 0o040755 | EXT2_S_IFDIR,
            i_uid: 0,
            i_size: self.block_size,
            i_atime: ticks,
            i_ctime: ticks,
            i_mtime: ticks,
            i_dtime: 0,
            i_gid: 0,
            i_links_count: 2,
            i_blocks: self.block_size / 512,
            i_flags: 0,
            i_osd1: 0,
            i_block: [0; 15],
            i_generation: 0,
            i_file_acl: 0,
            i_dir_acl: 0,
            i_faddr: 0,
            i_osd2: [0; 3],
        };
        inode.i_block[0] = block;
        self.write_inode(ino, &inode)?;

        let mut block_buf = alloc::vec![0u8; self.block_size as usize];
        let dot_size = Self::dir_entry_size(1);
        let dotdot_offset = dot_size;

        let dot_header = DirEntryHeader {
            inode: ino,
            rec_len: dot_size as u16,
            name_len: 1,
            file_type: 2,
        };
        unsafe {
            core::ptr::write_unaligned(block_buf.as_mut_ptr() as *mut DirEntryHeader, dot_header);
        }
        block_buf[core::mem::size_of::<DirEntryHeader>()] = b'.';

        let dotdot_header = DirEntryHeader {
            inode: parent_ino,
            rec_len: (self.block_size as usize - dotdot_offset) as u16,
            name_len: 2,
            file_type: 2,
        };
        unsafe {
            core::ptr::write_unaligned(
                block_buf.as_mut_ptr().add(dotdot_offset) as *mut DirEntryHeader,
                dotdot_header,
            );
        }
        block_buf[dotdot_offset + core::mem::size_of::<DirEntryHeader>()] = b'.';
        block_buf[dotdot_offset + core::mem::size_of::<DirEntryHeader>() + 1] = b'.';

        self.write_disk_block(block, &block_buf)?;
        self.add_dir_entry(parent_ino, name, ino, 2)?;
        Ok(VfsHandle {
            fs_idx: 0,
            inode: ino as u64,
        })
    }

    fn unlink(&mut self, path: &str) -> Result<(), VfsError> {
        let (parent_path, name) = Self::split_last(path);
        if name.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let parent_ino = self.lookup(parent_path)?.inode as u32;
        let ino = self
            .find_dir_entry(parent_ino, name)?
            .ok_or(VfsError::NotFound)?;

        let mut inode = self.read_inode(ino)?;
        if (inode.i_mode & EXT2_S_IFMT) == EXT2_S_IFDIR {
            // The target *is* a directory; NotADirectory says the opposite.
            return Err(VfsError::InvalidOperation);
        }

        self.free_inode_blocks(&mut inode)?;
        self.remove_dir_entry(parent_ino, name)?;
        self.free_inode(ino)?;
        Ok(())
    }

    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        let inode = self.read_inode(handle.inode as u32)?;
        if (inode.i_mode & EXT2_S_IFMT) != EXT2_S_IFDIR {
            return Err(VfsError::NotADirectory);
        }

        let dir_size = self.checked_dir_size(&inode)?;
        let mut dir_data = alloc::vec![0u8; dir_size];
        let read_len = self.read_inode_data(&inode, 0, &mut dir_data)?;
        if read_len != dir_size {
            return Err(VfsError::IoError);
        }

        let mut entries = Vec::new();
        let mut offset = 0;

        while offset < dir_data.len() {
            if offset + core::mem::size_of::<DirEntryHeader>() > dir_data.len() {
                break;
            }

            let header = unsafe {
                core::ptr::read_unaligned(dir_data.as_ptr().add(offset) as *const DirEntryHeader)
            };

            if header.inode != 0 {
                let name_start = offset + core::mem::size_of::<DirEntryHeader>();
                let name_end = name_start + header.name_len as usize;
                if name_end <= dir_data.len() {
                    let name = core::str::from_utf8(&dir_data[name_start..name_end])
                        .unwrap_or("")
                        .into();

                    let node_type = match header.file_type {
                        1 => VfsNodeType::File,
                        2 => VfsNodeType::Directory,
                        3 => VfsNodeType::CharDevice,
                        4 => VfsNodeType::BlockDevice,
                        5 => VfsNodeType::File,
                        6 => VfsNodeType::File,
                        7 => VfsNodeType::Symlink,
                        _ => VfsNodeType::File,
                    };

                    entries.push(VfsDirEntry { name, node_type });
                }
            }

            if header.rec_len == 0 {
                break;
            }
            offset += header.rec_len as usize;
        }

        Ok(entries)
    }

    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        let inode = self.read_inode(handle.inode as u32)?;
        let node_type = match inode.i_mode & EXT2_S_IFMT {
            EXT2_S_IFREG => VfsNodeType::File,
            EXT2_S_IFDIR => VfsNodeType::Directory,
            EXT2_S_IFLNK => VfsNodeType::Symlink,
            EXT2_S_IFCHR => VfsNodeType::CharDevice,
            EXT2_S_IFBLK => VfsNodeType::BlockDevice,
            _ => VfsNodeType::File,
        };

        let size = inode.i_size as u64;
        let total_size = if node_type == VfsNodeType::File
            && self
                .superblock
                .as_ref()
                .map(|sb| sb.s_rev_level >= 1)
                .unwrap_or(false)
        {
            size | ((inode.i_dir_acl as u64) << 32)
        } else {
            size
        };

        let (major, minor) =
            if node_type == VfsNodeType::CharDevice || node_type == VfsNodeType::BlockDevice {
                let dev = inode.i_block[0];
                ((dev >> 8) & 0xFF, dev & 0xFF)
            } else {
                (0, 0)
            };

        Ok(VfsNode {
            size: total_size,
            permissions: inode.i_mode & 0xFFF,
            uid: inode.i_uid as u32,
            gid: inode.i_gid as u32,
            mode: inode.i_mode,
            atime: inode.i_atime as u64,
            mtime: inode.i_mtime as u64,
            node_type,
            major,
            minor,
        })
    }

    fn rmdir(&mut self, path: &str) -> Result<(), VfsError> {
        let (parent_path, name) = Self::split_last(path);
        if name.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let parent_ino = self.lookup(parent_path)?.inode as u32;
        let ino = self
            .find_dir_entry(parent_ino, name)?
            .ok_or(VfsError::NotFound)?;

        let mut inode = self.read_inode(ino)?;
        if (inode.i_mode & EXT2_S_IFMT) != EXT2_S_IFDIR {
            return Err(VfsError::NotADirectory);
        }

        let dir_size = self.checked_dir_size(&inode)?;
        let mut dir_data = alloc::vec![0u8; dir_size];
        self.read_inode_data(&inode, 0, &mut dir_data)?;

        let mut offset = 0;
        while offset < dir_data.len() {
            let header = unsafe {
                core::ptr::read_unaligned(dir_data.as_ptr().add(offset) as *const DirEntryHeader)
            };
            if header.inode != 0 {
                let name_start = offset + core::mem::size_of::<DirEntryHeader>();
                let name_end = name_start + header.name_len as usize;
                if name_end <= dir_data.len() {
                    let entry_name =
                        core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
                    if entry_name != "." && entry_name != ".." {
                        return Err(VfsError::InvalidOperation);
                    }
                }
            }
            if header.rec_len == 0 {
                break;
            }
            offset += header.rec_len as usize;
        }

        self.free_inode_blocks(&mut inode)?;
        self.remove_dir_entry(parent_ino, name)?;
        self.free_inode(ino)?;
        Ok(())
    }

    fn rename(&mut self, old: &str, new: &str) -> Result<(), VfsError> {
        let (old_parent, old_name) = Self::split_last(old);
        let (new_parent, new_name) = Self::split_last(new);

        let old_parent_ino = self.lookup(old_parent)?.inode as u32;
        let new_parent_ino = self.lookup(new_parent)?.inode as u32;

        if old_parent_ino == new_parent_ino && old_name == new_name {
            return Ok(());
        }

        if self.find_dir_entry(new_parent_ino, new_name)?.is_some() {
            return Err(VfsError::AlreadyExists);
        }

        // Scan old_parent for the entry.
        let mut old_dir_inode = self.read_inode(old_parent_ino)?;
        let old_dir_size = self.checked_dir_size(&old_dir_inode)?;
        let mut old_dir_data = alloc::vec![0u8; old_dir_size];
        self.read_inode_data(&old_dir_inode, 0, &mut old_dir_data)?;

        let mut old_offset = None;
        let mut old_rec_len = 0;
        let mut old_ino = 0;
        let mut old_file_type = 0;
        let mut offset = 0;

        while offset + core::mem::size_of::<DirEntryHeader>() <= old_dir_data.len() {
            let header = unsafe {
                core::ptr::read_unaligned(old_dir_data.as_ptr().add(offset) as *const DirEntryHeader)
            };
            if header.inode != 0 {
                let name_start = offset + core::mem::size_of::<DirEntryHeader>();
                let name_end = name_start + header.name_len as usize;
                if name_end <= old_dir_data.len() {
                    let entry_name =
                        core::str::from_utf8(&old_dir_data[name_start..name_end]).unwrap_or("");
                    if entry_name == old_name {
                        old_offset = Some(offset);
                        old_rec_len = header.rec_len;
                        old_ino = header.inode;
                        old_file_type = header.file_type;
                        break;
                    }
                }
            }
            if header.rec_len == 0 {
                break;
            }
            offset += header.rec_len as usize;
        }

        let old_offset = old_offset.ok_or(VfsError::NotFound)?;

        if old_parent_ino == new_parent_ino {
            // Same-directory rename: reuse existing logic.
            let needed = Self::dir_entry_size(new_name.len());
            if needed <= old_rec_len as usize {
                let new_header = DirEntryHeader {
                    inode: old_ino,
                    rec_len: old_rec_len,
                    name_len: new_name.len() as u8,
                    file_type: old_file_type,
                };
                unsafe {
                    core::ptr::write_unaligned(
                        old_dir_data.as_mut_ptr().add(old_offset) as *mut DirEntryHeader,
                        new_header,
                    );
                }
                let name_start = old_offset + core::mem::size_of::<DirEntryHeader>();
                for b in &mut old_dir_data[name_start..name_start + old_name.len()] {
                    *b = 0;
                }
                old_dir_data[name_start..name_start + new_name.len()]
                    .copy_from_slice(new_name.as_bytes());
                self.write_inode_data(&mut old_dir_inode, 0, &old_dir_data)?;
                self.write_inode(old_parent_ino, &old_dir_inode)?;
            } else {
                self.remove_dir_entry(old_parent_ino, old_name)?;
                self.add_dir_entry(old_parent_ino, new_name, old_ino, old_file_type)?;
            }
        } else {
            // Cross-directory rename: remove from old, add to new.
            self.remove_dir_entry(old_parent_ino, old_name)?;
            self.add_dir_entry(new_parent_ino, new_name, old_ino, old_file_type)?;

            // If moving a directory, update its ".." entry to point to new_parent.
            let mut moved_inode = self.read_inode(old_ino)?;
            if (moved_inode.i_mode & EXT2_S_IFMT) == EXT2_S_IFDIR {
                let dir_size = self.checked_dir_size(&moved_inode)?;
                let mut dir_data = alloc::vec![0u8; dir_size];
                self.read_inode_data(&moved_inode, 0, &mut dir_data)?;
                let mut off = 0;
                while off + core::mem::size_of::<DirEntryHeader>() <= dir_data.len() {
                    let h = unsafe {
                        core::ptr::read_unaligned(
                            dir_data.as_ptr().add(off) as *const DirEntryHeader
                        )
                    };
                    if h.inode != 0 {
                        let ns = off + core::mem::size_of::<DirEntryHeader>();
                        let ne = ns + h.name_len as usize;
                        if ne <= dir_data.len() {
                            let en = core::str::from_utf8(&dir_data[ns..ne]).unwrap_or("");
                            if en == ".." {
                                let mut nh = h;
                                nh.inode = new_parent_ino;
                                unsafe {
                                    core::ptr::write_unaligned(
                                        dir_data.as_mut_ptr().add(off) as *mut DirEntryHeader,
                                        nh,
                                    );
                                }
                                self.write_inode_data(&mut moved_inode, 0, &dir_data)?;
                                self.write_inode(old_ino, &moved_inode)?;
                                // Update link counts: old_parent loses one, new_parent gains one.
                                let mut op = self.read_inode(old_parent_ino)?;
                                op.i_links_count = op.i_links_count.saturating_sub(1);
                                self.write_inode(old_parent_ino, &op)?;
                                let mut np = self.read_inode(new_parent_ino)?;
                                np.i_links_count = np.i_links_count.saturating_add(1);
                                self.write_inode(new_parent_ino, &np)?;
                                break;
                            }
                        }
                    }
                    if h.rec_len == 0 {
                        break;
                    }
                    off += h.rec_len as usize;
                }
            }
        }

        Ok(())
    }

    fn mknod(
        &mut self,
        path: &str,
        node_type: VfsNodeType,
        major: u32,
        minor: u32,
    ) -> Result<VfsHandle, VfsError> {
        let (parent_path, name) = Self::split_last(path);
        if name.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        if matches!(node_type, VfsNodeType::Directory | VfsNodeType::Symlink) {
            return Err(VfsError::InvalidOperation);
        }
        let parent_ino = self.lookup(parent_path)?.inode as u32;

        if self.find_dir_entry(parent_ino, name)?.is_some() {
            return Err(VfsError::AlreadyExists);
        }

        let ino = self.allocate_inode()?;
        let mode = match node_type {
            VfsNodeType::CharDevice => EXT2_S_IFCHR,
            VfsNodeType::BlockDevice => EXT2_S_IFBLK,
            VfsNodeType::Pipe => EXT2_S_IFIFO,
            _ => EXT2_S_IFREG,
        } | 0o666;

        let ticks = crate::drivers::interrupts::ticks() as u32;
        let mut inode = Inode {
            i_mode: mode,
            i_uid: 0,
            i_size: 0,
            i_atime: ticks,
            i_ctime: ticks,
            i_mtime: ticks,
            i_dtime: 0,
            i_gid: 0,
            i_links_count: 1,
            i_blocks: 0,
            i_flags: 0,
            i_osd1: 0,
            i_block: [0; 15],
            i_generation: 0,
            i_file_acl: 0,
            i_dir_acl: 0,
            i_faddr: 0,
            i_osd2: [0; 3],
        };

        if node_type == VfsNodeType::CharDevice || node_type == VfsNodeType::BlockDevice {
            let dev = ((major & 0xFF) << 8) | (minor & 0xFF);
            inode.i_block[0] = dev;
        }

        self.write_inode(ino, &inode)?;

        let file_type = match node_type {
            VfsNodeType::Directory => 2,
            VfsNodeType::CharDevice => 3,
            VfsNodeType::BlockDevice => 4,
            VfsNodeType::Symlink => 7,
            VfsNodeType::Pipe => 5, // FIFO
            _ => 1,
        };

        self.add_dir_entry(parent_ino, name, ino, file_type)?;
        Ok(VfsHandle {
            fs_idx: 0,
            inode: ino as u64,
        })
    }

    fn sync(&mut self) -> Result<(), VfsError> {
        // T4: flushing the buffer cache is the ext2 metadata/data sync path.
        self.buffer_cache.lock().sync();
        Ok(())
    }

    fn fsync(&mut self, _handle: VfsHandle) -> Result<(), VfsError> {
        // Ext2 does not track per-inode dirty cache state;
        // fsync flushes the entire buffer cache for durability.
        self.buffer_cache.lock().sync();
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::test_assert;

    /// Device extent large enough that the device bound cannot be what a test
    /// is observing. Tests that isolate one non-device superblock field pass
    /// this, so a failure there is unambiguously about that field.
    const ANY_DEVICE: u64 = u64::MAX;

    /// A superblock describing a legal filesystem of `blocks_count` blocks at
    /// `1024 << log_block_size` bytes each. Each test mutates the one field it
    /// is about, so every other field has to be in range or the test observes
    /// the wrong rejection.
    ///
    /// `s_inodes_per_group` and `s_inodes_count` used to default to `0`, and
    /// `validate_superblock` has rejected `s_inodes_per_group == 0` since
    /// commit ea6e899 — so `test_valid_log_block_size_is_accepted` (all six
    /// `s_log_block_size` values) and `test_inode_size_ignored_when_rev_level_0`
    /// have been asserting `is_ok()` against a superblock this file refuses.
    /// Nothing caught it: `kernel/seal-os` is outside the root workspace, its
    /// `#[cfg(test)]` functions never compile, and the in-kernel harness these
    /// are registered with has never executed in this project's history. The
    /// assertions are unchanged; the fixture they run against is now a
    /// filesystem that can actually exist.
    fn make_superblock(blocks_count: u32, log_block_size: u32) -> Superblock {
        Superblock {
            s_inodes_count: 128,
            s_blocks_count: blocks_count,
            s_r_blocks_count: 0,
            s_free_blocks_count: 0,
            s_free_inodes_count: 0,
            s_first_data_block: 1,
            s_log_block_size: log_block_size,
            s_log_frag_size: 0,
            s_blocks_per_group: 8192,
            s_frags_per_group: 8192,
            s_inodes_per_group: 128,
            s_mtime: 0,
            s_wtime: 0,
            s_mnt_count: 0,
            s_max_mnt_count: 0,
            s_magic: EXT2_MAGIC,
            s_state: 0,
            s_errors: 0,
            s_minor_rev_level: 0,
            s_lastcheck: 0,
            s_checkinterval: 0,
            s_creator_os: 0,
            s_rev_level: 0,
            s_def_resuid: 0,
            s_def_resgid: 0,
            s_first_ino: 0,
            s_inode_size: 128,
            s_block_group_nr: 0,
            s_feature_compat: 0,
            s_feature_incompat: 0,
            s_feature_ro_compat: 0,
            s_uuid: [0; 16],
            s_volume_name: [0; 16],
            s_last_mounted: [0; 64],
            s_algo_bitmap: 0,
            padding: [0; 788],
        }
    }

    /// Build a mounted-in-memory `Ext2Fs` (no disk I/O) whose superblock
    /// describes a filesystem of `blocks_count` blocks at `1024 << log_block_size`
    /// bytes each — enough state for `checked_dir_size` to compute its ceiling.
    fn make_fs(blocks_count: u32, log_block_size: u32) -> Ext2Fs {
        let block_size = 1024u32 << log_block_size;
        Ext2Fs {
            dev_num: 0,
            superblock: Some(make_superblock(blocks_count, log_block_size)),
            block_size,
            bgd_block: 0,
            inodes_per_group: 0,
            inode_size: 128,
            buffer_cache: Mutex::new(BufferCache::new(1, block_size as usize)),
        }
    }

    fn make_dir_inode(i_size: u32) -> Inode {
        Inode {
            i_mode: EXT2_S_IFDIR,
            i_uid: 0,
            i_size,
            i_atime: 0,
            i_ctime: 0,
            i_mtime: 0,
            i_dtime: 0,
            i_gid: 0,
            i_links_count: 0,
            i_blocks: 0,
            i_flags: 0,
            i_osd1: 0,
            i_block: [0; 15],
            i_generation: 0,
            i_file_acl: 0,
            i_dir_acl: 0,
            i_faddr: 0,
            i_osd2: [0; 3],
        }
    }

    /// RED: the defect this closes. Every directory-entry parser
    /// (`find_dir_entry`, `add_dir_entry`, `remove_dir_entry`, `lookup`,
    /// `readdir`, `rmdir`, `rename`) used to do
    /// `alloc::vec![0u8; inode.i_size as usize]` with `i_size` read straight
    /// off disk and no upper bound. A filesystem with 4096 1 KiB blocks
    /// (~4 MiB real capacity) but a directory inode corrupted to claim
    /// `i_size = 0xFFFF_FFF0` (~4 GiB) would reach that allocation before a
    /// single byte is read from disk — and this `no_std` kernel registers no
    /// `#[alloc_error_handler]`, so the allocation failure aborts rather than
    /// returning an error. `checked_dir_size` must reject that inode instead
    /// of handing back a size to allocate.
    fn test_oversized_dir_i_size_is_rejected() -> TestResult {
        let fs = make_fs(4096, 0); // 4096 * 1 KiB = 4 MiB filesystem capacity
        let bad_inode = make_dir_inode(0xFFFF_FFF0);
        let result = fs.checked_dir_size(&bad_inode);
        test_assert!(
            result.is_err(),
            "i_size beyond filesystem capacity must be rejected, not allocated"
        );
        TestResult::Pass
    }

    /// GREEN: a directory inode whose `i_size` fits inside the filesystem's
    /// real capacity is legitimate and must still be accepted, unchanged.
    fn test_in_bounds_dir_i_size_is_accepted() -> TestResult {
        let fs = make_fs(4096, 0); // 4 MiB filesystem capacity
        let good_inode = make_dir_inode(4096); // 4 KiB directory, well inside 4 MiB
        let result = fs.checked_dir_size(&good_inode);
        test_assert!(result.is_ok(), "in-bounds i_size must be accepted");
        test_assert!(
            result.unwrap() == 4096,
            "checked size must match i_size when in bounds"
        );
        TestResult::Pass
    }

    /// RED: the defect this closes. `checked_dir_size` bounds directory
    /// `i_size` only, by its own doc comment and its eight call sites, all
    /// directory-entry parsers. `read_inode_data` — reached from
    /// `FileSystem::read` for *regular* files — never bounded `i_size`
    /// against filesystem capacity at all. A corrupted regular-file inode
    /// claiming a huge size, combined with a caller-chosen offset inside
    /// that claimed size, drives `get_disk_block_from_inode`'s
    /// triple-indirect branch to compute `dind_idx`/`ind_idx` past the real
    /// `ptrs_per_block` entries of the block, producing an out-of-bounds
    /// `buf[off]` index — and this kernel builds `panic = "abort"`, so
    /// that halts the machine. `offset = 0` alone is enough to observe the
    /// gap without ever reaching that dangerous index arithmetic: pre-fix,
    /// `read_inode_data` treated `offset (0) < total_size (huge)` as
    /// license to proceed and silently returned zeroed "hole" bytes for a
    /// size no real disk of this capacity could hold, instead of refusing
    /// the read.
    fn test_oversized_regular_file_size_is_rejected() -> TestResult {
        let fs = make_fs(4096, 0); // 4096 * 1 KiB = 4 MiB filesystem capacity
        let mut inode = make_dir_inode(0xFFFF_FFF0); // ~4 GiB claimed size
        inode.i_mode = EXT2_S_IFREG;
        let mut buf = [0u8; 16];
        let result = fs.read_inode_data(&inode, 0, &mut buf);
        test_assert!(
            result.is_err(),
            "a regular-file i_size beyond filesystem capacity must be rejected before any block lookup, not read as zeros"
        );
        TestResult::Pass
    }

    /// GREEN: a regular-file inode whose size fits inside the filesystem's
    /// real capacity must still read normally — unallocated (hole) blocks
    /// read back as zero, unchanged from before this fix.
    fn test_in_bounds_regular_file_size_is_accepted() -> TestResult {
        let fs = make_fs(4096, 0); // 4 MiB filesystem capacity
        let mut inode = make_dir_inode(16); // 16-byte file, well inside 4 MiB
        inode.i_mode = EXT2_S_IFREG;
        let mut buf = [0xAAu8; 16];
        let result = fs.read_inode_data(&inode, 0, &mut buf);
        test_assert!(result.is_ok(), "in-bounds regular-file size must still be read");
        test_assert!(
            result.unwrap() == 16,
            "must read the full requested length within i_size"
        );
        test_assert!(buf == [0u8; 16], "unallocated (hole) blocks must read as zero");
        TestResult::Pass
    }

    /// RED: the defect this closes. `get_or_allocate_data_block`'s
    /// triple-indirect capacity guard is
    /// `rel < ptrs_per_block * ptrs_per_block * ptrs_per_block`, computed
    /// in `u32`. `ptrs_per_block` is `block_size / 4`; for block sizes
    /// 8192, 16384, and 32768 (`s_log_block_size` 3..=5, all accepted by
    /// `validate_superblock`), `ptrs_per_block` is a power of two >= 2048,
    /// so its cube is an exact multiple of `2^32` and — with neither Cargo
    /// profile in this crate setting `overflow-checks` — wraps to exactly
    /// `0` in release. The guard becomes `rel < 0`, false for every
    /// unsigned `rel`, so even `rel == 0` (the very first triple-indirect
    /// data block) is rejected: the branch is permanently unreachable and
    /// the write path returns `Err(IoError)` for any offset needing it.
    /// Widening the same product to `u64` — which cannot wrap for any
    /// `ptrs_per_block` this file can produce (max `8192`, cube `~5.5e11`,
    /// far under `u64::MAX`) — restores it.
    fn test_triple_indirect_guard_reachable_after_widening() -> TestResult {
        for &log_block_size in &[3u32, 4, 5] {
            let block_size = 1024u32 << log_block_size;
            let ptrs_per_block = block_size / 4;
            let rel: u32 = 0; // the very first triple-indirect data block

            // Pre-fix arithmetic: the exact `u32` product this file used to compute.
            let wrapped_bound = ptrs_per_block
                .wrapping_mul(ptrs_per_block)
                .wrapping_mul(ptrs_per_block);
            test_assert!(
                wrapped_bound == 0,
                "regression precondition: ptrs_per_block^3 must wrap to 0 in u32 for this test to mean anything"
            );
            test_assert!(
                !(rel < wrapped_bound),
                "RED: the unfixed u32 guard rejects even rel == 0, proving the branch was unreachable"
            );

            // Post-fix arithmetic: the shipped function itself, not a copy of
            // its expression — a local copy would keep passing if the file's
            // own computation narrowed back to `u32`.
            let fixed_bound = Ext2Fs::triple_indirect_capacity(ptrs_per_block);
            test_assert!(
                (rel as u64) < fixed_bound,
                "GREEN: the widened guard accepts rel == 0, restoring the triple-indirect branch"
            );
        }
        TestResult::Pass
    }

    /// GREEN: the widened guard must still reject a `rel` past the true
    /// triple-indirect capacity, not merely become unconditionally true —
    /// the fix must preserve the intended bound, not just avoid the wrap.
    /// Uses the smallest block size this file accepts (1024,
    /// `ptrs_per_block` 256, cube `16_777_216` — ext2's real maximum
    /// triple-indirect block count for 1 KiB blocks), where the true
    /// boundary is small enough to exceed while still fitting in a `u32`
    /// `rel`.
    fn test_triple_indirect_guard_still_bounded() -> TestResult {
        let ptrs_per_block = 1024u32 / 4; // 256
        let cube = Ext2Fs::triple_indirect_capacity(ptrs_per_block);
        test_assert!(
            cube == 16_777_216,
            "known-value sanity check on the intended bound itself"
        );

        let in_range: u64 = cube - 1;
        let out_of_range: u64 = cube;
        test_assert!(
            in_range < cube,
            "the last legal triple-indirect index must stay in bounds"
        );
        test_assert!(
            !(out_of_range < cube),
            "the fix must still reject a rel at the true capacity boundary, not just avoid the u32 wrap"
        );
        TestResult::Pass
    }

    /// RED: the defect this closes. `get_or_allocate_data_block` (write) has
    /// carried a triple-indirect capacity guard since the branch was written;
    /// `get_disk_block_from_inode` (read) never had one. Its triple-indirect
    /// branch is the fall-through, so `file_block` arrives unbounded and
    /// `dind_idx = file_block / ptrs_per_block^2` can exceed `ptrs_per_block`,
    /// indexing `buf[off]` past the end of the block just read — a panic, and
    /// both Cargo profiles here set `panic = "abort"`.
    ///
    /// The `checked_size` bound added for regular files does not close this:
    /// its ceiling is the *volume's* capacity, and on any volume with more
    /// blocks than `12 + 256 + 256^2 + 256^3 = 16_843_020` (about 16 GiB at
    /// the 1024-byte block size) a size well inside that ceiling still names a
    /// file block no `i_block` entry can address. Reproduced on a real mounted
    /// image, 20,000,000 blocks of 1024 bytes, in the host harness for this
    /// file:
    ///
    /// ```text
    /// thread 'hosttests::unaddressable_file_block_is_refused_not_indexed' panicked at
    /// kernel/seal-os/src/fs/ext2.rs:506:49:
    /// index out of bounds: the len is 1024 but the index is 1024
    /// ```
    ///
    /// This in-kernel form needs no block device: with `i_block[14] == 0` the
    /// unfixed function returned `Ok(0)` — "hole", fabricated zero data for a
    /// block the inode cannot reach — before touching the disk, and the guard
    /// is placed ahead of that pointer read so the refusal is observable here.
    fn test_unaddressable_file_block_is_refused() -> TestResult {
        let fs = make_fs(4096, 0); // 1024-byte blocks, 256 pointers per block
        let mut inode = make_dir_inode(0);
        inode.i_mode = EXT2_S_IFREG;

        let ptrs_per_block = 1024u32 / 4;
        let first_unaddressable = 12
            + ptrs_per_block
            + ptrs_per_block * ptrs_per_block
            + (Ext2Fs::triple_indirect_capacity(ptrs_per_block) as u32);
        test_assert!(
            first_unaddressable == 16_843_020,
            "known-value sanity check on the capacity of 15 block pointers at 1024-byte blocks"
        );

        let result = fs.get_disk_block_from_inode(&inode, first_unaddressable);
        test_assert!(
            result.is_err(),
            "a file block past triple-indirect capacity must be refused, not indexed and not reported as a hole"
        );
        TestResult::Pass
    }

    /// GREEN: the last block the pointers *do* reach must still resolve — the
    /// guard is a capacity bound, not a blanket refusal of the triple-indirect
    /// branch, and an off-by-one either way shows up here. `i_block[14] == 0`,
    /// so a legitimately sparse file reads back as a hole exactly as before.
    fn test_last_addressable_file_block_still_resolves() -> TestResult {
        let fs = make_fs(4096, 0);
        let mut inode = make_dir_inode(0);
        inode.i_mode = EXT2_S_IFREG;

        let ptrs_per_block = 1024u32 / 4;
        let last_addressable = 11
            + ptrs_per_block
            + ptrs_per_block * ptrs_per_block
            + (Ext2Fs::triple_indirect_capacity(ptrs_per_block) as u32);
        test_assert!(
            last_addressable == 16_843_019,
            "known-value sanity check on the last addressable file block"
        );

        let result = fs.get_disk_block_from_inode(&inode, last_addressable);
        test_assert!(
            matches!(result, Ok(0)),
            "the last addressable file block must still resolve, as a hole when i_block[14] is unset"
        );
        TestResult::Pass
    }

    /// RED: the defect this closes. `mount()` (ext2.rs) read `s_log_block_size`
    /// straight off disk and only validated the magic number:
    ///
    /// ```text
    /// let sb = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const Superblock) };
    /// if sb.s_magic != EXT2_MAGIC {
    ///     return Err(VfsError::IoError);
    /// }
    /// self.superblock = Some(sb);
    /// self.block_size = 1024 << sb.s_log_block_size;
    /// ```
    ///
    /// `1024u32 << 22 == 0`: `1024` is `2^10`, so the shift computes
    /// `2^32`, which overflows `u32` and — with neither `[profile.dev]` nor
    /// `[profile.release]` in Cargo.toml setting `overflow-checks` — wraps
    /// to `0` in release. `self.block_size == 0` then reaches its first
    /// divisor at `read_bgd` (`self.block_size / size_of::<BlockGroupDescriptor>()`,
    /// ext2.rs `read_bgd`), which panics on integer division by zero
    /// unconditionally (not gated by overflow-checks), and both Cargo
    /// profiles here set `panic = "abort"` — so mounting a filesystem with
    /// a corrupted or hostile `s_log_block_size` would halt the kernel on
    /// the very first block-group read.
    ///
    /// This is a static assertion, not an executed mount: `mount()` needs a
    /// live block device, which this in-kernel harness has none of. Instead
    /// this exercises `validate_superblock`, the exact function `mount()`
    /// now calls to compute `block_size` before any of that divisor code
    /// runs — no QEMU, no disk I/O, this file's own arithmetic only.
    fn test_oversized_log_block_size_is_rejected() -> TestResult {
        // The contradicting arithmetic itself: this is what the unfixed
        // `mount()` did before handing block_size to every divisor site.
        // Routed through `black_box` so the shift amount isn't visible to
        // the constant folder — otherwise clippy's `eq_op` (denied in this
        // crate) flags the fully-folded `0 == 0` as comparing a value to
        // itself, even though the two sides are written differently and
        // the point is exactly to demonstrate that they collapse.
        let n = core::hint::black_box(22u32);
        test_assert!(
            1024u32 << n == 0,
            "regression precondition: 1024 << 22 must wrap to 0 for this test to mean anything"
        );

        let sb = make_superblock(4096, 22); // s_log_block_size = 22
        let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
        test_assert!(
            result.is_err(),
            "s_log_block_size = 22 must be refused at mount, not yield block_size == 0"
        );
        TestResult::Pass
    }

    /// A previous reviewer mis-described `s_log_block_size >= 32` as
    /// undefined behaviour. It is not: the shift amount is masked to
    /// `n % 32` by `<<`'s defined semantics (`unchecked_shl` would be UB;
    /// `<<` does not lower to it). `1024u32 << 32` therefore behaves as
    /// `1024u32 << 0 == 1024` — a *wrong* block size, not a zero one, and
    /// still rejected here on correctness grounds rather than UB.
    fn test_shift_overflow_log_block_size_is_rejected() -> TestResult {
        let sb = make_superblock(4096, 32); // s_log_block_size = 32
        let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
        test_assert!(
            result.is_err(),
            "s_log_block_size = 32 must be refused, even though the masked shift is defined, not UB"
        );
        TestResult::Pass
    }

    /// GREEN: every `s_log_block_size` ext2 actually mounts with (0..=6 per
    /// spec, narrowed to 0..=5 here — see `validate_superblock`'s doc
    /// comment for why 6 is rejected too) must still be accepted, and must
    /// compute the exact `1024 << n` block size.
    fn test_valid_log_block_size_is_accepted() -> TestResult {
        for n in 0..=5u32 {
            let sb = make_superblock(4096, n);
            let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
            test_assert!(result.is_ok(), "in-range s_log_block_size must mount");
            test_assert!(
                result.unwrap() == 1024u32 << n,
                "block_size must equal 1024 << s_log_block_size when in range"
            );
        }
        TestResult::Pass
    }

    /// The narrower-than-spec ceiling: `add_dir_entry`'s new-block path
    /// writes `rec_len: self.block_size as u16`. `65536u32 as u16 == 0`,
    /// which every dir-entry walker in this file reads as end-of-directory
    /// — so `s_log_block_size == 6` (block_size 65536), legal per the ext2
    /// spec, must still be refused by this file's own `validate_superblock`.
    fn test_spec_legal_but_file_unsafe_log_block_size_is_rejected() -> TestResult {
        test_assert!(
            65536u32 as u16 == 0,
            "regression precondition: block_size 65536 must truncate to 0 as u16"
        );
        let sb = make_superblock(4096, 6); // s_log_block_size = 6, block_size = 65536
        let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
        test_assert!(
            result.is_err(),
            "s_log_block_size = 6 must be refused: block_size 65536 truncates to 0 as u16 in add_dir_entry's rec_len"
        );
        TestResult::Pass
    }

    /// `s_inodes_per_group == 0` divides by zero at `read_inode`/
    /// `write_inode` (`(ino - 1) / self.inodes_per_group`) and at
    /// `allocate_inode`/`free_inode`. Must be refused at mount instead of
    /// panicking on the first inode lookup.
    fn test_zero_inodes_per_group_is_rejected() -> TestResult {
        let mut sb = make_superblock(4096, 2);
        sb.s_inodes_per_group = 0;
        let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
        test_assert!(result.is_err(), "s_inodes_per_group = 0 must be refused");
        TestResult::Pass
    }

    /// `s_blocks_per_group == 0` divides by zero at `allocate_block`
    /// (`s_blocks_count.div_ceil(s_blocks_per_group)`) and at `free_block`
    /// (`(block - s_first_data_block) / s_blocks_per_group`). Same defect
    /// class as `s_inodes_per_group`, found by auditing every divisor site
    /// in this file rather than from the field checklist this fix started
    /// from.
    fn test_zero_blocks_per_group_is_rejected() -> TestResult {
        let mut sb = make_superblock(4096, 2);
        sb.s_blocks_per_group = 0;
        let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
        test_assert!(result.is_err(), "s_blocks_per_group = 0 must be refused");
        TestResult::Pass
    }

    /// `s_inode_size` (only read when `s_rev_level >= 1`) is a divisor via
    /// `inodes_per_block = self.block_size / self.inode_size` in
    /// `read_inode`/`write_inode`. Zero divides directly; a value larger
    /// than `block_size` floors `inodes_per_block` to `0`, which is then
    /// itself used as a divisor two lines later — an indirect divide by
    /// zero either way.
    fn test_invalid_inode_size_is_rejected_when_rev_level_ge_1() -> TestResult {
        let mut sb = make_superblock(4096, 2); // block_size = 4096
        sb.s_rev_level = 1;

        sb.s_inode_size = 0;
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_err(),
            "s_inode_size = 0 must be refused when s_rev_level >= 1"
        );

        sb.s_inode_size = 8192; // > block_size (4096)
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_err(),
            "s_inode_size > block_size must be refused: it floors inodes_per_block to 0"
        );

        // Below `size_of::<Inode>()` the divisions all succeed and the damage
        // moves to the raw read they scale: `read_inode` places the last
        // inode of a block at `(block_size / inode_size - 1) * inode_size`
        // and reads 128 bytes from there. At `s_inode_size = 64` that is
        // offset 4032 of a 4096-byte block, so the last 32 bytes come from
        // past the end of the buffer — through `read_unaligned`, which will
        // not trap on the way out.
        sb.s_inode_size = 64;
        test_assert!(
            (4096 / 64 - 1) * 64 + core::mem::size_of::<Inode>() > 4096,
            "regression precondition: an inode_size of 64 must overrun the block"
        );
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_err(),
            "s_inode_size < size_of::<Inode>() must be refused: read_inode reads 128 bytes past the block"
        );

        sb.s_inode_size = core::mem::size_of::<Inode>() as u16;
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_ok(),
            "the smallest inode size ext2 defines must still mount"
        );
        TestResult::Pass
    }

    /// `s_inode_size` is never read when `s_rev_level == 0` (`mount()`
    /// hardcodes `128` on that path), so an out-of-range value there must
    /// not block the mount.
    fn test_inode_size_ignored_when_rev_level_0() -> TestResult {
        let mut sb = make_superblock(4096, 2);
        sb.s_rev_level = 0;
        sb.s_inode_size = 0; // would be rejected if this field were read
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_ok(),
            "s_inode_size must be ignored on the rev_level 0 path"
        );
        TestResult::Pass
    }

    /// `s_first_data_block` is added un-checked into `bgd_block` in
    /// `mount()` (`sb.s_first_data_block + 1`); an out-of-volume value can
    /// wrap that addition with overflow-checks off. Bounding it to inside
    /// the volume (`< s_blocks_count`) is this file's own justification,
    /// independent of the wrap: `bgd_block` must reference a real block.
    fn test_first_data_block_out_of_range_is_rejected() -> TestResult {
        let mut sb = make_superblock(100, 2); // 100-block filesystem
        sb.s_first_data_block = 100; // == s_blocks_count: outside the volume
        let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
        test_assert!(
            result.is_err(),
            "s_first_data_block >= s_blocks_count must be refused"
        );
        TestResult::Pass
    }

    /// GREEN: a fully in-range superblock — the shape `mount()` accepts
    /// every day — must still mount cleanly through `validate_superblock`.
    fn test_ordinary_superblock_is_accepted() -> TestResult {
        let mut sb = make_superblock(4096, 2); // 4096 blocks, 4 KiB blocks
        sb.s_inodes_per_group = 1024;
        sb.s_blocks_per_group = 8192;
        sb.s_rev_level = 1;
        sb.s_inode_size = 256;
        sb.s_first_data_block = 0;
        let result = Ext2Fs::validate_superblock(&sb, ANY_DEVICE);
        test_assert!(result.is_ok(), "an ordinary in-range superblock must mount");
        test_assert!(
            result.unwrap() == 4096,
            "block_size must be 4096 for s_log_block_size = 2"
        );
        TestResult::Pass
    }

    /// RED: the defect this closes. `s_blocks_count` is read straight off
    /// disk and `mount()` compared it against nothing. It is the sole
    /// multiplicand of `checked_size`'s ceiling
    /// (`block_size * s_blocks_count`), which is the entire basis of the two
    /// bounds this file already carries — the directory `i_size` bound and
    /// the regular-file `total_size` bound added in 532e121. So an image on
    /// a 1 MiB device claiming `s_blocks_count = u32::MAX` does not merely
    /// lie about its size: it lifts that ceiling to 4 TiB and thereby
    /// re-opens both earlier fixes, letting a directory inode claiming
    /// `i_size = 0xFFFF_FFF0` through `checked_dir_size` and into
    /// `alloc::vec![0u8; 4 GiB]`. This `no_std` kernel registers no
    /// `#[alloc_error_handler]`, so that allocation failing aborts.
    /// Bounding the claim against the extent of the device it was read from
    /// makes the ceiling a physical fact again.
    fn test_blocks_count_beyond_device_is_rejected() -> TestResult {
        let device_sectors = 2048u64; // a 1 MiB device
        let mut sb = make_superblock(1024, 0); // 1 KiB blocks
        sb.s_blocks_count = u32::MAX;

        // What that claim does to the ceiling the earlier fixes rest on.
        let ceiling = 1024u64 * sb.s_blocks_count as u64;
        test_assert!(
            ceiling > device_sectors * 512,
            "regression precondition: the claimed capacity must exceed the device"
        );
        test_assert!(
            ceiling >= 0xFFFF_FFF0,
            "regression precondition: the inflated ceiling must clear a 4 GiB i_size"
        );

        test_assert!(
            Ext2Fs::validate_superblock(&sb, device_sectors).is_err(),
            "s_blocks_count beyond the device extent must be refused"
        );
        TestResult::Pass
    }

    /// GREEN: the ext2 parity fixture (`fs::parity::format_ext2`) — 1024
    /// blocks of 1 KiB on a 2048-sector `MemDisk`, exactly filling it — must
    /// still mount. This is the shape the `Verify FAT/ext2 parity proof` CI
    /// gate depends on, and it sits on the boundary the check uses, so an
    /// off-by-one in either direction shows up here.
    fn test_blocks_count_filling_the_device_is_accepted() -> TestResult {
        let sb = make_superblock(1024, 0);
        // Routed through `black_box` so the constant folder cannot collapse
        // this to `2048 == 2048`, which clippy's `eq_op` (denied in this
        // crate) reads as comparing a value to itself.
        let blocks = core::hint::black_box(1024u64);
        test_assert!(
            blocks * (1024 / 512) == 2048,
            "regression precondition: the fixture exactly fills its 2048-sector device"
        );
        test_assert!(
            Ext2Fs::validate_superblock(&sb, 2048).is_ok(),
            "a filesystem that exactly fills its device must mount"
        );
        test_assert!(
            Ext2Fs::validate_superblock(&sb, 2047).is_err(),
            "one sector short of the claim must be refused"
        );
        TestResult::Pass
    }

    /// `s_blocks_per_group` and `s_inodes_per_group` bound `local` in
    /// `free_block`/`free_inode`, and `local / 8` then indexes a `Vec` that
    /// is exactly one block long. Only `== 0` was rejected, so
    /// `s_blocks_per_group = 0x8000_0000` plus a corrupt `i_block` entry of
    /// `u32::MAX` indexes `buf[268435455]` of a 1024-byte buffer — a panic,
    /// and this kernel's profiles set `panic = "abort"`. One bitmap block
    /// holds `block_size * 8` bits, which is both ext2's real ceiling for a
    /// group and precisely the bound that keeps the index inside the buffer.
    fn test_oversized_group_counts_are_rejected() -> TestResult {
        // The index arithmetic `free_block` performs, at the largest value
        // the bound accepts, for every block size this file mounts.
        for log_block_size in 0..=5u32 {
            let block_size = 1024u32 << log_block_size;
            let local = (block_size * 8) - 1; // worst case under the bound
            test_assert!(
                local / 8 < block_size,
                "the accepted maximum must keep byte_idx inside the bitmap block"
            );
        }

        let mut sb = make_superblock(4096, 0); // block_size 1024, 8192 bits
        sb.s_blocks_per_group = 0x8000_0000;
        test_assert!(
            (u32::MAX - 1) % sb.s_blocks_per_group / 8 >= 1024,
            "regression precondition: this s_blocks_per_group must index past the block"
        );
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_err(),
            "s_blocks_per_group beyond one bitmap block must be refused"
        );

        let mut sb = make_superblock(4096, 0);
        sb.s_inodes_per_group = 8193; // one past 1024 * 8
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_err(),
            "s_inodes_per_group beyond one bitmap block must be refused"
        );

        let mut sb = make_superblock(4096, 0);
        sb.s_blocks_per_group = 8192; // exactly one bitmap block
        sb.s_inodes_per_group = 8192;
        sb.s_inodes_count = 8192;
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_ok(),
            "a group that exactly fills its bitmap block must still mount"
        );
        TestResult::Pass
    }

    /// `s_inodes_count` bounds `allocate_inode`'s group scan
    /// (`s_inodes_count.div_ceil(s_inodes_per_group)`) the same way
    /// `s_blocks_count` bounds `allocate_block`'s, and nothing checked it: on
    /// the one-group fixture, `u32::MAX` inodes makes that loop scan
    /// 33,554,432 groups, each iteration a `read_bgd` reading a block far
    /// outside the block-group descriptor table and interpreting whatever it
    /// finds as a descriptor. Tying it to the group count the block side
    /// yields — ext2's own invariant — bounds it by the now device-bounded
    /// block count.
    fn test_inodes_count_beyond_group_capacity_is_rejected() -> TestResult {
        let mut sb = make_superblock(1024, 0); // one group, 128 inodes in it
        sb.s_inodes_count = u32::MAX;
        test_assert!(
            sb.s_inodes_count.div_ceil(sb.s_inodes_per_group) == 33_554_432,
            "regression precondition: the unbounded count drives a 33554432-group scan"
        );
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_err(),
            "more inodes than the filesystem's groups can hold must be refused"
        );

        let mut sb = make_superblock(1024, 0);
        sb.s_inodes_count = sb.s_inodes_per_group; // exactly one group's worth
        test_assert!(
            Ext2Fs::validate_superblock(&sb, ANY_DEVICE).is_ok(),
            "an inode count the groups can hold must still mount"
        );
        TestResult::Pass
    }

    /// RED: the defect this closes. The last thing `mount()` does is
    /// `*self.buffer_cache.lock() = BufferCache::new(..)`, and `BufferCache`
    /// has no `Drop` impl (`fs/buffer_cache.rs`), so a second `mount()` drops
    /// every dirty buffer — inode table, bitmaps, directory blocks, file data
    /// — without writing any of it back, discarding writes that already
    /// returned `Ok` to their caller. Latent at HEAD: `fs::init_vfs`,
    /// `apps::installer`, and `fs::parity::measure` each build a fresh
    /// `Ext2Fs::new` and mount it once, so no in-tree caller reaches it. The
    /// guard is on `mount` rather than on those three because `mount` is
    /// `pub` on a `pub` struct.
    ///
    /// This runs the real `mount()`, not a stand-in: the guard is the first
    /// thing it does, so the refusal is observable with no block device
    /// attached — the `read_block` on the next line is never reached.
    fn test_second_mount_is_refused() -> TestResult {
        let mut fs = make_fs(4096, 0); // already carries a mounted superblock
        test_assert!(
            matches!(fs.mount(), Err(VfsError::InvalidOperation)),
            "mounting an already-mounted Ext2Fs must be refused, not silently discard the buffer cache"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "ext2::oversized_dir_i_size_is_rejected",
            test_oversized_dir_i_size_is_rejected,
        );
        crate::testing::register_test(
            "ext2::in_bounds_dir_i_size_is_accepted",
            test_in_bounds_dir_i_size_is_accepted,
        );
        crate::testing::register_test(
            "ext2::oversized_regular_file_size_is_rejected",
            test_oversized_regular_file_size_is_rejected,
        );
        crate::testing::register_test(
            "ext2::in_bounds_regular_file_size_is_accepted",
            test_in_bounds_regular_file_size_is_accepted,
        );
        crate::testing::register_test(
            "ext2::triple_indirect_guard_reachable_after_widening",
            test_triple_indirect_guard_reachable_after_widening,
        );
        crate::testing::register_test(
            "ext2::triple_indirect_guard_still_bounded",
            test_triple_indirect_guard_still_bounded,
        );
        crate::testing::register_test(
            "ext2::unaddressable_file_block_is_refused",
            test_unaddressable_file_block_is_refused,
        );
        crate::testing::register_test(
            "ext2::last_addressable_file_block_still_resolves",
            test_last_addressable_file_block_still_resolves,
        );
        crate::testing::register_test(
            "ext2::oversized_log_block_size_is_rejected",
            test_oversized_log_block_size_is_rejected,
        );
        crate::testing::register_test(
            "ext2::shift_overflow_log_block_size_is_rejected",
            test_shift_overflow_log_block_size_is_rejected,
        );
        crate::testing::register_test(
            "ext2::valid_log_block_size_is_accepted",
            test_valid_log_block_size_is_accepted,
        );
        crate::testing::register_test(
            "ext2::spec_legal_but_file_unsafe_log_block_size_is_rejected",
            test_spec_legal_but_file_unsafe_log_block_size_is_rejected,
        );
        crate::testing::register_test(
            "ext2::zero_inodes_per_group_is_rejected",
            test_zero_inodes_per_group_is_rejected,
        );
        crate::testing::register_test(
            "ext2::zero_blocks_per_group_is_rejected",
            test_zero_blocks_per_group_is_rejected,
        );
        crate::testing::register_test(
            "ext2::invalid_inode_size_is_rejected_when_rev_level_ge_1",
            test_invalid_inode_size_is_rejected_when_rev_level_ge_1,
        );
        crate::testing::register_test(
            "ext2::inode_size_ignored_when_rev_level_0",
            test_inode_size_ignored_when_rev_level_0,
        );
        crate::testing::register_test(
            "ext2::first_data_block_out_of_range_is_rejected",
            test_first_data_block_out_of_range_is_rejected,
        );
        crate::testing::register_test(
            "ext2::ordinary_superblock_is_accepted",
            test_ordinary_superblock_is_accepted,
        );
        crate::testing::register_test(
            "ext2::blocks_count_beyond_device_is_rejected",
            test_blocks_count_beyond_device_is_rejected,
        );
        crate::testing::register_test(
            "ext2::blocks_count_filling_the_device_is_accepted",
            test_blocks_count_filling_the_device_is_accepted,
        );
        crate::testing::register_test(
            "ext2::oversized_group_counts_are_rejected",
            test_oversized_group_counts_are_rejected,
        );
        crate::testing::register_test(
            "ext2::inodes_count_beyond_group_capacity_is_rejected",
            test_inodes_count_beyond_group_capacity_is_rejected,
        );
        crate::testing::register_test(
            "ext2::second_mount_is_refused",
            test_second_mount_is_refused,
        );
    }
}
