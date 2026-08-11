// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Persistence layer — superblock + inode table + bitmap allocator + WAL journal.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;

use super::encoder::{ManifoldPayload, SpherePoint};
use super::journal::{compute_manifold_embedding, TopologicalJournal};
use super::manifold_fs::{Inode, InodeKind, InodeMetadata};
use crate::drivers::block::{read_block, write_block, BlockError};
use core::convert::TryInto;
use spin::Mutex;

const SECTOR_SIZE: usize = 512;
const INODE_RECORD_SIZE: usize = 256;
const MAX_EXTENT_BLOCKS: u32 = 256;
const MAGIC: [u8; 4] = *b"MNFD";
const VERSION: u32 = 1;
const JOURNAL_BLOCKS: u64 = 1024;
const NONE_ID: u64 = u64::MAX;
const AHCI_MANIFOLD_START_LBA: u64 = 2048 + ((64 * 1024 * 1024) / SECTOR_SIZE as u64);
const JOURNAL_NAME_OFFSET: usize = 98;
const JOURNAL_CHECKSUM_OFFSET: usize = JOURNAL_NAME_OFFSET + 128;
const INODE_NAME_OFFSET: usize = 90;
const INODE_SIBLING_NEXT_OFFSET: usize = 224;
const INODE_SIBLING_PREV_OFFSET: usize = 232;
const INODE_DIR_FIRST_CHILD_OFFSET: usize = 240;
// The owner occupies bytes the format has never written and has never read:
// the last eight of the 256-byte inode record, and eight of the 282 that
// trail the journal entry's checksum, four-byte aligned. A disk written by a
// build that predates the owner holds zero there, and zero is root, so the
// on-disk version stays at 1 and every existing filesystem still mounts.
const INODE_UID_OFFSET: usize = 248;
const INODE_GID_OFFSET: usize = 252;
const JOURNAL_UID_OFFSET: usize = JOURNAL_CHECKSUM_OFFSET + 6;
const JOURNAL_GID_OFFSET: usize = JOURNAL_UID_OFFSET + 4;

fn put_u16(buf: &mut [u8], offset: usize, value: u16) {
    buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(buf: &mut [u8], offset: usize, value: i32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(buf: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        buf.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn get_u32(buf: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        buf.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn get_i32(buf: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_le_bytes(
        buf.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn get_u64(buf: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        buf.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

/// Trait abstracting over real AHCI and mock block devices.
pub trait BlockStoreBackend: Send + Sync {
    fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
    fn write_sector(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;
}

/// Adapter for the global block device registry (AHCI device 0x800).
pub struct AhciBackend;

impl BlockStoreBackend for AhciBackend {
    fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        read_block(0x800, AHCI_MANIFOLD_START_LBA + lba, buf)
    }
    fn write_sector(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        write_block(0x800, AHCI_MANIFOLD_START_LBA + lba, buf)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountError {
    NoDevice,
    NoSuperblock,
}

/// In-memory mock for host testing.
pub struct MockBackend {
    sectors: Mutex<Vec<[u8; SECTOR_SIZE]>>,
}

impl MockBackend {
    pub fn new(num_sectors: usize) -> Self {
        let mut sectors = Vec::with_capacity(num_sectors);
        for _ in 0..num_sectors {
            sectors.push([0u8; SECTOR_SIZE]);
        }
        Self {
            sectors: Mutex::new(sectors),
        }
    }
}

impl BlockStoreBackend for MockBackend {
    fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let sectors = self.sectors.lock();
        let idx = lba as usize;
        if idx >= sectors.len() || buf.len() != SECTOR_SIZE {
            return Err(BlockError::InvalidLba);
        }
        buf.copy_from_slice(&sectors[idx]);
        Ok(())
    }
    fn write_sector(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        let mut sectors = self.sectors.lock();
        let idx = lba as usize;
        if idx >= sectors.len() || buf.len() != SECTOR_SIZE {
            return Err(BlockError::InvalidLba);
        }
        sectors[idx].copy_from_slice(buf);
        Ok(())
    }
}

// ── Superblock ──────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Superblock {
    magic: [u8; 4],
    version: u32,
    block_size: u32,
    _pad1: [u8; 4],
    inode_count: u64,
    free_bitmap_start: u64,
    free_bitmap_blocks: u64,
    journal_start: u64,
    journal_blocks: u64,
    inode_table_start: u64,
    inode_table_blocks: u64,
    data_start: u64,
    total_blocks: u64,
    generation: u64,
    crc32: u32,
    _pad2: [u8; 412],
}

impl Superblock {
    fn new(total_blocks: u64) -> Self {
        let block_size = SECTOR_SIZE as u32;
        let journal_start = 1;
        let free_bitmap_start = journal_start + JOURNAL_BLOCKS;
        let free_bitmap_blocks = 256;
        let inode_table_start = free_bitmap_start + free_bitmap_blocks;
        let max_inodes = total_blocks.saturating_sub(inode_table_start);
        let inode_table_blocks =
            (max_inodes * INODE_RECORD_SIZE as u64).div_ceil(block_size as u64);
        let data_start = inode_table_start + inode_table_blocks;

        Self {
            magic: MAGIC,
            version: VERSION,
            block_size,
            _pad1: [0; 4],
            inode_count: max_inodes,
            free_bitmap_start,
            free_bitmap_blocks,
            journal_start,
            journal_blocks: JOURNAL_BLOCKS,
            inode_table_start,
            inode_table_blocks,
            data_start,
            total_blocks,
            generation: 1,
            crc32: 0,
            _pad2: [0; 412],
        }
    }

    fn valid(&self) -> bool {
        self.magic == MAGIC && self.version == VERSION && self.block_size as usize == SECTOR_SIZE
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: Superblock is repr(C) and exactly 512 bytes; any bit pattern is valid for byte slice.
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }

    fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < size_of::<Self>() {
            return None;
        }
        // SAFETY: buf has correct length; read_unaligned avoids alignment requirements.
        let sb = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const Self) };
        if sb.valid() {
            Some(sb)
        } else {
            None
        }
    }
}

// ── JournalEntry ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct JournalEntry {
    seq: u64,
    op: u8,
    committed: u8,
    _pad1: [u8; 6],
    inode_id: u64,
    parent: u64,
    kind: u8,
    _pad2: [u8; 7],
    data_lba: u64,
    data_blocks: u32,
    voronoi_cell: u32,
    original_size: u64,
    permissions: u16,
    _pad3: [u8; 6],
    created_ms: u64,
    modified_ms: u64,
    content_hash: u64,
    name_len: u16,
    name: [u8; 128],
    uid: u32,
    gid: u32,
    checksum: u32,
    _pad4: [u8; 282],
}

impl JournalEntry {
    /// The write-ahead entry for a metadata update, describing `rec` exactly as
    /// the inode table holds it. Every update entry is built here so that the
    /// journal and the inode table cannot disagree about a field — including
    /// the owner, which a remount replays over the table it just read.
    fn update_for(seq: u64, rec: &InodeRecord) -> Self {
        let mut entry = Self {
            seq,
            op: 1,
            committed: 0,
            _pad1: [0; 6],
            inode_id: rec.id,
            parent: rec.parent,
            kind: rec.kind,
            _pad2: [0; 7],
            data_lba: rec.data_lba,
            data_blocks: rec.data_blocks,
            voronoi_cell: rec.voronoi_cell,
            original_size: rec.original_size,
            permissions: rec.permissions,
            _pad3: [0; 6],
            created_ms: rec.created_ms,
            modified_ms: rec.modified_ms,
            content_hash: rec.content_hash,
            name_len: rec.name_len,
            name: rec.name,
            uid: rec.uid,
            gid: rec.gid,
            checksum: 0,
            _pad4: [0; 282],
        };
        entry.checksum = entry.checksum();
        entry
    }

    fn checksum(&self) -> u32 {
        let sector = self.encode_sector(0);
        Self::checksum_sector(&sector)
    }

    fn checksum_sector(sector: &[u8; SECTOR_SIZE]) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        for (idx, &b) in sector.iter().enumerate() {
            let b = if (JOURNAL_CHECKSUM_OFFSET..JOURNAL_CHECKSUM_OFFSET + 4).contains(&idx) {
                0
            } else {
                b
            };
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h
    }

    fn encode_sector(&self, checksum: u32) -> [u8; SECTOR_SIZE] {
        let mut sector = [0u8; SECTOR_SIZE];
        put_u64(&mut sector, 0, self.seq);
        sector[8] = self.op;
        sector[9] = self.committed;
        put_u64(&mut sector, 16, self.inode_id);
        put_u64(&mut sector, 24, self.parent);
        sector[32] = self.kind;
        put_u64(&mut sector, 40, self.data_lba);
        put_u32(&mut sector, 48, self.data_blocks);
        put_u32(&mut sector, 52, self.voronoi_cell);
        put_u64(&mut sector, 56, self.original_size);
        put_u16(&mut sector, 64, self.permissions);
        put_u64(&mut sector, 72, self.created_ms);
        put_u64(&mut sector, 80, self.modified_ms);
        put_u64(&mut sector, 88, self.content_hash);
        put_u16(&mut sector, 96, self.name_len);
        sector[JOURNAL_NAME_OFFSET..JOURNAL_NAME_OFFSET + 128].copy_from_slice(&self.name);
        put_u32(&mut sector, JOURNAL_UID_OFFSET, self.uid);
        put_u32(&mut sector, JOURNAL_GID_OFFSET, self.gid);
        put_u32(&mut sector, JOURNAL_CHECKSUM_OFFSET, checksum);
        sector
    }

    fn to_sector(self) -> [u8; SECTOR_SIZE] {
        let mut sector = self.encode_sector(0);
        let checksum = Self::checksum_sector(&sector);
        put_u32(&mut sector, JOURNAL_CHECKSUM_OFFSET, checksum);
        sector
    }

    fn from_sector(buf: &[u8]) -> Option<Self> {
        let sector: &[u8; SECTOR_SIZE] = buf.try_into().ok()?;
        let checksum = get_u32(sector, JOURNAL_CHECKSUM_OFFSET)?;
        if checksum != Self::checksum_sector(sector) {
            return None;
        }
        let mut name = [0u8; 128];
        name.copy_from_slice(&sector[JOURNAL_NAME_OFFSET..JOURNAL_NAME_OFFSET + 128]);
        Some(Self {
            seq: get_u64(sector, 0)?,
            op: sector[8],
            committed: sector[9],
            _pad1: [0; 6],
            inode_id: get_u64(sector, 16)?,
            parent: get_u64(sector, 24)?,
            kind: sector[32],
            _pad2: [0; 7],
            data_lba: get_u64(sector, 40)?,
            data_blocks: get_u32(sector, 48)?,
            voronoi_cell: get_u32(sector, 52)?,
            original_size: get_u64(sector, 56)?,
            permissions: get_u16(sector, 64)?,
            _pad3: [0; 6],
            created_ms: get_u64(sector, 72)?,
            modified_ms: get_u64(sector, 80)?,
            content_hash: get_u64(sector, 88)?,
            name_len: get_u16(sector, 96)?,
            name,
            uid: get_u32(sector, JOURNAL_UID_OFFSET)?,
            gid: get_u32(sector, JOURNAL_GID_OFFSET)?,
            checksum,
            _pad4: [0; 282],
        })
    }
}

// ── InodeRecord ─────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct InodeRecord {
    id: u64,
    kind: u8,
    _pad1: [u8; 7],
    parent: u64,
    data_lba: u64,
    data_blocks: u32,
    voronoi_cell: u32,
    cluster_id: i32,
    _pad2: [u8; 4],
    original_size: u64,
    permissions: u16,
    _pad3: [u8; 6],
    created_ms: u64,
    modified_ms: u64,
    content_hash: u64,
    name_len: u16,
    name: [u8; 128],
    sibling_next: u64,
    sibling_prev: u64,
    dir_first_child: u64,
    uid: u32,
    gid: u32,
}

impl InodeRecord {
    /// Build a record from `ino`. The owner is set to root, because `Inode`
    /// does not carry one: the record is where ownership lives, so a caller
    /// rebuilding a record for a node that already has one must carry it over
    /// from the old record — `BlockStore::write_inode` does.
    fn from_inode(ino: &Inode, data_lba: u64, data_blocks: u32) -> Self {
        let mut name = [0u8; 128];
        let name_bytes = ino.name.as_bytes();
        let name_len = name_bytes.len().min(128);
        name[..name_len].copy_from_slice(&name_bytes[..name_len]);
        Self {
            id: ino.id,
            kind: match ino.kind {
                InodeKind::File => 0,
                InodeKind::Directory => 1,
            },
            _pad1: [0; 7],
            parent: ino.parent,
            data_lba,
            data_blocks,
            voronoi_cell: ino.voronoi_cell as u32,
            cluster_id: ino.cluster_id,
            _pad2: [0; 4],
            original_size: ino.metadata.original_size,
            permissions: ino.metadata.permissions,
            _pad3: [0; 6],
            created_ms: ino.metadata.created_ms,
            modified_ms: ino.metadata.modified_ms,
            content_hash: ino.payload.content_hash,
            name_len: name_len as u16,
            name,
            sibling_next: ino.sibling_next.unwrap_or(NONE_ID),
            sibling_prev: ino.sibling_prev.unwrap_or(NONE_ID),
            dir_first_child: ino.dir_first_child.unwrap_or(NONE_ID),
            uid: 0,
            gid: 0,
        }
    }

    fn to_inode(self, payload: ManifoldPayload, data: Vec<u8>) -> Inode {
        let name_len = self.name_len.min(128) as usize;
        let name = String::from_utf8_lossy(&self.name[..name_len]).into_owned();
        Inode {
            id: self.id,
            name,
            kind: match self.kind {
                1 => InodeKind::Directory,
                _ => InodeKind::File,
            },
            payload,
            data,
            metadata: InodeMetadata {
                created_ms: self.created_ms,
                modified_ms: self.modified_ms,
                original_size: self.original_size,
                permissions: self.permissions,
            },
            voronoi_cell: self.voronoi_cell as usize,
            cluster_id: self.cluster_id,
            parent: self.parent,
            sibling_next: if self.sibling_next == NONE_ID {
                None
            } else {
                Some(self.sibling_next)
            },
            sibling_prev: if self.sibling_prev == NONE_ID {
                None
            } else {
                Some(self.sibling_prev)
            },
            dir_first_child: if self.dir_first_child == NONE_ID {
                None
            } else {
                Some(self.dir_first_child)
            },
        }
    }

    fn to_record(self) -> [u8; INODE_RECORD_SIZE] {
        let mut record = [0u8; INODE_RECORD_SIZE];
        put_u64(&mut record, 0, self.id);
        record[8] = self.kind;
        put_u64(&mut record, 16, self.parent);
        put_u64(&mut record, 24, self.data_lba);
        put_u32(&mut record, 32, self.data_blocks);
        put_u32(&mut record, 36, self.voronoi_cell);
        put_i32(&mut record, 40, self.cluster_id);
        put_u64(&mut record, 48, self.original_size);
        put_u16(&mut record, 56, self.permissions);
        put_u64(&mut record, 64, self.created_ms);
        put_u64(&mut record, 72, self.modified_ms);
        put_u64(&mut record, 80, self.content_hash);
        put_u16(&mut record, 88, self.name_len);
        record[INODE_NAME_OFFSET..INODE_NAME_OFFSET + 128].copy_from_slice(&self.name);
        put_u64(&mut record, INODE_SIBLING_NEXT_OFFSET, self.sibling_next);
        put_u64(&mut record, INODE_SIBLING_PREV_OFFSET, self.sibling_prev);
        put_u64(
            &mut record,
            INODE_DIR_FIRST_CHILD_OFFSET,
            self.dir_first_child,
        );
        put_u32(&mut record, INODE_UID_OFFSET, self.uid);
        put_u32(&mut record, INODE_GID_OFFSET, self.gid);
        record
    }

    fn from_record(buf: &[u8]) -> Option<Self> {
        if buf.len() < INODE_RECORD_SIZE {
            return None;
        }
        let id = get_u64(buf, 0)?;
        let kind = *buf.get(8)?;
        let name_len = get_u16(buf, 88)?;
        if kind > 1 || (id == 0 && name_len == 0) {
            return None;
        }
        let mut name = [0u8; 128];
        name.copy_from_slice(buf.get(INODE_NAME_OFFSET..INODE_NAME_OFFSET + 128)?);
        Some(Self {
            id,
            kind,
            _pad1: [0; 7],
            parent: get_u64(buf, 16)?,
            data_lba: get_u64(buf, 24)?,
            data_blocks: get_u32(buf, 32)?,
            voronoi_cell: get_u32(buf, 36)?,
            cluster_id: get_i32(buf, 40)?,
            _pad2: [0; 4],
            original_size: get_u64(buf, 48)?,
            permissions: get_u16(buf, 56)?,
            _pad3: [0; 6],
            created_ms: get_u64(buf, 64)?,
            modified_ms: get_u64(buf, 72)?,
            content_hash: get_u64(buf, 80)?,
            name_len,
            name,
            sibling_next: get_u64(buf, INODE_SIBLING_NEXT_OFFSET)?,
            sibling_prev: get_u64(buf, INODE_SIBLING_PREV_OFFSET)?,
            dir_first_child: get_u64(buf, INODE_DIR_FIRST_CHILD_OFFSET)?,
            uid: get_u32(buf, INODE_UID_OFFSET)?,
            gid: get_u32(buf, INODE_GID_OFFSET)?,
        })
    }
}

// ── BlockStore ──────────────────────────────────────────────────────────────

/// Persistence layer with bitmap allocation and WAL; allocation and I/O scale with blocks touched.
pub struct BlockStore {
    backend: Option<Box<dyn BlockStoreBackend>>,
    superblock: Option<Superblock>,
    inode_records: BTreeMap<u64, InodeRecord>,
    data_extents: BTreeMap<u64, Vec<u8>>,
    payload_extents: BTreeMap<u64, ManifoldPayload>,
    freelist: Vec<u64>,
    dirty_inodes: Vec<u64>,
    deleted_inodes: Vec<u64>,
    bitmap: Vec<u8>,
    journal_seq: u64,
    journal_pos: u64,
    topo_journal: Option<TopologicalJournal>,
    data_write_ops: u64,
}

impl BlockStore {
    pub fn new() -> Self {
        Self {
            backend: None,
            superblock: None,
            inode_records: BTreeMap::new(),
            data_extents: BTreeMap::new(),
            payload_extents: BTreeMap::new(),
            freelist: Vec::new(),
            dirty_inodes: Vec::new(),
            deleted_inodes: Vec::new(),
            bitmap: Vec::new(),
            journal_seq: 0,
            journal_pos: 0,
            topo_journal: Some(TopologicalJournal::new()),
            data_write_ops: 0,
        }
    }

    pub fn with_mock(num_sectors: usize) -> Self {
        let mut s = Self::new();
        s.backend = Some(Box::new(MockBackend::new(num_sectors)));
        let _ = s.format(num_sectors as u64);
        s.topo_journal = Some(TopologicalJournal::new());
        s
    }

    /// Mount an existing filesystem from `backend`, reading only.
    ///
    /// A superblock that is absent, carries foreign data, or announces a
    /// version this build does not understand is refused with
    /// `MountError::NoSuperblock`. Mounting never formats: a format erases the
    /// superblock, the journal, and the free bitmap, so inferring one from an
    /// unrecognised sector turns a recoverable fault — a transient read, a disk
    /// holding somebody else's filesystem, a future on-disk version — into
    /// unrecoverable data loss. The manifold region is provisioned deliberately
    /// by the image builder (`kernel/seal-mkimage`, `create_manifold_superblock`),
    /// which is the only caller that has established the device is blank.
    fn mount_backend(backend: Box<dyn BlockStoreBackend>) -> Result<Self, MountError> {
        let mut buf = [0u8; SECTOR_SIZE];
        backend.read_sector(0, &mut buf).map_err(|e| match e {
            BlockError::NoDevice => MountError::NoDevice,
            _ => MountError::NoSuperblock,
        })?;
        if Superblock::from_bytes(&buf).is_none() {
            return Err(MountError::NoSuperblock);
        }
        let mut s = Self::new();
        s.backend = Some(backend);
        s.read_superblock().map_err(|_| MountError::NoSuperblock)?;
        s.read_bitmap().map_err(|_| MountError::NoSuperblock)?;
        s.read_inode_table().map_err(|_| MountError::NoSuperblock)?;
        s.replay_journal().map_err(|_| MountError::NoSuperblock)?;
        s.topo_journal = Some(TopologicalJournal::new());
        if let Some(ref mut journal) = s.topo_journal {
            if let Some(ref backend) = s.backend {
                if let Some(ref sb) = s.superblock {
                    let _ = journal.replay(backend.as_ref(), sb.journal_start, sb.journal_blocks);
                }
            }
        }
        Ok(s)
    }

    pub fn mount_ahci() -> Result<Self, BlockError> {
        Self::mount_backend(Box::new(AhciBackend)).map_err(|e| match e {
            MountError::NoDevice => BlockError::NoDevice,
            MountError::NoSuperblock => BlockError::IoError,
        })
    }

    pub fn try_mount_ahci() -> Result<Self, MountError> {
        Self::mount_backend(Box::new(AhciBackend))
    }

    pub fn format(&mut self, total_blocks: u64) -> Result<(), BlockError> {
        if total_blocks < JOURNAL_BLOCKS + 10 {
            return Err(BlockError::InvalidLba);
        }
        self.superblock = Some(Superblock::new(total_blocks));
        self.inode_records.clear();
        self.data_extents.clear();
        self.payload_extents.clear();
        self.freelist.clear();
        self.dirty_inodes.clear();
        self.deleted_inodes.clear();

        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let bitmap_bytes = (sb.free_bitmap_blocks as usize) * SECTOR_SIZE;
        self.bitmap = vec![0u8; bitmap_bytes];
        self.mark_used(0, sb.data_start)?;

        self.journal_seq = 0;
        self.journal_pos = sb.journal_start;
        self.topo_journal = Some(TopologicalJournal::new());

        if let Some(ref backend) = self.backend {
            backend.write_sector(0, sb.as_bytes())?;
            let zero = [0u8; SECTOR_SIZE];
            for i in sb.journal_start..sb.journal_start + sb.journal_blocks {
                backend.write_sector(i, &zero)?;
            }
            for i in sb.free_bitmap_start..sb.free_bitmap_start + sb.free_bitmap_blocks {
                backend.write_sector(i, &zero)?;
            }
        }
        Ok(())
    }

    fn mark_used(&mut self, lba: u64, count: u64) -> Result<(), BlockError> {
        for i in lba..lba + count {
            let idx = i as usize;
            let byte = idx / 8;
            let bit = idx % 8;
            if byte >= self.bitmap.len() {
                return Err(BlockError::InvalidLba);
            }
            self.bitmap[byte] |= 1 << bit;
        }
        Ok(())
    }

    fn mark_free(&mut self, lba: u64, count: u64) -> Result<(), BlockError> {
        for i in lba..lba + count {
            let idx = i as usize;
            let byte = idx / 8;
            let bit = idx % 8;
            if byte >= self.bitmap.len() {
                return Err(BlockError::InvalidLba);
            }
            self.bitmap[byte] &= !(1 << bit);
        }
        Ok(())
    }

    fn is_free(&self, lba: u64) -> Result<bool, BlockError> {
        let idx = lba as usize;
        let byte = idx / 8;
        let bit = idx % 8;
        if byte >= self.bitmap.len() {
            return Err(BlockError::InvalidLba);
        }
        Ok((self.bitmap[byte] >> bit) & 1 == 0)
    }

    pub fn alloc_blocks(&mut self, count: u32) -> Result<u64, BlockError> {
        let sb = self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let start = sb.data_start;
        let end = sb.total_blocks;
        let mut run_start = 0u64;
        let mut run_len = 0u32;

        for lba in start..end {
            if self.is_free(lba)? {
                if run_len == 0 {
                    run_start = lba;
                }
                run_len += 1;
                if run_len >= count {
                    self.mark_used(run_start, count as u64)?;
                    return Ok(run_start);
                }
            } else {
                run_len = 0;
            }
        }
        Err(BlockError::IoError)
    }

    pub fn free_blocks(&mut self, lba: u64, count: u32) -> Result<(), BlockError> {
        self.mark_free(lba, count as u64)
    }

    fn read_superblock(&mut self) -> Result<(), BlockError> {
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let mut buf = [0u8; SECTOR_SIZE];
        backend.read_sector(0, &mut buf)?;
        let sb = Superblock::from_bytes(&buf).ok_or(BlockError::IoError)?;
        self.superblock = Some(sb);
        Ok(())
    }

    fn read_bitmap(&mut self) -> Result<(), BlockError> {
        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let mut bitmap = Vec::with_capacity((sb.free_bitmap_blocks as usize) * SECTOR_SIZE);
        for i in 0..sb.free_bitmap_blocks {
            let mut buf = [0u8; SECTOR_SIZE];
            backend.read_sector(sb.free_bitmap_start + i, &mut buf)?;
            bitmap.extend_from_slice(&buf);
        }
        self.bitmap = bitmap;
        Ok(())
    }

    fn write_bitmap(&mut self) -> Result<(), BlockError> {
        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        for i in 0..sb.free_bitmap_blocks {
            let start = (i as usize) * SECTOR_SIZE;
            let end = start + SECTOR_SIZE;
            if end > self.bitmap.len() {
                break;
            }
            backend.write_sector(sb.free_bitmap_start + i, &self.bitmap[start..end])?;
        }
        Ok(())
    }

    fn read_inode_table(&mut self) -> Result<(), BlockError> {
        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let records_per_sector = SECTOR_SIZE / INODE_RECORD_SIZE;

        let mut sector = [0u8; SECTOR_SIZE];
        for block in 0..sb.inode_table_blocks {
            backend.read_sector(sb.inode_table_start + block, &mut sector)?;
            for rec in 0..records_per_sector {
                let offset = rec * INODE_RECORD_SIZE;
                if let Some(record) =
                    InodeRecord::from_record(&sector[offset..offset + INODE_RECORD_SIZE])
                {
                    self.inode_records.insert(record.id, record);
                }
            }
        }
        Ok(())
    }

    fn replay_journal(&mut self) -> Result<(), BlockError> {
        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let mut buf = [0u8; SECTOR_SIZE];
        let mut max_seq = 0u64;

        for i in 0..sb.journal_blocks {
            {
                let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
                backend.read_sector(sb.journal_start + i, &mut buf)?;
            }
            if let Some(entry) = JournalEntry::from_sector(&buf) {
                if entry.committed != 0 && entry.seq > max_seq {
                    max_seq = entry.seq;
                    if entry.op == 2 {
                        self.inode_records.remove(&entry.inode_id);
                        if entry.data_lba != 0 && entry.data_blocks > 0 {
                            self.data_extents.remove(&entry.data_lba);
                            self.payload_extents.remove(&entry.data_lba);
                            let _ = self.mark_free(entry.data_lba, entry.data_blocks as u64);
                        }
                        continue;
                    }
                    if entry.op != 1 {
                        continue;
                    }
                    let payload = if entry.data_blocks > 0 && entry.data_lba != 0 {
                        match self.read_data_extent(entry.data_lba, entry.data_blocks as usize) {
                            Ok(ref buf) => {
                                bytes_to_payload(buf).unwrap_or_else(|| ManifoldPayload {
                                    points: vec![SpherePoint::zero()],
                                    point_count: 1,
                                    betti_0: 1,
                                    original_size: entry.original_size,
                                    content_hash: entry.content_hash,
                                })
                            }
                            Err(_) => ManifoldPayload {
                                points: vec![SpherePoint::zero()],
                                point_count: 1,
                                betti_0: 1,
                                original_size: entry.original_size,
                                content_hash: entry.content_hash,
                            },
                        }
                    } else {
                        ManifoldPayload {
                            points: vec![SpherePoint::zero()],
                            point_count: 1,
                            betti_0: 1,
                            original_size: entry.original_size,
                            content_hash: entry.content_hash,
                        }
                    };

                    let data = if entry.data_blocks > 0 && entry.data_lba != 0 {
                        match self.read_data_extent(entry.data_lba, entry.data_blocks as usize) {
                            Ok(ref extent) => {
                                let pbytes = payload_bytes(&payload);
                                if extent.len() > pbytes {
                                    extent[pbytes..].to_vec()
                                } else {
                                    Vec::new()
                                }
                            }
                            Err(_) => Vec::new(),
                        }
                    } else {
                        Vec::new()
                    };

                    let name_len = entry.name_len.min(128) as usize;
                    let name = String::from_utf8_lossy(&entry.name[..name_len]).into_owned();
                    let inode = Inode {
                        id: entry.inode_id,
                        name,
                        kind: match entry.kind {
                            1 => InodeKind::Directory,
                            _ => InodeKind::File,
                        },
                        payload,
                        data,
                        metadata: InodeMetadata {
                            created_ms: entry.created_ms,
                            modified_ms: entry.modified_ms,
                            original_size: entry.original_size,
                            permissions: entry.permissions,
                        },
                        voronoi_cell: entry.voronoi_cell as usize,
                        cluster_id: entry.voronoi_cell as i32,
                        parent: entry.parent,
                        sibling_next: None,
                        sibling_prev: None,
                        dir_first_child: None,
                    };

                    let mut record =
                        InodeRecord::from_inode(&inode, entry.data_lba, entry.data_blocks);
                    // The entry is the authority here: it is replayed over
                    // whatever the inode table held, owner included.
                    record.uid = entry.uid;
                    record.gid = entry.gid;
                    self.inode_records.insert(entry.inode_id, record);
                    if entry.data_lba != 0 && entry.data_blocks > 0 {
                        let _ = self.read_data_extent(entry.data_lba, entry.data_blocks as usize);
                    }
                }
            }
        }
        self.journal_seq = max_seq;
        Ok(())
    }

    fn mark_dirty_inode(&mut self, inode_id: u64) {
        if !self.dirty_inodes.contains(&inode_id) {
            self.dirty_inodes.push(inode_id);
        }
    }

    fn mark_deleted_inode(&mut self, inode_id: u64) {
        self.dirty_inodes.retain(|&id| id != inode_id);
        if !self.deleted_inodes.contains(&inode_id) {
            self.deleted_inodes.push(inode_id);
        }
    }

    fn write_journal_entry(&mut self, entry: &JournalEntry) -> Result<(), BlockError> {
        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let lba = sb.journal_start + (self.journal_pos % sb.journal_blocks);
        let sector = entry.to_sector();
        backend.write_sector(lba, &sector)?;
        self.journal_pos = (self.journal_pos + 1) % sb.journal_blocks;
        Ok(())
    }

    fn commit_journal(&mut self) -> Result<(), BlockError> {
        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let mut buf = [0u8; SECTOR_SIZE];

        for i in 0..sb.journal_blocks {
            backend.read_sector(sb.journal_start + i, &mut buf)?;
            if let Some(mut entry) = JournalEntry::from_sector(&buf) {
                if entry.committed == 0 {
                    entry.committed = 1;
                    entry.checksum = entry.checksum();
                    let sector = entry.to_sector();
                    backend.write_sector(sb.journal_start + i, &sector)?;
                }
            }
        }
        Ok(())
    }

    fn read_data_extent(&mut self, lba: u64, blocks: usize) -> Result<Vec<u8>, BlockError> {
        if let Some(extent) = self.data_extents.get(&lba) {
            return Ok(extent.clone());
        }
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let mut extent = Vec::with_capacity(blocks * SECTOR_SIZE);
        for i in 0..blocks {
            let mut sector = [0u8; SECTOR_SIZE];
            backend.read_sector(lba + i as u64, &mut sector)?;
            extent.extend_from_slice(&sector);
        }
        self.data_extents.insert(lba, extent.clone());
        Ok(extent)
    }

    fn write_data_extent(
        &mut self,
        lba: u64,
        blocks: usize,
        data: &[u8],
    ) -> Result<(), BlockError> {
        // Record topological journal entry for atomicity.
        // Avoid double-borrow of self by reading extent before touching journal.
        let before = if self.topo_journal.is_some() {
            Some(match self.read_data_extent(lba, blocks) {
                Ok(extent) => extent,
                Err(_) => vec![0u8; blocks * SECTOR_SIZE],
            })
        } else {
            None
        };
        if let (Some(journal), Some(before)) = (self.topo_journal.as_mut(), before) {
            let embedding = compute_manifold_embedding(data);
            journal.record_change(lba as u32, &before, data, embedding);
        }

        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        for i in 0..blocks {
            let start = i * SECTOR_SIZE;
            let end = (start + SECTOR_SIZE).min(data.len());
            let mut sector = [0u8; SECTOR_SIZE];
            sector[..end - start].copy_from_slice(&data[start..end]);
            backend.write_sector(lba + i as u64, &sector)?;
        }
        self.data_write_ops = self.data_write_ops.saturating_add(1);
        self.data_extents.insert(lba, data.to_vec());
        Ok(())
    }

    pub fn data_write_ops(&self) -> u64 {
        self.data_write_ops
    }

    pub fn write_inode(&mut self, inode: &Inode, data: &[u8]) -> Result<(), BlockError> {
        let payload = &inode.payload;
        let payload_buf = payload_to_bytes(payload);
        let original_size = data.len() as u64;
        let total = payload_buf.len() + data.len();
        let blocks_needed = total.div_ceil(SECTOR_SIZE).max(1) as u32;
        let blocks = blocks_needed.min(MAX_EXTENT_BLOCKS);

        let old_record = self.inode_records.get(&inode.id).copied();
        let (lba, blocks) = if let Some(old) = old_record {
            if old.data_blocks >= blocks && old.data_lba != 0 {
                let mut extent = Vec::with_capacity((old.data_blocks as usize) * SECTOR_SIZE);
                extent.extend_from_slice(&payload_buf);
                extent.extend_from_slice(data);
                extent.resize((old.data_blocks as usize) * SECTOR_SIZE, 0);
                self.write_data_extent(old.data_lba, old.data_blocks as usize, &extent)?;
                self.payload_extents.insert(old.data_lba, payload.clone());
                (old.data_lba, old.data_blocks)
            } else {
                if old.data_lba != 0 && old.data_blocks > 0 {
                    let _ = self.free_blocks(old.data_lba, old.data_blocks);
                }
                let new_lba = self.alloc_blocks(blocks)?;
                let mut extent = Vec::with_capacity((blocks as usize) * SECTOR_SIZE);
                extent.extend_from_slice(&payload_buf);
                extent.extend_from_slice(data);
                extent.resize((blocks as usize) * SECTOR_SIZE, 0);
                self.write_data_extent(new_lba, blocks as usize, &extent)?;
                self.payload_extents.insert(new_lba, payload.clone());
                (new_lba, blocks)
            }
        } else {
            let new_lba = self.alloc_blocks(blocks)?;
            let mut extent = Vec::with_capacity((blocks as usize) * SECTOR_SIZE);
            extent.extend_from_slice(&payload_buf);
            extent.extend_from_slice(data);
            extent.resize((blocks as usize) * SECTOR_SIZE, 0);
            self.write_data_extent(new_lba, blocks as usize, &extent)?;
            self.payload_extents.insert(new_lba, payload.clone());
            (new_lba, blocks)
        };

        let mut record = InodeRecord::from_inode(inode, lba, blocks);
        record.original_size = original_size;
        if let Some(old) = old_record {
            // Ownership lives in the record, not in the inode, so rebuilding
            // the record from the inode must carry the old owner over rather
            // than quietly hand the node back to root.
            record.uid = old.uid;
            record.gid = old.gid;
        }
        self.inode_records.insert(inode.id, record);
        self.deleted_inodes.retain(|&id| id != inode.id);
        self.mark_dirty_inode(inode.id);

        // WAL: write journal entry before metadata is durable.
        self.journal_seq += 1;
        let entry = JournalEntry::update_for(self.journal_seq, &record);
        self.write_journal_entry(&entry)?;

        Ok(())
    }

    pub fn delete_inode(&mut self, inode: &Inode) -> Result<(), BlockError> {
        let Some(old) = self.inode_records.get(&inode.id).copied() else {
            return if self.backend.is_none() {
                Ok(())
            } else {
                Err(BlockError::InvalidLba)
            };
        };

        self.journal_seq += 1;
        let mut name = [0u8; 128];
        let name_bytes = inode.name.as_bytes();
        let name_len = name_bytes.len().min(128);
        name[..name_len].copy_from_slice(&name_bytes[..name_len]);

        let mut entry = JournalEntry {
            seq: self.journal_seq,
            op: 2,
            committed: 0,
            _pad1: [0; 6],
            inode_id: inode.id,
            parent: inode.parent,
            kind: match inode.kind {
                InodeKind::File => 0,
                InodeKind::Directory => 1,
            },
            _pad2: [0; 7],
            data_lba: old.data_lba,
            data_blocks: old.data_blocks,
            voronoi_cell: inode.voronoi_cell as u32,
            original_size: inode.metadata.original_size,
            permissions: inode.metadata.permissions,
            _pad3: [0; 6],
            created_ms: inode.metadata.created_ms,
            modified_ms: inode.metadata.modified_ms,
            content_hash: inode.payload.content_hash,
            name_len: name_len as u16,
            name,
            uid: old.uid,
            gid: old.gid,
            checksum: 0,
            _pad4: [0; 282],
        };
        entry.checksum = entry.checksum();
        self.write_journal_entry(&entry)?;

        self.inode_records.remove(&inode.id);
        if old.data_lba != 0 && old.data_blocks > 0 {
            self.data_extents.remove(&old.data_lba);
            self.payload_extents.remove(&old.data_lba);
            self.free_blocks(old.data_lba, old.data_blocks)?;
        }
        self.mark_deleted_inode(inode.id);
        Ok(())
    }

    pub fn update_inode_metadata(&mut self, inode: &Inode) -> Result<(), BlockError> {
        let Some(old) = self.inode_records.get(&inode.id).copied() else {
            return if self.backend.is_none() {
                Ok(())
            } else {
                Err(BlockError::InvalidLba)
            };
        };
        let mut record = InodeRecord::from_inode(inode, old.data_lba, old.data_blocks);
        record.original_size = old.original_size;
        record.uid = old.uid;
        record.gid = old.gid;
        self.inode_records.insert(inode.id, record);
        self.deleted_inodes.retain(|&id| id != inode.id);
        self.mark_dirty_inode(inode.id);

        self.journal_seq += 1;
        let entry = JournalEntry::update_for(self.journal_seq, &record);
        self.write_journal_entry(&entry)?;

        Ok(())
    }

    /// Ownership recorded on disk for `inode_id`, as `(uid, gid)`.
    ///
    /// A node no record covers, and a record written before the format carried
    /// an owner, both read as root: those bytes are zero and zero is root. An
    /// owner that cannot be read is root, never unowned and never world, so a
    /// missing one narrows access rather than widening it.
    pub fn record_owner(&self, inode_id: u64) -> (u32, u32) {
        self.inode_records
            .get(&inode_id)
            .map_or((0, 0), |rec| (rec.uid, rec.gid))
    }

    /// Record `uid` and `gid` as the owner of `inode_id`, write-ahead first.
    ///
    /// The write-ahead entry is not optional: a remount reads the inode table
    /// and then replays the journal over it, so an owner recorded only in the
    /// table would be overwritten by the older owner the last entry carries.
    pub fn set_record_owner(
        &mut self,
        inode_id: u64,
        uid: u32,
        gid: u32,
    ) -> Result<(), BlockError> {
        let Some(mut record) = self.inode_records.get(&inode_id).copied() else {
            return if self.backend.is_none() {
                Ok(())
            } else {
                Err(BlockError::InvalidLba)
            };
        };
        record.uid = uid;
        record.gid = gid;
        self.inode_records.insert(inode_id, record);
        self.mark_dirty_inode(inode_id);

        self.journal_seq += 1;
        let entry = JournalEntry::update_for(self.journal_seq, &record);
        self.write_journal_entry(&entry)
    }

    pub fn read_data(&self, inode_id: u64) -> Option<(ManifoldPayload, Vec<u8>)> {
        let record = self.inode_records.get(&inode_id)?;
        let extent = self.data_extents.get(&record.data_lba)?;
        let payload = self.payload_extents.get(&record.data_lba)?.clone();
        let pbytes = payload_bytes(&payload);
        let data = if extent.len() > pbytes && record.original_size > 0 {
            let end = (pbytes + record.original_size as usize).min(extent.len());
            extent[pbytes..end].to_vec()
        } else {
            Vec::new()
        };
        Some((payload, data))
    }

    pub fn read_payload(&self, inode_id: u64) -> Option<ManifoldPayload> {
        let record = self.inode_records.get(&inode_id)?;
        self.payload_extents.get(&record.data_lba).cloned()
    }

    pub fn flush_inode(&mut self, inode_id: u64) -> Result<(), BlockError> {
        let record = match self.inode_records.get(&inode_id).copied() {
            Some(r) => r,
            None => return Ok(()),
        };
        self.write_inode_record_to_table(inode_id, &record.to_record())
    }

    fn clear_inode(&mut self, inode_id: u64) -> Result<(), BlockError> {
        self.write_inode_record_to_table(inode_id, &[0u8; INODE_RECORD_SIZE])
    }

    fn write_inode_record_to_table(
        &mut self,
        inode_id: u64,
        record: &[u8; INODE_RECORD_SIZE],
    ) -> Result<(), BlockError> {
        let sb = *self.superblock.as_ref().ok_or(BlockError::NoDevice)?;
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let records_per_sector = SECTOR_SIZE / INODE_RECORD_SIZE;
        let table_idx = inode_id & 0xFFFF_FFFF;
        let block = table_idx / (records_per_sector as u64);
        let rec = (table_idx % (records_per_sector as u64)) as usize;
        if block >= sb.inode_table_blocks {
            return Err(BlockError::InvalidLba);
        }
        let lba = sb.inode_table_start + block;
        let mut sector = [0u8; SECTOR_SIZE];
        backend.read_sector(lba, &mut sector)?;
        let offset = rec * INODE_RECORD_SIZE;
        sector[offset..offset + INODE_RECORD_SIZE].copy_from_slice(record);
        backend.write_sector(lba, &sector)?;
        Ok(())
    }

    pub fn flush_all(&mut self) -> Result<(), BlockError> {
        // T4: commit topological journal before making metadata durable.
        if let Some(ref mut journal) = self.topo_journal {
            if journal.is_dirty() {
                if let Some(ref sb) = self.superblock {
                    if let Some(ref backend) = self.backend {
                        journal.commit(backend.as_ref(), sb.journal_start, sb.journal_blocks)?;
                    }
                }
            }
        }

        let dirty = self.dirty_inodes.clone();
        for id in &dirty {
            self.flush_inode(*id)?;
        }
        let deleted = self.deleted_inodes.clone();
        for id in &deleted {
            self.clear_inode(*id)?;
        }
        self.dirty_inodes.clear();
        self.deleted_inodes.clear();
        self.commit_journal()?;
        self.write_bitmap()?;
        if let Some(ref sb) = self.superblock {
            if let Some(ref backend) = self.backend {
                backend.write_sector(0, sb.as_bytes())?;
            }
        }
        Ok(())
    }

    pub fn sync(&mut self) -> Result<(), BlockError> {
        self.flush_all()
    }

    pub fn is_mounted(&self) -> bool {
        self.superblock.is_some()
    }

    pub fn load_inodes(&mut self) -> Vec<Inode> {
        let mut inodes = Vec::new();
        let records: Vec<InodeRecord> = self.inode_records.values().copied().collect();
        for record in records {
            let payload = match self.payload_extents.get(&record.data_lba) {
                Some(p) => p.clone(),
                None => self
                    .read_data_extent(record.data_lba, record.data_blocks as usize)
                    .ok()
                    .and_then(|extent| bytes_to_payload(&extent))
                    .unwrap_or_else(|| ManifoldPayload {
                        points: vec![SpherePoint::zero()],
                        point_count: 1,
                        betti_0: 1,
                        original_size: record.original_size,
                        content_hash: record.content_hash,
                    }),
            };
            let extent = self
                .data_extents
                .get(&record.data_lba)
                .cloned()
                .or_else(|| {
                    self.read_data_extent(record.data_lba, record.data_blocks as usize)
                        .ok()
                });
            let data = extent
                .map(|extent| {
                    let pbytes = payload_bytes(&payload);
                    if extent.len() > pbytes && record.original_size > 0 {
                        let end = (pbytes + record.original_size as usize).min(extent.len());
                        extent[pbytes..end].to_vec()
                    } else {
                        Vec::new()
                    }
                })
                .unwrap_or_default();
            inodes.push(record.to_inode(payload, data));
        }
        inodes
    }
}

impl Default for BlockStore {
    fn default() -> Self {
        Self::new()
    }
}

fn payload_bytes(payload: &ManifoldPayload) -> usize {
    24 + payload.point_count * 24
}

fn payload_to_bytes(payload: &ManifoldPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(payload_bytes(payload));
    buf.extend_from_slice(&(payload.point_count as u32).to_le_bytes());
    buf.extend_from_slice(&(payload.betti_0 as u32).to_le_bytes());
    buf.extend_from_slice(&payload.original_size.to_le_bytes());
    buf.extend_from_slice(&payload.content_hash.to_le_bytes());
    for pt in &payload.points {
        for &c in &pt.coords {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    buf
}

fn bytes_to_payload(buf: &[u8]) -> Option<ManifoldPayload> {
    if buf.len() < 24 {
        return None;
    }
    let point_count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let betti_0 = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let original_size = u64::from_le_bytes(buf[8..16].try_into().ok()?);
    let content_hash = u64::from_le_bytes(buf[16..24].try_into().ok()?);
    let expected = 24 + point_count * 24;
    if buf.len() < expected {
        return None;
    }
    let mut points = Vec::with_capacity(point_count);
    for i in 0..point_count {
        let off = 24 + i * 24;
        let x = f64::from_le_bytes(buf[off..off + 8].try_into().ok()?);
        let y = f64::from_le_bytes(buf[off + 8..off + 16].try_into().ok()?);
        let z = f64::from_le_bytes(buf[off + 16..off + 24].try_into().ok()?);
        points.push(SpherePoint { coords: [x, y, z] });
    }
    Some(ManifoldPayload {
        points,
        point_count,
        betti_0,
        original_size,
        content_hash,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::super::encoder;
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};
    use core::sync::atomic::{AtomicUsize, Ordering};

    fn dummy_inode(id: u64, name: &str) -> Inode {
        Inode {
            id,
            name: String::from(name),
            kind: InodeKind::File,
            payload: encoder::encode_text(name),
            data: vec![1, 2, 3],
            metadata: InodeMetadata {
                created_ms: 0,
                modified_ms: 0,
                original_size: 3,
                permissions: 0o644,
            },
            voronoi_cell: 0,
            cluster_id: 0,
            parent: 0,
            sibling_next: None,
            sibling_prev: None,
            dir_first_child: None,
        }
    }

    fn test_roundtrip() -> TestResult {
        let mut store = BlockStore::with_mock(8192);
        let inode = dummy_inode(1, "test");
        match store.write_inode(&inode, b"hello world") {
            Ok(()) => {}
            Err(_) => return TestResult::Fail("write_inode failed"),
        };
        let (payload, data) = match store.read_data(1) {
            Some(v) => v,
            None => return TestResult::Fail("read_data returned None"),
        };
        test_assert_eq!(data, b"hello world");
        test_assert!(payload.point_count > 0);
        TestResult::Pass
    }

    fn test_journal_sector_roundtrip() -> TestResult {
        let mut entry = JournalEntry {
            seq: 7,
            op: 1,
            committed: 0,
            _pad1: [0; 6],
            inode_id: 0x1_0000_0001,
            parent: 0x1_0000_0000,
            kind: 0,
            _pad2: [0; 7],
            data_lba: 2048,
            data_blocks: 2,
            voronoi_cell: 3,
            original_size: 11,
            permissions: 0o644,
            _pad3: [0; 6],
            created_ms: 1,
            modified_ms: 2,
            content_hash: 0xAA55,
            name_len: 8,
            name: [0; 128],
            uid: 0,
            gid: 0,
            checksum: 0,
            _pad4: [0; 282],
        };
        entry.name[..8].copy_from_slice(b"test.bin");
        entry.checksum = entry.checksum();
        let sector = entry.to_sector();
        test_assert_eq!(sector.len(), SECTOR_SIZE);
        let parsed = match JournalEntry::from_sector(&sector) {
            Some(parsed) => parsed,
            None => return TestResult::Fail("journal sector did not round-trip"),
        };
        test_assert_eq!(parsed.inode_id, entry.inode_id);
        test_assert_eq!(parsed.parent, entry.parent);
        test_assert_eq!(parsed.name_len, entry.name_len);

        let mut tampered = sector;
        tampered[0] ^= 0x80;
        test_assert!(
            JournalEntry::from_sector(&tampered).is_none(),
            "tampered journal sector should fail checksum"
        );
        TestResult::Pass
    }

    fn test_generational_inode_flush() -> TestResult {
        let mut store = BlockStore::with_mock(8192);
        let inode = dummy_inode(0x1_0000_0001, "gen");
        if store.write_inode(&inode, b"payload").is_err() {
            return TestResult::Fail("write_inode failed for generational inode");
        }
        if store.flush_all().is_err() {
            return TestResult::Fail("flush_all failed for generational inode");
        }
        let backend = store.backend.take();
        let mut remounted = BlockStore::new();
        remounted.backend = backend;
        if remounted.read_superblock().is_err() {
            return TestResult::Fail("remount read_superblock failed");
        }
        if remounted.read_bitmap().is_err() {
            return TestResult::Fail("remount read_bitmap failed");
        }
        if remounted.read_inode_table().is_err() {
            return TestResult::Fail("remount read_inode_table failed");
        }
        if remounted.replay_journal().is_err() {
            return TestResult::Fail("remount replay_journal failed");
        }
        let inodes = remounted.load_inodes();
        let Some(inode) = inodes.iter().find(|inode| inode.id == 0x1_0000_0001) else {
            return TestResult::Fail("generational inode missing after remount");
        };
        test_assert_eq!(inode.name, "gen");
        test_assert_eq!(inode.data, b"payload");
        TestResult::Pass
    }

    fn test_cold_inode_table_load_restores_data_without_journal() -> TestResult {
        let mut store = BlockStore::with_mock(8192);
        let inode = dummy_inode(0x1_0000_0002, "cold");
        if store.write_inode(&inode, b"durable bytes").is_err() {
            return TestResult::Fail("write_inode failed");
        }
        if store.flush_all().is_err() {
            return TestResult::Fail("flush_all failed");
        }

        let Some(sb) = store.superblock else {
            return TestResult::Fail("missing superblock");
        };
        let zero = [0u8; SECTOR_SIZE];
        let Some(backend) = store.backend.as_ref() else {
            return TestResult::Fail("missing backend");
        };
        for i in 0..sb.journal_blocks {
            if backend.write_sector(sb.journal_start + i, &zero).is_err() {
                return TestResult::Fail("journal wipe failed");
            }
        }

        let backend = store.backend.take();
        let mut remounted = BlockStore::new();
        remounted.backend = backend;
        if remounted.read_superblock().is_err() {
            return TestResult::Fail("remount read_superblock failed");
        }
        if remounted.read_bitmap().is_err() {
            return TestResult::Fail("remount read_bitmap failed");
        }
        if remounted.read_inode_table().is_err() {
            return TestResult::Fail("remount read_inode_table failed");
        }
        if remounted.replay_journal().is_err() {
            return TestResult::Fail("remount replay_journal failed");
        }
        let inodes = remounted.load_inodes();
        let Some(inode) = inodes.iter().find(|inode| inode.id == 0x1_0000_0002) else {
            return TestResult::Fail("cold inode missing after remount");
        };
        test_assert_eq!(inode.name, "cold");
        test_assert_eq!(inode.data, b"durable bytes");
        TestResult::Pass
    }

    /// Ownership recorded on a node must still be on it after a remount, and a
    /// node whose owner was never recorded must come back root-owned. Both the
    /// inode table and the journal are exercised: `flush_all` makes the record
    /// durable and commits the write-ahead entry, and the remount reads the
    /// table and then replays the journal over it, so an owner that either half
    /// drops is lost.
    fn test_recorded_owner_survives_remount() -> TestResult {
        let mut store = BlockStore::with_mock(8192);
        let owned = dummy_inode(0x1_0000_0004, "owned");
        let plain = dummy_inode(0x1_0000_0005, "plain");
        if store.write_inode(&owned, b"owned bytes").is_err() {
            return TestResult::Fail("write_inode failed for the owned node");
        }
        if store.write_inode(&plain, b"plain bytes").is_err() {
            return TestResult::Fail("write_inode failed for the unowned node");
        }

        // Nothing has recorded an owner yet, so every node is root-owned.
        test_assert_eq!(store.record_owner(owned.id), (0, 0));
        if store.set_record_owner(owned.id, 1000, 1001).is_err() {
            return TestResult::Fail("set_record_owner rejected a recorded inode");
        }
        test_assert_eq!(store.record_owner(owned.id), (1000, 1001));

        // A later write of the same node must not drop the owner it carries.
        if store.write_inode(&owned, b"owned bytes").is_err() {
            return TestResult::Fail("rewrite of the owned node failed");
        }
        test_assert_eq!(store.record_owner(owned.id), (1000, 1001));

        if store.flush_all().is_err() {
            return TestResult::Fail("flush_all failed");
        }

        let backend = store.backend.take();
        let mut remounted = BlockStore::new();
        remounted.backend = backend;
        if remounted.read_superblock().is_err() {
            return TestResult::Fail("remount read_superblock failed");
        }
        if remounted.read_bitmap().is_err() {
            return TestResult::Fail("remount read_bitmap failed");
        }
        if remounted.read_inode_table().is_err() {
            return TestResult::Fail("remount read_inode_table failed");
        }
        if remounted.replay_journal().is_err() {
            return TestResult::Fail("remount replay_journal failed");
        }

        test_assert_eq!(remounted.record_owner(owned.id), (1000, 1001));
        // A node nobody claimed reads as root, and so does an id no record
        // holds: an absent owner is root, never unowned and never world.
        test_assert_eq!(remounted.record_owner(plain.id), (0, 0));
        test_assert_eq!(remounted.record_owner(0xDEAD_BEEF), (0, 0));

        // Everything else about both nodes survived the remount untouched.
        let inodes = remounted.load_inodes();
        let Some(restored) = inodes.iter().find(|i| i.id == owned.id) else {
            return TestResult::Fail("owned node missing after remount");
        };
        test_assert_eq!(restored.name, "owned");
        test_assert_eq!(restored.data, b"owned bytes");
        let Some(restored) = inodes.iter().find(|i| i.id == plain.id) else {
            return TestResult::Fail("unowned node missing after remount");
        };
        test_assert_eq!(restored.name, "plain");
        test_assert_eq!(restored.data, b"plain bytes");
        TestResult::Pass
    }

    /// The owner occupies bytes that were dead in the format before it, so a
    /// disk written by an older build reads back as root and every other field
    /// keeps its offset. Recording an owner may change exactly the eight bytes
    /// it lives in, and those eight must have been zero.
    fn test_owner_occupies_only_previously_dead_bytes() -> TestResult {
        let ino = dummy_inode(0x1_0000_0006, "legacy");
        // Every byte of both owners is non-zero, so a byte-by-byte comparison
        // sees the whole of each field rather than only its low half.
        const UID: u32 = 0x0102_0304;
        const GID: u32 = 0x0506_0708;

        let unowned = InodeRecord::from_inode(&ino, 4096, 1);
        let mut owned = unowned;
        owned.uid = UID;
        owned.gid = GID;
        let (a, b) = (unowned.to_record(), owned.to_record());
        let mut moved = 0;
        for i in 0..INODE_RECORD_SIZE {
            if a[i] != b[i] {
                moved += 1;
                test_assert_eq!(a[i], 0);
            }
        }
        test_assert_eq!(moved, 8);

        // The record an older build wrote — owner bytes zero — is root-owned,
        // and one carrying an owner round-trips it exactly.
        let Some(back) = InodeRecord::from_record(&a) else {
            return TestResult::Fail("unowned record did not decode");
        };
        test_assert_eq!((back.uid, back.gid), (0, 0));
        let Some(back) = InodeRecord::from_record(&b) else {
            return TestResult::Fail("owned record did not decode");
        };
        test_assert_eq!((back.uid, back.gid), (UID, GID));
        test_assert_eq!(back.id, ino.id);
        test_assert_eq!(back.name_len, 6);
        test_assert_eq!(back.data_lba, 4096);
        test_assert_eq!(back.permissions, 0o644);
        test_assert_eq!(back.dir_first_child, NONE_ID);

        // Same for the journal entry, checksum held out of the comparison so
        // only the owner bytes can differ.
        let mut unowned = JournalEntry {
            seq: 9,
            op: 1,
            committed: 0,
            _pad1: [0; 6],
            inode_id: ino.id,
            parent: 0,
            kind: 0,
            _pad2: [0; 7],
            data_lba: 4096,
            data_blocks: 1,
            voronoi_cell: 0,
            original_size: 3,
            permissions: 0o644,
            _pad3: [0; 6],
            created_ms: 0,
            modified_ms: 0,
            content_hash: 0,
            name_len: 6,
            name: [0; 128],
            uid: 0,
            gid: 0,
            checksum: 0,
            _pad4: [0; 282],
        };
        unowned.name[..6].copy_from_slice(b"legacy");
        let mut owned = unowned;
        owned.uid = UID;
        owned.gid = GID;
        let (a, b) = (unowned.encode_sector(0), owned.encode_sector(0));
        let mut moved = 0;
        for i in 0..SECTOR_SIZE {
            if a[i] != b[i] {
                moved += 1;
                test_assert_eq!(a[i], 0);
            }
        }
        test_assert_eq!(moved, 8);

        owned.checksum = owned.checksum();
        let Some(back) = JournalEntry::from_sector(&owned.to_sector()) else {
            return TestResult::Fail("owned journal sector did not decode");
        };
        test_assert_eq!((back.uid, back.gid), (UID, GID));
        test_assert_eq!(back.inode_id, ino.id);
        test_assert_eq!(back.name_len, 6);
        let Some(back) = JournalEntry::from_sector(&unowned.to_sector()) else {
            return TestResult::Fail("unowned journal sector did not decode");
        };
        test_assert_eq!((back.uid, back.gid), (0, 0));
        TestResult::Pass
    }

    /// Backend that counts writes and hands back whatever sector 0 it was told
    /// to hold, so a refused mount can be told from a mount that formatted.
    struct RefusalProbe {
        sector0: [u8; SECTOR_SIZE],
        read_fails: bool,
    }

    static PROBE_WRITES: AtomicUsize = AtomicUsize::new(0);

    impl BlockStoreBackend for RefusalProbe {
        fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
            if self.read_fails {
                return Err(BlockError::IoError);
            }
            if buf.len() != SECTOR_SIZE {
                return Err(BlockError::InvalidLba);
            }
            if lba == 0 {
                buf.copy_from_slice(&self.sector0);
            } else {
                buf.fill(0);
            }
            Ok(())
        }
        fn write_sector(&self, _lba: u64, _buf: &[u8]) -> Result<(), BlockError> {
            PROBE_WRITES.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// An unrecognised superblock must refuse the mount without writing. A
    /// mount that refuses is recoverable; a mount that formats is not.
    fn test_mount_refuses_unrecognised_superblock_without_writing() -> TestResult {
        let blank = [0u8; SECTOR_SIZE];

        let mut foreign = [0u8; SECTOR_SIZE];
        for (i, b) in foreign.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let mut future = Superblock::new(1024 * 1024);
        future.version = VERSION + 1;
        let mut future_sector = [0u8; SECTOR_SIZE];
        future_sector.copy_from_slice(future.as_bytes());

        for (accepted, wrote, sector, read_fails) in [
            (
                "mount accepted a blank device",
                "mount wrote to a blank device",
                blank,
                false,
            ),
            (
                "mount accepted foreign data as a superblock",
                "mount wrote over foreign data",
                foreign,
                false,
            ),
            (
                "mount accepted a future on-disk version",
                "mount wrote over a future on-disk version",
                future_sector,
                false,
            ),
            (
                "mount accepted an unreadable device",
                "mount wrote to a device it could not read",
                blank,
                true,
            ),
        ] {
            PROBE_WRITES.store(0, Ordering::SeqCst);
            let probe = RefusalProbe {
                sector0: sector,
                read_fails,
            };
            let mounted = BlockStore::mount_backend(Box::new(probe));
            test_assert!(mounted.is_err(), accepted);
            test_assert!(PROBE_WRITES.load(Ordering::SeqCst) == 0, wrote);
        }

        // A superblock this build does understand still mounts, and mounting it
        // writes nothing.
        PROBE_WRITES.store(0, Ordering::SeqCst);
        let sb = Superblock::new(1024 * 1024);
        let mut valid = [0u8; SECTOR_SIZE];
        valid.copy_from_slice(sb.as_bytes());
        let mounted = BlockStore::mount_backend(Box::new(RefusalProbe {
            sector0: valid,
            read_fails: false,
        }));
        test_assert!(mounted.is_ok(), "valid superblock should mount");
        test_assert_eq!(PROBE_WRITES.load(Ordering::SeqCst), 0);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("block_store::roundtrip", test_roundtrip);
        crate::testing::register_test(
            "block_store::mount_refuses_unrecognised_superblock_without_writing",
            test_mount_refuses_unrecognised_superblock_without_writing,
        );
        crate::testing::register_test(
            "block_store::journal_sector_roundtrip",
            test_journal_sector_roundtrip,
        );
        crate::testing::register_test(
            "block_store::generational_inode_flush",
            test_generational_inode_flush,
        );
        crate::testing::register_test(
            "block_store::cold_inode_table_load_restores_data_without_journal",
            test_cold_inode_table_load_restores_data_without_journal,
        );
        crate::testing::register_test(
            "block_store::recorded_owner_survives_remount",
            test_recorded_owner_survives_remount,
        );
        crate::testing::register_test(
            "block_store::owner_occupies_only_previously_dead_bytes",
            test_owner_occupies_only_previously_dead_bytes,
        );
    }
}
