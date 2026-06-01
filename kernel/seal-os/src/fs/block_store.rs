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

const SECTOR_SIZE: usize = 512;
const INODE_RECORD_SIZE: usize = 256;
const MAX_EXTENT_BLOCKS: u32 = 256;
const MAGIC: [u8; 4] = *b"MNFD";
const VERSION: u32 = 1;
const JOURNAL_BLOCKS: u64 = 1024;
const NONE_ID: u64 = u64::MAX;
const AHCI_MANIFOLD_START_LBA: u64 = 2048 + ((64 * 1024 * 1024) / SECTOR_SIZE as u64);

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
    sectors: Vec<[u8; SECTOR_SIZE]>,
}

impl MockBackend {
    pub fn new(num_sectors: usize) -> Self {
        let mut sectors = Vec::with_capacity(num_sectors);
        for _ in 0..num_sectors {
            sectors.push([0u8; SECTOR_SIZE]);
        }
        Self { sectors }
    }
}

impl BlockStoreBackend for MockBackend {
    fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let idx = lba as usize;
        if idx >= self.sectors.len() || buf.len() != SECTOR_SIZE {
            return Err(BlockError::InvalidLba);
        }
        buf.copy_from_slice(&self.sectors[idx]);
        Ok(())
    }
    fn write_sector(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        let idx = lba as usize;
        if idx >= self.sectors.len() || buf.len() != SECTOR_SIZE {
            return Err(BlockError::InvalidLba);
        }
        // SAFETY: idx is bounds-checked above; ptr points to valid mutable memory in Vec.
        unsafe {
            let ptr = self.sectors.as_ptr() as *mut [u8; SECTOR_SIZE];
            (*ptr.add(idx)).copy_from_slice(buf);
        }
        Ok(())
    }
}

unsafe impl Send for MockBackend {}
unsafe impl Sync for MockBackend {}

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
            (max_inodes * INODE_RECORD_SIZE as u64 + block_size as u64 - 1) / block_size as u64;
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
    checksum: u32,
    _pad4: [u8; 282],
}

impl JournalEntry {
    fn checksum(&self) -> u32 {
        let bytes = self.as_bytes();
        let mut h: u32 = 0x811c_9dc5;
        for &b in &bytes[..SECTOR_SIZE - 4] {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h
    }

    fn verify(&self) -> bool {
        self.checksum == self.checksum()
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: JournalEntry is repr(C) and exactly 512 bytes.
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }

    fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < size_of::<Self>() {
            return None;
        }
        // SAFETY: buf has correct length.
        let entry = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const Self) };
        if entry.verify() {
            Some(entry)
        } else {
            None
        }
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
}

impl InodeRecord {
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
        }
    }

    fn to_inode(&self, payload: ManifoldPayload, data: Vec<u8>) -> Inode {
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

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: InodeRecord is repr(C) and exactly 256 bytes.
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
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

    pub fn mount_ahci() -> Result<Self, BlockError> {
        let backend = AhciBackend;
        let mut buf = [0u8; SECTOR_SIZE];
        backend.read_sector(0, &mut buf)?;
        if Superblock::from_bytes(&buf).is_none() {
            let mut s = Self::new();
            s.backend = Some(Box::new(backend));
            s.format(1024 * 1024)?;
            s.topo_journal = Some(TopologicalJournal::new());
            return Ok(s);
        }
        let mut s = Self::new();
        s.backend = Some(Box::new(backend));
        s.read_superblock()?;
        s.read_bitmap()?;
        s.read_inode_table()?;
        s.replay_journal()?;
        if let Some(ref mut journal) = s.topo_journal {
            if let Some(ref backend) = s.backend {
                let sb = s.superblock.as_ref().ok_or(BlockError::NoDevice)?;
                let _ = journal.replay(backend.as_ref(), sb.journal_start, sb.journal_blocks);
            }
        }
        Ok(s)
    }

    pub fn try_mount_ahci() -> Result<Self, MountError> {
        let backend = AhciBackend;
        let mut buf = [0u8; SECTOR_SIZE];
        backend.read_sector(0, &mut buf).map_err(|e| match e {
            BlockError::NoDevice => MountError::NoDevice,
            _ => MountError::NoSuperblock,
        })?;
        if Superblock::from_bytes(&buf).is_none() {
            return Err(MountError::NoSuperblock);
        }
        let mut s = Self::new();
        s.backend = Some(Box::new(backend));
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

        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
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
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
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
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
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
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let records_per_sector = SECTOR_SIZE / INODE_RECORD_SIZE;

        let mut sector = [0u8; SECTOR_SIZE];
        for block in 0..sb.inode_table_blocks {
            backend.read_sector(sb.inode_table_start + block, &mut sector)?;
            for rec in 0..records_per_sector {
                let offset = rec * INODE_RECORD_SIZE;
                // SAFETY: offset is within sector; InodeRecord is 256 bytes.
                let record = unsafe {
                    core::ptr::read_unaligned(sector.as_ptr().add(offset) as *const InodeRecord)
                };
                let is_valid = record.kind <= 1 && (record.id != 0 || record.name_len > 0);
                if is_valid {
                    self.inode_records.insert(record.id, record);
                }
            }
        }
        Ok(())
    }

    fn write_inode_table(&mut self) -> Result<(), BlockError> {
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let records_per_sector = SECTOR_SIZE / INODE_RECORD_SIZE;

        let zero = [0u8; SECTOR_SIZE];
        for block in 0..sb.inode_table_blocks {
            backend.write_sector(sb.inode_table_start + block, &zero)?;
        }

        for (&id, record) in &self.inode_records {
            let block = id / (records_per_sector as u64);
            let rec = (id % (records_per_sector as u64)) as usize;
            if block >= sb.inode_table_blocks {
                continue;
            }
            let lba = sb.inode_table_start + block;
            let mut sector = [0u8; SECTOR_SIZE];
            backend.read_sector(lba, &mut sector)?;
            let offset = rec * INODE_RECORD_SIZE;
            sector[offset..offset + INODE_RECORD_SIZE].copy_from_slice(record.as_bytes());
            backend.write_sector(lba, &sector)?;
        }
        Ok(())
    }

    fn replay_journal(&mut self) -> Result<(), BlockError> {
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
        let mut buf = [0u8; SECTOR_SIZE];
        let mut max_seq = 0u64;

        for i in 0..sb.journal_blocks {
            {
                let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
                backend.read_sector(sb.journal_start + i, &mut buf)?;
            }
            if let Some(entry) = JournalEntry::from_bytes(&buf) {
                if entry.committed != 0 && entry.seq > max_seq {
                    max_seq = entry.seq;
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

                    let record = InodeRecord::from_inode(&inode, entry.data_lba, entry.data_blocks);
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

    fn write_journal_entry(&mut self, entry: &JournalEntry) -> Result<(), BlockError> {
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let lba = sb.journal_start + (self.journal_pos % sb.journal_blocks);
        backend.write_sector(lba, entry.as_bytes())?;
        self.journal_pos = (self.journal_pos + 1) % sb.journal_blocks;
        Ok(())
    }

    fn commit_journal(&mut self) -> Result<(), BlockError> {
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let mut buf = [0u8; SECTOR_SIZE];

        for i in 0..sb.journal_blocks {
            backend.read_sector(sb.journal_start + i, &mut buf)?;
            if let Some(mut entry) = JournalEntry::from_bytes(&buf) {
                if entry.committed == 0 {
                    entry.committed = 1;
                    entry.checksum = entry.checksum();
                    backend.write_sector(sb.journal_start + i, entry.as_bytes())?;
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
        let total = payload_buf.len() + data.len();
        let blocks_needed = ((total + SECTOR_SIZE - 1) / SECTOR_SIZE).max(1) as u32;
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

        let record = InodeRecord::from_inode(inode, lba, blocks);
        self.inode_records.insert(inode.id, record);
        self.dirty_inodes.push(inode.id);

        // WAL: write journal entry before metadata is durable.
        self.journal_seq += 1;
        let mut name = [0u8; 128];
        let name_bytes = inode.name.as_bytes();
        let name_len = name_bytes.len().min(128);
        name[..name_len].copy_from_slice(&name_bytes[..name_len]);

        let mut entry = JournalEntry {
            seq: self.journal_seq,
            op: 1,
            committed: 0,
            _pad1: [0; 6],
            inode_id: inode.id,
            parent: inode.parent,
            kind: match inode.kind {
                InodeKind::File => 0,
                InodeKind::Directory => 1,
            },
            _pad2: [0; 7],
            data_lba: lba,
            data_blocks: blocks,
            voronoi_cell: inode.voronoi_cell as u32,
            original_size: inode.metadata.original_size,
            permissions: inode.metadata.permissions,
            _pad3: [0; 6],
            created_ms: inode.metadata.created_ms,
            modified_ms: inode.metadata.modified_ms,
            content_hash: inode.payload.content_hash,
            name_len: name_len as u16,
            name,
            checksum: 0,
            _pad4: [0; 282],
        };
        entry.checksum = entry.checksum();
        self.write_journal_entry(&entry)?;

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
        let record = InodeRecord::from_inode(inode, old.data_lba, old.data_blocks);
        self.inode_records.insert(inode.id, record);
        self.dirty_inodes.push(inode.id);

        self.journal_seq += 1;
        let mut name = [0u8; 128];
        let name_bytes = inode.name.as_bytes();
        let name_len = name_bytes.len().min(128);
        name[..name_len].copy_from_slice(&name_bytes[..name_len]);

        let mut entry = JournalEntry {
            seq: self.journal_seq,
            op: 1,
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
            checksum: 0,
            _pad4: [0; 282],
        };
        entry.checksum = entry.checksum();
        self.write_journal_entry(&entry)?;

        Ok(())
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
        let sb = self
            .superblock
            .as_ref()
            .ok_or(BlockError::NoDevice)?
            .clone();
        let backend = self.backend.as_ref().ok_or(BlockError::NoDevice)?;
        let records_per_sector = SECTOR_SIZE / INODE_RECORD_SIZE;
        let block = inode_id / (records_per_sector as u64);
        let rec = (inode_id % (records_per_sector as u64)) as usize;
        if block >= sb.inode_table_blocks {
            return Err(BlockError::InvalidLba);
        }
        let lba = sb.inode_table_start + block;
        let mut sector = [0u8; SECTOR_SIZE];
        backend.read_sector(lba, &mut sector)?;
        let offset = rec * INODE_RECORD_SIZE;
        sector[offset..offset + INODE_RECORD_SIZE].copy_from_slice(record.as_bytes());
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

        let dirty: Vec<u64> = self.dirty_inodes.drain(..).collect();
        for id in &dirty {
            self.flush_inode(*id)?;
        }
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

    pub fn load_inodes(&self) -> Vec<Inode> {
        let mut inodes = Vec::new();
        for (&_id, record) in &self.inode_records {
            let payload = match self.payload_extents.get(&record.data_lba) {
                Some(p) => p.clone(),
                None => ManifoldPayload {
                    points: vec![SpherePoint::zero()],
                    point_count: 1,
                    betti_0: 1,
                    original_size: record.original_size,
                    content_hash: record.content_hash,
                },
            };
            let data = if let Some(extent) = self.data_extents.get(&record.data_lba) {
                let pbytes = payload_bytes(&payload);
                if extent.len() > pbytes && record.original_size > 0 {
                    let end = (pbytes + record.original_size as usize).min(extent.len());
                    extent[pbytes..end].to_vec()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            inodes.push(record.to_inode(payload, data));
        }
        inodes
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

    pub fn register_all() {
        crate::testing::register_test("block_store::roundtrip", test_roundtrip);
    }
}
