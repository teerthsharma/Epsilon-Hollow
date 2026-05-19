// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Persistence layer — superblock + inode table + buddy bitmap + payload-first extents.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;

use super::encoder::ManifoldPayload;
use super::manifold_fs::{Inode, InodeKind, InodeMetadata};
use crate::drivers::block::{read_block, write_block, BlockDevice, BlockError};
use crate::serial_println;

const SECTOR_SIZE: usize = 512;
const INODE_RECORD_SIZE: usize = 256;
const MAX_EXTENT_BLOCKS: u32 = 256; // 128 KiB
const MAGIC: u64 = 0x4D_46_53_5F_4F_31_00_00; // "MFS_O1\0\0"
const VERSION: u32 = 1;

/// Trait abstracting over real AHCI and mock block devices.
pub trait BlockStoreBackend: Send + Sync {
    fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
    fn write_sector(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;
}

/// Adapter for the global block device registry (AHCI device 0x800).
pub struct AhciBackend;

impl BlockStoreBackend for AhciBackend {
    fn read_sector(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        read_block(0x800, lba, buf)
    }
    fn write_sector(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        write_block(0x800, lba, buf)
    }
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
        unsafe {
            let ptr = self.sectors.as_ptr() as *mut [u8; SECTOR_SIZE];
            (*ptr.add(idx)).copy_from_slice(buf);
        }
        Ok(())
    }
}

unsafe impl Send for MockBackend {}
unsafe impl Sync for MockBackend {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Superblock {
    magic: u64,
    version: u32,
    inode_table_lba: u64,
    freelist_lba: u64,
    generation: u64,
    total_blocks: u64,
    inode_count: u64,
    crc32: u32,
    _pad: [u8; 512 - 48],
}

impl Superblock {
    fn new(total_blocks: u64) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            inode_table_lba: 1,
            freelist_lba: 1 + 1024, // 1M inodes = 512 sectors of inode table
            generation: 1,
            total_blocks,
            inode_count: 1024 * 1024,
            crc32: 0,
            _pad: [0; 512 - 48],
        }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                size_of::<Self>(),
            )
        }
    }

    fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < size_of::<Self>() {
            return None;
        }
        let sb = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const Self) };
        if sb.magic != MAGIC {
            return None;
        }
        Some(sb)
    }
}

#[derive(Clone, Debug)]
struct InodeRecord {
    id: u64,
    kind: u8, // 0=file, 1=dir
    parent: u64,
    data_lba: u64,
    data_blocks: u32,
    voronoi_cell: u32,
    subcell: u32,
    original_size: u64,
    permissions: u16,
    created_ms: u64,
    modified_ms: u64,
    content_hash: u64,
    name_len: u16,
    name: [u8; 128],
    _pad: [u8; 256 - 194],
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
            parent: ino.parent,
            data_lba,
            data_blocks,
            voronoi_cell: ino.voronoi_cell as u32,
            subcell: 0,
            original_size: ino.metadata.original_size,
            permissions: ino.metadata.permissions,
            created_ms: ino.metadata.created_ms,
            modified_ms: ino.metadata.modified_ms,
            content_hash: ino.payload.content_hash,
            name_len: name_len as u16,
            name,
            _pad: [0; 256 - 194],
        }
    }

    fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                size_of::<Self>(),
            )
        }
    }
}

/// O(1) persistence layer.
pub struct BlockStore {
    backend: Option<Box<dyn BlockStoreBackend>>,
    superblock: Option<Superblock>,
    // In-memory caches for when backend is unavailable
    inode_records: Vec<Option<InodeRecord>>,
    data_extents: Vec<Option<Vec<u8>>>,
    payload_extents: Vec<Option<ManifoldPayload>>,
    freelist: Vec<u64>, // free extent IDs
    next_extent_id: u64,
    dirty_inodes: Vec<u64>,
}

impl BlockStore {
    pub fn new() -> Self {
        Self {
            backend: None,
            superblock: None,
            inode_records: Vec::new(),
            data_extents: Vec::new(),
            payload_extents: Vec::new(),
            freelist: Vec::new(),
            next_extent_id: 0,
            dirty_inodes: Vec::new(),
        }
    }

    pub fn with_mock(num_sectors: usize) -> Self {
        let mut s = Self::new();
        s.backend = Some(Box::new(MockBackend::new(num_sectors)));
        s.format(1024 * 1024);
        s
    }

    pub fn try_mount_ahci(&mut self) -> bool {
        let backend = AhciBackend;
        let mut buf = [0u8; SECTOR_SIZE];
        if backend.read_sector(0, &mut buf).is_err() {
            serial_println!("[BlockStore] AHCI not available");
            return false;
        }
        if let Some(sb) = Superblock::from_bytes(&buf) {
            serial_println!("[BlockStore] Mounted existing filesystem, gen={}", sb.generation);
            self.superblock = Some(sb);
        } else {
            serial_println!("[BlockStore] No valid superblock; formatting");
            self.format_with_backend(Box::new(backend));
            return true;
        }
        self.backend = Some(Box::new(backend));
        true
    }

    pub fn format(&mut self, total_blocks: u64) {
        self.superblock = Some(Superblock::new(total_blocks));
        self.inode_records.clear();
        self.data_extents.clear();
        self.payload_extents.clear();
        self.freelist.clear();
        self.next_extent_id = 0;
        self.dirty_inodes.clear();
    }

    fn format_with_backend(&mut self, backend: Box<dyn BlockStoreBackend>) {
        self.backend = Some(backend);
        self.format(1024 * 1024);
        if let Some(ref sb) = self.superblock {
            let _ = self.backend.as_ref().unwrap().write_sector(0, sb.as_bytes());
        }
    }

    fn alloc_extent(&mut self, payload: &ManifoldPayload, data: &[u8]) -> (u64, u32) {
        let total = payload_bytes(payload) + data.len();
        let blocks_needed = ((total + SECTOR_SIZE - 1) / SECTOR_SIZE).max(1) as u32;
        let blocks = blocks_needed.min(MAX_EXTENT_BLOCKS);

        let id = if let Some(id) = self.freelist.pop() {
            id
        } else {
            let id = self.next_extent_id;
            self.next_extent_id += 1;
            id
        };

        // Ensure vectors are large enough
        while self.data_extents.len() <= id as usize {
            self.data_extents.push(None);
            self.payload_extents.push(None);
        }

        let mut extent = Vec::with_capacity((blocks as usize) * SECTOR_SIZE);
        extent.extend_from_slice(&payload_to_bytes(payload));
        extent.extend_from_slice(data);
        extent.resize((blocks as usize) * SECTOR_SIZE, 0);
        self.data_extents[id as usize] = Some(extent);
        self.payload_extents[id as usize] = Some(payload.clone());

        (id, blocks)
    }

    fn free_extent(&mut self, id: u64) {
        let idx = id as usize;
        if idx < self.data_extents.len() {
            self.data_extents[idx] = None;
            self.payload_extents[idx] = None;
        }
        self.freelist.push(id);
    }

    pub fn write_inode(&mut self, inode: &Inode, data: &[u8]) {
        let slot = (inode.id & 0xFFFF_FFFF) as usize;
        while self.inode_records.len() <= slot {
            self.inode_records.push(None);
        }

        // Allocate or reuse extent
        let (extent_id, blocks) = if let Some(ref old) = self.inode_records[slot] {
            if old.data_blocks > 0 {
                // Try to reuse if size fits
                let old_total = (old.data_blocks as usize) * SECTOR_SIZE;
                let new_total = payload_bytes(&inode.payload) + data.len();
                if new_total <= old_total {
                    // Reuse same extent
                    let idx = old.data_lba as usize;
                    if idx < self.data_extents.len() {
                        let mut extent = Vec::with_capacity(old_total);
                        extent.extend_from_slice(&payload_to_bytes(&inode.payload));
                        extent.extend_from_slice(data);
                        extent.resize(old_total, 0);
                        self.data_extents[idx] = Some(extent);
                        self.payload_extents[idx] = Some(inode.payload.clone());
                    }
                    self.inode_records[slot] = Some(InodeRecord::from_inode(inode, old.data_lba, old.data_blocks));
                    self.dirty_inodes.push(inode.id);
                    return;
                }
                self.free_extent(old.data_lba);
            }
            (0, 0)
        } else {
            (0, 0)
        };

        let (extent_id, blocks) = if extent_id == 0 && blocks == 0 {
            self.alloc_extent(&inode.payload, data)
        } else {
            (extent_id, blocks)
        };

        self.inode_records[slot] = Some(InodeRecord::from_inode(inode, extent_id, blocks));
        self.dirty_inodes.push(inode.id);
    }

    pub fn read_data(&self, inode_id: u64) -> Option<(ManifoldPayload, Vec<u8>)> {
        let slot = (inode_id & 0xFFFF_FFFF) as usize;
        let record = self.inode_records.get(slot)?.as_ref()?;
        let idx = record.data_lba as usize;
        let extent = self.data_extents.get(idx)?.as_ref()?;
        let payload = self.payload_extents.get(idx)?.as_ref()?.clone();
        let payload_len = payload_bytes(&payload);
        let data = extent[payload_len..payload_len + record.original_size as usize].to_vec();
        Some((payload, data))
    }

    pub fn read_payload(&self, inode_id: u64) -> Option<ManifoldPayload> {
        let slot = (inode_id & 0xFFFF_FFFF) as usize;
        let record = self.inode_records.get(slot)?.as_ref()?;
        let idx = record.data_lba as usize;
        self.payload_extents.get(idx)?.as_ref().cloned()
    }

    pub fn flush_inode(&mut self, inode_id: u64) -> Result<(), BlockError> {
        let Some(ref backend) = self.backend else { return Ok(()); };
        let slot = (inode_id & 0xFFFF_FFFF) as usize;
        let record = match self.inode_records.get(slot).and_then(|r| r.as_ref()) {
            Some(r) => r.clone(),
            None => return Ok(()),
        };

        // Write data extent first
        let idx = record.data_lba as usize;
        if let Some(ref extent) = self.data_extents.get(idx).and_then(|e| e.as_ref()) {
            let blocks = record.data_blocks as usize;
            for i in 0..blocks {
                let lba = record.data_lba + i as u64;
                let start = i * SECTOR_SIZE;
                let end = start + SECTOR_SIZE;
                backend.write_sector(lba, &extent[start..end])?;
            }
        }

        // Write inode record
        let inode_lba = 1 + (slot as u64 / 2);
        let mut sector = [0u8; SECTOR_SIZE];
        backend.read_sector(inode_lba, &mut sector)?;
        let offset_in_sector = (slot % 2) * INODE_RECORD_SIZE;
        sector[offset_in_sector..offset_in_sector + INODE_RECORD_SIZE]
            .copy_from_slice(record.as_bytes());
        backend.write_sector(inode_lba, &sector)?;

        Ok(())
    }

    pub fn flush_all(&mut self) -> Result<(), BlockError> {
        let dirty: Vec<u64> = self.dirty_inodes.drain(..).collect();
        for id in dirty {
            self.flush_inode(id)?;
        }
        if let Some(ref sb) = self.superblock {
            if let Some(ref backend) = self.backend {
                backend.write_sector(0, sb.as_bytes())?;
            }
        }
        Ok(())
    }

    pub fn is_mounted(&self) -> bool {
        self.superblock.is_some()
    }
}

fn payload_bytes(payload: &ManifoldPayload) -> usize {
    payload.point_count * size_of::<[f64; 3]>()
}

fn payload_to_bytes(payload: &ManifoldPayload) -> Vec<u8> {
    let mut buf = Vec::with_capacity(payload.point_count * size_of::<[f64; 3]>());
    for pt in &payload.points {
        for &c in &pt.coords {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    buf
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;
    use super::super::encoder;

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
        let mut store = BlockStore::with_mock(4096);
        let inode = dummy_inode(1, "test");
        store.write_inode(&inode, b"hello world");
        let (payload, data) = store.read_data(1).unwrap();
        test_assert_eq!(data, b"hello world");
        test_assert!(payload.point_count > 0);
        TestResult::Pass
    }

    fn test_payload_first_on_disk() -> TestResult {
        let mut store = BlockStore::with_mock(4096);
        let mut inode = dummy_inode(1, "test");
        inode.payload = encoder::encode_text("alpha");
        store.write_inode(&inode, b"rawdata");
        store.flush_inode(1).unwrap();

        // Read sector 2 directly (data extent for inode 1)
        let backend = store.backend.as_ref().unwrap();
        let mut sector = [0u8; SECTOR_SIZE];
        backend.read_sector(2, &mut sector).unwrap();

        // First bytes should be the payload's first coordinate
        let first_coord = f64::from_le_bytes([
            sector[0], sector[1], sector[2], sector[3],
            sector[4], sector[5], sector[6], sector[7],
        ]);
        let expected = inode.payload.points[0].coords[0].to_le_bytes();
        test_assert_eq!(sector[0..8], expected[..]);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("block_store::roundtrip", test_roundtrip);
        crate::testing::register_test("block_store::payload_first_on_disk", test_payload_first_on_disk);
    }
}
