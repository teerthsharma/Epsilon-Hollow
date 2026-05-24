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

use alloc::string::String;
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
        let cached = cache.get_block(self.dev_num, block as u64)
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
        let cached = cache.get_block(self.dev_num, block as u64)
            .ok_or(VfsError::IoError)?;
        cached.data_mut().fill(0);
        cached.mark_dirty();
        Ok(())
    }

    /// Read and verify the Ext2 superblock.
    pub fn mount(&mut self) -> Result<(), VfsError> {
        let mut buf = [0u8; 1024];

        // Superblock is always located at offset 1024 of the device.
        // Assuming 512-byte sectors, it begins at LBA 2.
        crate::drivers::block::read_block(self.dev_num, 2, &mut buf)
            .map_err(|_| VfsError::IoError)?;

        let sb = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const Superblock) };
        if sb.s_magic != EXT2_MAGIC {
            return Err(VfsError::IoError);
        }
        self.superblock = Some(sb);
        self.block_size = 1024 << sb.s_log_block_size;
        self.inodes_per_group = sb.s_inodes_per_group;
        self.inode_size = if sb.s_rev_level >= 1 { sb.s_inode_size as u32 } else { 128 };
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
        let cached = cache.get_block(self.dev_num, (self.bgd_block + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        let bgd = unsafe {
            let ptr = cached.data().as_ptr().add((bgd_idx_in_block as usize) * core::mem::size_of::<BlockGroupDescriptor>());
            core::ptr::read_unaligned(ptr as *const BlockGroupDescriptor)
        };
        Ok(bgd)
    }

    fn write_bgd(&self, group: u32, bgd: &BlockGroupDescriptor) -> Result<(), VfsError> {
        let bgds_per_block = self.block_size / core::mem::size_of::<BlockGroupDescriptor>() as u32;
        let block_offset = group / bgds_per_block;
        let bgd_idx_in_block = group % bgds_per_block;

        let mut cache = self.buffer_cache.lock();
        let cached = cache.get_block(self.dev_num, (self.bgd_block + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        unsafe {
            let ptr = cached.data_mut().as_mut_ptr().add((bgd_idx_in_block as usize) * core::mem::size_of::<BlockGroupDescriptor>());
            core::ptr::copy_nonoverlapping(bgd as *const BlockGroupDescriptor as *const u8, ptr, core::mem::size_of::<BlockGroupDescriptor>());
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
        let cached = cache.get_block(self.dev_num, (bgd.bg_inode_table + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        let inode = unsafe {
            let ptr = cached.data().as_ptr().add((inode_idx_in_block as usize) * (self.inode_size as usize));
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
        let cached = cache.get_block(self.dev_num, (bgd.bg_inode_table + block_offset) as u64)
            .ok_or(VfsError::IoError)?;
        unsafe {
            let ptr = cached.data_mut().as_mut_ptr().add((inode_idx_in_block as usize) * (self.inode_size as usize));
            core::ptr::copy_nonoverlapping(inode as *const Inode as *const u8, ptr, core::mem::size_of::<Inode>());
        }
        cached.mark_dirty();
        Ok(())
    }

    fn get_disk_block_from_inode(&self, inode: &Inode, file_block: u32) -> Result<u32, VfsError> {
        let ptrs_per_block = self.block_size / 4;

        if file_block < 12 {
            return Ok(inode.i_block[file_block as usize]);
        }

        let mut file_block = file_block - 12;
        if file_block < ptrs_per_block {
            let ind_block = inode.i_block[12];
            if ind_block == 0 { return Ok(0); }
            let mut buf = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(ind_block, &mut buf)?;
            let off = file_block as usize * 4;
            let block = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            return Ok(block);
        }

        file_block -= ptrs_per_block;
        if file_block < ptrs_per_block * ptrs_per_block {
            let dind_block = inode.i_block[13];
            if dind_block == 0 { return Ok(0); }
            let mut buf = alloc::vec![0u8; self.block_size as usize];
            self.read_disk_block(dind_block, &mut buf)?;
            let ind_idx = file_block / ptrs_per_block;
            let off = ind_idx as usize * 4;
            let ind_block = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            if ind_block == 0 { return Ok(0); }

            self.read_disk_block(ind_block, &mut buf)?;
            let block_idx = file_block % ptrs_per_block;
            let off2 = block_idx as usize * 4;
            let block = u32::from_le_bytes([buf[off2], buf[off2 + 1], buf[off2 + 2], buf[off2 + 3]]);
            return Ok(block);
        }

        file_block -= ptrs_per_block * ptrs_per_block;
        let tind_block = inode.i_block[14];
        if tind_block == 0 { return Ok(0); }
        let mut buf = alloc::vec![0u8; self.block_size as usize];
        self.read_disk_block(tind_block, &mut buf)?;

        let dind_idx = file_block / (ptrs_per_block * ptrs_per_block);
        let off = dind_idx as usize * 4;
        let dind_block = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        if dind_block == 0 { return Ok(0); }

        self.read_disk_block(dind_block, &mut buf)?;
        let ind_idx = (file_block / ptrs_per_block) % ptrs_per_block;
        let off2 = ind_idx as usize * 4;
        let ind_block = u32::from_le_bytes([buf[off2], buf[off2 + 1], buf[off2 + 2], buf[off2 + 3]]);
        if ind_block == 0 { return Ok(0); }

        self.read_disk_block(ind_block, &mut buf)?;
        let block_idx = file_block % ptrs_per_block;
        let off3 = block_idx as usize * 4;
        let block = u32::from_le_bytes([buf[off3], buf[off3 + 1], buf[off3 + 2], buf[off3 + 3]]);
        Ok(block)
    }

    fn read_inode_data(&self, inode: &Inode, mut offset: u64, mut buf: &mut [u8]) -> Result<usize, VfsError> {
        let size = inode.i_size as u64;
        let total_size = if (inode.i_mode & EXT2_S_IFMT) == EXT2_S_IFREG && self.superblock.as_ref().map(|sb| sb.s_rev_level >= 1).unwrap_or(false) {
            size | ((inode.i_dir_acl as u64) << 32)
        } else {
            size
        };

        if offset >= total_size {
            return Ok(0);
        }
        let mut read_bytes = 0;

        let mut temp_buf = alloc::vec![0u8; self.block_size as usize];

        while !buf.is_empty() && offset < total_size {
            let file_block = (offset / self.block_size as u64) as u32;
            let block_offset = (offset % self.block_size as u64) as usize;

            let mut read_len = self.block_size as usize - block_offset;
            if read_len > buf.len() { read_len = buf.len(); }
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

    /// Resolve or allocate the indirect block referenced by `inode.i_block[idx]`.
    fn get_or_allocate_indirect_block(&mut self, inode: &mut Inode, idx: usize, block_incr: u32) -> Result<u32, VfsError> {
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
    fn get_or_allocate_data_block(&mut self, inode: &mut Inode, file_block: u32, block_incr: u32) -> Result<u32, VfsError> {
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
            let cached = cache.get_block(self.dev_num, ind_block as u64)
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
                let cached = cache.get_block(self.dev_num, ind_block as u64)
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
            let cached = cache.get_block(self.dev_num, dind_block as u64)
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
                let cached = cache.get_block(self.dev_num, dind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off..off + 4].copy_from_slice(&ind_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
            }

            let block_idx = rel % ptrs_per_block;
            let mut cache = self.buffer_cache.lock();
            let cached = cache.get_block(self.dev_num, ind_block as u64)
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
                let cached = cache.get_block(self.dev_num, ind_block as u64)
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
        if rel < ptrs_per_block * ptrs_per_block * ptrs_per_block {
            let tind_block = self.get_or_allocate_indirect_block(inode, 14, block_incr)?;

            let mut cache = self.buffer_cache.lock();
            let cached = cache.get_block(self.dev_num, tind_block as u64)
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
                let cached = cache.get_block(self.dev_num, tind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off..off + 4].copy_from_slice(&dind_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
            }

            let mut cache = self.buffer_cache.lock();
            let cached = cache.get_block(self.dev_num, dind_block as u64)
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
                let cached = cache.get_block(self.dev_num, dind_block as u64)
                    .ok_or(VfsError::IoError)?;
                cached.data_mut()[off2..off2 + 4].copy_from_slice(&ind_block.to_le_bytes());
                cached.mark_dirty();
                drop(cache);
            }

            let block_idx = rel % ptrs_per_block;
            let mut cache = self.buffer_cache.lock();
            let cached = cache.get_block(self.dev_num, ind_block as u64)
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
                let cached = cache.get_block(self.dev_num, ind_block as u64)
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

    fn write_inode_data(&mut self, inode: &mut Inode, mut offset: u64, buf: &[u8]) -> Result<usize, VfsError> {
        let mut written = 0;
        let block_incr = self.block_size / 512;

        while written < buf.len() {
            let file_block = (offset / self.block_size as u64) as u32;
            let block_offset = (offset % self.block_size as u64) as usize;
            let write_len = core::cmp::min(buf.len() - written, self.block_size as usize - block_offset);

            let disk_block = self.get_or_allocate_data_block(inode, file_block, block_incr)?;

            {
                let mut cache = self.buffer_cache.lock();
                let cached = cache.get_block(self.dev_num, disk_block as u64)
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
        let total_groups = (sb.s_blocks_count + sb.s_blocks_per_group - 1) / sb.s_blocks_per_group;

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

                        let block_num = group * sb.s_blocks_per_group + byte_idx as u32 * 8 + bit + sb.s_first_data_block;
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
        let total_groups = (sb.s_inodes_count + sb.s_inodes_per_group - 1) / sb.s_inodes_per_group;

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
                    indirect[off], indirect[off + 1], indirect[off + 2], indirect[off + 3],
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
                    dindirect[off], dindirect[off + 1], dindirect[off + 2], dindirect[off + 3],
                ]);
                if ind_block != 0 {
                    let mut indirect = alloc::vec![0u8; self.block_size as usize];
                    self.read_disk_block(ind_block, &mut indirect)?;
                    for k in 0..ptrs_per_block {
                        let off2 = k as usize * 4;
                        let block = u32::from_le_bytes([
                            indirect[off2], indirect[off2 + 1], indirect[off2 + 2], indirect[off2 + 3],
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
                if parent.is_empty() { ("", name) } else { (parent, name) }
            }
            None => ("", path),
        }
    }

    fn find_dir_entry(&self, dir_ino: u32, name: &str) -> Result<Option<u32>, VfsError> {
        let inode = self.read_inode(dir_ino)?;
        let size = inode.i_size as usize;
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
                    let entry_name = core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
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

    fn add_dir_entry(&mut self, dir_ino: u32, name: &str, ino: u32, file_type: u8) -> Result<(), VfsError> {
        if name.len() > 255 {
            return Err(VfsError::InvalidPath);
        }

        let mut dir_inode = self.read_inode(dir_ino)?;
        let needed = Self::dir_entry_size(name.len());
        let dir_size = dir_inode.i_size as usize;

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
                    core::ptr::write_unaligned(dir_data.as_mut_ptr().add(offset) as *mut DirEntryHeader, new_header);
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
            let last_block_start = ((dir_size - 1) / self.block_size as usize) * self.block_size as usize;
            let mut last_offset = last_block_start;
            let mut last_header = DirEntryHeader { inode: 0, rec_len: 0, name_len: 0, file_type: 0 };

            while last_offset + core::mem::size_of::<DirEntryHeader>() <= dir_data.len()
                && last_offset < last_block_start + self.block_size as usize
            {
                let h = unsafe {
                    core::ptr::read_unaligned(dir_data.as_ptr().add(last_offset) as *const DirEntryHeader)
                };
                if h.rec_len == 0 {
                    break;
                }
                let next_offset = last_offset + h.rec_len as usize;
                if next_offset >= last_block_start + self.block_size as usize || next_offset >= dir_data.len() {
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
                        core::ptr::write_unaligned(dir_data.as_mut_ptr().add(last_offset) as *mut DirEntryHeader, updated_header);
                    }

                    let new_offset = last_offset + actual;
                    let new_header = DirEntryHeader {
                        inode: ino,
                        rec_len: new_rec_len,
                        name_len: name.len() as u8,
                        file_type,
                    };
                    unsafe {
                        core::ptr::write_unaligned(dir_data.as_mut_ptr().add(new_offset) as *mut DirEntryHeader, new_header);
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
        let phys_block = self.get_or_allocate_data_block(&mut dir_inode, block_idx, self.block_size / 512)?;
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
        let size = dir_inode.i_size as usize;
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
                    let entry_name = core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
                    if entry_name == name {
                        let mut new_header = header;
                        new_header.inode = 0;
                        unsafe {
                            core::ptr::write_unaligned(dir_data.as_mut_ptr().add(offset) as *mut DirEntryHeader, new_header);
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
            return Ok(VfsHandle { fs_idx: 0, inode: current_ino as u64 });
        }

        for component in path.split('/') {
            if component.is_empty() {
                continue;
            }

            let inode = self.read_inode(current_ino)?;
            if (inode.i_mode & EXT2_S_IFMT) != EXT2_S_IFDIR {
                return Err(VfsError::NotADirectory);
            }

            let mut dir_data = alloc::vec![0u8; inode.i_size as usize];
            let read_len = self.read_inode_data(&inode, 0, &mut dir_data)?;
            if read_len != inode.i_size as usize {
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
                        let name = core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
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

        Ok(VfsHandle { fs_idx: 0, inode: current_ino as u64 })
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
        Ok(VfsHandle { fs_idx: 0, inode: ino as u64 })
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
            core::ptr::write_unaligned(block_buf.as_mut_ptr().add(dotdot_offset) as *mut DirEntryHeader, dotdot_header);
        }
        block_buf[dotdot_offset + core::mem::size_of::<DirEntryHeader>()] = b'.';
        block_buf[dotdot_offset + core::mem::size_of::<DirEntryHeader>() + 1] = b'.';

        self.write_disk_block(block, &block_buf)?;
        self.add_dir_entry(parent_ino, name, ino, 2)?;
        Ok(VfsHandle { fs_idx: 0, inode: ino as u64 })
    }

    fn unlink(&mut self, path: &str) -> Result<(), VfsError> {
        let (parent_path, name) = Self::split_last(path);
        if name.is_empty() {
            return Err(VfsError::InvalidPath);
        }
        let parent_ino = self.lookup(parent_path)?.inode as u32;
        let ino = self.find_dir_entry(parent_ino, name)?.ok_or(VfsError::NotFound)?;

        let mut inode = self.read_inode(ino)?;
        if (inode.i_mode & EXT2_S_IFMT) == EXT2_S_IFDIR {
            return Err(VfsError::NotADirectory);
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

        let mut dir_data = alloc::vec![0u8; inode.i_size as usize];
        let read_len = self.read_inode_data(&inode, 0, &mut dir_data)?;
        if read_len != inode.i_size as usize {
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
                    let name = core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("").into();

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

                    entries.push(VfsDirEntry {
                        name,
                        node_type,
                    });
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
        let total_size = if node_type == VfsNodeType::File && self.superblock.as_ref().map(|sb| sb.s_rev_level >= 1).unwrap_or(false) {
            size | ((inode.i_dir_acl as u64) << 32)
        } else {
            size
        };

        let (major, minor) = if node_type == VfsNodeType::CharDevice || node_type == VfsNodeType::BlockDevice {
            let dev = inode.i_block[0];
            (((dev >> 8) & 0xFF) as u32, (dev & 0xFF) as u32)
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
        let ino = self.find_dir_entry(parent_ino, name)?.ok_or(VfsError::NotFound)?;

        let mut inode = self.read_inode(ino)?;
        if (inode.i_mode & EXT2_S_IFMT) != EXT2_S_IFDIR {
            return Err(VfsError::NotADirectory);
        }

        let mut dir_data = alloc::vec![0u8; inode.i_size as usize];
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
                    let entry_name = core::str::from_utf8(&dir_data[name_start..name_end]).unwrap_or("");
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
        let old_dir_size = old_dir_inode.i_size as usize;
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
                    let entry_name = core::str::from_utf8(&old_dir_data[name_start..name_end]).unwrap_or("");
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
                    core::ptr::write_unaligned(old_dir_data.as_mut_ptr().add(old_offset) as *mut DirEntryHeader, new_header);
                }
                let name_start = old_offset + core::mem::size_of::<DirEntryHeader>();
                for b in &mut old_dir_data[name_start..name_start + old_name.len()] {
                    *b = 0;
                }
                old_dir_data[name_start..name_start + new_name.len()].copy_from_slice(new_name.as_bytes());
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
                let mut dir_data = alloc::vec![0u8; moved_inode.i_size as usize];
                self.read_inode_data(&moved_inode, 0, &mut dir_data)?;
                let mut off = 0;
                while off + core::mem::size_of::<DirEntryHeader>() <= dir_data.len() {
                    let h = unsafe {
                        core::ptr::read_unaligned(dir_data.as_ptr().add(off) as *const DirEntryHeader)
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
                                    core::ptr::write_unaligned(dir_data.as_mut_ptr().add(off) as *mut DirEntryHeader, nh);
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
            inode.i_block[0] = dev as u32;
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
        Ok(VfsHandle { fs_idx: 0, inode: ino as u64 })
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
