// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! FAT12/16/32 filesystem driver (read-only v1).

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::drivers::block;
use crate::fs::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatType {
    Fat12,
    Fat16,
    Fat32,
}

/// BIOS Parameter Block (BPB) — on-disk structure.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BiosParameterBlock {
    pub boot_jump: [u8; 3],
    pub oem_name: [u8; 8],
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    pub reserved_sectors: u16,
    pub num_fats: u8,
    pub root_entries: u16,
    pub total_sectors_16: u16,
    pub media: u8,
    pub fat_size_16: u16,
    pub sectors_per_track: u16,
    pub num_heads: u16,
    pub hidden_sectors: u32,
    pub total_sectors_32: u32,
    pub fat_size_32: u32,
    pub root_cluster: u32,
    pub _padding: [u8; 468],
}

/// On-disk 32-byte directory entry.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct DirEntry {
    pub name: [u8; 8],
    pub ext: [u8; 3],
    pub attr: u8,
    pub nt_res: u8,
    pub crt_time_tenth: u8,
    pub crt_time: u16,
    pub crt_date: u16,
    pub lst_acc_date: u16,
    pub first_cluster_high: u16,
    pub wrt_time: u16,
    pub wrt_date: u16,
    pub first_cluster_low: u16,
    pub size: u32,
}

#[derive(Debug, Clone)]
struct FatDirEntry {
    name: String,
    attr: u8,
    first_cluster: u32,
    size: u32,
}

pub struct FatFs {
    dev_num: u32,
    bpb: BiosParameterBlock,
    fat_type: FatType,
    fat_start: u64,
    root_dir_start: u64,
    data_start: u64,
    clusters: u32,
    sectors_per_fat: u32,
    root_dir_sectors: u32,
    bytes_per_sector: u16,
    sectors_per_cluster: u8,
    root_cluster: u32,
}

impl FatFs {
    pub fn new(dev_num: u32) -> Self {
        Self {
            dev_num,
            bpb: BiosParameterBlock {
                boot_jump: [0; 3],
                oem_name: [0; 8],
                bytes_per_sector: 0,
                sectors_per_cluster: 0,
                reserved_sectors: 0,
                num_fats: 0,
                root_entries: 0,
                total_sectors_16: 0,
                media: 0,
                fat_size_16: 0,
                sectors_per_track: 0,
                num_heads: 0,
                hidden_sectors: 0,
                total_sectors_32: 0,
                fat_size_32: 0,
                root_cluster: 0,
                _padding: [0; 468],
            },
            fat_type: FatType::Fat12,
            fat_start: 0,
            root_dir_start: 0,
            data_start: 0,
            clusters: 0,
            sectors_per_fat: 0,
            root_dir_sectors: 0,
            bytes_per_sector: 0,
            sectors_per_cluster: 0,
            root_cluster: 0,
        }
    }

    pub fn mount(&mut self) -> Result<(), VfsError> {
        let mut buf = [0u8; 512];
        block::read_block(self.dev_num, 0, &mut buf).map_err(|_| VfsError::IoError)?;
        if buf[510] != 0x55 || buf[511] != 0xAA {
            return Err(VfsError::NotSupported);
        }
        let bpb = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const BiosParameterBlock) };
        if bpb.bytes_per_sector == 0 || bpb.sectors_per_cluster == 0 {
            return Err(VfsError::NotSupported);
        }
        self.bpb = bpb;
        self.bytes_per_sector = bpb.bytes_per_sector;
        self.sectors_per_cluster = bpb.sectors_per_cluster;

        let total_sectors = if bpb.total_sectors_16 == 0 {
            bpb.total_sectors_32
        } else {
            bpb.total_sectors_16 as u32
        };
        let fat_size = if bpb.fat_size_16 == 0 {
            bpb.fat_size_32
        } else {
            bpb.fat_size_16 as u32
        };
        if fat_size == 0 || total_sectors == 0 {
            return Err(VfsError::NotSupported);
        }

        let root_dir_sectors =
            ((bpb.root_entries as u32 * 32) + (bpb.bytes_per_sector as u32 - 1))
                / bpb.bytes_per_sector as u32;
        let data_sectors = total_sectors.saturating_sub(
            (bpb.reserved_sectors as u32) + (bpb.num_fats as u32 * fat_size) + root_dir_sectors,
        );
        self.clusters = data_sectors / bpb.sectors_per_cluster as u32;

        self.fat_type = if self.clusters < 4085 {
            FatType::Fat12
        } else if self.clusters < 65525 {
            FatType::Fat16
        } else {
            FatType::Fat32
        };

        self.fat_start = bpb.reserved_sectors as u64;
        self.root_dir_start = self.fat_start + ((bpb.num_fats as u64) * (fat_size as u64));
        self.data_start = self.root_dir_start + root_dir_sectors as u64;
        self.sectors_per_fat = fat_size;
        self.root_dir_sectors = root_dir_sectors;

        if self.fat_type == FatType::Fat32 {
            self.root_cluster = bpb.root_cluster;
        } else {
            self.root_cluster = 0;
        }
        Ok(())
    }

    fn cluster_to_sector(&self, cluster: u32) -> u64 {
        self.data_start + ((cluster.saturating_sub(2)) as u64) * (self.sectors_per_cluster as u64)
    }

    fn read_fat_entry(&self, cluster: u32) -> Result<u32, VfsError> {
        let bytes_per_sector = self.bytes_per_sector as u32;
        match self.fat_type {
            FatType::Fat12 => {
                let fat_offset = cluster * 12 / 8;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; (self.bytes_per_sector as usize) * 2];
                block::read_block(self.dev_num, fat_sector, &mut buf[..self.bytes_per_sector as usize])
                    .map_err(|_| VfsError::IoError)?;
                if entry_offset == self.bytes_per_sector as usize - 1 {
                    block::read_block(
                        self.dev_num,
                        fat_sector + 1,
                        &mut buf[self.bytes_per_sector as usize..],
                    )
                    .map_err(|_| VfsError::IoError)?;
                }
                let entry = if cluster % 2 == 0 {
                    (buf[entry_offset] as u16 | ((buf[entry_offset + 1] as u16 & 0x0F) << 8))
                        as u32
                } else {
                    ((buf[entry_offset] as u16 >> 4)
                        | ((buf[entry_offset + 1] as u16) << 4)) as u32
                };
                Ok(entry)
            }
            FatType::Fat16 => {
                let fat_offset = cluster * 2;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, fat_sector, &mut buf).map_err(|_| VfsError::IoError)?;
                let entry = u16::from_le_bytes([buf[entry_offset], buf[entry_offset + 1]]) as u32;
                Ok(entry)
            }
            FatType::Fat32 => {
                let fat_offset = cluster * 4;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, fat_sector, &mut buf).map_err(|_| VfsError::IoError)?;
                let entry = u32::from_le_bytes([
                    buf[entry_offset],
                    buf[entry_offset + 1],
                    buf[entry_offset + 2],
                    buf[entry_offset + 3],
                ]) & 0x0FFFFFFF;
                Ok(entry)
            }
        }
    }

    fn next_cluster(&self, cluster: u32) -> Result<Option<u32>, VfsError> {
        let entry = self.read_fat_entry(cluster)?;
        let eoc = match self.fat_type {
            FatType::Fat12 => entry >= 0xFF8,
            FatType::Fat16 => entry >= 0xFFF8,
            FatType::Fat32 => entry >= 0x0FFFFFF8,
        };
        if eoc || entry == 0 {
            Ok(None)
        } else {
            Ok(Some(entry))
        }
    }

    fn read_file(&self, cluster: u32, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> {
        if buf.is_empty() || cluster < 2 {
            return Ok(0);
        }
        let cluster_size = (self.sectors_per_cluster as u64) * (self.bytes_per_sector as u64);

        let mut cluster = cluster;
        let mut skipped = 0u64;
        while skipped + cluster_size <= offset {
            match self.next_cluster(cluster)? {
                Some(next) => {
                    cluster = next;
                    skipped += cluster_size;
                }
                None => return Ok(0),
            }
        }

        let mut sector_offset = (offset - skipped) as usize;
        let mut buf_written = 0usize;

        while buf_written < buf.len() {
            let sector_in_cluster = sector_offset / self.bytes_per_sector as usize;
            let byte_in_sector = sector_offset % self.bytes_per_sector as usize;
            let sector = self.cluster_to_sector(cluster) + sector_in_cluster as u64;
            let mut sector_buf = alloc::vec![0u8; self.bytes_per_sector as usize];
            block::read_block(self.dev_num, sector, &mut sector_buf).map_err(|_| VfsError::IoError)?;
            let to_copy = core::cmp::min(
                buf.len() - buf_written,
                self.bytes_per_sector as usize - byte_in_sector,
            );
            buf[buf_written..buf_written + to_copy]
                .copy_from_slice(&sector_buf[byte_in_sector..byte_in_sector + to_copy]);
            buf_written += to_copy;
            sector_offset += to_copy;

            if sector_offset >= cluster_size as usize {
                match self.next_cluster(cluster)? {
                    Some(next) => {
                        cluster = next;
                        sector_offset = 0;
                    }
                    None => break,
                }
            }
        }
        Ok(buf_written)
    }

    fn parse_dir_entries(data: &[u8]) -> Vec<FatDirEntry> {
        let mut entries = Vec::new();
        let mut offset = 0;
        while offset + 32 <= data.len() {
            let raw = unsafe { core::ptr::read_unaligned(data.as_ptr().add(offset) as *const DirEntry) };
            offset += 32;
            if raw.name[0] == 0x00 {
                break;
            }
            if raw.name[0] == 0xE5 {
                continue;
            }
            if raw.attr == 0x0F {
                continue;
            }
            if raw.attr & 0x08 != 0 {
                continue;
            }

            let name_len = raw.name.iter().position(|&b| b == b' ').unwrap_or(raw.name.len());
            let name = core::str::from_utf8(&raw.name[..name_len]).unwrap_or("").to_string();
            let ext_len = raw.ext.iter().position(|&b| b == b' ').unwrap_or(raw.ext.len());
            let ext = core::str::from_utf8(&raw.ext[..ext_len]).unwrap_or("").to_string();
            let full_name = if ext.is_empty() {
                name
            } else {
                format!("{}.{}", name, ext)
            };

            let first_cluster =
                ((raw.first_cluster_high as u32) << 16) | (raw.first_cluster_low as u32);
            entries.push(FatDirEntry {
                name: full_name,
                attr: raw.attr,
                first_cluster,
                size: raw.size,
            });
        }
        entries
    }

    fn read_root_dir(&self) -> Result<Vec<FatDirEntry>, VfsError> {
        let size = self.bpb.root_entries as usize * 32;
        let sectors = (size + self.bytes_per_sector as usize - 1) / self.bytes_per_sector as usize;
        let mut data = alloc::vec![0u8; size];
        for i in 0..sectors {
            let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
            block::read_block(self.dev_num, self.root_dir_start + i as u64, &mut buf)
                .map_err(|_| VfsError::IoError)?;
            let start = i * self.bytes_per_sector as usize;
            let end = core::cmp::min(start + self.bytes_per_sector as usize, size);
            data[start..end].copy_from_slice(&buf[..end - start]);
        }
        Ok(Self::parse_dir_entries(&data))
    }

    fn read_dir_cluster(&self, cluster: u32) -> Result<Vec<FatDirEntry>, VfsError> {
        let mut data = Vec::new();
        let mut cluster = cluster;
        loop {
            let sector = self.cluster_to_sector(cluster);
            for s in 0..self.sectors_per_cluster {
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, sector + s as u64, &mut buf)
                    .map_err(|_| VfsError::IoError)?;
                data.extend_from_slice(&buf);
            }
            match self.next_cluster(cluster)? {
                Some(next) => cluster = next,
                None => break,
            }
        }
        Ok(Self::parse_dir_entries(&data))
    }

    fn find_entry_by_cluster(&self, cluster: u32) -> Result<FatDirEntry, VfsError> {
        let mut queue = VecDeque::new();
        let root_entries = if self.fat_type != FatType::Fat32 {
            self.read_root_dir()?
        } else {
            self.read_dir_cluster(self.root_cluster)?
        };
        queue.push_back(root_entries);

        while let Some(entries) = queue.pop_front() {
            for e in entries {
                if e.first_cluster == cluster {
                    return Ok(e);
                }
                if e.attr & 0x10 != 0 && e.first_cluster >= 2 {
                    let sub = self.read_dir_cluster(e.first_cluster)?;
                    queue.push_back(sub);
                }
            }
        }
        Err(VfsError::NotFound)
    }

    fn lookup_path(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let path = path.trim_matches('/');
        if path.is_empty() {
            let inode = if self.fat_type == FatType::Fat32 {
                self.root_cluster as u64
            } else {
                1
            };
            return Ok(VfsHandle { fs_idx: 0, inode });
        }

        let mut current_cluster = if self.fat_type == FatType::Fat32 {
            self.root_cluster
        } else {
            0
        };

        let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        for (i, component) in components.iter().enumerate() {
            let entries = if current_cluster == 0 {
                self.read_root_dir()?
            } else {
                self.read_dir_cluster(current_cluster)?
            };

            let mut found = false;
            for e in entries {
                if e.name == *component {
                    if i == components.len() - 1 {
                        return Ok(VfsHandle {
                            fs_idx: 0,
                            inode: e.first_cluster as u64,
                        });
                    } else {
                        if e.attr & 0x10 == 0 {
                            return Err(VfsError::NotADirectory);
                        }
                        current_cluster = e.first_cluster;
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return Err(VfsError::NotFound);
            }
        }
        Err(VfsError::NotFound)
    }
}

impl FileSystem for FatFs {
    fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        self.lookup_path(path)
    }

    fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> {
        if handle.inode == 1 && self.fat_type != FatType::Fat32 {
            return Err(VfsError::NotSupported);
        }
        if self.fat_type == FatType::Fat32 && handle.inode == self.root_cluster as u64 {
            return Err(VfsError::NotSupported);
        }
        let entry = self.find_entry_by_cluster(handle.inode as u32)?;
        if entry.attr & 0x10 != 0 {
            return Err(VfsError::NotSupported);
        }
        let cluster = entry.first_cluster;
        if cluster < 2 {
            return Ok(0);
        }
        let max_read = (entry.size as u64).saturating_sub(offset);
        if max_read == 0 {
            return Ok(0);
        }
        let to_read = core::cmp::min(buf.len(), max_read as usize);
        self.read_file(cluster, &mut buf[..to_read], offset)
    }

    fn write(&mut self, _handle: VfsHandle, _buf: &[u8], _offset: u64) -> Result<usize, VfsError> {
        Err(VfsError::NotSupported)
    }

    fn create(&mut self, _path: &str) -> Result<VfsHandle, VfsError> {
        Err(VfsError::NotSupported)
    }

    fn mkdir(&mut self, _path: &str) -> Result<VfsHandle, VfsError> {
        Err(VfsError::NotSupported)
    }

    fn unlink(&mut self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn rmdir(&mut self, _path: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn rename(&mut self, _old: &str, _new: &str) -> Result<(), VfsError> {
        Err(VfsError::NotSupported)
    }

    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        let is_dir = if handle.inode == 1 && self.fat_type != FatType::Fat32 {
            true
        } else if self.fat_type == FatType::Fat32 && handle.inode == self.root_cluster as u64 {
            true
        } else {
            match self.find_entry_by_cluster(handle.inode as u32) {
                Ok(e) => e.attr & 0x10 != 0,
                Err(_) => false,
            }
        };
        if !is_dir {
            return Err(VfsError::NotADirectory);
        }

        let entries = if handle.inode == 1 && self.fat_type != FatType::Fat32 {
            self.read_root_dir()?
        } else {
            self.read_dir_cluster(handle.inode as u32)?
        };
        let mut result = Vec::new();
        for e in entries {
            let node_type = if e.attr & 0x10 != 0 {
                VfsNodeType::Directory
            } else {
                VfsNodeType::File
            };
            result.push(VfsDirEntry {
                name: e.name,
                node_type,
            });
        }
        Ok(result)
    }

    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        if handle.inode == 1 && self.fat_type != FatType::Fat32 {
            return Ok(VfsNode {
                size: 0,
                permissions: 0o755,
                uid: 0,
                gid: 0,
                mode: 0o755,
                atime: 0,
                mtime: 0,
                node_type: VfsNodeType::Directory,
                major: 0,
                minor: 0,
            });
        }
        if self.fat_type == FatType::Fat32 && handle.inode == self.root_cluster as u64 {
            return Ok(VfsNode {
                size: 0,
                permissions: 0o755,
                uid: 0,
                gid: 0,
                mode: 0o755,
                atime: 0,
                mtime: 0,
                node_type: VfsNodeType::Directory,
                major: 0,
                minor: 0,
            });
        }
        let entry = self.find_entry_by_cluster(handle.inode as u32)?;
        let node_type = if entry.attr & 0x10 != 0 {
            VfsNodeType::Directory
        } else {
            VfsNodeType::File
        };
        Ok(VfsNode {
            size: entry.size as u64,
            permissions: if node_type == VfsNodeType::Directory { 0o755 } else { 0o644 },
            uid: 0,
            gid: 0,
            mode: if node_type == VfsNodeType::Directory { 0o755 } else { 0o644 },
            atime: 0,
            mtime: 0,
            node_type,
            major: 0,
            minor: 0,
        })
    }

    fn mknod(
        &mut self,
        _path: &str,
        _node_type: VfsNodeType,
        _major: u32,
        _minor: u32,
    ) -> Result<VfsHandle, VfsError> {
        Err(VfsError::NotSupported)
    }
}

// Mount integration helper (optional).
// Example usage in `fs::init_vfs` or `lib.rs`:
//
//     let mut fat = fat::FatFs::new(dev_num);
//     if fat.mount().is_ok() {
//         vfs.mount("/fat", Box::new(fat))?;
//     }
