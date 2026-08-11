// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! FAT12/16/32 filesystem driver with VFS read/write/create/delete/rename support.

use crate::drivers::block;
use crate::fs::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

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
            return Err(VfsError::IoError);
        }
        let bpb = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const BiosParameterBlock) };
        if bpb.bytes_per_sector == 0 || bpb.sectors_per_cluster == 0 {
            return Err(VfsError::IoError);
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
            return Err(VfsError::IoError);
        }

        let root_dir_sectors = (bpb.root_entries as u32 * 32).div_ceil(bpb.bytes_per_sector as u32);
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

    /// Rejects a cluster number that no well-formed volume can name.
    ///
    /// FAT numbers data clusters `2..=clusters + 1`: 0 and 1 are the reserved
    /// media-descriptor entries, and anything above the volume's cluster count
    /// addresses sectors outside the data region. Every cluster number this
    /// driver handles arrives off the disk unverified — as a directory entry's
    /// first cluster or as a FAT next-pointer — so the three places that turn
    /// one into a disk address (`cluster_to_sector`, `read_fat_entry`,
    /// `write_fat_entry`) check it here first. That covers the start of every
    /// walk as well as each hop, without a check at each of the walks.
    fn check_cluster(&self, cluster: u32) -> Result<(), VfsError> {
        if cluster < 2 || cluster > self.clusters.saturating_add(1) {
            return Err(VfsError::IoError);
        }
        Ok(())
    }

    fn cluster_to_sector(&self, cluster: u32) -> Result<u64, VfsError> {
        self.check_cluster(cluster)?;
        Ok(self.data_start + ((cluster - 2) as u64) * (self.sectors_per_cluster as u64))
    }

    fn read_fat_entry(&self, cluster: u32) -> Result<u32, VfsError> {
        self.check_cluster(cluster)?;
        let bytes_per_sector = self.bytes_per_sector as u32;
        match self.fat_type {
            FatType::Fat12 => {
                let fat_offset = cluster * 12 / 8;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; (self.bytes_per_sector as usize) * 2];
                block::read_block(
                    self.dev_num,
                    fat_sector,
                    &mut buf[..self.bytes_per_sector as usize],
                )
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
                    (buf[entry_offset] as u16 | ((buf[entry_offset + 1] as u16 & 0x0F) << 8)) as u32
                } else {
                    ((buf[entry_offset] as u16 >> 4) | ((buf[entry_offset + 1] as u16) << 4)) as u32
                };
                Ok(entry)
            }
            FatType::Fat16 => {
                let fat_offset = cluster * 2;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, fat_sector, &mut buf)
                    .map_err(|_| VfsError::IoError)?;
                let entry = u16::from_le_bytes([buf[entry_offset], buf[entry_offset + 1]]) as u32;
                Ok(entry)
            }
            FatType::Fat32 => {
                let fat_offset = cluster * 4;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, fat_sector, &mut buf)
                    .map_err(|_| VfsError::IoError)?;
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

    /// Follows one link of a cluster chain while guarding against corruption.
    ///
    /// A well-formed chain references each of the filesystem's data clusters
    /// at most once, so `self.clusters` is a hard upper bound on legitimate
    /// chain length — revisiting a cluster (most commonly a cycle, e.g.
    /// cluster 5 -> 6 -> 5 -> ...) can only happen on a corrupt on-disk FAT.
    /// Every chain-walking loop in this file shares this one helper instead
    /// of calling `next_cluster` directly, so the bound applies everywhere
    /// uniformly. `steps` is threaded through by the caller so a walk that
    /// spans multiple loops (e.g. a seek loop followed by a read loop over
    /// the same chain) shares a single budget rather than resetting it.
    ///
    /// The cap alone is a poor defence on a large volume — a cluster pointing
    /// at itself would still cost `self.clusters` FAT reads before tripping it
    /// — so the pointer is rejected outright when it names its own cluster or
    /// a cluster the volume does not have.
    fn next_cluster_bounded(
        &self,
        cluster: u32,
        steps: &mut u32,
    ) -> Result<Option<u32>, VfsError> {
        *steps += 1;
        if *steps > self.clusters {
            return Err(VfsError::IoError);
        }
        match self.next_cluster(cluster)? {
            None => Ok(None),
            Some(next) if next == cluster => Err(VfsError::IoError),
            Some(next) => {
                self.check_cluster(next)?;
                Ok(Some(next))
            }
        }
    }

    fn read_file(&self, cluster: u32, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> {
        if buf.is_empty() || cluster < 2 {
            return Ok(0);
        }
        let cluster_size = (self.sectors_per_cluster as u64) * (self.bytes_per_sector as u64);

        let mut cluster = cluster;
        let mut skipped = 0u64;
        let mut steps = 0u32;
        while skipped + cluster_size <= offset {
            match self.next_cluster_bounded(cluster, &mut steps)? {
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
            let sector = self.cluster_to_sector(cluster)? + sector_in_cluster as u64;
            let mut sector_buf = alloc::vec![0u8; self.bytes_per_sector as usize];
            block::read_block(self.dev_num, sector, &mut sector_buf)
                .map_err(|_| VfsError::IoError)?;
            let to_copy = core::cmp::min(
                buf.len() - buf_written,
                self.bytes_per_sector as usize - byte_in_sector,
            );
            buf[buf_written..buf_written + to_copy]
                .copy_from_slice(&sector_buf[byte_in_sector..byte_in_sector + to_copy]);
            buf_written += to_copy;
            sector_offset += to_copy;

            if sector_offset >= cluster_size as usize {
                match self.next_cluster_bounded(cluster, &mut steps)? {
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
            let raw =
                unsafe { core::ptr::read_unaligned(data.as_ptr().add(offset) as *const DirEntry) };
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

            let name_len = raw
                .name
                .iter()
                .position(|&b| b == b' ')
                .unwrap_or(raw.name.len());
            let name = core::str::from_utf8(&raw.name[..name_len])
                .unwrap_or("")
                .to_string();
            let ext_len = raw
                .ext
                .iter()
                .position(|&b| b == b' ')
                .unwrap_or(raw.ext.len());
            let ext = core::str::from_utf8(&raw.ext[..ext_len])
                .unwrap_or("")
                .to_string();
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

    fn name_to_83(name: &str) -> ([u8; 8], [u8; 3]) {
        let parts: Vec<&str> = name.split('.').collect();
        let base = parts[0].to_ascii_uppercase();
        let mut n = [b' '; 8];
        for (i, b) in base.bytes().take(8).enumerate() {
            n[i] = b;
        }
        let mut e = [b' '; 3];
        if parts.len() > 1 {
            let ext = parts[parts.len() - 1].to_ascii_uppercase();
            for (i, b) in ext.bytes().take(3).enumerate() {
                e[i] = b;
            }
        }
        (n, e)
    }

    fn write_fat_entry(&mut self, cluster: u32, value: u32) -> Result<(), VfsError> {
        self.check_cluster(cluster)?;
        let bytes_per_sector = self.bytes_per_sector as u32;
        match self.fat_type {
            FatType::Fat12 => {
                let fat_offset = cluster * 12 / 8;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; (self.bytes_per_sector as usize) * 2];
                block::read_block(
                    self.dev_num,
                    fat_sector,
                    &mut buf[..self.bytes_per_sector as usize],
                )
                .map_err(|_| VfsError::IoError)?;
                if entry_offset == self.bytes_per_sector as usize - 1 {
                    block::read_block(
                        self.dev_num,
                        fat_sector + 1,
                        &mut buf[self.bytes_per_sector as usize..],
                    )
                    .map_err(|_| VfsError::IoError)?;
                }
                if cluster % 2 == 0 {
                    buf[entry_offset] = (value & 0x0FF) as u8;
                    buf[entry_offset + 1] =
                        (buf[entry_offset + 1] & 0xF0) | (((value >> 8) & 0x0F) as u8);
                } else {
                    buf[entry_offset] = (buf[entry_offset] & 0x0F) | (((value << 4) & 0xF0) as u8);
                    buf[entry_offset + 1] = ((value >> 4) & 0x0FF) as u8;
                }
                block::write_block(
                    self.dev_num,
                    fat_sector,
                    &buf[..self.bytes_per_sector as usize],
                )
                .map_err(|_| VfsError::IoError)?;
                if entry_offset == self.bytes_per_sector as usize - 1 {
                    block::write_block(
                        self.dev_num,
                        fat_sector + 1,
                        &buf[self.bytes_per_sector as usize..self.bytes_per_sector as usize + 1],
                    )
                    .map_err(|_| VfsError::IoError)?;
                }
            }
            FatType::Fat16 => {
                let fat_offset = cluster * 2;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, fat_sector, &mut buf)
                    .map_err(|_| VfsError::IoError)?;
                buf[entry_offset..entry_offset + 2].copy_from_slice(&(value as u16).to_le_bytes());
                block::write_block(self.dev_num, fat_sector, &buf)
                    .map_err(|_| VfsError::IoError)?;
            }
            FatType::Fat32 => {
                let fat_offset = cluster * 4;
                let fat_sector = self.fat_start + (fat_offset / bytes_per_sector) as u64;
                let entry_offset = (fat_offset % bytes_per_sector) as usize;
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, fat_sector, &mut buf)
                    .map_err(|_| VfsError::IoError)?;
                let existing = u32::from_le_bytes([
                    buf[entry_offset],
                    buf[entry_offset + 1],
                    buf[entry_offset + 2],
                    buf[entry_offset + 3],
                ]);
                let new_val = (existing & 0xF0000000) | (value & 0x0FFFFFFF);
                buf[entry_offset..entry_offset + 4].copy_from_slice(&new_val.to_le_bytes());
                block::write_block(self.dev_num, fat_sector, &buf)
                    .map_err(|_| VfsError::IoError)?;
            }
        }
        Ok(())
    }

    fn find_free_cluster(&self) -> Result<u32, VfsError> {
        for c in 2..self.clusters {
            let entry = self.read_fat_entry(c)?;
            if entry == 0 {
                return Ok(c);
            }
        }
        Err(VfsError::IoError)
    }

    fn allocate_cluster(&mut self) -> Result<u32, VfsError> {
        let c = self.find_free_cluster()?;
        let eoc = match self.fat_type {
            FatType::Fat12 => 0x0FFF,
            FatType::Fat16 => 0xFFFF,
            FatType::Fat32 => 0x0FFFFFFF,
        };
        self.write_fat_entry(c, eoc)?;
        // Zero the cluster.
        let sector = self.cluster_to_sector(c)?;
        let zero = alloc::vec![0u8; self.bytes_per_sector as usize];
        for s in 0..self.sectors_per_cluster {
            block::write_block(self.dev_num, sector + s as u64, &zero)
                .map_err(|_| VfsError::IoError)?;
        }
        Ok(c)
    }

    fn free_cluster_chain(&mut self, cluster: u32) -> Result<(), VfsError> {
        let mut c = cluster;
        let mut steps = 0u32;
        while c >= 2 {
            let next = self.next_cluster_bounded(c, &mut steps)?;
            self.write_fat_entry(c, 0)?;
            c = next.unwrap_or(0);
        }
        Ok(())
    }

    fn write_clusters(
        &mut self,
        start_cluster: u32,
        buf: &[u8],
        offset: u64,
    ) -> Result<usize, VfsError> {
        if start_cluster < 2 || buf.is_empty() {
            return Ok(0);
        }
        let cluster_size = (self.sectors_per_cluster as u64) * (self.bytes_per_sector as u64);
        let mut cluster = start_cluster;
        let mut skipped = 0u64;
        let mut steps = 0u32;
        while skipped + cluster_size <= offset {
            match self.next_cluster_bounded(cluster, &mut steps)? {
                Some(next) => {
                    cluster = next;
                    skipped += cluster_size;
                }
                None => {
                    let new_c = self.allocate_cluster()?;
                    self.write_fat_entry(cluster, new_c)?;
                    cluster = new_c;
                    skipped += cluster_size;
                }
            }
        }

        let mut sector_offset = (offset - skipped) as usize;
        let mut buf_written = 0usize;

        while buf_written < buf.len() {
            let sector_in_cluster = sector_offset / self.bytes_per_sector as usize;
            let byte_in_sector = sector_offset % self.bytes_per_sector as usize;
            let sector = self.cluster_to_sector(cluster)? + sector_in_cluster as u64;
            let mut sector_buf = alloc::vec![0u8; self.bytes_per_sector as usize];
            block::read_block(self.dev_num, sector, &mut sector_buf)
                .map_err(|_| VfsError::IoError)?;
            let to_copy = core::cmp::min(
                buf.len() - buf_written,
                self.bytes_per_sector as usize - byte_in_sector,
            );
            sector_buf[byte_in_sector..byte_in_sector + to_copy]
                .copy_from_slice(&buf[buf_written..buf_written + to_copy]);
            block::write_block(self.dev_num, sector, &sector_buf).map_err(|_| VfsError::IoError)?;
            buf_written += to_copy;
            sector_offset += to_copy;

            if sector_offset >= cluster_size as usize {
                match self.next_cluster_bounded(cluster, &mut steps)? {
                    Some(next) => cluster = next,
                    None => {
                        let new_c = self.allocate_cluster()?;
                        self.write_fat_entry(cluster, new_c)?;
                        cluster = new_c;
                    }
                }
                sector_offset = 0;
            }
        }
        Ok(buf_written)
    }

    fn read_dir_raw(&self, cluster: u32) -> Result<Vec<u8>, VfsError> {
        let mut data = Vec::new();
        if cluster == 0 {
            let size = self.bpb.root_entries as usize * 32;
            let sectors = size.div_ceil(self.bytes_per_sector as usize);
            data.resize(size, 0);
            for i in 0..sectors {
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, self.root_dir_start + i as u64, &mut buf)
                    .map_err(|_| VfsError::IoError)?;
                let start = i * self.bytes_per_sector as usize;
                let end = core::cmp::min(start + self.bytes_per_sector as usize, size);
                data[start..end].copy_from_slice(&buf[..end - start]);
            }
        } else {
            let mut c = cluster;
            let mut steps = 0u32;
            loop {
                let sector = self.cluster_to_sector(c)?;
                for s in 0..self.sectors_per_cluster {
                    let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                    block::read_block(self.dev_num, sector + s as u64, &mut buf)
                        .map_err(|_| VfsError::IoError)?;
                    data.extend_from_slice(&buf);
                }
                match self.next_cluster_bounded(c, &mut steps)? {
                    Some(next) => c = next,
                    None => break,
                }
            }
        }
        Ok(data)
    }

    fn write_dir_raw(&mut self, cluster: u32, data: &[u8]) -> Result<(), VfsError> {
        if cluster == 0 {
            let size = self.bpb.root_entries as usize * 32;
            let sectors = size.div_ceil(self.bytes_per_sector as usize);
            for i in 0..sectors {
                let start = i * self.bytes_per_sector as usize;
                let end = core::cmp::min(start + self.bytes_per_sector as usize, size);
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                buf[..end - start].copy_from_slice(&data[start..end]);
                block::write_block(self.dev_num, self.root_dir_start + i as u64, &buf)
                    .map_err(|_| VfsError::IoError)?;
            }
        } else {
            let mut c = cluster;
            let mut steps = 0u32;
            let _cluster_size =
                (self.sectors_per_cluster as usize) * (self.bytes_per_sector as usize);
            let mut offset = 0;
            while offset < data.len() {
                let sector = self.cluster_to_sector(c)?;
                for s in 0..self.sectors_per_cluster {
                    let start = offset;
                    let end = core::cmp::min(start + self.bytes_per_sector as usize, data.len());
                    let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                    if end > start {
                        buf[..end - start].copy_from_slice(&data[start..end]);
                    }
                    block::write_block(self.dev_num, sector + s as u64, &buf)
                        .map_err(|_| VfsError::IoError)?;
                    offset += self.bytes_per_sector as usize;
                }
                if offset < data.len() {
                    match self.next_cluster_bounded(c, &mut steps)? {
                        Some(next) => c = next,
                        None => {
                            let new_c = self.allocate_cluster()?;
                            self.write_fat_entry(c, new_c)?;
                            c = new_c;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn find_entry_in_dir(
        &self,
        dir_cluster: u32,
        name: &str,
    ) -> Result<(usize, DirEntry, u32), VfsError> {
        let data = self.read_dir_raw(dir_cluster)?;
        let mut offset = 0;
        while offset + 32 <= data.len() {
            let raw =
                unsafe { core::ptr::read_unaligned(data.as_ptr().add(offset) as *const DirEntry) };
            if raw.name[0] == 0x00 {
                break;
            }
            if raw.name[0] != 0xE5 && raw.attr != 0x0F && raw.attr & 0x08 == 0 {
                let name_len = raw
                    .name
                    .iter()
                    .position(|&b| b == b' ')
                    .unwrap_or(raw.name.len());
                let base = core::str::from_utf8(&raw.name[..name_len]).unwrap_or("");
                let ext_len = raw
                    .ext
                    .iter()
                    .position(|&b| b == b' ')
                    .unwrap_or(raw.ext.len());
                let ext = core::str::from_utf8(&raw.ext[..ext_len]).unwrap_or("");
                let full_name = if ext.is_empty() {
                    base.to_string()
                } else {
                    format!("{}.{}", base, ext)
                };
                if full_name.eq_ignore_ascii_case(name) {
                    return Ok((offset, raw, dir_cluster));
                }
            }
            offset += 32;
        }
        Err(VfsError::NotFound)
    }

    fn add_dir_entry(
        &mut self,
        dir_cluster: u32,
        name: &str,
        attr: u8,
        first_cluster: u32,
        size: u32,
    ) -> Result<(), VfsError> {
        let (n, e) = Self::name_to_83(name);
        let entry = DirEntry {
            name: n,
            ext: e,
            attr,
            nt_res: 0,
            crt_time_tenth: 0,
            crt_time: 0,
            crt_date: 0,
            lst_acc_date: 0,
            first_cluster_high: (first_cluster >> 16) as u16,
            wrt_time: 0,
            wrt_date: 0,
            first_cluster_low: (first_cluster & 0xFFFF) as u16,
            size,
        };

        let mut data = self.read_dir_raw(dir_cluster)?;
        let mut offset = 0;
        let mut free_offset = None;
        while offset + 32 <= data.len() {
            let b = data[offset];
            if b == 0x00 || b == 0xE5 {
                free_offset = Some(offset);
                break;
            }
            offset += 32;
        }
        if free_offset.is_none() {
            // Need to extend directory.
            if dir_cluster == 0 {
                // FAT12/16 root dir is fixed size.
                return Err(VfsError::IoError);
            }
            let new_c = self.allocate_cluster()?;
            let mut tail = dir_cluster;
            let mut steps = 0u32;
            while let Some(next) = self.next_cluster_bounded(tail, &mut steps)? {
                tail = next;
            }
            self.write_fat_entry(tail, new_c)?;
            let cluster_size =
                (self.sectors_per_cluster as usize) * (self.bytes_per_sector as usize);
            data.resize(data.len() + cluster_size, 0);
            free_offset = Some(offset);
        }
        let off = free_offset.unwrap();
        unsafe {
            core::ptr::write_unaligned(data.as_mut_ptr().add(off) as *mut DirEntry, entry);
        }
        self.write_dir_raw(dir_cluster, &data)?;
        Ok(())
    }

    fn remove_dir_entry(&mut self, dir_cluster: u32, name: &str) -> Result<(), VfsError> {
        let (offset, _raw, _dc) = self
            .find_entry_in_dir(dir_cluster, name)
            .map_err(|_| VfsError::NotFound)?;
        let mut data = self.read_dir_raw(dir_cluster)?;
        data[offset] = 0xE5;
        self.write_dir_raw(dir_cluster, &data)?;
        Ok(())
    }

    fn split_path(path: &str) -> (&str, &str) {
        let path = path.trim_matches('/');
        match path.rfind('/') {
            Some(pos) => (&path[..pos], &path[pos + 1..]),
            None => ("", path),
        }
    }

    fn resolve_parent_and_name(&self, path: &str) -> Result<(u32, String), VfsError> {
        let (parent_path, name) = Self::split_path(path);
        let parent_cluster = if parent_path.is_empty() {
            if self.fat_type == FatType::Fat32 {
                self.root_cluster
            } else {
                0
            }
        } else {
            let handle = self.lookup_path(parent_path)?;
            if handle.inode == 1 && self.fat_type != FatType::Fat32 {
                0
            } else if self.fat_type == FatType::Fat32 && handle.inode == self.root_cluster as u64 {
                self.root_cluster
            } else {
                handle.inode as u32
            }
        };
        Ok((parent_cluster, String::from(name)))
    }

    fn read_root_dir(&self) -> Result<Vec<FatDirEntry>, VfsError> {
        let size = self.bpb.root_entries as usize * 32;
        let sectors = size.div_ceil(self.bytes_per_sector as usize);
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
        let mut steps = 0u32;
        loop {
            let sector = self.cluster_to_sector(cluster)?;
            for s in 0..self.sectors_per_cluster {
                let mut buf = alloc::vec![0u8; self.bytes_per_sector as usize];
                block::read_block(self.dev_num, sector + s as u64, &mut buf)
                    .map_err(|_| VfsError::IoError)?;
                data.extend_from_slice(&buf);
            }
            match self.next_cluster_bounded(cluster, &mut steps)? {
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
        // "." points at the directory itself and ".." at its parent, so the
        // loop below refuses to descend into either — but skipping them is not
        // enough: any two directories whose entries name each other's cluster
        // form the same endless descent under ordinary names. Remembering
        // which directories have been walked bounds the search by the number
        // of directories on the volume.
        // ponytail: linear scan; a bitmap over clusters if directory counts
        // ever make this show up in a profile.
        let mut visited = alloc::vec![self.root_cluster];

        while let Some(entries) = queue.pop_front() {
            for e in entries {
                if e.first_cluster == cluster {
                    return Ok(e);
                }
                if e.attr & 0x10 != 0
                    && e.first_cluster >= 2
                    && e.name != "."
                    && e.name != ".."
                    && !visited.contains(&e.first_cluster)
                {
                    visited.push(e.first_cluster);
                    let sub = self.read_dir_cluster(e.first_cluster)?;
                    queue.push_back(sub);
                }
            }
        }
        Err(VfsError::NotFound)
    }

    fn find_entry_location_by_cluster(&self, cluster: u32) -> Result<(u32, usize), VfsError> {
        let mut queue = VecDeque::new();
        let root_data = self.read_dir_raw(if self.fat_type != FatType::Fat32 {
            0
        } else {
            self.root_cluster
        })?;
        let root_cluster = if self.fat_type != FatType::Fat32 {
            0
        } else {
            self.root_cluster
        };
        queue.push_back((root_data, root_cluster));
        // Same cycle as in `find_entry_by_cluster`: two directories naming
        // each other under ordinary names descend forever without this.
        let mut visited = alloc::vec![root_cluster];

        while let Some((data, dir_cluster)) = queue.pop_front() {
            let mut offset = 0;
            while offset + 32 <= data.len() {
                let raw = unsafe {
                    core::ptr::read_unaligned(data.as_ptr().add(offset) as *const DirEntry)
                };
                if raw.name[0] == 0x00 {
                    break;
                }
                if raw.name[0] != 0xE5 && raw.attr != 0x0F && raw.attr & 0x08 == 0 {
                    let fc =
                        ((raw.first_cluster_high as u32) << 16) | (raw.first_cluster_low as u32);
                    if fc == cluster && cluster >= 2 {
                        return Ok((dir_cluster, offset));
                    }
                    // Skip "." and ".." — a valid 8.3 name never starts with a
                    // dot, and descending into them cycles forever.
                    if raw.attr & 0x10 != 0
                        && fc >= 2
                        && raw.name[0] != b'.'
                        && !visited.contains(&fc)
                    {
                        visited.push(fc);
                        if let Ok(sub) = self.read_dir_raw(fc) {
                            queue.push_back((sub, fc));
                        }
                    }
                }
                offset += 32;
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
                // FAT stores 8.3 names upper-cased, so path resolution must fold
                // case the same way `find_entry_in_dir` does — otherwise
                // `create("/a.txt")` succeeds and `lookup("/a.txt")` then fails.
                if e.name.eq_ignore_ascii_case(component) {
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
            return Err(VfsError::InvalidOperation);
        }
        if self.fat_type == FatType::Fat32 && handle.inode == self.root_cluster as u64 {
            return Err(VfsError::InvalidOperation);
        }
        let entry = self.find_entry_by_cluster(handle.inode as u32)?;
        if entry.attr & 0x10 != 0 {
            return Err(VfsError::InvalidOperation);
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

    fn write(&mut self, handle: VfsHandle, buf: &[u8], offset: u64) -> Result<usize, VfsError> {
        if handle.inode == 1 && self.fat_type != FatType::Fat32 {
            return Err(VfsError::InvalidOperation);
        }
        if self.fat_type == FatType::Fat32 && handle.inode == self.root_cluster as u64 {
            return Err(VfsError::InvalidOperation);
        }
        let entry = self.find_entry_by_cluster(handle.inode as u32)?;
        if entry.attr & 0x10 != 0 {
            return Err(VfsError::InvalidOperation);
        }
        let cluster = if entry.first_cluster < 2 {
            let new_c = self.allocate_cluster()?;
            // Update the directory entry with the new first cluster.
            let (dir_cluster, off) = self.find_entry_location_by_cluster(handle.inode as u32)?;
            let mut data = self.read_dir_raw(dir_cluster)?;
            let fc_off = off + 26;
            data[fc_off..fc_off + 2].copy_from_slice(&(new_c as u16).to_le_bytes());
            let fc_hi_off = off + 20;
            data[fc_hi_off..fc_hi_off + 2].copy_from_slice(&((new_c >> 16) as u16).to_le_bytes());
            self.write_dir_raw(dir_cluster, &data)?;
            new_c
        } else {
            entry.first_cluster
        };
        let written = self.write_clusters(cluster, buf, offset)?;
        let new_size = core::cmp::max(entry.size as u64, offset + written as u64);
        let (dir_cluster, off) = self.find_entry_location_by_cluster(handle.inode as u32)?;
        let mut data = self.read_dir_raw(dir_cluster)?;
        let size_off = off + 28;
        data[size_off..size_off + 4].copy_from_slice(&(new_size as u32).to_le_bytes());
        self.write_dir_raw(dir_cluster, &data)?;
        Ok(written)
    }

    fn create(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let (parent_cluster, name) = self.resolve_parent_and_name(path)?;
        if self.find_entry_in_dir(parent_cluster, &name).is_ok() {
            return Err(VfsError::AlreadyExists);
        }
        let new_c = self.allocate_cluster()?;
        self.add_dir_entry(parent_cluster, &name, 0x20, new_c, 0)?;
        Ok(VfsHandle {
            fs_idx: 0,
            inode: new_c as u64,
        })
    }

    fn mkdir(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let (parent_cluster, name) = self.resolve_parent_and_name(path)?;
        if self.find_entry_in_dir(parent_cluster, &name).is_ok() {
            return Err(VfsError::AlreadyExists);
        }
        let new_c = self.allocate_cluster()?;
        self.add_dir_entry(parent_cluster, &name, 0x10, new_c, 0)?;
        // Create . and .. entries in new directory.
        let dot = DirEntry {
            name: *b".       ",
            ext: [b' '; 3],
            attr: 0x10,
            nt_res: 0,
            crt_time_tenth: 0,
            crt_time: 0,
            crt_date: 0,
            lst_acc_date: 0,
            first_cluster_high: (new_c >> 16) as u16,
            wrt_time: 0,
            wrt_date: 0,
            first_cluster_low: (new_c & 0xFFFF) as u16,
            size: 0,
        };
        let dotdot = DirEntry {
            name: *b"..      ",
            ext: [b' '; 3],
            attr: 0x10,
            nt_res: 0,
            crt_time_tenth: 0,
            crt_time: 0,
            crt_date: 0,
            lst_acc_date: 0,
            first_cluster_high: (parent_cluster >> 16) as u16,
            wrt_time: 0,
            wrt_date: 0,
            first_cluster_low: (parent_cluster & 0xFFFF) as u16,
            size: 0,
        };
        let mut dir_data =
            alloc::vec![0u8; self.bytes_per_sector as usize * self.sectors_per_cluster as usize];
        unsafe {
            core::ptr::write_unaligned(dir_data.as_mut_ptr() as *mut DirEntry, dot);
            core::ptr::write_unaligned(dir_data.as_mut_ptr().add(32) as *mut DirEntry, dotdot);
        }
        self.write_dir_raw(new_c, &dir_data)?;
        Ok(VfsHandle {
            fs_idx: 0,
            inode: new_c as u64,
        })
    }

    fn unlink(&mut self, path: &str) -> Result<(), VfsError> {
        let (parent_cluster, name) = self.resolve_parent_and_name(path)?;
        let (_offset, raw, _) = self.find_entry_in_dir(parent_cluster, &name)?;
        if raw.attr & 0x10 != 0 {
            return Err(VfsError::InvalidOperation);
        }
        let first_cluster =
            ((raw.first_cluster_high as u32) << 16) | (raw.first_cluster_low as u32);
        if first_cluster >= 2 {
            self.free_cluster_chain(first_cluster)?;
        }
        self.remove_dir_entry(parent_cluster, &name)?;
        Ok(())
    }

    fn rmdir(&mut self, path: &str) -> Result<(), VfsError> {
        let (parent_cluster, name) = self.resolve_parent_and_name(path)?;
        let (_offset, raw, _) = self.find_entry_in_dir(parent_cluster, &name)?;
        if raw.attr & 0x10 == 0 {
            return Err(VfsError::NotADirectory);
        }
        let first_cluster =
            ((raw.first_cluster_high as u32) << 16) | (raw.first_cluster_low as u32);
        // Directory must be empty (only . and .. allowed).
        if first_cluster >= 2 {
            let entries = self.read_dir_cluster(first_cluster)?;
            for e in entries {
                if e.name != "." && e.name != ".." {
                    return Err(VfsError::InvalidOperation);
                }
            }
            self.free_cluster_chain(first_cluster)?;
        }
        self.remove_dir_entry(parent_cluster, &name)?;
        Ok(())
    }

    fn rename(&mut self, old: &str, new: &str) -> Result<(), VfsError> {
        let (old_parent, old_name) = self.resolve_parent_and_name(old)?;
        let (new_parent, new_name) = self.resolve_parent_and_name(new)?;
        if old_parent != new_parent {
            return Err(VfsError::InvalidOperation);
        }
        if old_name == new_name {
            return Ok(());
        }
        let (offset, _raw, _) = self.find_entry_in_dir(old_parent, &old_name)?;
        if self.find_entry_in_dir(new_parent, &new_name).is_ok() {
            return Err(VfsError::AlreadyExists);
        }
        let (new_n, new_e) = Self::name_to_83(&new_name);
        let mut data = self.read_dir_raw(old_parent)?;
        let name_off = offset;
        data[name_off..name_off + 8].copy_from_slice(&new_n);
        data[name_off + 8..name_off + 11].copy_from_slice(&new_e);
        self.write_dir_raw(old_parent, &data)?;
        Ok(())
    }

    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        // Either root-directory form counts as a directory: the fixed inode 1
        // on FAT12/16, or the root cluster on FAT32.
        let is_dir = if (handle.inode == 1 && self.fat_type != FatType::Fat32)
            || (self.fat_type == FatType::Fat32 && handle.inode == self.root_cluster as u64)
        {
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
            permissions: if node_type == VfsNodeType::Directory {
                0o755
            } else {
                0o644
            },
            uid: 0,
            gid: 0,
            mode: if node_type == VfsNodeType::Directory {
                0o755
            } else {
                0o644
            },
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
        Err(VfsError::InvalidOperation)
    }
}

// Mount integration helper (optional).
// Example usage in `fs::init_vfs` or `lib.rs`:
//
//     let mut fat = fat::FatFs::new(dev_num);
//     if fat.mount().is_ok() {
//         vfs.mount("/fat", Box::new(fat))?;
//     }

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::drivers::block::{register_block_device, BlockDevice, BlockError};
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};
    use alloc::boxed::Box;
    use spin::Mutex;

    /// Minimal in-memory block device. The chain-walk tests only care about
    /// FAT-entry bytes, not a fully mountable image, so this stands in for
    /// AHCI without building one.
    struct TestDisk {
        data: Mutex<Vec<u8>>,
        sector_size: u64,
    }

    impl TestDisk {
        fn new(sectors: usize, sector_size: usize) -> Self {
            Self {
                data: Mutex::new(alloc::vec![0u8; sectors * sector_size]),
                sector_size: sector_size as u64,
            }
        }

        /// Pokes a raw FAT16 entry directly into the backing bytes, i.e. sets
        /// up the on-disk state a corrupt (or legitimate) chain would leave.
        fn set_fat16_entry(&self, fat_start_lba: u64, cluster: u32, value: u16) {
            let offset =
                fat_start_lba as usize * self.sector_size as usize + cluster as usize * 2;
            let mut data = self.data.lock();
            data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }

        /// Writes a raw 8.3 directory entry at slot `index` of sector `lba`.
        /// Only the fields the directory walks read are filled in.
        fn set_dir_entry(&self, lba: u64, index: usize, name: &[u8; 8], attr: u8, first: u32) {
            let off = lba as usize * self.sector_size as usize + index * 32;
            let mut data = self.data.lock();
            data[off..off + 8].copy_from_slice(name);
            data[off + 8..off + 11].copy_from_slice(b"   ");
            data[off + 11] = attr;
            data[off + 20..off + 22].copy_from_slice(&((first >> 16) as u16).to_le_bytes());
            data[off + 26..off + 28].copy_from_slice(&((first & 0xFFFF) as u16).to_le_bytes());
        }
    }

    impl BlockDevice for TestDisk {
        fn sector_size(&self) -> u64 {
            self.sector_size
        }
        fn num_sectors(&self) -> u64 {
            self.data.lock().len() as u64 / self.sector_size
        }
        fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
            let data = self.data.lock();
            let start = (lba * self.sector_size) as usize;
            let end = start + buf.len();
            if end > data.len() {
                return Err(BlockError::InvalidLba);
            }
            buf.copy_from_slice(&data[start..end]);
            Ok(())
        }
        fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
            let mut data = self.data.lock();
            let start = (lba * self.sector_size) as usize;
            let end = start + buf.len();
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

    /// Builds a `FatFs` with hand-set FAT16 geometry over a fresh,
    /// uniquely-numbered `TestDisk`, bypassing `mount()` (which needs a full
    /// on-disk BPB) since these tests only exercise the cluster-chain walk.
    /// Returns the disk too, so the caller can poke raw FAT entries.
    fn make_fs(dev_num: u32) -> (FatFs, &'static TestDisk) {
        let disk: &'static TestDisk = Box::leak(Box::new(TestDisk::new(16, 512)));
        register_block_device(dev_num, disk);
        let mut fs = FatFs::new(dev_num);
        fs.fat_type = FatType::Fat16;
        fs.fat_start = 1;
        fs.bytes_per_sector = 512;
        fs.sectors_per_cluster = 1;
        fs.data_start = 2;
        // Small on purpose: the cap under test is derived directly from this.
        fs.clusters = 8;
        (fs, disk)
    }

    fn test_cyclic_chain_fails_closed_instead_of_hanging() -> TestResult {
        let (fs, disk) = make_fs(0xFA70);

        // The exact defect from the bug report: cluster 5 points at 6 and
        // cluster 6 points back at 5. `read_dir_raw` is a pure read walk (it
        // never rewrites the FAT as it goes), so nothing about this loop
        // self-heals — before the fix this alternates 5, 6, 5, 6... forever.
        disk.set_fat16_entry(fs.fat_start, 5, 6);
        disk.set_fat16_entry(fs.fat_start, 6, 5);

        let result = fs.read_dir_raw(5);
        test_assert!(
            matches!(result, Err(VfsError::IoError)),
            "cyclic FAT chain must fail closed, not loop forever or truncate silently"
        );
        TestResult::Pass
    }

    fn test_valid_chain_within_cap_still_reads() -> TestResult {
        let (fs, disk) = make_fs(0xFA71);

        // 2 -> 3 -> 4 -> EOC: a legitimate 3-cluster chain, well under the
        // 8-cluster cap. The bound must not reject real, non-corrupt chains.
        disk.set_fat16_entry(fs.fat_start, 2, 3);
        disk.set_fat16_entry(fs.fat_start, 3, 4);
        disk.set_fat16_entry(fs.fat_start, 4, 0xFFFF);

        let result = fs.read_dir_raw(2);
        let data = match result {
            Ok(d) => d,
            Err(_) => return TestResult::Fail("valid chain within cap must not error"),
        };
        test_assert_eq!(data.len(), 3 * 512);
        TestResult::Pass
    }

    /// Lays out a two-directory cycle that uses no "." or ".." entry, so the
    /// existing dot-name skip in the directory walks cannot see it:
    ///
    ///   root (sector 2) -> "A"    dir, cluster 2
    ///   cluster 2        -> "B"    dir, cluster 3
    ///   cluster 3        -> "BACK" dir, cluster 2
    ///
    /// Descending it endlessly yields A, B, A, B... Both FAT entries are EOC,
    /// so the per-chain step cap never fires — the cycle is in the directory
    /// tree, not in a single cluster chain.
    fn make_cyclic_dir_tree(dev_num: u32) -> FatFs {
        let (mut fs, disk) = make_fs(dev_num);
        fs.root_dir_start = 2;
        fs.data_start = 3;
        fs.bpb.root_entries = 16;
        disk.set_dir_entry(2, 0, b"A       ", 0x10, 2);
        disk.set_dir_entry(3, 0, b"B       ", 0x10, 3);
        disk.set_dir_entry(4, 0, b"BACK    ", 0x10, 2);
        disk.set_fat16_entry(fs.fat_start, 2, 0xFFFF);
        disk.set_fat16_entry(fs.fat_start, 3, 0xFFFF);
        fs
    }

    fn test_self_referential_cluster_rejected_at_the_pointer() -> TestResult {
        let (mut fs, disk) = make_fs(0xFA72);
        // A volume big enough that the step cap is useless as a defence: a
        // cluster pointing at itself would burn 2^20 FAT reads before the cap
        // noticed, which on real hardware is a dead machine either way. The
        // pointer itself has to be rejected.
        fs.clusters = 1 << 20;
        disk.set_fat16_entry(fs.fat_start, 5, 5);

        let result = fs.read_dir_raw(5);
        test_assert!(
            matches!(result, Err(VfsError::IoError)),
            "a cluster pointing at itself must be rejected at the pointer, not by the step cap"
        );
        TestResult::Pass
    }

    fn test_next_pointer_past_volume_rejected() -> TestResult {
        let (fs, disk) = make_fs(0xFA73);
        // Cluster 12 is off the end of an 8-cluster volume but still lands on
        // a sector the device happens to have, so the read succeeds and the
        // walk silently serves data from outside the filesystem.
        disk.set_fat16_entry(fs.fat_start, 2, 12);

        let result = fs.read_dir_raw(2);
        test_assert!(
            matches!(result, Err(VfsError::IoError)),
            "a next pointer past the last cluster of the volume must be rejected"
        );
        TestResult::Pass
    }

    fn test_next_pointer_below_first_data_cluster_rejected() -> TestResult {
        let (fs, disk) = make_fs(0xFA74);
        // Clusters 0 and 1 are reserved FAT entries, never data. Following a
        // pointer to one reads whatever sector the arithmetic lands on.
        disk.set_fat16_entry(fs.fat_start, 2, 1);

        let result = fs.read_dir_raw(2);
        test_assert!(
            matches!(result, Err(VfsError::IoError)),
            "a next pointer below the first data cluster must be rejected"
        );
        TestResult::Pass
    }

    fn test_directory_cycle_does_not_hang_entry_search() -> TestResult {
        let fs = make_cyclic_dir_tree(0xFA75);

        // Cluster 9 exists on this volume but is in no directory, so the
        // search has to exhaust the tree — which it can only do if it stops
        // re-descending into directories it has already walked.
        test_assert!(
            matches!(fs.find_entry_by_cluster(9), Err(VfsError::NotFound)),
            "an absent cluster must terminate the directory search, not cycle"
        );
        // Control: the cycle guard must not prune a directory that is really
        // reachable. "B" lives at cluster 3, one level down.
        match fs.find_entry_by_cluster(3) {
            Ok(e) => test_assert_eq!(e.name, "B"),
            Err(_) => return TestResult::Fail("reachable entry must still be found"),
        }
        TestResult::Pass
    }

    fn test_directory_cycle_does_not_hang_entry_location() -> TestResult {
        let fs = make_cyclic_dir_tree(0xFA76);

        test_assert!(
            matches!(
                fs.find_entry_location_by_cluster(9),
                Err(VfsError::NotFound)
            ),
            "an absent cluster must terminate the location search, not cycle"
        );
        // Control: "B" is at slot 0 of the directory whose cluster is 2.
        match fs.find_entry_location_by_cluster(3) {
            Ok((dir_cluster, offset)) => {
                test_assert_eq!(dir_cluster, 2);
                test_assert_eq!(offset, 0);
            }
            Err(_) => return TestResult::Fail("reachable entry must still be located"),
        }
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "fat::cyclic_chain_fails_closed_instead_of_hanging",
            test_cyclic_chain_fails_closed_instead_of_hanging,
        );
        crate::testing::register_test(
            "fat::valid_chain_within_cap_still_reads",
            test_valid_chain_within_cap_still_reads,
        );
        crate::testing::register_test(
            "fat::self_referential_cluster_rejected_at_the_pointer",
            test_self_referential_cluster_rejected_at_the_pointer,
        );
        crate::testing::register_test(
            "fat::next_pointer_past_volume_rejected",
            test_next_pointer_past_volume_rejected,
        );
        crate::testing::register_test(
            "fat::next_pointer_below_first_data_cluster_rejected",
            test_next_pointer_below_first_data_cluster_rejected,
        );
        crate::testing::register_test(
            "fat::directory_cycle_does_not_hang_entry_search",
            test_directory_cycle_does_not_hang_entry_search,
        );
        crate::testing::register_test(
            "fat::directory_cycle_does_not_hang_entry_location",
            test_directory_cycle_does_not_hang_entry_location,
        );
    }
}
