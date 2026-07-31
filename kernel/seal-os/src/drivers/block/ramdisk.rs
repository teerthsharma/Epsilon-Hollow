// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Memory-backed block device.
//!
//! Used as the install scratch target so the raw partitioning and formatting
//! path can be exercised end to end under QEMU without a second physical disk
//! and without ever addressing the live boot disk.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use spin::Mutex;

use super::{register_block_device, BlockDevice, BlockError, BLOCK_DEVICES};

/// Device number of the install scratch disk.
pub const SCRATCH_DEV_NUM: u32 = 0x5CA0;

const SECTOR: usize = 512;

pub struct RamDisk {
    sectors: u64,
    data: Mutex<Vec<u8>>,
}

impl RamDisk {
    pub fn new(sectors: u64) -> Self {
        Self {
            sectors,
            data: Mutex::new(vec![0u8; (sectors as usize) * SECTOR]),
        }
    }

    fn range(&self, lba: u64, len: usize) -> Result<(usize, usize), BlockError> {
        if len == 0 || len % SECTOR != 0 {
            return Err(BlockError::InvalidLba);
        }
        let count = (len / SECTOR) as u64;
        if lba.saturating_add(count) > self.sectors {
            return Err(BlockError::InvalidLba);
        }
        let start = (lba as usize) * SECTOR;
        Ok((start, start + len))
    }
}

impl BlockDevice for RamDisk {
    fn sector_size(&self) -> u64 {
        SECTOR as u64
    }

    fn num_sectors(&self) -> u64 {
        self.sectors
    }

    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let (start, end) = self.range(lba, buf.len())?;
        buf.copy_from_slice(&self.data.lock()[start..end]);
        Ok(())
    }

    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        let (start, end) = self.range(lba, buf.len())?;
        self.data.lock()[start..end].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

/// Register the scratch disk once, returning its device number.
///
/// Idempotent: a second call reuses the disk already registered, so the size
/// argument only applies to the first call of a boot.
pub fn ensure_scratch(sectors: u64) -> Result<u32, BlockError> {
    if BLOCK_DEVICES.lock().get(SCRATCH_DEV_NUM).is_some() {
        return Ok(SCRATCH_DEV_NUM);
    }
    if sectors == 0 {
        return Err(BlockError::InvalidLba);
    }
    let disk: &'static RamDisk = Box::leak(Box::new(RamDisk::new(sectors)));
    register_block_device(SCRATCH_DEV_NUM, disk);
    Ok(SCRATCH_DEV_NUM)
}
