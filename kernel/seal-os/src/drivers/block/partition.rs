// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Partition view over a parent block device.
//!
//! A partition is registered as its own device number, so a filesystem driver
//! addresses it from LBA 0 with no notion of the GPT beneath it, and every
//! access is clamped to the partition extent.

use alloc::boxed::Box;

use super::{
    read_block, register_block_device, write_block, BlockDevice, BlockError, BLOCK_DEVICES,
    BOOT_DEV_NUM,
};

/// Device number of partition 1 on the install scratch disk.
pub const SCRATCH_ROOT_DEV_NUM: u32 = 0x5CA1;

const SECTOR: usize = 512;

pub struct Partition {
    parent: u32,
    start_lba: u64,
    sectors: u64,
}

impl Partition {
    fn map(&self, lba: u64, len: usize) -> Result<u64, BlockError> {
        if len == 0 || len % SECTOR != 0 {
            return Err(BlockError::InvalidLba);
        }
        let count = (len / SECTOR) as u64;
        if lba.saturating_add(count) > self.sectors {
            return Err(BlockError::InvalidLba);
        }
        Ok(self.start_lba + lba)
    }
}

impl BlockDevice for Partition {
    fn sector_size(&self) -> u64 {
        SECTOR as u64
    }

    fn num_sectors(&self) -> u64 {
        self.sectors
    }

    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let parent_lba = self.map(lba, buf.len())?;
        read_block(self.parent, parent_lba, buf)
    }

    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        let parent_lba = self.map(lba, buf.len())?;
        write_block(self.parent, parent_lba, buf)
    }

    fn flush(&self) -> Result<(), BlockError> {
        Ok(())
    }
}

/// Register a partition view as `dev_num`.
///
/// Refuses to build a view onto the boot device: a partition device would
/// otherwise be a second name for the live disk and slip past the install
/// target check.
pub fn attach(dev_num: u32, parent: u32, start_lba: u64, sectors: u64) -> Result<(), BlockError> {
    if parent == BOOT_DEV_NUM || dev_num == BOOT_DEV_NUM || dev_num == parent {
        return Err(BlockError::Refused);
    }
    if sectors == 0 {
        return Err(BlockError::InvalidLba);
    }
    let parent_sectors = {
        let registry = BLOCK_DEVICES.lock();
        if registry.get(dev_num).is_some() {
            return Ok(());
        }
        registry
            .get(parent)
            .ok_or(BlockError::NoDevice)?
            .num_sectors()
    };
    if start_lba.saturating_add(sectors) > parent_sectors {
        return Err(BlockError::InvalidLba);
    }
    let part: &'static Partition = Box::leak(Box::new(Partition {
        parent,
        start_lba,
        sectors,
    }));
    register_block_device(dev_num, part);
    Ok(())
}
