// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Block device abstraction layer.

pub mod ahci;

use alloc::vec::Vec;
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    NoDevice,
    IoError,
    InvalidLba,
    Timeout,
    Unsupported,
}

/// A block device capable of reading and writing sectors.
pub trait BlockDevice: Send + Sync {
    fn sector_size(&self) -> u64;
    fn num_sectors(&self) -> u64;
    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;
    fn flush(&self) -> Result<(), BlockError>;
}

/// Registry of all detected block devices.
pub struct BlockDeviceRegistry {
    devices: Vec<(u32, &'static dyn BlockDevice)>,
}

impl BlockDeviceRegistry {
    pub const fn new() -> Self {
        Self {
            devices: Vec::new(),
        }
    }

    pub fn register(&mut self, dev_num: u32, device: &'static dyn BlockDevice) {
        self.devices.push((dev_num, device));
    }

    pub fn get(&self, dev_num: u32) -> Option<&'static dyn BlockDevice> {
        self.devices
            .iter()
            .find(|(n, _)| *n == dev_num)
            .map(|(_, d)| *d)
    }
}

static BLOCK_DEVICES: Mutex<BlockDeviceRegistry> = Mutex::new(BlockDeviceRegistry::new());

/// Register a block device with the global registry.
pub fn register_block_device(dev_num: u32, device: &'static dyn BlockDevice) {
    BLOCK_DEVICES.lock().register(dev_num, device);
}

/// Read a single 512-byte sector from the device numbered `dev_num`.
pub fn read_block(dev_num: u32, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
    BLOCK_DEVICES
        .lock()
        .get(dev_num)
        .ok_or(BlockError::NoDevice)?
        .read_sector(lba, buf)
}

/// Write a single 512-byte sector to the device numbered `dev_num`.
pub fn write_block(dev_num: u32, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
    BLOCK_DEVICES
        .lock()
        .get(dev_num)
        .ok_or(BlockError::NoDevice)?
        .write_sector(lba, buf)
}
