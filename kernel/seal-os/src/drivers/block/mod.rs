// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Block device abstraction layer.

pub mod ahci;
pub mod partition;
pub mod ramdisk;
pub mod virtio_blk;

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    NoDevice,
    IoError,
    InvalidLba,
    Timeout,
    Unsupported,
    Busy,
    /// The write was rejected because the device is not the armed install target.
    Refused,
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

impl Default for BlockDeviceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

static BLOCK_DEVICES: Mutex<BlockDeviceRegistry> = Mutex::new(BlockDeviceRegistry::new());

/// Register a block device with the global registry.
pub fn register_block_device(dev_num: u32, device: &'static dyn BlockDevice) {
    BLOCK_DEVICES.lock().register(dev_num, device);
}

/// Look a device up and release the registry lock before touching it, so that
/// stacked devices (a partition view over its parent) cannot self-deadlock.
fn resolve(dev_num: u32) -> Result<&'static dyn BlockDevice, BlockError> {
    let device = BLOCK_DEVICES.lock().get(dev_num);
    device.ok_or(BlockError::NoDevice)
}

/// Read from the device numbered `dev_num`.
pub fn read_block(dev_num: u32, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
    resolve(dev_num)?.read_sectors(lba, buf)
}

/// Write to the device numbered `dev_num`.
pub fn write_block(dev_num: u32, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
    resolve(dev_num)?.write_sectors(lba, buf)
}

/// Unique GUID of the GPT partition firmware loaded Seal OS from, taken from
/// the HD node of the UEFI LoadedImage device path. Unset when the boot media
/// had no GPT partition (El Torito CD, network boot).
static BOOT_PARTUUID: spin::Once<[u8; 16]> = spin::Once::new();

/// Record the boot partition. Called once, from the UEFI entry, before
/// ExitBootServices.
pub fn set_boot_partuuid(guid: [u8; 16]) {
    BOOT_PARTUUID.call_once(|| guid);
}

pub fn boot_partuuid() -> Option<[u8; 16]> {
    BOOT_PARTUUID.get().copied()
}

/// True when `dev_num` carries the partition firmware booted from, whatever
/// device number or transport it arrived under. That disk is never a legal
/// install target. A read error is an error, never "not the boot disk".
pub fn is_boot_disk(dev_num: u32) -> Result<bool, BlockError> {
    match boot_partuuid() {
        Some(guid) => crate::fs::gpt::holds_partition(dev_num, &guid),
        None => Ok(false),
    }
}

/// The registered device carrying the boot partition, if any.
pub fn boot_disk() -> Option<u32> {
    let devices: Vec<u32> = BLOCK_DEVICES.lock().devices.iter().map(|d| d.0).collect();
    devices
        .into_iter()
        .find(|&dev| is_boot_disk(dev) == Ok(true))
}

const NO_INSTALL_TARGET: u32 = u32::MAX;

/// The one device raw install writes are currently allowed to touch.
static INSTALL_TARGET: AtomicU32 = AtomicU32::new(NO_INSTALL_TARGET);

/// `(dev_num, num_sectors)` for every registered device.
pub fn list_devices() -> Vec<(u32, u64)> {
    BLOCK_DEVICES
        .lock()
        .devices
        .iter()
        .map(|(num, device)| (*num, device.num_sectors()))
        .collect()
}

/// True when a device with this number is registered.
pub fn device_exists(dev_num: u32) -> bool {
    BLOCK_DEVICES.lock().get(dev_num).is_some()
}

/// Arm `dev_num` as the sole destination for raw install writes.
///
/// Refuses the boot disk and any device that is not registered. The previous
/// target is left untouched when the request is refused.
pub fn arm_install_target(dev_num: u32) -> Result<(), BlockError> {
    if !device_exists(dev_num) {
        return Err(BlockError::NoDevice);
    }
    if is_boot_disk(dev_num)? {
        return Err(BlockError::Refused);
    }
    INSTALL_TARGET.store(dev_num, Ordering::SeqCst);
    Ok(())
}

/// Drop the armed install target; every raw install write is refused again.
pub fn disarm_install_target() {
    INSTALL_TARGET.store(NO_INSTALL_TARGET, Ordering::SeqCst);
}

/// The armed install target, if any.
pub fn install_target() -> Option<u32> {
    match INSTALL_TARGET.load(Ordering::SeqCst) {
        NO_INSTALL_TARGET => None,
        dev_num => Some(dev_num),
    }
}

/// Raw install write. This is the only door through which the partitioner and
/// the filesystem formatter reach a disk, so the target check happens once,
/// here, for every caller. `arm_install_target` never arms the boot disk.
pub fn write_install_block(dev_num: u32, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
    if install_target() != Some(dev_num) {
        return Err(BlockError::Refused);
    }
    write_block(dev_num, lba, buf)
}
