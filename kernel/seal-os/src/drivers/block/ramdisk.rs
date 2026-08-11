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
    /// Fails if `sectors * SECTOR` would overflow `usize`. Left unchecked,
    /// that overflow wraps the backing allocation down to (near) zero bytes
    /// while `self.sectors` — the value `range()` bound-checks reads and
    /// writes against — kept the caller's full, untruncated count, so a
    /// later `read_sectors`/`write_sectors` call would pass its bounds check
    /// and then index past the end of an undersized `Vec`.
    pub fn new(sectors: u64) -> Result<Self, BlockError> {
        let bytes = (sectors as usize)
            .checked_mul(SECTOR)
            .ok_or(BlockError::InvalidLba)?;
        Ok(Self {
            sectors,
            data: Mutex::new(vec![0u8; bytes]),
        })
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
    let disk: &'static RamDisk = Box::leak(Box::new(RamDisk::new(sectors)?));
    register_block_device(SCRATCH_DEV_NUM, disk);
    Ok(SCRATCH_DEV_NUM)
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::test_assert_eq;
    use crate::testing::TestResult;

    /// Regression: `RamDisk::new` used to compute the backing `Vec`'s byte
    /// length as `(sectors as usize) * SECTOR` with no overflow check, while
    /// storing the caller's raw `sectors` (the value `range()` bound-checks
    /// against) unmodified. `1u64 << 55` makes that multiplication wrap to
    /// 0 on a 64-bit `usize`, so the `Vec` came out empty while `self.sectors`
    /// still claimed 2^55 — a later `read_sectors(0, ..)` would pass the
    /// bounds check and then index past the end of an empty `Vec`. Not
    /// reachable through any call site in this crate today (the only caller,
    /// `ensure_scratch`, is only ever invoked with the constant
    /// `SCRATCH_SECTORS = 8192`); this guards the constructor itself.
    fn test_new_rejects_overflowing_size() -> TestResult {
        test_assert!(RamDisk::new(1u64 << 55).is_err());
        TestResult::Pass
    }

    /// Sanity check that the `Result`-ification above didn't break ordinary
    /// use: a normal-sized disk still constructs and round-trips data.
    fn test_new_normal_size_read_write_roundtrip() -> TestResult {
        let disk = match RamDisk::new(8) {
            Ok(d) => d,
            Err(_) => return TestResult::Fail("RamDisk::new(8) must not fail"),
        };
        let written = [0xABu8; SECTOR];
        test_assert!(disk.write_sectors(0, &written).is_ok());
        let mut read_back = [0u8; SECTOR];
        test_assert!(disk.read_sectors(0, &mut read_back).is_ok());
        test_assert_eq!(written, read_back);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "ramdisk::new_rejects_overflowing_size",
            test_new_rejects_overflowing_size,
        );
        crate::testing::register_test(
            "ramdisk::new_normal_size_read_write_roundtrip",
            test_new_normal_size_read_write_roundtrip,
        );
    }
}
