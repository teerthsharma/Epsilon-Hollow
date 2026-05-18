// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! USB mass storage class driver — bulk-only transport, SCSI READ/WRITE.
//! NOTE: This is a simulated driver. No real USB mass storage hardware is accessed.

use alloc::vec::Vec;

pub const MASS_STORAGE_CLASS: u8 = 8;
pub const SCSI_SUBCLASS: u8 = 6;
pub const BBB_PROTOCOL: u8 = 0x50; // Bulk-Only

pub struct MassStorageDriver {
    capacity_sectors: u64,
    sector_size: u32,
    connected: bool,
}

impl MassStorageDriver {
    pub fn new() -> Self {
        Self {
            capacity_sectors: 0,
            sector_size: 512,
            connected: false,
        }
    }

    pub fn inquiry(&self) -> &'static str {
        "[Sim] Seal OS USB Mass Storage"
    }

    pub fn read_capacity(&self) -> (u64, u32) {
        // [Sim] Returns stored values; no real SCSI READ CAPACITY command issued
        (self.capacity_sectors, self.sector_size)
    }

    pub fn read_sectors(&self, _lba: u64, _count: u32) -> Vec<u8> {
        // [Sim] No real bulk transfer — cannot read from simulated device
        Vec::new()
    }

    pub fn write_sectors(&self, _lba: u64, _data: &[u8]) -> Result<(), &'static str> {
        Err("[Sim] write_sectors: no real USB mass storage hardware present")
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }
}
