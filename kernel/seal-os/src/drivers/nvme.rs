// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! NVMe controller driver stub — honest PCI probe only.

use crate::drivers::pci::get_device_by_class;
use crate::serial_println;

pub struct NvmeController {
    pub bar0: u64,
    pub vendor_id: u16,
    pub device_id: u16,
}

impl NvmeController {
    pub fn probe() -> Option<Self> {
        serial_println!("[NVMe] Probing PCI class 0x01/0x08 (Non-Volatile Memory Controller)...");

        let dev = match get_device_by_class(0x01, 0x08, 0x02) {
            Some(d) => d,
            None => {
                serial_println!("[NVMe] No NVMe controller detected");
                return None;
            }
        };

        serial_println!(
            "[NVMe] Found controller at {:02x}:{:02x}.{} — {:04X}:{:04X}",
            dev.bus,
            dev.device,
            dev.function,
            dev.vendor_id,
            dev.device_id
        );

        Some(Self {
            bar0: dev.bar_address(0),
            vendor_id: dev.vendor_id,
            device_id: dev.device_id,
        })
    }
}

pub fn init() -> Option<()> {
    let _ctrl = NvmeController::probe()?;
    serial_println!("[NVMe] Controller detected — full driver not yet implemented");
    None
}
