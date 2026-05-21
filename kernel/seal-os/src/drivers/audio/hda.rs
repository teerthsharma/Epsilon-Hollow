// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Intel High Definition Audio (HDA) driver stub — honest PCI probe only.

use crate::drivers::pci::get_device_by_class;
use crate::serial_println;

pub struct HdaController {
    pub bar0: u64,
    pub vendor_id: u16,
    pub device_id: u16,
}

impl HdaController {
    pub fn probe() -> Option<Self> {
        serial_println!("[HDA] Probing PCI class 0x04/0x03 (High Definition Audio)...");

        let dev = match get_device_by_class(0x04, 0x03, 0x00) {
            Some(d) => d,
            None => {
                serial_println!("[HDA] No HDA controller detected");
                return None;
            }
        };

        serial_println!(
            "[HDA] Found controller at {:02x}:{:02x}.{} — {:04X}:{:04X}",
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
    let _ctrl = HdaController::probe()?;
    serial_println!("[HDA] Controller detected — full driver not yet implemented");
    None
}
