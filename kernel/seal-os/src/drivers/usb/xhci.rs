// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! xHCI (USB 3.x) host controller driver — MMIO register interface.
//! NOTE: Register structs are defined but hardware init is not yet implemented.

use crate::drivers::pci::get_device_by_class;
use crate::serial_println;

pub const XHCI_CLASS: u8 = 0x0C;
pub const XHCI_SUBCLASS: u8 = 0x03;
pub const XHCI_PROG_IF: u8 = 0x30;

#[repr(C)]
pub struct XhciCapRegs {
    pub caplength: u8,
    pub _rsvd: u8,
    pub hci_version: u16,
    pub hcsparams1: u32,
    pub hcsparams2: u32,
    pub hcsparams3: u32,
    pub hccparams1: u32,
    pub dboff: u32,
    pub rtsoff: u32,
    pub hccparams2: u32,
}

pub struct XhciController {
    base_addr: u64,
    max_slots: u8,
    max_ports: u8,
    initialized: bool,
}

impl XhciController {
    pub fn new(bar0: u64) -> Self {
        Self {
            base_addr: bar0,
            max_slots: 0,
            max_ports: 0,
            initialized: false,
        }
    }

    /// Honest reset — returns error because full xHCI init is not yet implemented.
    pub fn reset(&mut self) -> Result<(), &'static str> {
        if !self.initialized {
            return Err("xHCI reset not yet implemented");
        }
        Ok(())
    }

    pub fn max_ports(&self) -> u8 {
        self.max_ports
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn probe() -> Option<Self> {
        serial_println!("[xHCI] Probing PCI class 0x0C/0x03 (USB xHCI host controller)...");

        let dev = match get_device_by_class(XHCI_CLASS, XHCI_SUBCLASS, XHCI_PROG_IF) {
            Some(d) => d,
            None => {
                serial_println!("[xHCI] No xHCI controller detected");
                return None;
            }
        };

        serial_println!(
            "[xHCI] Found controller at {:02x}:{:02x}.{} — {:04X}:{:04X}",
            dev.bus,
            dev.device,
            dev.function,
            dev.vendor_id,
            dev.device_id
        );

        Some(Self::new(dev.bar_address(0)))
    }
}

pub fn init() -> Option<()> {
    let mut ctrl = XhciController::probe()?;
    serial_println!("[xHCI] Controller detected — full driver not yet implemented");
    // Do not fake initialization success
    if let Err(e) = ctrl.reset() {
        serial_println!("[xHCI] {}", e);
    }
    None
}
