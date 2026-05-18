// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! USB subsystem — xHCI host controller, HID, mass storage.
//! NOTE: This is a simulated subsystem. No real USB hardware is accessed.

pub mod xhci;
pub mod descriptor;
pub mod hid;
pub mod mass_storage;

use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UsbSpeed {
    Low,      // 1.5 Mbps
    Full,     // 12 Mbps
    High,     // 480 Mbps
    Super,    // 5 Gbps
    SuperPlus, // 10 Gbps
}

impl UsbSpeed {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "Low (1.5 Mbps)",
            Self::Full => "Full (12 Mbps)",
            Self::High => "High (480 Mbps)",
            Self::Super => "Super (5 Gbps)",
            Self::SuperPlus => "Super+ (10 Gbps)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct UsbDevice {
    pub address: u8,
    pub speed: UsbSpeed,
    pub vendor_id: u16,
    pub product_id: u16,
    pub class: u8,
    pub subclass: u8,
    pub protocol: u8,
    pub port: u8,
}

impl UsbDevice {
    pub fn is_hid(&self) -> bool {
        self.class == 3
    }

    pub fn is_mass_storage(&self) -> bool {
        self.class == 8
    }

    pub fn is_hub(&self) -> bool {
        self.class == 9
    }
}

pub struct UsbSubsystem {
    devices: Vec<UsbDevice>,
    xhci_bar: u32,
    initialized: bool,
}

impl UsbSubsystem {
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            xhci_bar: 0,
            initialized: false,
        }
    }

    pub fn init_from_pci(&mut self, bar0: u32) {
        self.xhci_bar = bar0;
        self.initialized = true;
    }

    pub fn enumerate_ports(&mut self) {
        // No real port enumeration — no xHCI hardware present
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

pub fn init() {
    // USB subsystem init is a no-op — no real hardware to initialize
}
