// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! USB subsystem — xHCI host controller, HID, mass storage.

pub mod descriptor;
pub mod hid;
pub mod mass_storage;
pub mod xhci;

use alloc::vec::Vec;
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UsbSpeed {
    Low,       // 1.5 Mbps
    Full,      // 12 Mbps
    High,      // 480 Mbps
    Super,     // 5 Gbps
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

static USB_SUBSYSTEM: Mutex<UsbSubsystem> = Mutex::new(UsbSubsystem::new());

pub struct UsbSubsystem {
    devices: Vec<UsbDevice>,
    hid_devices: Vec<hid::HidDevice>,
    initialized: bool,
}

impl UsbSubsystem {
    pub const fn new() -> Self {
        Self {
            devices: Vec::new(),
            hid_devices: Vec::new(),
            initialized: false,
        }
    }

    pub fn init(&mut self) {
        if xhci::init().is_some() {
            self.initialized = true;
            crate::serial_println!("[USB] xHCI initialized");
        }
    }

    pub fn poll(&mut self) {
        if !self.initialized {
            return;
        }
        xhci::with_xhci(|ctrl| {
            ctrl.poll_ports(&mut self.devices, &mut self.hid_devices);
            ctrl.poll_hid(&mut self.hid_devices);
        });
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn devices(&self) -> &Vec<UsbDevice> {
        &self.devices
    }

    pub fn hid_count(&self) -> usize {
        self.hid_devices.len()
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

pub fn init() {
    USB_SUBSYSTEM.lock().init();
}

pub fn poll() {
    USB_SUBSYSTEM.lock().poll();
    mass_storage::poll();
}

pub fn device_count() -> usize {
    USB_SUBSYSTEM.lock().device_count()
}

pub fn devices() -> Vec<UsbDevice> {
    USB_SUBSYSTEM.lock().devices().clone()
}
