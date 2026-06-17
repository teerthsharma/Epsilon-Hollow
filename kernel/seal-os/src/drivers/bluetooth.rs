// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![allow(dead_code)] // REASON: USB Bluetooth class constants for future HCI driver completion

//! Bluetooth driver — PCI probe and honest status reporting.
//! Real HCI firmware upload and L2CAP/ATT processing are not yet implemented;
//! the driver detects hardware but returns empty scans until firmware support
//! is added.

use alloc::string::String;
use alloc::vec::Vec;

const USB_CLASS_WIRELESS: u8 = 0xE0;
const USB_SUBCLASS_BT: u8 = 0x01;

#[derive(Debug, Clone)]
pub struct BtDevice {
    pub name: String,
    pub device_type: BtDeviceType,
    pub rssi_dbm: i32,
    pub paired: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum BtDeviceType {
    Audio,
    Hid,
    Le,
    Unknown,
}

impl BtDeviceType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Audio => "Audio",
            Self::Hid => "HID",
            Self::Le => "LE",
            Self::Unknown => "Unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BtState {
    Disabled,
    Enabled,
    Scanning,
    Pairing,
    NoHardware,
}

pub struct BluetoothDriver {
    state: BtState,
    paired_devices: Vec<BtDevice>,
    detected: bool,
    last_scan: Vec<BtDevice>,
}

impl BluetoothDriver {
    pub fn new() -> Self {
        Self {
            state: BtState::NoHardware,
            paired_devices: Vec::new(),
            detected: false,
            last_scan: Vec::new(),
        }
    }

    pub fn probe_pci(&mut self) {
        let devices = crate::drivers::pci::get_devices();
        for dev in &devices {
            if dev.class == 0x02 && dev.subclass == 0x80 {
                self.detected = true;
                self.state = BtState::Enabled;
                return;
            }
        }
        self.state = BtState::NoHardware;
    }

    pub fn status_string(&self) -> String {
        match self.state {
            BtState::NoHardware => String::from("Bluetooth: no adapter detected"),
            BtState::Disabled => String::from("Bluetooth: disabled"),
            BtState::Enabled => {
                alloc::format!("Bluetooth: enabled ({} paired)", self.paired_devices.len())
            }
            BtState::Scanning => String::from("Bluetooth: scanning..."),
            BtState::Pairing => String::from("Bluetooth: pairing..."),
        }
    }

    /// Scan for nearby Bluetooth devices.
    /// Currently returns empty because HCI firmware upload and inquiry/page
    /// procedures are not yet implemented. Hardware detection works.
    pub fn scan(&mut self) -> Vec<BtDevice> {
        if self.state == BtState::NoHardware {
            return Vec::new();
        }
        let prev = self.state;
        self.state = BtState::Scanning;
        // TODO: implement real HCI inquiry and LE scanning once firmware
        // upload and L2CAP/ATT layers are available.
        self.state = prev;
        Vec::new()
    }

    pub fn pair(&mut self, _name: &str) -> Result<(), String> {
        if self.state == BtState::NoHardware {
            return Err(String::from("Bluetooth: no adapter detected"));
        }
        Err(String::from(
            "Bluetooth: pairing not implemented — HCI firmware and L2CAP/ATT are pending",
        ))
    }

    pub fn unpair(&mut self, name: &str) {
        self.paired_devices.retain(|d| d.name != name);
    }

    pub fn state(&self) -> BtState {
        self.state
    }

    pub fn paired_count(&self) -> usize {
        self.paired_devices.len()
    }

    pub fn paired_devices(&self) -> &[BtDevice] {
        &self.paired_devices
    }
}

pub fn init() {
    let mut driver = BluetoothDriver::new();
    driver.probe_pci();
    crate::serial_println!("{}", driver.status_string());
}
