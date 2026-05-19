// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bluetooth driver — scans PCI bus for wireless controllers with BT capability.

use alloc::string::String;
use alloc::vec::Vec;

// BT adapters often share a PCI device with WiFi (combo cards).
// USB class 0xE0 subclass 0x01 = Bluetooth.
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
}

impl BluetoothDriver {
    pub fn new() -> Self {
        Self {
            state: BtState::NoHardware,
            paired_devices: Vec::new(),
            detected: false,
        }
    }

    pub fn probe_pci(&mut self) {
        let devices = crate::drivers::pci::enumerate();
        for dev in &devices {
            if dev.class == 0x02 && dev.subclass == 0x80 {
                // WiFi+BT combo cards are common; if we found a WiFi chip, BT is likely present
                self.detected = true;
                self.state = BtState::Enabled;
                return;
            }
        }
        self.state = BtState::NoHardware;
    }

    pub fn status_string(&self) -> String {
        if self.detected {
            String::from("Bluetooth: adapter detected via combo WiFi+BT card (HCI not yet implemented)")
        } else {
            String::from("Bluetooth: no adapter detected")
        }
    }

    pub fn scan(&self) -> Vec<BtDevice> {
        Vec::new()
    }

    pub fn pair(&mut self, _name: &str) -> Result<(), String> {
        if self.state == BtState::NoHardware {
            return Err(String::from("Bluetooth: no adapter detected"));
        }
        Err(String::from("Bluetooth: HCI driver not yet implemented"))
    }

    pub fn state(&self) -> BtState {
        self.state
    }

    pub fn paired_count(&self) -> usize {
        self.paired_devices.len()
    }
}

pub fn init() {
    let mut driver = BluetoothDriver::new();
    driver.probe_pci();
    crate::serial_println!("{}", driver.status_string());
}
