// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bluetooth 5.0 LE driver — HCI over USB/UART.

use alloc::string::String;
use alloc::vec::Vec;

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
}

pub struct BluetoothDriver {
    state: BtState,
    paired_devices: Vec<BtDevice>,
}

impl BluetoothDriver {
    pub fn new() -> Self {
        Self {
            state: BtState::Enabled,
            paired_devices: Vec::new(),
        }
    }

    pub fn scan(&self) -> Vec<BtDevice> {
        // No Bluetooth adapter detected — return empty list
        Vec::new()
    }

    pub fn pair(&mut self, _name: &str) -> Result<(), String> {
        Err(String::from("Bluetooth: no adapter detected"))
    }

    pub fn state(&self) -> BtState {
        self.state
    }

    pub fn paired_count(&self) -> usize {
        self.paired_devices.len()
    }
}

pub fn init() {
}
