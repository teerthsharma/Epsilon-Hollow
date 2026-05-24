// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bluetooth driver — PCI probe + topological device simulation.
//! Real HCI firmware upload is out of scope; this driver provides a fully
//! functional state machine and simulated device topology.

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
        let devices = crate::drivers::pci::enumerate();
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
            BtState::Enabled => alloc::format!("Bluetooth: enabled ({} paired)", self.paired_devices.len()),
            BtState::Scanning => String::from("Bluetooth: scanning..."),
            BtState::Pairing => String::from("Bluetooth: pairing..."),
        }
    }

    /// Topological scan: generate deterministic simulated BLE devices.
    pub fn scan(&mut self) -> Vec<BtDevice> {
        if self.state == BtState::NoHardware {
            return Vec::new();
        }
        let prev = self.state;
        self.state = BtState::Scanning;

        let devices = vec![
            BtDevice {
                name: String::from("TopoMouse"),
                device_type: BtDeviceType::Hid,
                rssi_dbm: -42,
                paired: false,
            },
            BtDevice {
                name: String::from("SpectralHeadset"),
                device_type: BtDeviceType::Audio,
                rssi_dbm: -55,
                paired: false,
            },
            BtDevice {
                name: String::from("ManifoldSensor"),
                device_type: BtDeviceType::Le,
                rssi_dbm: -68,
                paired: false,
            },
        ];

        // Merge with already-paired devices, marking them paired.
        let mut result = self.paired_devices.clone();
        for d in &devices {
            if !result.iter().any(|p| p.name == d.name) {
                result.push(d.clone());
            }
        }

        self.state = prev;
        self.last_scan = result.clone();
        result
    }

    pub fn pair(&mut self, name: &str) -> Result<(), String> {
        if self.state == BtState::NoHardware {
            return Err(String::from("Bluetooth: no adapter detected"));
        }
        if self.last_scan.is_empty() {
            let _ = self.scan();
        }
        if let Some(dev) = self.last_scan.iter().find(|d| d.name == name) {
            if self.paired_devices.iter().any(|p| p.name == name) {
                return Ok(());
            }
            self.state = BtState::Pairing;
            let mut paired = dev.clone();
            paired.paired = true;
            self.paired_devices.push(paired);
            self.state = BtState::Enabled;
            Ok(())
        } else {
            Err(alloc::format!("Bluetooth: device '{}' not found", name))
        }
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
