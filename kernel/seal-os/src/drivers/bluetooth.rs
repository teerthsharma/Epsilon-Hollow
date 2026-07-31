// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![allow(dead_code)] // REASON: USB Bluetooth class constants for future HCI driver completion

//! Bluetooth driver — PCI probe, atlas chart request, honest status reporting.
//!
//! Same contract as the WiFi driver: the adapter's HCI chart is requested from
//! the atlas by PCI ID. No chart, no adapter. Seal OS ships no vendor charts,
//! so `chart_missing` is the normal state on an unprovisioned system.
//!
//! Even with a chart resident, HCI upload and L2CAP/ATT are unimplemented.
//! `scan()` returns nothing in either case and never fabricates devices.

use alloc::string::String;
use alloc::vec::Vec;

use crate::atlas::{self, ChartError, ChartRef};

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
    /// Adapter present, HCI chart refused or absent. The adapter stays down.
    ChartMissing,
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
    vendor_id: u16,
    device_id: u16,
    chart_name: Option<String>,
    chart: Option<ChartRef>,
    chart_error: Option<ChartError>,
}

/// Atlas chart name for a Bluetooth adapter, derived from its PCI IDs.
pub fn chart_name_for(vendor_id: u16, device_id: u16) -> String {
    alloc::format!("bt-{:04x}-{:04x}.chart", vendor_id, device_id)
}

impl BluetoothDriver {
    pub fn new() -> Self {
        Self {
            state: BtState::NoHardware,
            paired_devices: Vec::new(),
            detected: false,
            last_scan: Vec::new(),
            vendor_id: 0,
            device_id: 0,
            chart_name: None,
            chart: None,
            chart_error: None,
        }
    }

    pub fn probe_pci(&mut self) {
        let devices = crate::drivers::pci::get_devices();
        for dev in &devices {
            if dev.class == 0x02 && dev.subclass == 0x80 {
                self.detected = true;
                self.vendor_id = dev.vendor_id;
                self.device_id = dev.device_id;
                self.request_chart();
                return;
            }
        }
        self.state = BtState::NoHardware;
    }

    /// Ask the atlas for this adapter's HCI chart. Fail-closed.
    fn request_chart(&mut self) {
        let name = chart_name_for(self.vendor_id, self.device_id);
        match atlas::request_chart(&name) {
            Ok(chart) => {
                self.chart = Some(chart);
                self.chart_error = None;
                self.state = BtState::Enabled;
            }
            Err(e) => {
                self.chart = None;
                self.chart_error = Some(e);
                self.state = BtState::ChartMissing;
            }
        }
        self.chart_name = Some(name);
    }

    /// Chart name this adapter needs, once probed.
    pub fn chart_name(&self) -> Option<&str> {
        self.chart_name.as_deref()
    }

    pub fn chart_error(&self) -> Option<ChartError> {
        self.chart_error
    }

    /// Size of the resident chart, or 0 when none is provisioned.
    pub fn chart_bytes(&self) -> usize {
        self.chart.as_ref().map(|c| c.bytes.len()).unwrap_or(0)
    }

    /// True only when the adapter has a verified chart resident.
    pub fn is_up(&self) -> bool {
        self.chart.is_some()
            && matches!(
                self.state,
                BtState::Enabled | BtState::Scanning | BtState::Pairing
            )
    }

    pub fn state_tag(&self) -> &'static str {
        match self.state {
            BtState::NoHardware => "no_hardware",
            BtState::Disabled => "disabled",
            BtState::ChartMissing => "chart_missing",
            BtState::Enabled => "enabled",
            BtState::Scanning => "scanning",
            BtState::Pairing => "pairing",
        }
    }

    pub fn status_string(&self) -> String {
        match self.state {
            BtState::NoHardware => String::from("Bluetooth: no adapter detected"),
            BtState::Disabled => String::from("Bluetooth: disabled"),
            BtState::ChartMissing => alloc::format!(
                "Bluetooth: adapter {:04X}:{:04X} down — atlas chart '{}' {}",
                self.vendor_id,
                self.device_id,
                self.chart_name.as_deref().unwrap_or("?"),
                self.chart_error
                    .map(|e| e.tag())
                    .unwrap_or("not_provisioned")
            ),
            BtState::Enabled => {
                alloc::format!("Bluetooth: enabled ({} paired)", self.paired_devices.len())
            }
            BtState::Scanning => String::from("Bluetooth: scanning..."),
            BtState::Pairing => String::from("Bluetooth: pairing..."),
        }
    }

    /// Scan for nearby Bluetooth devices.
    ///
    /// Always empty. Without a chart the adapter is down; with a chart the HCI
    /// upload and inquiry/page procedures are still unimplemented. This
    /// function never fabricates devices.
    pub fn scan(&mut self) -> Vec<BtDevice> {
        if !self.is_up() {
            return Vec::new();
        }
        let prev = self.state;
        self.state = BtState::Scanning;
        // TODO: chart upload over HCI, then inquiry and LE scanning.
        self.state = prev;
        Vec::new()
    }

    pub fn pair(&mut self, _name: &str) -> Result<(), String> {
        if self.state == BtState::NoHardware {
            return Err(String::from("Bluetooth: no adapter detected"));
        }
        if !self.is_up() {
            return Err(alloc::format!(
                "Bluetooth: atlas chart '{}' {} — install the vendor chart package to bring the adapter up",
                self.chart_name.as_deref().unwrap_or("?"),
                self.chart_error
                    .map(|e| e.tag())
                    .unwrap_or("not_provisioned")
            ));
        }
        Err(String::from(
            "Bluetooth: chart resident but L2CAP/ATT pairing is not implemented",
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
