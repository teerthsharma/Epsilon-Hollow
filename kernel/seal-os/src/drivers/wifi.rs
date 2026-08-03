// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! WiFi driver — PCI probe, bundle section request, honest status reporting.
//!
//! The driver detects hardware and then asks the bundle for the vendor section
//! (firmware) matching the PCI IDs. No section, no radio: the driver reports
//! `section_missing` and every operation fails with the section name. Seal OS ships
//! no vendor sections, so on an unprovisioned system this is the normal outcome.
//!
//! Even with a section resident, 802.11 association is not implemented — the
//! section is fetched and verified, not yet uploaded to the device. `scan()`
//! returns nothing in either case, and nothing in this file generates network
//! entries.

use alloc::string::String;
use alloc::vec::Vec;

use crate::bundle::{self, SectionError, SectionRef};

const WIFI_CHIPSETS: &[(u16, u16, &str)] = &[
    (0x8086, 0x2723, "Intel Wi-Fi 6 AX200"),
    (0x8086, 0x2725, "Intel Wi-Fi 6E AX210"),
    (0x8086, 0x7AF0, "Intel Wi-Fi 6E AX211"),
    (0x8086, 0x51F0, "Intel Wi-Fi 7 BE200"),
    (0x8086, 0x24FD, "Intel Wireless 8265"),
    (0x8086, 0x24F3, "Intel Wireless 8260"),
    (0x8086, 0x095A, "Intel Wireless 7265"),
    (0x14E4, 0x43A0, "Broadcom BCM4360"),
    (0x14E4, 0x4365, "Broadcom BCM43142"),
    (0x168C, 0x003E, "Qualcomm Atheros QCA6174"),
    (0x168C, 0x0046, "Qualcomm Atheros QCA9984"),
    (0x10EC, 0xC822, "Realtek RTL8822CE"),
    (0x10EC, 0x8852, "Realtek RTL8852AE"),
];

const PCI_CLASS_NETWORK: u8 = 0x02;
const PCI_SUBCLASS_WIRELESS: u8 = 0x80;

#[derive(Debug, Clone)]
pub struct WifiNetwork {
    pub ssid: String,
    pub signal_dbm: i32,
    pub security: WifiSecurity,
    pub channel: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WifiSecurity {
    Open,
    Wpa2,
    Wpa3,
}

impl WifiSecurity {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Wpa2 => "WPA2",
            Self::Wpa3 => "WPA3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WifiState {
    Disabled,
    /// Hardware present, vendor section refused or absent. The radio stays down.
    SectionMissing,
    Disconnected,
    Scanning,
    Connecting,
    Connected,
    NoHardware,
}

pub struct WifiDriver {
    state: WifiState,
    connected_ssid: Option<String>,
    ip_addr: Option<[u8; 4]>,
    pci_bar: u64,
    chipset_name: Option<&'static str>,
    vendor_id: u16,
    device_id: u16,
    section_name: Option<String>,
    section: Option<SectionRef>,
    section_error: Option<SectionError>,
}

/// Bundle section name for a wireless controller, derived from its PCI IDs.
pub fn section_name_for(vendor_id: u16, device_id: u16) -> String {
    alloc::format!("wifi-{:04x}-{:04x}.section", vendor_id, device_id)
}

impl WifiDriver {
    pub fn new() -> Self {
        Self {
            state: WifiState::NoHardware,
            connected_ssid: None,
            ip_addr: None,
            pci_bar: 0,
            chipset_name: None,
            vendor_id: 0,
            device_id: 0,
            section_name: None,
            section: None,
            section_error: None,
        }
    }

    pub fn probe_pci(&mut self) {
        let devices = crate::drivers::pci::get_devices();
        for dev in &devices {
            if dev.class == PCI_CLASS_NETWORK && dev.subclass == PCI_SUBCLASS_WIRELESS {
                self.vendor_id = dev.vendor_id;
                self.device_id = dev.device_id;
                self.pci_bar = dev.bar_address(0);
                self.chipset_name = WIFI_CHIPSETS
                    .iter()
                    .find(|(v, d, _)| *v == dev.vendor_id && *d == dev.device_id)
                    .map(|(_, _, name)| *name);
                self.request_section();
                return;
            }
        }
        self.state = WifiState::NoHardware;
    }

    /// Ask the bundle for this controller's section. Fail-closed: on any error the
    /// radio stays down and the error is kept for reporting.
    fn request_section(&mut self) {
        let name = section_name_for(self.vendor_id, self.device_id);
        match bundle::request_section(&name) {
            Ok(section) => {
                self.section = Some(section);
                self.section_error = None;
                self.state = WifiState::Disconnected;
            }
            Err(e) => {
                self.section = None;
                self.section_error = Some(e);
                self.state = WifiState::SectionMissing;
            }
        }
        self.section_name = Some(name);
    }

    pub fn init_from_pci(&mut self, bar0: u32) {
        self.pci_bar = bar0 as u64;
        self.request_section();
    }

    /// Section name this controller needs, once probed.
    pub fn section_name(&self) -> Option<&str> {
        self.section_name.as_deref()
    }

    pub fn section_error(&self) -> Option<SectionError> {
        self.section_error
    }

    /// Size of the resident section, or 0 when none is provisioned.
    pub fn section_bytes(&self) -> usize {
        self.section.as_ref().map(|c| c.bytes.len()).unwrap_or(0)
    }

    /// True only when the radio has a verified section resident.
    pub fn is_up(&self) -> bool {
        self.section.is_some()
            && matches!(
                self.state,
                WifiState::Disconnected
                    | WifiState::Scanning
                    | WifiState::Connecting
                    | WifiState::Connected
            )
    }

    pub fn state_tag(&self) -> &'static str {
        match self.state {
            WifiState::NoHardware => "no_hardware",
            WifiState::Disabled => "disabled",
            WifiState::SectionMissing => "section_missing",
            WifiState::Disconnected => "disconnected",
            WifiState::Scanning => "scanning",
            WifiState::Connecting => "connecting",
            WifiState::Connected => "connected",
        }
    }

    pub fn status_string(&self) -> String {
        match self.state {
            WifiState::NoHardware => String::from("WiFi: no wireless hardware detected"),
            WifiState::Disabled => String::from("WiFi: disabled"),
            WifiState::SectionMissing => alloc::format!(
                "WiFi: {} ({:04X}:{:04X}) down — bundle section '{}' {}",
                self.chipset_name.unwrap_or("unknown wireless controller"),
                self.vendor_id,
                self.device_id,
                self.section_name.as_deref().unwrap_or("?"),
                self.section_error
                    .map(|e| e.tag())
                    .unwrap_or("not_provisioned")
            ),
            WifiState::Disconnected => match self.chipset_name {
                Some(name) => alloc::format!(
                    "WiFi: {} ({:04X}:{:04X}) disconnected",
                    name,
                    self.vendor_id,
                    self.device_id
                ),
                None => alloc::format!(
                    "WiFi: unknown wireless controller {:04X}:{:04X} disconnected",
                    self.vendor_id,
                    self.device_id
                ),
            },
            WifiState::Scanning => String::from("WiFi: scanning..."),
            WifiState::Connecting => alloc::format!(
                "WiFi: connecting to {}...",
                self.connected_ssid.as_deref().unwrap_or("?")
            ),
            WifiState::Connected => alloc::format!(
                "WiFi: connected to {} (IP {}.{}.{}.{})",
                self.connected_ssid.as_deref().unwrap_or("?"),
                self.ip_addr.map_or(0, |a| a[0]),
                self.ip_addr.map_or(0, |a| a[1]),
                self.ip_addr.map_or(0, |a| a[2]),
                self.ip_addr.map_or(0, |a| a[3]),
            ),
        }
    }

    /// Scan for available networks.
    ///
    /// Always empty. Without a section the radio is down; with a section the
    /// upload and 802.11 frame path are still unimplemented. This function
    /// never fabricates entries — an empty result means no scan happened.
    pub fn scan(&mut self) -> Vec<WifiNetwork> {
        if !self.is_up() {
            return Vec::new();
        }
        let prev = self.state;
        self.state = WifiState::Scanning;
        // TODO: section upload to device SRAM, then 802.11 active/passive scan.
        self.state = prev;
        Vec::new()
    }

    pub fn connect(&mut self, _ssid: &str, _password: &str) -> Result<(), String> {
        if self.state == WifiState::NoHardware {
            return Err(String::from("WiFi: no wireless hardware detected"));
        }
        if !self.is_up() {
            return Err(alloc::format!(
                "WiFi: bundle section '{}' {} — install the vendor section package to bring the radio up",
                self.section_name.as_deref().unwrap_or("?"),
                self.section_error
                    .map(|e| e.tag())
                    .unwrap_or("not_provisioned")
            ));
        }
        Err(String::from(
            "WiFi: section resident but 802.11 association is not implemented",
        ))
    }

    pub fn disconnect(&mut self) {
        self.state = if self.section.is_some() {
            WifiState::Disconnected
        } else if self.chipset_name.is_some() {
            WifiState::SectionMissing
        } else {
            WifiState::NoHardware
        };
        self.connected_ssid = None;
        self.ip_addr = None;
    }

    pub fn state(&self) -> WifiState {
        self.state
    }

    pub fn connected_ssid(&self) -> Option<&str> {
        self.connected_ssid.as_deref()
    }

    pub fn ip_addr(&self) -> Option<[u8; 4]> {
        self.ip_addr
    }
}

impl Default for WifiDriver {
    fn default() -> Self {
        Self::new()
    }
}

pub fn init() {
    let mut driver = WifiDriver::new();
    driver.probe_pci();
    crate::serial_println!("{}", driver.status_string());
}
