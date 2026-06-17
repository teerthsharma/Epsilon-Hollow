// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! WiFi driver — PCI probe and honest status reporting.
//! Real firmware upload and 802.11 frame processing are not yet implemented;
//! the driver detects hardware but returns empty scans until firmware support
//! is added.

use alloc::string::String;
use alloc::vec::Vec;

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
                self.state = WifiState::Disconnected;
                return;
            }
        }
        self.state = WifiState::NoHardware;
    }

    pub fn init_from_pci(&mut self, bar0: u32) {
        self.pci_bar = bar0 as u64;
        self.state = WifiState::Disconnected;
    }

    pub fn status_string(&self) -> String {
        match self.state {
            WifiState::NoHardware => String::from("WiFi: no wireless hardware detected"),
            WifiState::Disabled => String::from("WiFi: disabled"),
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
    /// Currently returns empty because firmware upload and 802.11 frame
    /// processing are not yet implemented. Hardware detection works.
    pub fn scan(&mut self) -> Vec<WifiNetwork> {
        if self.state == WifiState::NoHardware {
            return Vec::new();
        }
        let prev = self.state;
        self.state = WifiState::Scanning;
        // TODO: implement real 802.11 active/passive scan once firmware
        // upload and frame construction are available.
        self.state = prev;
        Vec::new()
    }

    pub fn connect(&mut self, _ssid: &str, _password: &str) -> Result<(), String> {
        if self.state == WifiState::NoHardware {
            return Err(String::from("WiFi: no wireless hardware detected"));
        }
        Err(String::from(
            "WiFi: connection not implemented — firmware upload and 802.11 association are pending",
        ))
    }

    pub fn disconnect(&mut self) {
        self.state = if self.chipset_name.is_some() {
            WifiState::Disconnected
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

pub fn init() {
    let mut driver = WifiDriver::new();
    driver.probe_pci();
    crate::serial_println!("{}", driver.status_string());
}
