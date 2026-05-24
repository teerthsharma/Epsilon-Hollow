// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! WiFi driver — PCI probe + topological scan simulation.
//! Real firmware upload is out of scope; this driver provides a fully
//! functional state machine and simulated network topology.

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
    last_scan: Vec<WifiNetwork>,
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
            last_scan: Vec::new(),
        }
    }

    pub fn probe_pci(&mut self) {
        let devices = crate::drivers::pci::enumerate();
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
                Some(name) => alloc::format!("WiFi: {} ({:04X}:{:04X}) disconnected", name, self.vendor_id, self.device_id),
                None => alloc::format!("WiFi: unknown wireless controller {:04X}:{:04X} disconnected", self.vendor_id, self.device_id),
            },
            WifiState::Scanning => String::from("WiFi: scanning..."),
            WifiState::Connecting => alloc::format!("WiFi: connecting to {}...", self.connected_ssid.as_deref().unwrap_or("?")),
            WifiState::Connected => alloc::format!("WiFi: connected to {} (IP {}.{}.{}.{})",
                self.connected_ssid.as_deref().unwrap_or("?"),
                self.ip_addr.map_or(0, |a| a[0]),
                self.ip_addr.map_or(0, |a| a[1]),
                self.ip_addr.map_or(0, |a| a[2]),
                self.ip_addr.map_or(0, |a| a[3]),
            ),
        }
    }

    /// Topological scan: generate a deterministic set of simulated networks
    /// based on the PCI vendor/device hash.  No firmware required.
    pub fn scan(&mut self) -> Vec<WifiNetwork> {
        if self.state == WifiState::NoHardware {
            return Vec::new();
        }
        let prev = self.state;
        self.state = WifiState::Scanning;

        let seed = (self.vendor_id as u64).wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(self.device_id as u64);
        let mut networks = Vec::new();

        let ssids = ["AlphaNet", "BetaWave", "GammaLink", "DeltaCore", "EpsilonMesh"];
        let securities = [WifiSecurity::Wpa3, WifiSecurity::Wpa2, WifiSecurity::Wpa2, WifiSecurity::Open, WifiSecurity::Wpa3];

        for i in 0..ssids.len() {
            let s = seed.wrapping_add(i as u64);
            let signal = -30 - ((s % 60) as i32); // -30 to -90 dBm
            let channel = 1 + ((s >> 8) % 11) as u8; // 1-11
            networks.push(WifiNetwork {
                ssid: String::from(ssids[i]),
                signal_dbm: signal,
                security: securities[i],
                channel,
            });
        }

        self.state = prev;
        self.last_scan = networks.clone();
        networks
    }

    pub fn connect(&mut self, ssid: &str, password: &str) -> Result<(), String> {
        if self.state == WifiState::NoHardware {
            return Err(String::from("WiFi: no wireless hardware detected"));
        }
        if self.last_scan.is_empty() {
            let _ = self.scan();
        }
        let net = self.last_scan.iter().find(|n| n.ssid == ssid);
        if net.is_none() {
            return Err(alloc::format!("WiFi: network '{}' not found", ssid));
        }
        let net = net.unwrap();
        if net.security != WifiSecurity::Open && password.is_empty() {
            return Err(alloc::format!("WiFi: password required for '{}'", ssid));
        }
        self.state = WifiState::Connecting;
        self.connected_ssid = Some(String::from(ssid));
        // Simulate DHCP assignment: IP derived from SSID hash.
        let ip = [
            192,
            168,
            1 + (ssid.bytes().fold(0u8, |a, b| a.wrapping_add(b)) % 250),
            100 + (password.len() as u8 % 100),
        ];
        self.ip_addr = Some(ip);
        self.state = WifiState::Connected;
        Ok(())
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
