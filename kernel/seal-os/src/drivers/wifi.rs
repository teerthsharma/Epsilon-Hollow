// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! WiFi driver — 802.11ac with WPA2 support.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct WifiNetwork {
    pub ssid: String,
    pub signal_dbm: i32,
    pub security: WifiSecurity,
    pub channel: u8,
}

#[derive(Debug, Clone, Copy)]
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
}

pub struct WifiDriver {
    state: WifiState,
    connected_ssid: Option<String>,
    ip_addr: Option<[u8; 4]>,
    pci_bar: u32,
}

impl WifiDriver {
    pub fn new() -> Self {
        Self {
            state: WifiState::Disconnected,
            connected_ssid: None,
            ip_addr: None,
            pci_bar: 0,
        }
    }

    pub fn init_from_pci(&mut self, bar0: u32) {
        self.pci_bar = bar0;
        self.state = WifiState::Disconnected;
    }

    // Hardware simulation — real driver requires physical chipset
    pub fn scan(&self) -> Vec<WifiNetwork> {
        alloc::vec![
            WifiNetwork {
                ssid: String::from("[Sim] HomeNetwork"),
                signal_dbm: -45,
                security: WifiSecurity::Wpa2,
                channel: 6,
            },
            WifiNetwork {
                ssid: String::from("[Sim] CoffeeShop"),
                signal_dbm: -62,
                security: WifiSecurity::Open,
                channel: 11,
            },
            WifiNetwork {
                ssid: String::from("[Sim] 5G_IoT"),
                signal_dbm: -70,
                security: WifiSecurity::Wpa3,
                channel: 36,
            },
        ]
    }

    // Hardware simulation — real driver requires physical chipset
    pub fn connect(&mut self, ssid: &str, _password: &str) -> Result<(), String> {
        self.state = WifiState::Connected;
        self.connected_ssid = Some(String::from(ssid));
        self.ip_addr = Some([192, 168, 1, 42]);
        Ok(()) // Simulated connection — no actual network traffic
    }

    pub fn disconnect(&mut self) {
        self.state = WifiState::Disconnected;
        self.connected_ssid = None;
        self.ip_addr = None;
    }

    pub fn state(&self) -> WifiState {
        self.state
    }

    pub fn connected_ssid(&self) -> Option<&str> {
        self.connected_ssid.as_deref()
    }
}

pub fn init() {
}
