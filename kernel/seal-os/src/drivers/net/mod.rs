// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Network stack — TCP/UDP/DHCP/DNS/HTTP/TLS.

pub mod tcp;
pub mod udp;
pub mod dhcp;
pub mod dns;
pub mod http;
pub mod tls;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetState {
    Down,
    Up,
    Connected,
}

pub struct NetworkStack {
    state: NetState,
    ip: [u8; 4],
    gateway: [u8; 4],
    dns: [u8; 4],
    subnet: [u8; 4],
}

impl NetworkStack {
    pub fn new() -> Self {
        Self {
            state: NetState::Down,
            ip: [0; 4],
            gateway: [0; 4],
            dns: [8, 8, 8, 8],
            subnet: [255, 255, 255, 0],
        }
    }

    /// [Sim] No network hardware — stores config locally but no NIC is programmed.
    pub fn configure(&mut self, ip: [u8; 4], gateway: [u8; 4], dns: [u8; 4]) {
        self.ip = ip;
        self.gateway = gateway;
        self.dns = dns;
        self.state = NetState::Connected;
    }

    pub fn state(&self) -> NetState {
        self.state
    }
}

pub fn init() {
}
