// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! DHCP client — automatic IP assignment.

pub struct DhcpClient {
    state: DhcpState,
    offered_ip: [u8; 4],
    server_ip: [u8; 4],
    gateway: [u8; 4],
    dns: [u8; 4],
    subnet: [u8; 4],
    lease_seconds: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DhcpState {
    Init,
    Discover,
    Offer,
    Request,
    Bound,
}

impl DhcpClient {
    pub fn new() -> Self {
        Self {
            state: DhcpState::Init,
            offered_ip: [0; 4],
            server_ip: [0; 4],
            gateway: [0; 4],
            dns: [8, 8, 8, 8],
            subnet: [255, 255, 255, 0],
            lease_seconds: 86400,
        }
    }

    /// [Sim] No network hardware — assigns hardcoded IP without real DHCP exchange.
    pub fn discover(&mut self) {
        self.state = DhcpState::Discover;
        // [Sim] No NIC present — skipping real DHCP DISCOVER broadcast on UDP port 67
        self.offered_ip = [192, 168, 1, 42];
        self.gateway = [192, 168, 1, 1];
        self.state = DhcpState::Bound;
    }

    pub fn ip(&self) -> [u8; 4] {
        self.offered_ip
    }

    pub fn gateway(&self) -> [u8; 4] {
        self.gateway
    }

    pub fn dns(&self) -> [u8; 4] {
        self.dns
    }

    pub fn is_bound(&self) -> bool {
        self.state == DhcpState::Bound
    }
}
