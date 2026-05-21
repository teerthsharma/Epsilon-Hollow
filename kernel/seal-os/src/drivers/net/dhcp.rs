// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! DHCP client wrapper -- delegates to net::dhcp state machine.

use crate::net::dhcp::DhcpState;

pub struct DhcpClient;

impl DhcpClient {
    pub fn new() -> Self {
        Self
    }

    pub fn discover(&mut self) {
        crate::net::dhcp::init();
        crate::net::dhcp::poll();
    }

    pub fn poll(&mut self) {
        crate::net::dhcp::poll();
    }

    pub fn ip(&self) -> [u8; 4] {
        crate::net::local_ip()
    }

    pub fn gateway(&self) -> [u8; 4] {
        crate::net::gateway()
    }

    pub fn dns(&self) -> [u8; 4] {
        crate::net::dns::server()
    }

    pub fn subnet(&self) -> [u8; 4] {
        crate::net::subnet()
    }

    pub fn is_bound(&self) -> bool {
        crate::net::dhcp::is_bound()
    }

    pub fn state(&self) -> DhcpState {
        crate::net::dhcp::state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dhcp_client_init() {
        let mut client = DhcpClient::new();
        client.discover();
        assert_eq!(client.state(), DhcpState::Discover);
    }
}
