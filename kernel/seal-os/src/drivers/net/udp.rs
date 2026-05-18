// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! UDP implementation — connectionless datagrams.

use alloc::vec::Vec;

pub struct UdpSocket {
    local_port: u16,
    recv_buffer: Vec<(([u8; 4], u16), Vec<u8>)>,
}

impl UdpSocket {
    pub fn new(port: u16) -> Self {
        Self {
            local_port: port,
            recv_buffer: Vec::new(),
        }
    }

    pub fn send_to(&self, _data: &[u8], _ip: [u8; 4], _port: u16) {
        // Packet assembly and NIC transmit
    }

    pub fn recv_from(&mut self) -> Option<(([u8; 4], u16), Vec<u8>)> {
        self.recv_buffer.pop()
    }

    pub fn port(&self) -> u16 {
        self.local_port
    }
}
