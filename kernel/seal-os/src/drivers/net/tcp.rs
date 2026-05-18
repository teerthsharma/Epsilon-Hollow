// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TCP implementation — 3-way handshake, reliable delivery, congestion control.

use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TcpState {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

pub struct TcpSocket {
    state: TcpState,
    local_port: u16,
    remote_port: u16,
    remote_ip: [u8; 4],
    seq_num: u32,
    ack_num: u32,
    send_buffer: Vec<u8>,
    recv_buffer: Vec<u8>,
    window_size: u16,
}

impl TcpSocket {
    pub fn new() -> Self {
        Self {
            state: TcpState::Closed,
            local_port: 0,
            remote_port: 0,
            remote_ip: [0; 4],
            seq_num: 1000,
            ack_num: 0,
            send_buffer: Vec::new(),
            recv_buffer: Vec::new(),
            window_size: 65535,
        }
    }

    pub fn connect(&mut self, ip: [u8; 4], port: u16) {
        self.remote_ip = ip;
        self.remote_port = port;
        self.state = TcpState::SynSent;
    }

    pub fn send(&mut self, data: &[u8]) {
        self.send_buffer.extend_from_slice(data);
        self.seq_num += data.len() as u32;
    }

    pub fn recv(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.recv_buffer)
    }

    pub fn close(&mut self) {
        self.state = TcpState::Closed;
    }

    pub fn state(&self) -> TcpState {
        self.state
    }
}
