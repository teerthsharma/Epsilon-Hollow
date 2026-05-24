// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! High-level TCP socket API -- wires to net::tcp and e1000 via net::transmit.

use alloc::vec::Vec;
use crate::net::tcp::TcpState;

pub struct TcpSocket {
    idx: Option<usize>,
}

impl TcpSocket {
    pub fn new() -> Self {
        Self { idx: None }
    }

    pub fn connect(&mut self, ip: crate::net::IpAddr, port: u16) {
        if self.idx.is_none() {
            self.idx = Some(crate::net::tcp::socket());
        }
        if let Some(idx) = self.idx {
            crate::net::tcp::connect(idx, ip, port);
        }
    }

    pub fn send(&mut self, data: &[u8]) {
        if let Some(idx) = self.idx {
            crate::net::tcp::send(idx, data);
        }
    }

    pub fn recv(&mut self, buf: &mut [u8]) -> usize {
        if let Some(idx) = self.idx {
            crate::net::tcp::recv(idx, buf)
        } else {
            0
        }
    }

    pub fn close(&mut self) {
        if let Some(idx) = self.idx {
            crate::net::tcp::close(idx);
        }
    }

    pub fn state(&self) -> TcpState {
        if let Some(idx) = self.idx {
            crate::net::tcp::state(idx)
        } else {
            TcpState::Closed
        }
    }
}

/// Build and transmit a raw TCP segment over IPv4/Ethernet directly.
/// Used for driver-level TX where the full frame is assembled here.
pub fn send_tcp_packet(remote_ip: crate::net::IpAddr, remote_port: u16, local_port: u16, seq: u32, ack: u32, flags: u16, payload: &[u8]) {
    #[repr(C, packed)]
    struct TcpHeader {
        src_port: u16,
        dst_port: u16,
        seq: u32,
        ack: u32,
        data_offset_flags: u16,
        window: u16,
        checksum: u16,
        urgent: u16,
    }

    let mut hdr = TcpHeader {
        src_port: local_port.to_be(),
        dst_port: remote_port.to_be(),
        seq: seq.to_be(),
        ack: ack.to_be(),
        data_offset_flags: ((5u16 << 12) | flags).to_be(),
        window: 65535u16.to_be(),
        checksum: 0,
        urgent: 0,
    };

    match remote_ip {
        crate::net::IpAddr::V4(remote_ip) => {
            let src_ip = crate::net::local_ip();
            let mut pseudo = Vec::with_capacity(12 + 20 + payload.len());
            pseudo.extend_from_slice(&src_ip);
            pseudo.extend_from_slice(&remote_ip);
            pseudo.push(0);
            pseudo.push(6);
            pseudo.extend_from_slice(&((20 + payload.len()) as u16).to_be_bytes());
            let hdr_bytes = unsafe {
                core::slice::from_raw_parts(&hdr as *const _ as *const u8, 20)
            };
            pseudo.extend_from_slice(hdr_bytes);
            pseudo.extend_from_slice(payload);
            let cksum = crate::net::ipv4::internet_checksum(&pseudo);
            unsafe { core::ptr::addr_of_mut!(hdr.checksum).write(cksum); }

            let mut tcp_seg = Vec::with_capacity(20 + payload.len());
            let hdr_bytes = unsafe {
                core::slice::from_raw_parts(&hdr as *const _ as *const u8, 20)
            };
            tcp_seg.extend_from_slice(hdr_bytes);
            tcp_seg.extend_from_slice(payload);
            crate::net::ipv4::send_ipv4_packet(remote_ip, 6, &tcp_seg);
        }
        crate::net::IpAddr::V6(remote_ip) => {
            let src_ip = crate::net::local_ip_v6();
            let mut pseudo = Vec::with_capacity(40 + 20 + payload.len());
            pseudo.extend_from_slice(&src_ip);
            pseudo.extend_from_slice(&remote_ip);
            pseudo.extend_from_slice(&((20 + payload.len()) as u32).to_be_bytes());
            pseudo.push(0); pseudo.push(0); pseudo.push(0);
            pseudo.push(6);
            let hdr_bytes = unsafe {
                core::slice::from_raw_parts(&hdr as *const _ as *const u8, 20)
            };
            pseudo.extend_from_slice(hdr_bytes);
            pseudo.extend_from_slice(payload);
            let cksum = crate::net::ipv4::internet_checksum(&pseudo);
            unsafe { core::ptr::addr_of_mut!(hdr.checksum).write(cksum); }

            let mut tcp_seg = Vec::with_capacity(20 + payload.len());
            let hdr_bytes = unsafe {
                core::slice::from_raw_parts(&hdr as *const _ as *const u8, 20)
            };
            tcp_seg.extend_from_slice(hdr_bytes);
            tcp_seg.extend_from_slice(payload);
            crate::net::ipv6::send_ipv6_packet(remote_ip, 6, &tcp_seg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_socket_lifecycle() {
        let mut sock = TcpSocket::new();
        assert_eq!(sock.state(), TcpState::Closed);
    }
}
