// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IPv4 protocol -- header, checksum, routing, encapsulation.

use alloc::vec::Vec;

#[repr(C, packed)]
pub struct Ipv4Header {
    pub ver_ihl: u8,
    pub tos: u8,
    pub total_len: u16,
    pub id: u16,
    pub flags_frag: u16,
    pub ttl: u8,
    pub protocol: u8,
    pub checksum: u16,
    pub src: [u8; 4],
    pub dst: [u8; 4],
}

impl Ipv4Header {
    pub fn new(protocol: u8, src: [u8; 4], dst: [u8; 4], payload_len: usize) -> Self {
        let total_len = (20 + payload_len) as u16;
        Self {
            ver_ihl: 0x45,
            tos: 0,
            total_len: total_len.to_be(),
            id: 0,
            flags_frag: 0x4000_u16.to_be(), // Don't fragment
            ttl: 64,
            protocol,
            checksum: 0,
            src,
            dst,
        }
    }

    pub fn compute_checksum(&mut self) {
        self.checksum = 0;
        let bytes = unsafe {
            core::slice::from_raw_parts(
                self as *const _ as *const u8,
                core::mem::size_of::<Ipv4Header>(),
            )
        };
        self.checksum = internet_checksum(bytes);
    }

    /// Parse an IPv4 header from a raw byte slice.
    /// Returns `None` if the slice is too short, the version is not 4,
    /// the IHL is < 5, or the header checksum is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 {
            return None;
        }
        let ver_ihl = bytes[0];
        let version = ver_ihl >> 4;
        let ihl = (ver_ihl & 0x0F) as usize;
        if version != 4 || ihl < 5 {
            return None;
        }
        let hdr_len = ihl * 4;
        if bytes.len() < hdr_len {
            return None;
        }
        // Verify checksum over the header only
        if internet_checksum(&bytes[..hdr_len]) != 0 {
            return None;
        }
        Some(Self {
            ver_ihl,
            tos: bytes[1],
            total_len: u16::from_be_bytes([bytes[2], bytes[3]]),
            id: u16::from_be_bytes([bytes[4], bytes[5]]),
            flags_frag: u16::from_be_bytes([bytes[6], bytes[7]]),
            ttl: bytes[8],
            protocol: bytes[9],
            checksum: u16::from_be_bytes([bytes[10], bytes[11]]),
            src: [bytes[12], bytes[13], bytes[14], bytes[15]],
            dst: [bytes[16], bytes[17], bytes[18], bytes[19]],
        })
    }
}

pub fn internet_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

pub fn send_ipv4_packet(dst: [u8; 4], protocol: u8, payload: &[u8]) {
    if dst == [127, 0, 0, 1] {
        // Loopback
        let mut pkt = Vec::with_capacity(20 + payload.len());
        let src = crate::net::local_ip();
        let mut hdr = Ipv4Header::new(protocol, src, dst, payload.len());
        hdr.compute_checksum();
        let hdr_bytes = unsafe {
            core::slice::from_raw_parts(
                &hdr as *const _ as *const u8,
                core::mem::size_of::<Ipv4Header>(),
            )
        };
        pkt.extend_from_slice(hdr_bytes);
        pkt.extend_from_slice(payload);
        handle_ipv4_packet(&pkt);
        return;
    }

    let src = crate::net::local_ip();
    let subnet = crate::net::subnet();
    let gateway = crate::net::gateway();

    let on_subnet = (0..4).all(|i| src[i] & subnet[i] == dst[i] & subnet[i]);
    let target_ip = if on_subnet { dst } else { gateway };

    let target_mac = match crate::net::arp::lookup(target_ip) {
        Some(mac) => mac,
        None => {
            // ARP request sent; packet dropped for now
            return;
        }
    };

    let src_mac = crate::net::local_mac();
    let mut frame = Vec::with_capacity(14 + 20 + payload.len());
    frame.extend_from_slice(&target_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&0x0800_u16.to_be_bytes());

    let mut hdr = Ipv4Header::new(protocol, src, dst, payload.len());
    hdr.compute_checksum();
    let hdr_bytes = unsafe {
        core::slice::from_raw_parts(
            &hdr as *const _ as *const u8,
            core::mem::size_of::<Ipv4Header>(),
        )
    };
    frame.extend_from_slice(hdr_bytes);
    frame.extend_from_slice(payload);
    crate::net::transmit(&frame);
}

pub fn handle_ipv4_packet(pkt: &[u8]) {
    if pkt.len() < core::mem::size_of::<Ipv4Header>() {
        return;
    }
    let hdr = unsafe { &*(pkt.as_ptr() as *const Ipv4Header) };
    let ihl = (hdr.ver_ihl & 0x0F) as usize * 4;
    if ihl < 20 || pkt.len() < ihl {
        return;
    }

    // Verify checksum
    let hdr_bytes = &pkt[..ihl];
    if internet_checksum(hdr_bytes) != 0 {
        return;
    }

    let total_len = u16::from_be(hdr.total_len) as usize;
    if total_len > pkt.len() {
        return;
    }

    let payload = &pkt[ihl..total_len];
    let protocol = hdr.protocol;
    let src = hdr.src;

    match protocol {
        1 => crate::net::icmp::handle_icmp_packet(src, payload),
        6 => crate::net::tcp::handle_tcp_packet(crate::net::IpAddr::V4(src), payload),
        17 => crate::net::udp::handle_udp_packet(crate::net::IpAddr::V4(src), payload),
        _ => {}
    }
}
