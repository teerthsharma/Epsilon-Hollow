// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! IPv6 protocol -- fixed 40-byte header, ICMPv6, NDP (ARP replacement).

use alloc::vec::Vec;
use spin::Mutex;

#[repr(C, packed)]
pub struct Ipv6Header {
    pub ver_tc_fl: [u8; 4], // version(4), traffic class(8), flow label(20)
    pub payload_len: u16,
    pub next_header: u8,
    pub hop_limit: u8,
    pub src: [u8; 16],
    pub dst: [u8; 16],
}

impl Ipv6Header {
    pub fn version(&self) -> u8 {
        self.ver_tc_fl[0] >> 4
    }
    pub fn traffic_class(&self) -> u8 {
        ((self.ver_tc_fl[0] & 0x0F) << 4) | ((self.ver_tc_fl[1] & 0xF0) >> 4)
    }
    pub fn flow_label(&self) -> u32 {
        ((self.ver_tc_fl[1] as u32 & 0x0F) << 16)
            | ((self.ver_tc_fl[2] as u32) << 8)
            | (self.ver_tc_fl[3] as u32)
    }

    pub fn new(next_header: u8, src: [u8; 16], dst: [u8; 16], payload_len: usize) -> Self {
        Self {
            ver_tc_fl: [0x60, 0, 0, 0], // version 6
            payload_len: (payload_len as u16).to_be(),
            next_header,
            hop_limit: 64,
            src,
            dst,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 40 {
            return None;
        }
        if bytes[0] >> 4 != 6 {
            return None;
        }
        Some(Self {
            ver_tc_fl: [bytes[0], bytes[1], bytes[2], bytes[3]],
            payload_len: u16::from_be_bytes([bytes[4], bytes[5]]),
            next_header: bytes[6],
            hop_limit: bytes[7],
            src: [
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15], bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21],
                bytes[22], bytes[23],
            ],
            dst: [
                bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30],
                bytes[31], bytes[32], bytes[33], bytes[34], bytes[35], bytes[36], bytes[37],
                bytes[38], bytes[39],
            ],
        })
    }
}

pub fn send_ipv6_packet(dst: [u8; 16], next_header: u8, payload: &[u8]) {
    if dst == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] {
        // Loopback
        let mut pkt = Vec::with_capacity(40 + payload.len());
        let src = crate::net::local_ip_v6();
        let hdr = Ipv6Header::new(next_header, src, dst, payload.len());
        let hdr_bytes = unsafe {
            core::slice::from_raw_parts(
                &hdr as *const _ as *const u8,
                core::mem::size_of::<Ipv6Header>(),
            )
        };
        pkt.extend_from_slice(hdr_bytes);
        pkt.extend_from_slice(payload);
        handle_ipv6_packet(&pkt);
        return;
    }

    let src = crate::net::local_ip_v6();
    let src_mac = crate::net::local_mac();
    let dst_mac = ndp_lookup(dst).unwrap_or([0x33, 0x33, 0x00, 0x00, 0x00, 0x01]);

    let mut frame = Vec::with_capacity(14 + 40 + payload.len());
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&0x86DD_u16.to_be_bytes());

    let hdr = Ipv6Header::new(next_header, src, dst, payload.len());
    let hdr_bytes = unsafe {
        core::slice::from_raw_parts(
            &hdr as *const _ as *const u8,
            core::mem::size_of::<Ipv6Header>(),
        )
    };
    frame.extend_from_slice(hdr_bytes);
    frame.extend_from_slice(payload);
    crate::net::transmit(&frame);
}

pub fn handle_ipv6_packet(pkt: &[u8]) {
    if pkt.len() < core::mem::size_of::<Ipv6Header>() {
        return;
    }
    let hdr = unsafe { &*(pkt.as_ptr() as *const Ipv6Header) };
    if hdr.version() != 6 {
        return;
    }
    let payload_len = u16::from_be(hdr.payload_len) as usize;
    if payload_len > pkt.len() - 40 {
        return;
    }
    let payload = &pkt[40..40 + payload_len];
    let next_header = hdr.next_header;
    let src = hdr.src;

    match next_header {
        58 => handle_icmpv6_packet(src, payload),
        6 => crate::net::tcp::handle_tcp_packet(crate::net::IpAddr::V6(src), payload),
        17 => crate::net::udp::handle_udp_packet(crate::net::IpAddr::V6(src), payload),
        _ => {
            // Unknown IPv6 next-header; drop silently
        }
    }
}

// ---------------------------------------------------------------------------
// ICMPv6
// ---------------------------------------------------------------------------

const ICMPV6_ECHO_REQUEST: u8 = 128;
const ICMPV6_ECHO_REPLY: u8 = 129;
const ICMPV6_NEIGHBOR_SOLICIT: u8 = 135;
const ICMPV6_NEIGHBOR_ADVERT: u8 = 136;

#[repr(C, packed)]
struct Icmpv6Echo {
    icmptype: u8,
    code: u8,
    checksum: u16,
    id: u16,
    seq: u16,
}

pub fn send_icmpv6_echo_request(dst: [u8; 16], seq: u16) {
    let src = crate::net::local_ip_v6();
    let mut pkt = Icmpv6Echo {
        icmptype: ICMPV6_ECHO_REQUEST,
        code: 0,
        checksum: 0,
        id: 0x1234_u16.to_be(),
        seq: seq.to_be(),
    };
    let mut pseudo = Vec::with_capacity(40 + 8);
    pseudo.extend_from_slice(&src);
    pseudo.extend_from_slice(&dst);
    pseudo.extend_from_slice(&8u32.to_be_bytes());
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(58);
    let bytes = unsafe { core::slice::from_raw_parts(&pkt as *const _ as *const u8, 8) };
    pseudo.extend_from_slice(bytes);
    pkt.checksum = crate::net::ipv4::internet_checksum(&pseudo);
    let bytes = unsafe { core::slice::from_raw_parts(&pkt as *const _ as *const u8, 8) };
    send_ipv6_packet(dst, 58, bytes);
}

fn send_icmpv6_echo_reply(dst: [u8; 16], id: u16, seq: u16, data: &[u8]) {
    let src = crate::net::local_ip_v6();
    let mut buf = Vec::with_capacity(8 + data.len());
    buf.push(ICMPV6_ECHO_REPLY);
    buf.push(0);
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&id.to_be_bytes());
    buf.extend_from_slice(&seq.to_be_bytes());
    buf.extend_from_slice(data);
    let mut pseudo = Vec::with_capacity(40 + buf.len());
    pseudo.extend_from_slice(&src);
    pseudo.extend_from_slice(&dst);
    pseudo.extend_from_slice(&((buf.len()) as u32).to_be_bytes());
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(58);
    pseudo.extend_from_slice(&buf);
    let checksum = crate::net::ipv4::internet_checksum(&pseudo);
    buf[2] = (checksum >> 8) as u8;
    buf[3] = (checksum & 0xFF) as u8;
    send_ipv6_packet(dst, 58, &buf);
}

fn send_neighbor_advertisement(target: [u8; 16], dst: [u8; 16]) {
    let src = crate::net::local_ip_v6();
    let mut buf = Vec::with_capacity(32);
    buf.push(ICMPV6_NEIGHBOR_ADVERT);
    buf.push(0);
    buf.push(0);
    buf.push(0); // checksum placeholder
                 // Flags: Solicited + Override
    buf.extend_from_slice(&0x6000_0000u32.to_be_bytes());
    buf.extend_from_slice(&target);
    // Source link-layer address option
    buf.push(2); // type
    buf.push(1); // length (1 * 8 bytes)
    let mac = crate::net::local_mac();
    buf.extend_from_slice(&mac);
    while buf.len() % 8 != 0 {
        buf.push(0);
    }

    let mut pseudo = Vec::with_capacity(40 + buf.len());
    pseudo.extend_from_slice(&src);
    pseudo.extend_from_slice(&dst);
    pseudo.extend_from_slice(&((buf.len()) as u32).to_be_bytes());
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(58);
    pseudo.extend_from_slice(&buf);
    let checksum = crate::net::ipv4::internet_checksum(&pseudo);
    buf[2] = (checksum >> 8) as u8;
    buf[3] = (checksum & 0xFF) as u8;
    send_ipv6_packet(dst, 58, &buf);
}

struct NdCacheEntry {
    mac: [u8; 6],
    expires_at: u64,
}

static ND_CACHE: Mutex<Vec<([u8; 16], NdCacheEntry)>> = Mutex::new(Vec::new());

fn ticks() -> u64 {
    crate::drivers::interrupts::ticks()
}

fn is_expired(e: &NdCacheEntry) -> bool {
    ticks().wrapping_sub(e.expires_at) < 0x8000_0000_0000_0000 && ticks() >= e.expires_at
}

pub fn ndp_lookup(ip: [u8; 16]) -> Option<[u8; 6]> {
    let cache = ND_CACHE.lock();
    for (cached_ip, entry) in cache.iter() {
        if *cached_ip == ip && !is_expired(entry) {
            return Some(entry.mac);
        }
    }
    drop(cache);
    let _ = send_neighbor_solicitation(ip);
    None
}

pub fn insert_ndp(ip: [u8; 16], mac: [u8; 6]) {
    let mut cache = ND_CACHE.lock();
    for (cached_ip, entry) in cache.iter_mut() {
        if *cached_ip == ip {
            entry.mac = mac;
            entry.expires_at = ticks().wrapping_add(300_000);
            return;
        }
    }
    cache.push((
        ip,
        NdCacheEntry {
            mac,
            expires_at: ticks().wrapping_add(300_000),
        },
    ));
}

fn send_neighbor_solicitation(target: [u8; 16]) -> Result<(), &'static str> {
    if !crate::net::has_nic() {
        return Err("no NIC");
    }
    let src = crate::net::local_ip_v6();
    let src_mac = crate::net::local_mac();
    // Solicited-node multicast address
    let dst = [
        0xFF, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0xFF, target[13],
        target[14], target[15],
    ];
    let dst_mac = [0x33, 0x33, dst[12], dst[13], dst[14], dst[15]];

    let mut buf = Vec::with_capacity(32);
    buf.push(ICMPV6_NEIGHBOR_SOLICIT);
    buf.push(0);
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&0u32.to_be_bytes()); // reserved
    buf.extend_from_slice(&target);
    // Source link-layer address option
    buf.push(1); // type
    buf.push(1); // length
    buf.extend_from_slice(&src_mac);
    while buf.len() % 8 != 0 {
        buf.push(0);
    }

    let mut pseudo = Vec::with_capacity(40 + buf.len());
    pseudo.extend_from_slice(&src);
    pseudo.extend_from_slice(&dst);
    pseudo.extend_from_slice(&((buf.len()) as u32).to_be_bytes());
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(0);
    pseudo.push(58);
    pseudo.extend_from_slice(&buf);
    let checksum = crate::net::ipv4::internet_checksum(&pseudo);
    buf[2] = (checksum >> 8) as u8;
    buf[3] = (checksum & 0xFF) as u8;

    let mut frame = Vec::with_capacity(14 + 40 + buf.len());
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&0x86DD_u16.to_be_bytes());
    let hdr = Ipv6Header::new(58, src, dst, buf.len());
    let hdr_bytes = unsafe { core::slice::from_raw_parts(&hdr as *const _ as *const u8, 40) };
    frame.extend_from_slice(hdr_bytes);
    frame.extend_from_slice(&buf);
    crate::net::transmit(&frame);
    Ok(())
}

pub fn handle_icmpv6_packet(src: [u8; 16], pkt: &[u8]) {
    if pkt.len() < 8 {
        return;
    }
    let icmptype = pkt[0];
    let code = pkt[1];
    if code != 0 {
        return;
    }
    match icmptype {
        ICMPV6_ECHO_REQUEST => {
            let id = u16::from_be_bytes([pkt[4], pkt[5]]);
            let seq = u16::from_be_bytes([pkt[6], pkt[7]]);
            send_icmpv6_echo_reply(src, id, seq, &pkt[8..]);
        }
        ICMPV6_ECHO_REPLY => {
            crate::serial_println!("[ICMPv6] Echo reply from {:02x}{:02x}:...", src[0], src[1]);
        }
        ICMPV6_NEIGHBOR_SOLICIT => {
            if pkt.len() < 24 {
                return;
            }
            let target = [
                pkt[8], pkt[9], pkt[10], pkt[11], pkt[12], pkt[13], pkt[14], pkt[15], pkt[16],
                pkt[17], pkt[18], pkt[19], pkt[20], pkt[21], pkt[22], pkt[23],
            ];
            let local = crate::net::local_ip_v6();
            if target == local {
                send_neighbor_advertisement(target, src);
            }
        }
        ICMPV6_NEIGHBOR_ADVERT => {
            if pkt.len() < 24 {
                return;
            }
            let target = [
                pkt[8], pkt[9], pkt[10], pkt[11], pkt[12], pkt[13], pkt[14], pkt[15], pkt[16],
                pkt[17], pkt[18], pkt[19], pkt[20], pkt[21], pkt[22], pkt[23],
            ];
            let mut mac = [0u8; 6];
            if pkt.len() >= 32 && pkt[24] == 2 {
                mac.copy_from_slice(&pkt[26..32]);
            }
            insert_ndp(target, mac);
        }
        _ => {
            // Unknown ICMPv6 type; drop silently
        }
    }
}
