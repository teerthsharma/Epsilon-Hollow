// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ARP protocol -- cache, requests, replies.

use alloc::vec::Vec;
use spin::Mutex;

const ARP_HTYPE_ETH: u16 = 1;
const ARP_PTYPE_IP: u16 = 0x0800;
const ARP_OP_REQUEST: u16 = 1;
const ARP_OP_REPLY: u16 = 2;

#[repr(C, packed)]
struct ArpPacket {
    htype: u16,
    ptype: u16,
    hlen: u8,
    plen: u8,
    oper: u16,
    sha: [u8; 6],
    spa: [u8; 4],
    tha: [u8; 6],
    tpa: [u8; 4],
}

struct ArpEntry {
    mac: [u8; 6],
    expires_at: u64,
}

static ARP_CACHE: Mutex<Vec<([u8; 4], ArpEntry)>> = Mutex::new(Vec::new());

pub fn init() {}

fn ticks() -> u64 {
    crate::drivers::interrupts::ticks()
}

fn is_expired(e: &ArpEntry) -> bool {
    ticks().wrapping_sub(e.expires_at) < 0x8000_0000_0000_0000 && ticks() >= e.expires_at
}

pub fn lookup(ip: [u8; 4]) -> Option<[u8; 6]> {
    let cache = ARP_CACHE.lock();
    for (cached_ip, entry) in cache.iter() {
        if *cached_ip == ip && !is_expired(entry) {
            return Some(entry.mac);
        }
    }
    drop(cache);
    let _ = send_arp_request(ip);
    None
}

pub fn insert(ip: [u8; 4], mac: [u8; 6]) {
    let mut cache = ARP_CACHE.lock();
    for (cached_ip, entry) in cache.iter_mut() {
        if *cached_ip == ip {
            entry.mac = mac;
            entry.expires_at = ticks().wrapping_add(300_000); // ~5 min in ticks
            return;
        }
    }
    cache.push((
        ip,
        ArpEntry {
            mac,
            expires_at: ticks().wrapping_add(300_000),
        },
    ));
}

fn send_arp_request(target_ip: [u8; 4]) -> Result<(), &'static str> {
    if !crate::net::has_nic() {
        return Err("no NIC");
    }
    let src_mac = crate::net::local_mac();
    let src_ip = crate::net::local_ip();

    let pkt = ArpPacket {
        htype: ARP_HTYPE_ETH.to_be(),
        ptype: ARP_PTYPE_IP.to_be(),
        hlen: 6,
        plen: 4,
        oper: ARP_OP_REQUEST.to_be(),
        sha: src_mac,
        spa: src_ip,
        tha: [0; 6],
        tpa: target_ip,
    };

    let mut frame = Vec::with_capacity(14 + core::mem::size_of::<ArpPacket>());
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&[0xFF; 6]);
    frame.extend_from_slice(&0x0806_u16.to_be_bytes());
    let pkt_bytes = unsafe {
        core::slice::from_raw_parts(
            &pkt as *const _ as *const u8,
            core::mem::size_of::<ArpPacket>(),
        )
    };
    frame.extend_from_slice(pkt_bytes);
    crate::drivers::net::transmit(&frame);
    Ok(())
}

fn send_arp_reply(target_mac: [u8; 6], target_ip: [u8; 4]) -> Result<(), &'static str> {
    if !crate::net::has_nic() {
        return Err("no NIC");
    }
    let src_mac = crate::net::local_mac();
    let src_ip = crate::net::local_ip();

    let pkt = ArpPacket {
        htype: ARP_HTYPE_ETH.to_be(),
        ptype: ARP_PTYPE_IP.to_be(),
        hlen: 6,
        plen: 4,
        oper: ARP_OP_REPLY.to_be(),
        sha: src_mac,
        spa: src_ip,
        tha: target_mac,
        tpa: target_ip,
    };

    let mut frame = Vec::with_capacity(14 + core::mem::size_of::<ArpPacket>());
    frame.extend_from_slice(&target_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&0x0806_u16.to_be_bytes());
    let pkt_bytes = unsafe {
        core::slice::from_raw_parts(
            &pkt as *const _ as *const u8,
            core::mem::size_of::<ArpPacket>(),
        )
    };
    frame.extend_from_slice(pkt_bytes);
    crate::drivers::net::transmit(&frame);
    Ok(())
}

pub fn handle_arp_packet(pkt: &[u8]) {
    if pkt.len() < core::mem::size_of::<ArpPacket>() {
        return;
    }
    let arp = unsafe { &*(pkt.as_ptr() as *const ArpPacket) };
    let htype = u16::from_be(arp.htype);
    let ptype = u16::from_be(arp.ptype);
    let oper = u16::from_be(arp.oper);

    if htype != ARP_HTYPE_ETH || ptype != ARP_PTYPE_IP {
        return;
    }

    insert(arp.spa, arp.sha);

    let local_ip = crate::net::local_ip();
    match oper {
        ARP_OP_REQUEST => {
            if arp.tpa == local_ip {
                let _ = send_arp_reply(arp.sha, arp.spa);
            }
        }
        ARP_OP_REPLY => {
            // Reply processed above by updating cache; no further action
        }
        _ => {
            // Unknown ARP operation; drop
        }
    }
}
