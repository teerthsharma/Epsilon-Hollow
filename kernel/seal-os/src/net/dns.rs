// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Simple DNS resolver over UDP.

use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

struct DnsCacheEntry {
    ip: [u8; 4],
    expires_at: u64,
}

static DNS_SERVER: Mutex<[u8; 4]> = Mutex::new([10, 0, 2, 3]);
static CACHE: Mutex<Vec<(String, DnsCacheEntry)>> = Mutex::new(Vec::new());
static DNS_SOCKET: Mutex<Option<usize>> = Mutex::new(None);
static QUERY_ID: Mutex<u16> = Mutex::new(1);

pub fn init() {}

pub fn set_server(ip: [u8; 4]) {
    *DNS_SERVER.lock() = ip;
}

fn ticks() -> u64 {
    crate::drivers::interrupts::ticks()
}

pub fn query(name: &str) -> Option<[u8; 4]> {
    let cache = CACHE.lock();
    for (cached_name, entry) in cache.iter() {
        if cached_name == name {
            if ticks().wrapping_sub(entry.expires_at) < 0x8000_0000_0000_0000 && ticks() >= entry.expires_at {
                continue;
            }
            return Some(entry.ip);
        }
    }
    drop(cache);

    let mut qid = QUERY_ID.lock();
    let id = *qid;
    *qid = id.wrapping_add(1);
    drop(qid);

    let mut pkt = Vec::new();
    pkt.extend_from_slice(&id.to_be_bytes());
    pkt.extend_from_slice(&[0x01, 0x00]); // flags: standard query
    pkt.extend_from_slice(&[0x00, 0x01]); // questions: 1
    pkt.extend_from_slice(&[0x00, 0x00]); // answer RRs
    pkt.extend_from_slice(&[0x00, 0x00]); // authority RRs
    pkt.extend_from_slice(&[0x00, 0x00]); // additional RRs

    for label in name.split('.') {
        pkt.push(label.len() as u8);
        pkt.extend_from_slice(label.as_bytes());
    }
    pkt.push(0); // end of name
    pkt.extend_from_slice(&[0x00, 0x01]); // type A
    pkt.extend_from_slice(&[0x00, 0x01]); // class IN

    let server = *DNS_SERVER.lock();
    let sock = crate::net::udp::socket();
    crate::net::udp::sendto(sock, &pkt, server, 53);
    None
}

pub fn handle_dns_response(pkt: &[u8]) {
    if pkt.len() < 12 {
        return;
    }
    let flags = u16::from_be_bytes([pkt[2], pkt[3]]);
    if flags & 0x8000 == 0 {
        return; // not a response
    }
    let qdcount = u16::from_be_bytes([pkt[4], pkt[5]]);
    let ancount = u16::from_be_bytes([pkt[6], pkt[7]]);

    let mut offset = 12;
    // Skip questions
    for _ in 0..qdcount {
        while offset < pkt.len() && pkt[offset] != 0 {
            if pkt[offset] & 0xC0 == 0xC0 {
                offset += 2;
                break;
            }
            offset += 1 + pkt[offset] as usize;
        }
        if offset < pkt.len() && pkt[offset] == 0 {
            offset += 1;
        }
        offset += 4; // type + class
    }

    // Parse answers
    for _ in 0..ancount {
        if offset + 10 > pkt.len() {
            break;
        }
        // Skip name
        while offset < pkt.len() && pkt[offset] != 0 {
            if pkt[offset] & 0xC0 == 0xC0 {
                offset += 2;
                break;
            }
            offset += 1 + pkt[offset] as usize;
        }
        if offset < pkt.len() && pkt[offset] == 0 {
            offset += 1;
        }
        if offset + 10 > pkt.len() {
            break;
        }
        let rtype = u16::from_be_bytes([pkt[offset], pkt[offset + 1]]);
        offset += 4; // type + class
        let ttl = u32::from_be_bytes([pkt[offset], pkt[offset + 1], pkt[offset + 2], pkt[offset + 3]]);
        offset += 4;
        let rdlen = u16::from_be_bytes([pkt[offset], pkt[offset + 1]]) as usize;
        offset += 2;
        if rtype == 1 && rdlen == 4 && offset + 4 <= pkt.len() {
            let ip = [pkt[offset], pkt[offset + 1], pkt[offset + 2], pkt[offset + 3]];
            let mut cache = CACHE.lock();
            if cache.len() >= 16 {
                cache.remove(0);
            }
            cache.push((
                String::from("resolved"),
                DnsCacheEntry {
                    ip,
                    expires_at: ticks().wrapping_add(ttl as u64 * 100),
                },
            ));
            return;
        }
        offset += rdlen;
    }
}
