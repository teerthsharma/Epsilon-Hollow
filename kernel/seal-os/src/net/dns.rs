// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Simple DNS resolver over UDP with BTreeMap TTL cache.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

struct DnsCacheEntry {
    ip: [u8; 4],
    expires_at: u64,
}

static DNS_SERVER: Mutex<[u8; 4]> = Mutex::new([10, 0, 2, 3]);
static CACHE: Mutex<BTreeMap<String, DnsCacheEntry>> = Mutex::new(BTreeMap::new());
#[allow(dead_code)] // REASON: UDP socket handle reserved for future async DNS resolver
static DNS_SOCKET: Mutex<Option<usize>> = Mutex::new(None);
static QUERY_ID: Mutex<u16> = Mutex::new(1);

pub fn init() {}

pub fn set_server(ip: [u8; 4]) {
    *DNS_SERVER.lock() = ip;
}

pub fn server() -> [u8; 4] {
    *DNS_SERVER.lock()
}

fn ticks() -> u64 {
    crate::drivers::interrupts::ticks()
}

static PENDING_QUERY: Mutex<Option<String>> = Mutex::new(None);

// ---------------------------------------------------------------------------
// NEW: Real DNS header and query structs with honest error handling.
// ---------------------------------------------------------------------------

/// DNS packet header (12 bytes, RFC 1035).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DnsHeader {
    pub id: u16,
    pub flags: u16,
    pub qdcount: u16,
    pub ancount: u16,
    pub nscount: u16,
    pub arcount: u16,
}

impl DnsHeader {
    /// Parse a DNS header from exactly 12 bytes.
    /// Returns `None` if the slice is too short.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }
        Some(Self {
            id: u16::from_be_bytes([bytes[0], bytes[1]]),
            flags: u16::from_be_bytes([bytes[2], bytes[3]]),
            qdcount: u16::from_be_bytes([bytes[4], bytes[5]]),
            ancount: u16::from_be_bytes([bytes[6], bytes[7]]),
            nscount: u16::from_be_bytes([bytes[8], bytes[9]]),
            arcount: u16::from_be_bytes([bytes[10], bytes[11]]),
        })
    }

    /// Serialize the header to a fixed 12-byte array.
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut b = [0u8; 12];
        b[0..2].copy_from_slice(&self.id.to_be_bytes());
        b[2..4].copy_from_slice(&self.flags.to_be_bytes());
        b[4..6].copy_from_slice(&self.qdcount.to_be_bytes());
        b[6..8].copy_from_slice(&self.ancount.to_be_bytes());
        b[8..10].copy_from_slice(&self.nscount.to_be_bytes());
        b[10..12].copy_from_slice(&self.arcount.to_be_bytes());
        b
    }

    /// True if the QR bit indicates a response.
    pub fn is_response(&self) -> bool {
        self.flags & 0x8000 != 0
    }
}

/// Represents a single DNS question.
#[derive(Debug, Clone, PartialEq)]
pub struct DnsQuery {
    pub name: String,
    pub qtype: u16,
    pub qclass: u16,
}

/// Honest DNS resolver.
/// Returns `Err("no network")` if no NIC is present.
/// Otherwise sends a query and returns `None` until a response arrives.
pub fn resolve(name: &str) -> Result<Option<[u8; 4]>, &'static str> {
    if !crate::net::has_nic() {
        return Err("no network");
    }

    let cache = CACHE.lock();
    if let Some(entry) = cache.get(name) {
        if ticks() < entry.expires_at {
            return Ok(Some(entry.ip));
        }
    }
    drop(cache);

    *PENDING_QUERY.lock() = Some(String::from(name));

    let mut qid = QUERY_ID.lock();
    let id = *qid;
    *qid = id.wrapping_add(1);
    drop(qid);

    let hdr = DnsHeader {
        id,
        flags: 0x0100, // standard query
        qdcount: 1,
        ancount: 0,
        nscount: 0,
        arcount: 0,
    };

    let mut pkt = Vec::new();
    pkt.extend_from_slice(&hdr.to_bytes());

    for label in name.split('.') {
        pkt.push(label.len() as u8);
        pkt.extend_from_slice(label.as_bytes());
    }
    pkt.push(0); // end of name
    pkt.extend_from_slice(&[0x00, 0x01]); // type A
    pkt.extend_from_slice(&[0x00, 0x01]); // class IN

    let server = *DNS_SERVER.lock();
    let sock = crate::net::udp::socket();
    crate::net::udp::sendto(sock, &pkt, crate::net::IpAddr::V4(server), 53);
    Ok(None)
}

// ---------------------------------------------------------------------------
// Legacy query function (preserved for compatibility).
// ---------------------------------------------------------------------------

pub fn query(name: &str) -> Option<[u8; 4]> {
    let cache = CACHE.lock();
    if let Some(entry) = cache.get(name) {
        if ticks() < entry.expires_at {
            return Some(entry.ip);
        }
    }
    drop(cache);

    *PENDING_QUERY.lock() = Some(String::from(name));

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
    crate::net::udp::sendto(sock, &pkt, crate::net::IpAddr::V4(server), 53);
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
        let ttl = u32::from_be_bytes([
            pkt[offset],
            pkt[offset + 1],
            pkt[offset + 2],
            pkt[offset + 3],
        ]);
        offset += 4;
        let rdlen = u16::from_be_bytes([pkt[offset], pkt[offset + 1]]) as usize;
        offset += 2;
        if rtype == 1 && rdlen == 4 && offset + 4 <= pkt.len() {
            let ip = [
                pkt[offset],
                pkt[offset + 1],
                pkt[offset + 2],
                pkt[offset + 3],
            ];
            let hostname = PENDING_QUERY
                .lock()
                .take()
                .unwrap_or_else(|| String::from("unknown"));
            let mut cache = CACHE.lock();
            if cache.len() >= 16 {
                let first = cache.keys().next().cloned();
                if let Some(k) = first {
                    cache.remove(&k);
                }
            }
            cache.insert(
                hostname,
                DnsCacheEntry {
                    ip,
                    expires_at: ticks().wrapping_add(ttl as u64 * 100),
                },
            );
            return;
        }
        offset += rdlen;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_dns_response(hostname: &str, ip: [u8; 4], ttl: u32) -> Vec<u8> {
        let mut pkt = Vec::new();
        pkt.extend_from_slice(&0x1234u16.to_be_bytes()); // id
        pkt.extend_from_slice(&0x8180u16.to_be_bytes()); // flags: response
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // qdcount
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // ancount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // nscount
        pkt.extend_from_slice(&0x0000u16.to_be_bytes()); // arcount
                                                         // question
        for label in hostname.split('.') {
            pkt.push(label.len() as u8);
            pkt.extend_from_slice(label.as_bytes());
        }
        pkt.push(0);
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // type A
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // class IN
                                                         // answer
        pkt.push(0xC0);
        pkt.push(0x0C); // pointer to question name
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // type A
        pkt.extend_from_slice(&0x0001u16.to_be_bytes()); // class IN
        pkt.extend_from_slice(&ttl.to_be_bytes());
        pkt.extend_from_slice(&0x0004u16.to_be_bytes()); // rdlen
        pkt.extend_from_slice(&ip);
        pkt
    }

    #[test]
    fn test_dns_parse_a_record() {
        let resp = build_dns_response("example.com", [93, 184, 216, 34], 300);
        *PENDING_QUERY.lock() = Some(String::from("example.com"));
        handle_dns_response(&resp);
        let cache = CACHE.lock();
        assert_eq!(
            cache.get("example.com").map(|e| e.ip),
            Some([93, 184, 216, 34])
        );
    }

    #[test]
    fn test_dns_cache_expiration() {
        CACHE.lock().insert(
            String::from("old.com"),
            DnsCacheEntry {
                ip: [1, 2, 3, 4],
                expires_at: 0, // expired
            },
        );
        let result = query("old.com");
        assert!(result.is_none());
    }

    #[test]
    fn test_dns_cache_hit() {
        CACHE.lock().insert(
            String::from("fresh.com"),
            DnsCacheEntry {
                ip: [5, 6, 7, 8],
                expires_at: u64::MAX,
            },
        );
        let result = query("fresh.com");
        assert_eq!(result, Some([5, 6, 7, 8]));
    }

    #[test]
    fn test_dns_header_roundtrip() {
        let hdr = DnsHeader {
            id: 0x1234,
            flags: 0x8180,
            qdcount: 1,
            ancount: 1,
            nscount: 0,
            arcount: 0,
        };
        let bytes = hdr.to_bytes();
        let parsed = DnsHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, hdr);
        assert!(parsed.is_response());
    }

    #[test]
    fn test_dns_resolve_no_network() {
        // In unit tests there is no real NIC, so resolve must fail honestly.
        let result = resolve("example.com");
        assert_eq!(result, Err("no network"));
    }
}
