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

/// Builds a complete ARP frame: 14-byte Ethernet header followed by the
/// 28-byte ARP packet.
///
/// The Ethernet header carries the destination address at bytes 0..6, the
/// source address at 6..12 and the ethertype at 12..14, so the destination
/// must be written first. `target_mac` is `None` when the peer's hardware
/// address is not yet known (a request): that sends the frame to the
/// broadcast address and zeroes the ARP target hardware address, per RFC 826.
fn build_arp_frame(oper: u16, target_mac: Option<[u8; 6]>, target_ip: [u8; 4]) -> Vec<u8> {
    let src_mac = crate::net::local_mac();
    let src_ip = crate::net::local_ip();

    let pkt = ArpPacket {
        htype: ARP_HTYPE_ETH.to_be(),
        ptype: ARP_PTYPE_IP.to_be(),
        hlen: 6,
        plen: 4,
        oper: oper.to_be(),
        sha: src_mac,
        spa: src_ip,
        tha: target_mac.unwrap_or([0; 6]),
        tpa: target_ip,
    };

    let mut frame = Vec::with_capacity(14 + core::mem::size_of::<ArpPacket>());
    frame.extend_from_slice(&target_mac.unwrap_or([0xFF; 6]));
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&0x0806_u16.to_be_bytes());
    let pkt_bytes = unsafe {
        core::slice::from_raw_parts(
            &pkt as *const _ as *const u8,
            core::mem::size_of::<ArpPacket>(),
        )
    };
    frame.extend_from_slice(pkt_bytes);
    frame
}

fn send_arp_request(target_ip: [u8; 4]) -> Result<(), &'static str> {
    if !crate::net::has_nic() {
        return Err("no NIC");
    }
    crate::drivers::net::transmit(&build_arp_frame(ARP_OP_REQUEST, None, target_ip));
    Ok(())
}

fn send_arp_reply(target_mac: [u8; 6], target_ip: [u8; 4]) -> Result<(), &'static str> {
    if !crate::net::has_nic() {
        return Err("no NIC");
    }
    crate::drivers::net::transmit(&build_arp_frame(ARP_OP_REPLY, Some(target_mac), target_ip));
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

// ---------------------------------------------------------------------------
// In-kernel tests. `#[cfg(test)]` never runs here (seal-os is excluded from the
// workspace), so these register with the QEMU harness instead:
// `testing/runner.rs` must call `crate::net::arp::tests::register_all()`.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// A request must go to the broadcast address with this machine's MAC as
    /// the source. The inverted order that shipped before put the local MAC in
    /// the destination field and the broadcast address in the source field,
    /// which no peer answers and which switches drop.
    fn test_request_frame_header_order() -> TestResult {
        let frame = build_arp_frame(ARP_OP_REQUEST, None, [10, 0, 2, 2]);
        test_assert!(
            frame.len() == 14 + core::mem::size_of::<ArpPacket>(),
            "ARP request frame is not 14 bytes of Ethernet header plus a 28-byte ARP packet"
        );
        test_assert!(
            frame[0..6] == [0xFF; 6],
            "ARP request Ethernet destination (bytes 0..6) is not the broadcast address"
        );
        test_assert!(
            frame[6..12] == crate::net::local_mac(),
            "ARP request Ethernet source (bytes 6..12) is not this machine's MAC"
        );
        test_assert!(
            frame[12..14] == [0x08, 0x06],
            "ARP request ethertype (bytes 12..14) is not 0x0806"
        );
        // ARP payload: sha at 14+8, tha at 14+18.
        test_assert!(
            frame[22..28] == crate::net::local_mac(),
            "ARP request sender hardware address is not this machine's MAC"
        );
        test_assert!(
            frame[32..38] == [0u8; 6],
            "ARP request target hardware address is not zeroed"
        );
        TestResult::Pass
    }

    /// A reply is unicast to the peer that asked, sourced from this machine.
    fn test_reply_frame_header_order() -> TestResult {
        let peer = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x01];
        let frame = build_arp_frame(ARP_OP_REPLY, Some(peer), [10, 0, 2, 2]);
        test_assert!(
            frame[0..6] == peer,
            "ARP reply Ethernet destination (bytes 0..6) is not the peer's MAC"
        );
        test_assert!(
            frame[6..12] == crate::net::local_mac(),
            "ARP reply Ethernet source (bytes 6..12) is not this machine's MAC"
        );
        test_assert!(
            frame[22..28] == crate::net::local_mac() && frame[32..38] == peer,
            "ARP reply payload hardware addresses are swapped"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "arp::request_frame_header_order",
            test_request_frame_header_order,
        );
        crate::testing::register_test(
            "arp::reply_frame_header_order",
            test_reply_frame_header_order,
        );
    }
}
