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
        // `internet_checksum` returns host order; every other field of this
        // header is stored network order because `send_ipv4_packet` blits the
        // struct straight onto the wire. Without `to_be` the checksum ships
        // byte-swapped and `handle_ipv4_packet` below rejects our own packets.
        self.checksum = internet_checksum(bytes).to_be();
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

/// RFC 1071 internet checksum over `data`, **returned in host byte order**.
///
/// The return value is a number, not wire bytes. How it must be stored depends
/// entirely on how the containing header reaches the wire, and this tree uses
/// both conventions:
///
/// * A header that is **blitted raw** -- `repr(C, packed)` reinterpreted as
///   bytes by `from_raw_parts`, as in `Ipv4Header`, `icmp::IcmpEcho`,
///   `udp::UdpHeader`, `ipv6::Icmpv6Echo` and `drivers::net::tcp`'s
///   `TcpHeader` -- holds every multi-byte field pre-swapped, so the checksum
///   is stored with `.to_be()` exactly like its `total_len` / `id` / `seq` /
///   `src_port` siblings.
/// * A header **serialized field by field** -- `net::tcp::TcpHeader::to_bytes`,
///   or the byte buffers built by `icmp::send_echo_reply` and the three
///   ICMPv6 senders in `ipv6` -- holds host order throughout and converts on
///   the way out, so the value is stored as returned.
///
/// Storing a raw-blitted field without `.to_be()` ships the checksum
/// byte-swapped, and `Ipv4Header::from_bytes` / `handle_ipv4_packet` then
/// reject the packet: the stack refuses its own traffic, loopback included.
///
/// Verification is direction-free -- recompute over the whole header and
/// compare to zero -- because a byte-swapped zero is still zero.
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

    // `total_len` is attacker-controlled and needs a floor as well as a ceiling:
    // it counts the header too, so anything below `ihl` would invert the payload
    // slice below and panic. A remote peer computes a valid header checksum for
    // free, so the checksum above is no barrier. Same shape as
    // `ipv6::handle_ipv6_packet`, which bounds its length before slicing.
    let total_len = u16::from_be(hdr.total_len) as usize;
    if total_len < ihl || total_len > pkt.len() {
        return;
    }

    let payload = &pkt[ihl..total_len];
    let protocol = hdr.protocol;
    let src = hdr.src;

    match protocol {
        1 => crate::net::icmp::handle_icmp_packet(src, payload),
        6 => crate::net::tcp::handle_tcp_packet(crate::net::IpAddr::V4(src), payload),
        17 => crate::net::udp::handle_udp_packet(crate::net::IpAddr::V4(src), payload),
        _ => {
            // Unknown IPv4 protocol; drop silently
        }
    }
}

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// `kernel/seal-os` is excluded from the workspace, so `cargo test --workspace`
// never builds this crate; these register into crate::testing::TEST_REGISTRY
// and execute under QEMU via testing::runner::test_main(). See WIRING note on
// `register_all` below -- runner.rs does not call it yet.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// Build an IPv4 frame with an attacker-chosen `ver_ihl` and `total_len`
    /// and a header checksum that verifies. A remote peer computes this
    /// checksum for free, so it gates nothing; the builder makes that explicit
    /// rather than letting a test pass because the checksum guard rejected the
    /// packet before it ever reached the code under test.
    fn forged(ver_ihl: u8, total_len: u16, frame_len: usize) -> Vec<u8> {
        let mut pkt = vec![0u8; frame_len];
        pkt[0] = ver_ihl;
        pkt[2..4].copy_from_slice(&total_len.to_be_bytes());
        pkt[8] = 64; // ttl
        pkt[9] = 17; // UDP
        let ihl = (ver_ihl & 0x0F) as usize * 4;
        let ck = internet_checksum(&pkt[..ihl]);
        pkt[10..12].copy_from_slice(&ck.to_be_bytes());
        pkt
    }

    /// `total_len` of 0 clears `total_len > pkt.len()` and used to produce
    /// `&pkt[20..0]` -- a reversed slice range, which panics. With
    /// `panic = "abort"` that is a remote kernel kill from one 60-byte frame,
    /// the Ethernet minimum, reachable through both the Allow and Log firewall
    /// verdicts in `net::mod::process_packet`. Reaching the assertion at all is
    /// part of the proof.
    fn test_total_len_zero_dropped() -> TestResult {
        let pkt = forged(0x45, 0, 60);
        test_assert!(
            Ipv4Header::from_bytes(&pkt).is_some(),
            "forged header does not pass version/IHL/checksum, so it never reaches the payload slice and proves nothing"
        );
        handle_ipv4_packet(&pkt);
        TestResult::Pass
    }

    /// `total_len` counts the header, so any value below the header length
    /// inverts the slice the same way. IHL 15 gives a 60-byte header against a
    /// declared total of 20: `&pkt[60..20]`.
    fn test_total_len_below_ihl_dropped() -> TestResult {
        let pkt = forged(0x4F, 20, 60);
        test_assert!(
            Ipv4Header::from_bytes(&pkt).is_some(),
            "forged header does not pass version/IHL/checksum, so it never reaches the payload slice and proves nothing"
        );
        handle_ipv4_packet(&pkt);
        // One below the header length, the tightest failing case.
        let pkt = forged(0x45, 19, 60);
        handle_ipv4_packet(&pkt);
        TestResult::Pass
    }

    /// Control on the accepting side of the new boundary, so the fix cannot
    /// have passed by turning every packet into a drop. `total_len == ihl` is a
    /// legal header-only datagram with an empty payload, and `total_len ==
    /// pkt.len()` is the ordinary case; both must still slice and dispatch.
    /// Both carry an all-zero UDP payload, which `handle_udp_packet` rejects on
    /// its own length field, so nothing downstream is disturbed.
    fn test_total_len_at_and_above_ihl_accepted() -> TestResult {
        let header_only = forged(0x45, 20, 60);
        test_assert!(
            internet_checksum(&header_only[..20]) == 0,
            "control packet checksum does not verify"
        );
        handle_ipv4_packet(&header_only);
        let full = forged(0x45, 60, 60);
        handle_ipv4_packet(&full);
        TestResult::Pass
    }

    /// The upper bound the fix must not have disturbed: a `total_len` longer
    /// than the frame is still a drop.
    fn test_total_len_above_frame_dropped() -> TestResult {
        let pkt = forged(0x45, 1500, 60);
        test_assert!(
            Ipv4Header::from_bytes(&pkt).is_some(),
            "forged header does not pass version/IHL/checksum, so it never reaches the payload slice and proves nothing"
        );
        handle_ipv4_packet(&pkt);
        TestResult::Pass
    }

    // WIRING: testing/runner.rs must call `crate::net::ipv4::tests::register_all()`
    // for this group to execute; it is not registered there yet.
    pub fn register_all() {
        crate::testing::register_test("ipv4::total_len_zero_dropped", test_total_len_zero_dropped);
        crate::testing::register_test(
            "ipv4::total_len_below_ihl_dropped",
            test_total_len_below_ihl_dropped,
        );
        crate::testing::register_test(
            "ipv4::total_len_at_and_above_ihl_accepted",
            test_total_len_at_and_above_ihl_accepted,
        );
        crate::testing::register_test(
            "ipv4::total_len_above_frame_dropped",
            test_total_len_above_frame_dropped,
        );
    }
}
