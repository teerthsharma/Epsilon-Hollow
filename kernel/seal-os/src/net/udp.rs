// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! UDP sockets -- datagram demux and buffering.

use alloc::vec::Vec;
use spin::Mutex;

const UDP_HEADER_LEN: usize = 8;

#[repr(C, packed)]
struct UdpHeader {
    src_port: u16,
    dst_port: u16,
    length: u16,
    checksum: u16,
}

pub struct UdpSocket {
    local_port: u16,
    remote_addr: Option<crate::net::IpAddr>,
    remote_port: Option<u16>,
    rx_buffer: Vec<(crate::net::IpAddr, u16, Vec<u8>)>,
}

impl UdpSocket {
    pub fn new(local_port: u16) -> Self {
        Self {
            local_port,
            remote_addr: None,
            remote_port: None,
            rx_buffer: Vec::new(),
        }
    }

    pub fn bind(&mut self, port: u16) {
        self.local_port = port;
    }

    pub fn connect(&mut self, addr: crate::net::IpAddr, port: u16) {
        self.remote_addr = Some(addr);
        self.remote_port = Some(port);
    }

    pub fn sendto(&self, buf: &[u8], dst_addr: crate::net::IpAddr, dst_port: u16) {
        let src_port = self.local_port;
        let length = (UDP_HEADER_LEN + buf.len()) as u16;
        let mut hdr = UdpHeader {
            src_port: src_port.to_be(),
            dst_port: dst_port.to_be(),
            length: length.to_be(),
            checksum: 0,
        };
        match dst_addr {
            crate::net::IpAddr::V4(dst_addr) => {
                let src_ip = crate::net::local_ip();
                let mut pseudo = Vec::with_capacity(12 + UDP_HEADER_LEN + buf.len());
                pseudo.extend_from_slice(&src_ip);
                pseudo.extend_from_slice(&dst_addr);
                pseudo.push(0);
                pseudo.push(17);
                pseudo.extend_from_slice(&length.to_be_bytes());
                let hdr_bytes = unsafe {
                    core::slice::from_raw_parts(
                        &hdr as *const _ as *const u8,
                        core::mem::size_of::<UdpHeader>(),
                    )
                };
                pseudo.extend_from_slice(hdr_bytes);
                pseudo.extend_from_slice(buf);
                hdr.checksum = crate::net::ipv4::internet_checksum(&pseudo);
                let mut pkt = Vec::with_capacity(UDP_HEADER_LEN + buf.len());
                let hdr_bytes = unsafe {
                    core::slice::from_raw_parts(
                        &hdr as *const _ as *const u8,
                        core::mem::size_of::<UdpHeader>(),
                    )
                };
                pkt.extend_from_slice(hdr_bytes);
                pkt.extend_from_slice(buf);
                crate::net::ipv4::send_ipv4_packet(dst_addr, 17, &pkt);
            }
            crate::net::IpAddr::V6(dst_addr) => {
                let src_ip = crate::net::local_ip_v6();
                let mut pseudo = Vec::with_capacity(40 + UDP_HEADER_LEN + buf.len());
                pseudo.extend_from_slice(&src_ip);
                pseudo.extend_from_slice(&dst_addr);
                pseudo.extend_from_slice(&(length as u32).to_be_bytes());
                pseudo.push(0);
                pseudo.push(0);
                pseudo.push(0);
                pseudo.push(17);
                let hdr_bytes = unsafe {
                    core::slice::from_raw_parts(
                        &hdr as *const _ as *const u8,
                        core::mem::size_of::<UdpHeader>(),
                    )
                };
                pseudo.extend_from_slice(hdr_bytes);
                pseudo.extend_from_slice(buf);
                hdr.checksum = crate::net::ipv4::internet_checksum(&pseudo);
                let mut pkt = Vec::with_capacity(UDP_HEADER_LEN + buf.len());
                let hdr_bytes = unsafe {
                    core::slice::from_raw_parts(
                        &hdr as *const _ as *const u8,
                        core::mem::size_of::<UdpHeader>(),
                    )
                };
                pkt.extend_from_slice(hdr_bytes);
                pkt.extend_from_slice(buf);
                crate::net::ipv6::send_ipv6_packet(dst_addr, 17, &pkt);
            }
        }
    }

    pub fn send(&self, buf: &[u8]) {
        if let (Some(addr), Some(port)) = (self.remote_addr, self.remote_port) {
            self.sendto(buf, addr, port);
        }
    }

    pub fn recvfrom(&mut self, buf: &mut [u8]) -> Option<(usize, crate::net::IpAddr, u16)> {
        if let Some((addr, port, data)) = self.rx_buffer.pop() {
            let len = data.len().min(buf.len());
            buf[..len].copy_from_slice(&data[..len]);
            Some((len, addr, port))
        } else {
            None
        }
    }

    pub fn recv(&mut self, buf: &mut [u8]) -> Option<usize> {
        self.recvfrom(buf).map(|(len, _, _)| len)
    }

    pub fn port(&self) -> u16 {
        self.local_port
    }

    pub fn push_packet(&mut self, src: crate::net::IpAddr, src_port: u16, data: Vec<u8>) {
        if self.rx_buffer.len() > 64 {
            self.rx_buffer.remove(0);
        }
        self.rx_buffer.push((src, src_port, data));
    }
}

static UDP_SOCKETS: Mutex<Vec<UdpSocket>> = Mutex::new(Vec::new());
const EPHEMERAL_PORT_BASE: u16 = 49152; // RFC 6335 dynamic/private range start

pub fn init() {}

/// Pick a fresh local ephemeral port (RFC 6335 dynamic/private range,
/// 49152-65535). Every candidate is randomized -- hardware entropy
/// (`drivers::entropy::getrandom`) when available, else the shared
/// boot-seeded PRNG fallback (`drivers::entropy::fallback_random_u64`, also
/// used by `net::dns`'s transaction IDs) -- there is no bare-counter path at
/// all, at any point: the same induced-lookup prediction that made a
/// monotonic DNS transaction ID guessable applies just as well to a
/// monotonic port counter, since a spoofing target that also runs a server
/// the kernel talks to sees the source port of every packet it receives.
/// Retries on collision with an already-open socket for up to 8 draws, then
/// returns the last (still-randomized) draw regardless -- a same-port
/// collision between two sockets is tolerated by `handle_udp_packet`'s
/// dispatch (first match wins). `net::dns` allocates a fresh socket per
/// query (a shared-socket variant was tried and reverted -- see the
/// `ponytail:` comment on `net::dns`'s `DNS_SERVER`/`CACHE` statics), so
/// this table still grows without bound under sustained DNS traffic and
/// collisions here are a real, if much smaller, residual: with no bare
/// counter left in this function, a collision degrades to "two sockets
/// briefly share a port," not to predictability.
fn allocate_ephemeral_port(sockets: &[UdpSocket]) -> u16 {
    let mut candidate = EPHEMERAL_PORT_BASE;
    for _ in 0..8 {
        let mut buf = [0u8; 2];
        let raw = if crate::drivers::entropy::getrandom(&mut buf) {
            u16::from_ne_bytes(buf)
        } else {
            crate::drivers::entropy::fallback_random_u64() as u16
        };
        candidate = EPHEMERAL_PORT_BASE.wrapping_add(raw % (u16::MAX - EPHEMERAL_PORT_BASE));
        if !sockets.iter().any(|s| s.local_port == candidate) {
            return candidate;
        }
    }
    candidate
}

pub fn socket() -> usize {
    let mut sockets = UDP_SOCKETS.lock();
    let port = allocate_ephemeral_port(&sockets);
    let idx = sockets.len();
    sockets.push(UdpSocket::new(port));
    idx
}

pub fn bind(idx: usize, port: u16) {
    let mut sockets = UDP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.bind(port);
    }
}

pub fn connect(idx: usize, addr: crate::net::IpAddr, port: u16) {
    let mut sockets = UDP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.connect(addr, port);
    }
}

pub fn sendto(idx: usize, buf: &[u8], dst_addr: crate::net::IpAddr, dst_port: u16) {
    let sockets = UDP_SOCKETS.lock();
    if let Some(sock) = sockets.get(idx) {
        sock.sendto(buf, dst_addr, dst_port);
    }
}

/// Local (ephemeral) port a socket is bound to. Callers that need to verify
/// an inbound packet was actually addressed to the port a query went out
/// from (e.g. `dns::handle_dns_response`'s destination-port check) read it
/// back through here rather than re-deriving it.
pub fn port(idx: usize) -> u16 {
    let sockets = UDP_SOCKETS.lock();
    sockets.get(idx).map(|sock| sock.port()).unwrap_or(0)
}

pub fn send(idx: usize, buf: &[u8]) {
    let sockets = UDP_SOCKETS.lock();
    if let Some(sock) = sockets.get(idx) {
        sock.send(buf);
    }
}

pub fn recvfrom(idx: usize, buf: &mut [u8]) -> Option<(usize, crate::net::IpAddr, u16)> {
    let mut sockets = UDP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.recvfrom(buf)
    } else {
        None
    }
}

pub fn recv(idx: usize, buf: &mut [u8]) -> Option<usize> {
    recvfrom(idx, buf).map(|(len, _, _)| len)
}

pub fn handle_udp_packet(src: crate::net::IpAddr, pkt: &[u8]) {
    if pkt.len() < UDP_HEADER_LEN {
        return;
    }
    let hdr = unsafe { &*(pkt.as_ptr() as *const UdpHeader) };
    let dst_port = u16::from_be(hdr.dst_port);
    let src_port = u16::from_be(hdr.src_port);
    let length = u16::from_be(hdr.length) as usize;
    if length < UDP_HEADER_LEN || length > pkt.len() {
        return;
    }
    let payload = &pkt[UDP_HEADER_LEN..length];

    if dst_port == crate::net::dhcp::DHCP_CLIENT_PORT {
        crate::net::dhcp::handle_dhcp_packet(payload);
        return;
    }
    if src_port == 53 {
        crate::net::dns::handle_dns_response(src, src_port, dst_port, payload);
        return;
    }
    let mut sockets = UDP_SOCKETS.lock();
    for sock in sockets.iter_mut() {
        if sock.local_port == dst_port {
            sock.push_packet(src, src_port, payload.to_vec());
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// See net/dns.rs's own test-module header for why (`kernel/seal-os` is
// excluded from the workspace). `testing/runner.rs` calls
// `crate::net::udp::tests::register_all()`, so these do run under the QEMU
// boot proof in CI.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// Exercises exactly the code path `allocate_ephemeral_port` (and
    /// `dns::random_query_id`) take when hardware entropy is unavailable.
    /// This harness has no hook to force RDRAND/RDSEED off, so this tests
    /// the shared fallback generator directly rather than simulating the
    /// hardware-absent branch specifically -- honest scope, not a claim that
    /// hardware entropy was actually disabled for this run.
    fn test_fallback_prng_not_sequential() -> TestResult {
        let mut vals: Vec<u64> = Vec::new();
        for _ in 0..8 {
            vals.push(crate::drivers::entropy::fallback_random_u64());
        }
        let looks_sequential = vals.windows(2).all(|w| w[1] == w[0].wrapping_add(1));
        test_assert!(
            !looks_sequential,
            "fallback_random_u64 looks like a monotonic counter, not randomized"
        );
        TestResult::Pass
    }

    /// Forces every one of `allocate_ephemeral_port`'s 8 randomized attempts
    /// to collide, by filling the visible socket table with one entry per
    /// port in the entire ephemeral range, driving it into the terminal
    /// retries-exhausted path. That path must still be a randomized draw --
    /// it must not fall back to a fixed or incrementing value (the defect
    /// this whole change exists to close).
    fn test_retry_exhaustion_still_randomized() -> TestResult {
        let full: Vec<UdpSocket> = (EPHEMERAL_PORT_BASE..=u16::MAX).map(UdpSocket::new).collect();
        let a = allocate_ephemeral_port(&full);
        let b = allocate_ephemeral_port(&full);
        let c = allocate_ephemeral_port(&full);
        test_assert!(
            !(a == b && b == c),
            "port allocation under retry exhaustion returned the same value repeatedly -- looks like a fixed/counter fallback, not a randomized draw"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "udp::fallback_prng_not_sequential",
            test_fallback_prng_not_sequential,
        );
        crate::testing::register_test(
            "udp::retry_exhaustion_still_randomized",
            test_retry_exhaustion_still_randomized,
        );
    }
}
