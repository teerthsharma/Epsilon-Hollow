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
        send_datagram(self.local_port, buf, dst_addr, dst_port);
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

/// Build one datagram and hand it to the IP layer.
///
/// **The `UDP_SOCKETS` guard must be dropped before calling this.**
///
/// `ipv4::send_ipv4_packet` does not queue a 127.0.0.1 destination -- it calls
/// `handle_ipv4_packet` on this stack, which dispatches protocol 17 into
/// `handle_udp_packet`, which locks `UDP_SOCKETS`. `ipv6::send_ipv6_packet`
/// does the same for `::1` through `handle_ipv6_packet`. That mutex is a
/// `spin::Mutex`: non-reentrant, no interrupt masking, no owner check. A sender
/// still holding it spins forever on a lock it owns itself -- no panic, no
/// fault, the thread simply stops while the timer keeps ticking. The DHCP
/// diversion at the top of `handle_udp_packet` reaches the same lock by a
/// second door, through `dhcp::handle_dhcp_packet`'s own `udp::socket()`.
///
/// Everything this needs from the socket is its local port, so the module-level
/// `sendto` and `send` read that out under the guard and call here once it has
/// dropped. Taking a `u16` rather than a `&UdpSocket` is deliberate: a borrow
/// of a table entry can only be held while the table is locked, so the old
/// signature made the deadlock the natural way to call it.
///
/// No transmit queue rides along with this, unlike `tcp::flush_tx`. TCP needed
/// one because `handle_tcp_packet` answers segments from inside its own
/// delivery while holding `TCP_SOCKETS`, so deferring to the caller was not
/// enough. No UDP receive path replies: `handle_udp_packet` only buffers, and
/// the two handlers it diverts to answer either nothing (`dns`) or the
/// broadcast address (`dhcp`), never loopback. Nesting is therefore not
/// reachable from here, and `ipv4::LOOPBACK_MAX_DEPTH` already bounds the stack
/// if a future handler makes it so.
fn send_datagram(src_port: u16, buf: &[u8], dst_addr: crate::net::IpAddr, dst_port: u16) {
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
            // Network order, like the three fields above: `hdr` is blitted raw.
            hdr.checksum = crate::net::ipv4::internet_checksum(&pseudo).to_be();
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
            // Network order, like the three fields above: `hdr` is blitted raw.
            hdr.checksum = crate::net::ipv4::internet_checksum(&pseudo).to_be();
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

/// Send `buf` to an explicit destination from socket `idx`.
///
/// The source port is read out under the guard and the datagram is built and
/// transmitted after it drops -- see `send_datagram`, which explains why
/// transmitting under the guard stalls the thread on a 127.0.0.1 or `::1`
/// destination.
pub fn sendto(idx: usize, buf: &[u8], dst_addr: crate::net::IpAddr, dst_port: u16) {
    let src_port = UDP_SOCKETS.lock().get(idx).map(UdpSocket::port);
    if let Some(src_port) = src_port {
        send_datagram(src_port, buf, dst_addr, dst_port);
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

/// Send `buf` to the address socket `idx` was connected to. Same lock
/// discipline as `sendto`: the destination is read out under the guard, the
/// transmit happens after it drops.
pub fn send(idx: usize, buf: &[u8]) {
    let target = {
        let sockets = UDP_SOCKETS.lock();
        sockets
            .get(idx)
            .and_then(|sock| match (sock.remote_addr, sock.remote_port) {
                (Some(addr), Some(port)) => Some((sock.local_port, addr, port)),
                _ => None,
            })
    };
    if let Some((src_port, addr, port)) = target {
        send_datagram(src_port, buf, addr, port);
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
        let full: Vec<UdpSocket> = (EPHEMERAL_PORT_BASE..=u16::MAX)
            .map(UdpSocket::new)
            .collect();
        let a = allocate_ephemeral_port(&full);
        let b = allocate_ephemeral_port(&full);
        let c = allocate_ephemeral_port(&full);
        test_assert!(
            !(a == b && b == c),
            "port allocation under retry exhaustion returned the same value repeatedly -- looks like a fixed/counter fallback, not a randomized draw"
        );
        TestResult::Pass
    }

    /// End-to-end proof that the stack accepts a packet it built itself.
    ///
    /// `sendto` to 127.0.0.1 routes through `ipv4::send_ipv4_packet`'s loopback
    /// branch straight into `handle_ipv4_packet`, which drops anything failing
    /// `internet_checksum(header) != 0`. So a datagram arriving in the socket's
    /// receive buffer is proof the transmit path stored its header checksum in
    /// the byte order the receive path verifies -- the defect this closes shipped
    /// it host-order, and every loopback datagram was silently dropped.
    ///
    /// A wrong-but-self-consistent byte order cannot satisfy this: the two ends
    /// are `Ipv4Header::compute_checksum` and `internet_checksum`, which agree
    /// only when the stored bytes are network order.
    ///
    /// Ports sit below the RFC 6335 ephemeral range so `allocate_ephemeral_port`
    /// cannot hand the same value to another socket and steal the dispatch, and
    /// avoid 68 (DHCP) and 53 (DNS), which `handle_udp_packet` diverts.
    ///
    /// The send goes through a table socket and the module-level `sendto`, which
    /// is what every in-tree caller uses. It could not before: `sendto` held
    /// `UDP_SOCKETS` across the transmit, the loopback path re-entered
    /// `handle_udp_packet`, and that took the same non-reentrant lock. The test
    /// sidestepped it with a stack-local `UdpSocket`, so the shipped entry point
    /// was the one path this proof did not cover.
    fn test_loopback_ipv4_round_trip_accepted() -> TestResult {
        const RX_PORT: u16 = 9000;
        const TX_PORT: u16 = 9001;
        const PAYLOAD: &[u8] = b"seal-loopback";

        let idx = socket();
        bind(idx, RX_PORT);
        let tx = socket();
        bind(tx, TX_PORT);

        sendto(tx, PAYLOAD, crate::net::IpAddr::V4([127, 0, 0, 1]), RX_PORT);

        let mut buf = [0u8; 32];
        let got = recvfrom(idx, &mut buf);
        test_assert!(
            got.is_some(),
            "loopback datagram never reached the socket -- handle_ipv4_packet rejected a header this tree built"
        );
        let (len, _src, src_port) = got.unwrap();
        test_assert!(
            &buf[..len] == PAYLOAD,
            "loopback datagram payload did not survive the round trip"
        );
        test_assert!(
            src_port == TX_PORT,
            "loopback datagram carried the wrong source port"
        );

        // Control on the rejecting side, so the assertions above cannot pass
        // because the receive path delivers to any socket regardless of the
        // header: a datagram addressed to a port nothing is bound to must not
        // land in this socket.
        sendto(
            tx,
            PAYLOAD,
            crate::net::IpAddr::V4([127, 0, 0, 1]),
            RX_PORT + 100,
        );
        test_assert!(
            recvfrom(idx, &mut buf).is_none(),
            "datagram for an unbound port was delivered to the wrong socket"
        );
        TestResult::Pass
    }

    /// A loopback transmit must not be made while `UDP_SOCKETS` is held.
    ///
    /// `ipv4::send_ipv4_packet` hands a 127.0.0.1 destination straight to
    /// `handle_ipv4_packet` on the caller's stack, which dispatches protocol 17
    /// back into `handle_udp_packet`, which locks `UDP_SOCKETS`. That mutex is a
    /// `spin::Mutex` -- non-reentrant -- so a sender still holding it spins on a
    /// lock it owns. `ipv6::send_ipv6_packet` has the same `::1` branch into
    /// `handle_ipv6_packet`, so both address families are driven here.
    ///
    /// The failure mode is a stall, not a wrong answer: this test cannot report
    /// it, it simply never returns. What it can report is everything that must
    /// remain true once the transmit is deferred --
    ///
    /// * the datagram is delivered, so the stall was not cured by dropping the
    ///   packet or by skipping the loopback branch;
    /// * `handle_udp_packet` reached the socket table, which is proof the lock
    ///   was free during the delivery and not merely released afterwards;
    /// * `loopback_nested` does not move, so the delivery was not made from
    ///   inside another delivery;
    /// * `loopback_dispatched` moves by exactly one per send, so no datagram was
    ///   delivered twice or silently swallowed;
    /// * `loopback_depth` returns to zero.
    ///
    /// Ports avoid 68 (DHCP) and 53 (DNS), which `handle_udp_packet` diverts
    /// before it ever reaches the table, and sit below the RFC 6335 ephemeral
    /// range so `allocate_ephemeral_port` cannot steal the dispatch.
    fn test_loopback_sendto_does_not_reenter_socket_lock() -> TestResult {
        const RX_PORT: u16 = 9002;
        const TX_PORT: u16 = 9003;
        const RX6_PORT: u16 = 9004;
        const PAYLOAD: &[u8] = b"seal-sendto-loopback";

        let rx = socket();
        bind(rx, RX_PORT);
        let tx = socket();
        bind(tx, TX_PORT);

        let dispatched_before = crate::net::ipv4::loopback_dispatched();
        let nested_before = crate::net::ipv4::loopback_nested();
        let dropped_before = crate::net::ipv4::loopback_dropped();

        sendto(tx, PAYLOAD, crate::net::IpAddr::V4([127, 0, 0, 1]), RX_PORT);

        let mut buf = [0u8; 64];
        let got = recvfrom(rx, &mut buf);
        test_assert!(
            got.is_some(),
            "the module-level sendto delivered nothing over 127.0.0.1 -- the transmit was deferred but never made"
        );
        let (len, _src, src_port) = got.unwrap();
        test_assert!(
            &buf[..len] == PAYLOAD,
            "the loopback datagram sent through the socket table did not survive the round trip"
        );
        test_assert!(
            src_port == TX_PORT,
            "the loopback datagram carried a source port other than the sending socket's"
        );
        test_assert!(
            crate::net::ipv4::loopback_nested() == nested_before,
            "a loopback datagram was delivered from inside another delivery -- the send path re-entered the receive path"
        );
        test_assert!(
            crate::net::ipv4::loopback_dropped() == dropped_before,
            "a loopback datagram hit LOOPBACK_MAX_DEPTH -- deliveries are nesting instead of being deferred"
        );
        test_assert!(
            crate::net::ipv4::loopback_dispatched() - dispatched_before == 1,
            "one sendto to 127.0.0.1 entered the loopback branch other than once"
        );
        test_assert!(
            crate::net::ipv4::loopback_depth() == 0,
            "loopback depth did not return to zero -- a delivery was counted in but not out"
        );

        // `send` is the second site with the same shape -- it read the
        // destination out of the socket and transmitted under the same guard --
        // so it is driven here too. Covering only `sendto` would leave a
        // connected socket stalling the kernel exactly as before.
        connect(tx, crate::net::IpAddr::V4([127, 0, 0, 1]), RX_PORT);
        send(tx, PAYLOAD);
        let got_connected = recvfrom(rx, &mut buf);
        test_assert!(
            got_connected.is_some(),
            "the module-level send delivered nothing over 127.0.0.1 from a connected socket"
        );
        let (len_c, _src_c, src_port_c) = got_connected.unwrap();
        test_assert!(
            &buf[..len_c] == PAYLOAD && src_port_c == TX_PORT,
            "the datagram from the connected socket did not survive the round trip intact"
        );

        // The IPv6 arm reaches `handle_udp_packet` through
        // `ipv6::handle_ipv6_packet`, which has no header checksum to verify --
        // so unlike the IPv4 arm it has been able to reach the socket lock for
        // as long as the `::1` branch has existed.
        let rx6 = socket();
        bind(rx6, RX6_PORT);
        let mut v6 = [0u8; 16];
        v6[15] = 1;
        sendto(tx, PAYLOAD, crate::net::IpAddr::V6(v6), RX6_PORT);
        let got6 = recvfrom(rx6, &mut buf);
        test_assert!(
            got6.is_some(),
            "the module-level sendto delivered nothing over ::1"
        );
        let (len6, _src6, src_port6) = got6.unwrap();
        test_assert!(
            &buf[..len6] == PAYLOAD && src_port6 == TX_PORT,
            "the ::1 datagram did not survive the round trip intact"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "udp::fallback_prng_not_sequential",
            test_fallback_prng_not_sequential,
        );
        crate::testing::register_test(
            "udp::loopback_ipv4_round_trip_accepted",
            test_loopback_ipv4_round_trip_accepted,
        );
        crate::testing::register_test(
            "udp::retry_exhaustion_still_randomized",
            test_retry_exhaustion_still_randomized,
        );
        crate::testing::register_test(
            "udp::loopback_sendto_does_not_reenter_socket_lock",
            test_loopback_sendto_does_not_reenter_socket_lock,
        );
    }
}
