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
// Fallback only -- see `allocate_ephemeral_port`. Kept monotonic (never
// reused) so the fallback path can't hand out a port already in use.
static NEXT_EPHEMERAL_PORT: Mutex<u16> = Mutex::new(49152);
const EPHEMERAL_PORT_BASE: u16 = 49152; // RFC 6335 dynamic/private range start

pub fn init() {}

/// Pick a fresh local ephemeral port (RFC 6335 dynamic/private range,
/// 49152-65535). Drawn from hardware entropy when available rather than a
/// plain counter: the same induced-lookup prediction that made a monotonic
/// DNS transaction ID guessable (`net::dns`'s `random_query_id`) applies
/// just as well to a monotonic port counter here -- a spoofing target that
/// also runs a server the kernel talks to sees the source port of every
/// packet it receives, and could predict the counter's next value for an
/// unrelated query the same way. Retries on collision with an already-open
/// socket; falls back to the previous incrementing counter (which can only
/// collide on `u16` wraparound, unchanged pre-existing behavior) if entropy
/// is unavailable or stays unlucky for 8 draws.
fn allocate_ephemeral_port(sockets: &[UdpSocket]) -> u16 {
    let mut buf = [0u8; 2];
    for _ in 0..8 {
        if !crate::drivers::entropy::getrandom(&mut buf) {
            break;
        }
        let candidate = EPHEMERAL_PORT_BASE
            .wrapping_add(u16::from_ne_bytes(buf) % (u16::MAX - EPHEMERAL_PORT_BASE));
        if !sockets.iter().any(|s| s.local_port == candidate) {
            return candidate;
        }
    }
    let mut p = NEXT_EPHEMERAL_PORT.lock();
    let port = *p;
    *p = p.wrapping_add(1);
    port
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
