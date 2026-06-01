// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal TCP state machine with retransmission timer.

use alloc::vec::Vec;
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TcpState {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

#[repr(C, packed)]
pub struct TcpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub seq: u32,
    pub ack: u32,
    pub data_offset_flags: u16,
    pub window: u16,
    pub checksum: u16,
    pub urgent: u16,
}

impl TcpHeader {
    pub fn data_offset(&self) -> usize {
        ((u16::from_be(self.data_offset_flags) >> 12) as usize) * 4
    }
    pub fn flags(&self) -> u16 {
        u16::from_be(self.data_offset_flags) & 0x3F
    }

    /// Parse a TCP header from exactly 20 bytes.
    /// Returns `None` if the slice is too short.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 20 {
            return None;
        }
        Some(Self {
            src_port: u16::from_be_bytes([bytes[0], bytes[1]]),
            dst_port: u16::from_be_bytes([bytes[2], bytes[3]]),
            seq: u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            ack: u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            data_offset_flags: u16::from_be_bytes([bytes[12], bytes[13]]),
            window: u16::from_be_bytes([bytes[14], bytes[15]]),
            checksum: u16::from_be_bytes([bytes[16], bytes[17]]),
            urgent: u16::from_be_bytes([bytes[18], bytes[19]]),
        })
    }

    /// Serialize the TCP header to a fixed 20-byte array.
    pub fn to_bytes(&self) -> [u8; 20] {
        let mut b = [0u8; 20];
        // SAFETY: All fields are at naturally aligned offsets within the struct.
        // We use addr_of! to avoid creating references to packed struct fields.
        unsafe {
            b[0..2].copy_from_slice(
                &core::ptr::addr_of!((*self).src_port)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[2..4].copy_from_slice(
                &core::ptr::addr_of!((*self).dst_port)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[4..8].copy_from_slice(
                &core::ptr::addr_of!((*self).seq)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[8..12].copy_from_slice(
                &core::ptr::addr_of!((*self).ack)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[12..14].copy_from_slice(
                &core::ptr::addr_of!((*self).data_offset_flags)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[14..16].copy_from_slice(
                &core::ptr::addr_of!((*self).window)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[16..18].copy_from_slice(
                &core::ptr::addr_of!((*self).checksum)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[18..20].copy_from_slice(
                &core::ptr::addr_of!((*self).urgent)
                    .read_unaligned()
                    .to_be_bytes(),
            );
        }
        b
    }
}

const FLAG_FIN: u16 = 1;
const FLAG_SYN: u16 = 2;
const _FLAG_RST: u16 = 4;
const FLAG_PSH: u16 = 8;
const FLAG_ACK: u16 = 16;

struct RetransmitEntry {
    seq: u32,
    flags: u16,
    payload: Vec<u8>,
    time: u64,
}

pub struct TcpSocket {
    state: TcpState,
    local_port: u16,
    remote_port: u16,
    remote_ip: crate::net::IpAddr,
    seq_num: u32,
    ack_num: u32,
    rx_buffer: Vec<u8>,
    tx_buffer: Vec<u8>,
    retransmit_queue: Vec<RetransmitEntry>,
    rto: u64,
    pending_accept: Vec<usize>,
}

impl TcpSocket {
    pub fn new(local_port: u16) -> Self {
        Self {
            state: TcpState::Closed,
            local_port,
            remote_port: 0,
            remote_ip: crate::net::IpAddr::V4([0; 4]),
            seq_num: 1000,
            ack_num: 0,
            rx_buffer: Vec::new(),
            tx_buffer: Vec::new(),
            retransmit_queue: Vec::new(),
            rto: 1000,
            pending_accept: Vec::new(),
        }
    }

    pub fn connect(&mut self, ip: crate::net::IpAddr, port: u16) {
        self.remote_ip = ip;
        self.remote_port = port;
        self.state = TcpState::SynSent;
        self.send_tcp_packet(FLAG_SYN, &[]);
    }

    pub fn listen(&mut self) {
        self.state = TcpState::Listen;
    }

    pub fn send(&mut self, data: &[u8]) {
        if self.state == TcpState::Established {
            self.send_tcp_packet(FLAG_ACK | FLAG_PSH, data);
            self.seq_num += data.len() as u32;
        }
    }

    pub fn recv(&mut self, buf: &mut [u8]) -> usize {
        let len = self.rx_buffer.len().min(buf.len());
        buf[..len].copy_from_slice(&self.rx_buffer[..len]);
        self.rx_buffer.drain(..len);
        len
    }

    pub fn close(&mut self) {
        if self.state == TcpState::Established {
            self.send_tcp_packet(FLAG_FIN | FLAG_ACK, &[]);
            self.state = TcpState::FinWait1;
        }
    }

    pub fn state(&self) -> TcpState {
        self.state
    }

    pub fn local_port(&self) -> u16 {
        self.local_port
    }

    pub fn push_rx(&mut self, data: &[u8]) {
        self.rx_buffer.extend_from_slice(data);
    }

    fn send_tcp_packet(&mut self, flags: u16, payload: &[u8]) {
        let mut hdr = TcpHeader {
            src_port: self.local_port.to_be(),
            dst_port: self.remote_port.to_be(),
            seq: self.seq_num.to_be(),
            ack: self.ack_num.to_be(),
            data_offset_flags: ((5u16 << 12) | flags).to_be(),
            window: 65535u16.to_be(),
            checksum: 0,
            urgent: 0,
        };
        match self.remote_ip {
            crate::net::IpAddr::V4(remote_ip) => {
                let src_ip = crate::net::local_ip();
                let mut pseudo = Vec::with_capacity(12 + 20 + payload.len());
                pseudo.extend_from_slice(&src_ip);
                pseudo.extend_from_slice(&remote_ip);
                pseudo.push(0);
                pseudo.push(6);
                pseudo.extend_from_slice(&((20 + payload.len()) as u16).to_be_bytes());
                let hdr_bytes = hdr.to_bytes();
                pseudo.extend_from_slice(&hdr_bytes);
                pseudo.extend_from_slice(payload);
                let cksum = crate::net::ipv4::internet_checksum(&pseudo);
                unsafe {
                    core::ptr::addr_of_mut!(hdr.checksum).write(cksum);
                }
                let mut pkt = Vec::with_capacity(20 + payload.len());
                pkt.extend_from_slice(&hdr.to_bytes());
                pkt.extend_from_slice(payload);
                crate::net::ipv4::send_ipv4_packet(remote_ip, 6, &pkt);
            }
            crate::net::IpAddr::V6(remote_ip) => {
                let src_ip = crate::net::local_ip_v6();
                let mut pseudo = Vec::with_capacity(40 + 20 + payload.len());
                pseudo.extend_from_slice(&src_ip);
                pseudo.extend_from_slice(&remote_ip);
                pseudo.extend_from_slice(&((20 + payload.len()) as u32).to_be_bytes());
                pseudo.push(0);
                pseudo.push(0);
                pseudo.push(0);
                pseudo.push(6);
                let hdr_bytes = hdr.to_bytes();
                pseudo.extend_from_slice(&hdr_bytes);
                pseudo.extend_from_slice(payload);
                let cksum = crate::net::ipv4::internet_checksum(&pseudo);
                unsafe {
                    core::ptr::addr_of_mut!(hdr.checksum).write(cksum);
                }
                let mut pkt = Vec::with_capacity(20 + payload.len());
                pkt.extend_from_slice(&hdr.to_bytes());
                pkt.extend_from_slice(payload);
                crate::net::ipv6::send_ipv6_packet(remote_ip, 6, &pkt);
            }
        }

        if !payload.is_empty() || flags & FLAG_SYN != 0 || flags & FLAG_FIN != 0 {
            self.retransmit_queue.push(RetransmitEntry {
                seq: self.seq_num,
                flags,
                payload: payload.to_vec(),
                time: crate::drivers::interrupts::ticks(),
            });
        }
    }

    fn purge_acked(&mut self, ack: u32) {
        self.retransmit_queue.retain(|e| {
            let end = e.seq + e.payload.len() as u32;
            end > ack || (ack < 1000 && end < 1000)
        });
    }

    fn retransmit_expired(&mut self) {
        let now = crate::drivers::interrupts::ticks();
        let found = self
            .retransmit_queue
            .iter()
            .find(|entry| now.wrapping_sub(entry.time) >= self.rto)
            .map(|entry| (entry.seq, entry.flags, entry.payload.clone()));
        if let Some((seq, flags, payload)) = found {
            self.seq_num = seq;
            self.send_tcp_packet(flags, &payload);
        }
    }
}

static TCP_SOCKETS: Mutex<Vec<TcpSocket>> = Mutex::new(Vec::new());
static NEXT_TCP_PORT: Mutex<u16> = Mutex::new(40000);
static PENDING_SYN_QUEUE: Mutex<Vec<(u16, crate::net::IpAddr, u16, u32)>> = Mutex::new(Vec::new());

pub fn init() {}

pub fn socket() -> usize {
    let mut sockets = TCP_SOCKETS.lock();
    let port = {
        let mut p = NEXT_TCP_PORT.lock();
        let port = *p;
        *p += 1;
        port
    };
    let idx = sockets.len();
    sockets.push(TcpSocket::new(port));
    idx
}

pub fn bind(idx: usize, port: u16) {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.local_port = port;
    }
}

pub fn listen(idx: usize) {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.listen();
    }
}

pub fn accept(idx: usize) -> Option<usize> {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        if sock.state == TcpState::Listen {
            return sock.pending_accept.pop();
        }
    }
    None
}

pub fn connect(idx: usize, ip: crate::net::IpAddr, port: u16) {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.connect(ip, port);
    }
}

pub fn send(idx: usize, buf: &[u8]) {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.send(buf);
    }
}

pub fn recv(idx: usize, buf: &mut [u8]) -> usize {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.recv(buf)
    } else {
        0
    }
}

pub fn close(idx: usize) {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        sock.close();
    }
}

pub fn state(idx: usize) -> TcpState {
    let sockets = TCP_SOCKETS.lock();
    sockets.get(idx).map_or(TcpState::Closed, |s| s.state())
}

pub struct TcpPacketFixtureProof {
    pub ok: bool,
    pub listener_first: bool,
    pub payload_bytes: usize,
    pub rx_bytes: usize,
    pub accepted_state: TcpState,
    pub cleanup_ok: bool,
}

pub fn packet_fixture_proof() -> TcpPacketFixtureProof {
    const PAYLOAD: &[u8] = b"tick";

    let remote = crate::net::IpAddr::V4([192, 0, 2, 1]);
    let remote_port = 80u16;
    let (base_len, fixture_port) = {
        let sockets = TCP_SOCKETS.lock();
        let Some(port) =
            (49_152u16..=65_535).find(|port| !sockets.iter().any(|s| s.local_port == *port))
        else {
            return TcpPacketFixtureProof {
                ok: false,
                listener_first: false,
                payload_bytes: PAYLOAD.len(),
                rx_bytes: 0,
                accepted_state: TcpState::Closed,
                cleanup_ok: true,
            };
        };
        (sockets.len(), port)
    };

    {
        let mut sockets = TCP_SOCKETS.lock();
        let mut listener = TcpSocket::new(fixture_port);
        listener.listen();

        let mut accepted = TcpSocket::new(fixture_port);
        accepted.remote_ip = remote;
        accepted.remote_port = remote_port;
        accepted.state = TcpState::SynReceived;
        accepted.seq_num = 1000;
        accepted.ack_num = 501;

        sockets.push(listener);
        sockets.push(accepted);
    }

    let packet =
        make_tcp_packet(remote_port, fixture_port, FLAG_ACK | FLAG_PSH, 501, 1001, PAYLOAD);
    handle_tcp_packet(remote, &packet);

    let mut rx = [0u8; PAYLOAD.len()];
    let mut sockets = TCP_SOCKETS.lock();
    let listener_idx = base_len;
    let accepted_idx = base_len + 1;
    let listener_first = sockets
        .get(listener_idx)
        .is_some_and(|sock| sock.state == TcpState::Listen)
        && accepted_idx > listener_idx;
    let accepted_state = sockets
        .get(accepted_idx)
        .map_or(TcpState::Closed, |sock| sock.state);
    let rx_bytes = sockets
        .get_mut(accepted_idx)
        .map_or(0, |sock| sock.recv(&mut rx));
    let payload_matches = rx_bytes == PAYLOAD.len() && &rx[..rx_bytes] == PAYLOAD;
    if sockets.len() >= base_len + 2 {
        sockets.truncate(base_len);
    }
    let cleanup_ok = sockets.len() == base_len;
    let ok = listener_first
        && accepted_state == TcpState::Established
        && rx_bytes == PAYLOAD.len()
        && payload_matches
        && cleanup_ok;

    TcpPacketFixtureProof {
        ok,
        listener_first,
        payload_bytes: PAYLOAD.len(),
        rx_bytes,
        accepted_state,
        cleanup_ok,
    }
}

fn make_tcp_packet(
    src_port: u16,
    dst_port: u16,
    flags: u16,
    seq: u32,
    ack: u32,
    payload: &[u8],
) -> Vec<u8> {
    let hdr = TcpHeader {
        src_port: src_port.to_be(),
        dst_port: dst_port.to_be(),
        seq: seq.to_be(),
        ack: ack.to_be(),
        data_offset_flags: ((5u16 << 12) | flags).to_be(),
        window: 65535u16.to_be(),
        checksum: 0,
        urgent: 0,
    };
    let mut packet = Vec::with_capacity(20 + payload.len());
    packet.extend_from_slice(&hdr.to_bytes());
    packet.extend_from_slice(payload);
    packet
}

pub fn poll() {
    let pending: Vec<_> = {
        let mut q = PENDING_SYN_QUEUE.lock();
        q.drain(..).collect()
    };

    for (dst_port, remote_ip, remote_port, seq) in pending {
        let mut sockets = TCP_SOCKETS.lock();
        if let Some(listener_idx) = sockets
            .iter()
            .position(|s| s.local_port == dst_port && s.state == TcpState::Listen)
        {
            let new_port = {
                let mut p = NEXT_TCP_PORT.lock();
                let port = *p;
                *p += 1;
                port
            };
            let new_idx = sockets.len();
            let mut new_sock = TcpSocket::new(new_port);
            new_sock.local_port = dst_port;
            new_sock.remote_ip = remote_ip;
            new_sock.remote_port = remote_port;
            new_sock.state = TcpState::SynReceived;
            new_sock.ack_num = seq + 1;
            sockets.push(new_sock);
            if let Some(new_sock) = sockets.get_mut(new_idx) {
                new_sock.send_tcp_packet(FLAG_SYN | FLAG_ACK, &[]);
                new_sock.seq_num += 1;
            }
            if let Some(listener) = sockets.get_mut(listener_idx) {
                listener.pending_accept.push(new_idx);
            }
        }
    }

    let mut sockets = TCP_SOCKETS.lock();
    for sock in sockets.iter_mut() {
        if !sock.retransmit_queue.is_empty() {
            sock.retransmit_expired();
        }
    }
}

pub fn handle_tcp_packet(src: crate::net::IpAddr, pkt: &[u8]) {
    if pkt.len() < 20 {
        return;
    }
    // SAFETY: pkt is at least 20 bytes; read_unaligned avoids UB on unaligned pointers.
    let hdr = unsafe { core::ptr::read_unaligned(pkt.as_ptr() as *const TcpHeader) };
    let data_offset = hdr.data_offset();
    if data_offset < 20 || pkt.len() < data_offset {
        return;
    }
    let flags = hdr.flags();
    let dst_port = u16::from_be(hdr.dst_port);
    let src_port = u16::from_be(hdr.src_port);
    let payload = &pkt[data_offset..];

    let mut pending_syn = None;
    {
        let mut sockets = TCP_SOCKETS.lock();
        let exact_idx = sockets.iter().position(|sock| {
            sock.local_port == dst_port
                && sock.remote_port == src_port
                && sock.remote_ip == src
                && sock.state != TcpState::Listen
        });
        let local_active_idx = sockets
            .iter()
            .position(|sock| sock.local_port == dst_port && sock.state != TcpState::Listen);
        let listener_idx = sockets
            .iter()
            .position(|sock| sock.local_port == dst_port && sock.state == TcpState::Listen);
        if let Some(idx) = exact_idx.or(local_active_idx).or(listener_idx) {
            if let Some(sock) = sockets.get_mut(idx) {
                match sock.state {
                    TcpState::Listen => {
                        if flags & FLAG_SYN != 0 {
                            let seq = u32::from_be(hdr.seq);
                            pending_syn = Some((dst_port, src, src_port, seq));
                        }
                    }
                    TcpState::SynSent => {
                        if flags & FLAG_SYN != 0 && flags & FLAG_ACK != 0 {
                            sock.ack_num = u32::from_be(hdr.seq) + 1;
                            sock.seq_num += 1;
                            sock.send_tcp_packet(FLAG_ACK, &[]);
                            sock.state = TcpState::Established;
                        }
                    }
                    TcpState::SynReceived => {
                        if flags & FLAG_ACK != 0 {
                            sock.state = TcpState::Established;
                            sock.ack_num = u32::from_be(hdr.seq) + payload.len() as u32;
                            if !payload.is_empty() {
                                sock.push_rx(payload);
                                sock.send_tcp_packet(FLAG_ACK, &[]);
                            }
                        }
                    }
                    TcpState::Established => {
                        if flags & FLAG_FIN != 0 {
                            sock.ack_num = u32::from_be(hdr.seq) + 1;
                            sock.send_tcp_packet(FLAG_ACK, &[]);
                            sock.state = TcpState::CloseWait;
                        } else if !payload.is_empty() || flags & FLAG_ACK != 0 {
                            sock.ack_num = u32::from_be(hdr.seq) + payload.len() as u32;
                            if !payload.is_empty() {
                                sock.push_rx(payload);
                            }
                            if flags & FLAG_PSH != 0 || !payload.is_empty() {
                                sock.send_tcp_packet(FLAG_ACK, &[]);
                            }
                            if flags & FLAG_ACK != 0 {
                                let ack = u32::from_be(hdr.ack);
                                sock.purge_acked(ack);
                            }
                        }
                    }
                    TcpState::FinWait1 => {
                        if flags & FLAG_FIN != 0 && flags & FLAG_ACK != 0 {
                            sock.ack_num = u32::from_be(hdr.seq) + 1;
                            sock.send_tcp_packet(FLAG_ACK, &[]);
                            sock.state = TcpState::TimeWait;
                        } else if flags & FLAG_ACK != 0 {
                            sock.state = TcpState::FinWait2;
                        }
                    }
                    TcpState::FinWait2 => {
                        if flags & FLAG_FIN != 0 {
                            sock.ack_num = u32::from_be(hdr.seq) + 1;
                            sock.send_tcp_packet(FLAG_ACK, &[]);
                            sock.state = TcpState::TimeWait;
                        }
                    }
                    TcpState::CloseWait => {
                        if flags & FLAG_ACK != 0 {
                            sock.send_tcp_packet(FLAG_FIN | FLAG_ACK, &[]);
                            sock.state = TcpState::LastAck;
                        }
                    }
                    TcpState::LastAck => {
                        if flags & FLAG_ACK != 0 {
                            sock.state = TcpState::Closed;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(syn) = pending_syn {
        PENDING_SYN_QUEUE.lock().push(syn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_tcp_for_test() {
        TCP_SOCKETS.lock().clear();
        PENDING_SYN_QUEUE.lock().clear();
        *NEXT_TCP_PORT.lock() = 40000;
    }

    fn make_tcp_header(flags: u16, seq: u32, ack: u32, dst_port: u16) -> Vec<u8> {
        let hdr = TcpHeader {
            src_port: 80u16.to_be(),
            dst_port: dst_port.to_be(),
            seq: seq.to_be(),
            ack: ack.to_be(),
            data_offset_flags: ((5u16 << 12) | flags).to_be(),
            window: 65535u16.to_be(),
            checksum: 0,
            urgent: 0,
        };
        hdr.to_bytes().to_vec()
    }

    #[test]
    fn test_syn_sent_to_established() {
        let mut sock = TcpSocket::new(1234);
        sock.remote_port = 80;
        sock.state = TcpState::SynSent;
        let pkt = make_tcp_header(FLAG_SYN | FLAG_ACK, 500, 1001, 1234);
        let mut sockets = TCP_SOCKETS.lock();
        let idx = sockets.len();
        sockets.push(sock);
        drop(sockets);
        handle_tcp_packet(crate::net::IpAddr::V4([192, 168, 1, 1]), &pkt);
        let sockets = TCP_SOCKETS.lock();
        assert_eq!(sockets[idx].state, TcpState::Established);
        assert_eq!(sockets[idx].ack_num, 501);
        assert_eq!(sockets[idx].seq_num, 1001);
    }

    #[test]
    fn test_established_to_closewait() {
        let mut sock = TcpSocket::new(1234);
        sock.remote_port = 80;
        sock.state = TcpState::Established;
        let pkt = make_tcp_header(FLAG_FIN | FLAG_ACK, 500, 1000, 1234);
        let mut sockets = TCP_SOCKETS.lock();
        let idx = sockets.len();
        sockets.push(sock);
        drop(sockets);
        handle_tcp_packet(crate::net::IpAddr::V4([192, 168, 1, 1]), &pkt);
        let sockets = TCP_SOCKETS.lock();
        assert_eq!(sockets[idx].state, TcpState::CloseWait);
        assert_eq!(sockets[idx].ack_num, 501);
    }

    #[test]
    fn test_fin_wait1_to_timewait() {
        let mut sock = TcpSocket::new(1234);
        sock.remote_port = 80;
        sock.state = TcpState::FinWait1;
        let pkt = make_tcp_header(FLAG_FIN | FLAG_ACK, 500, 1000, 1234);
        let mut sockets = TCP_SOCKETS.lock();
        let idx = sockets.len();
        sockets.push(sock);
        drop(sockets);
        handle_tcp_packet(crate::net::IpAddr::V4([192, 168, 1, 1]), &pkt);
        let sockets = TCP_SOCKETS.lock();
        assert_eq!(sockets[idx].state, TcpState::TimeWait);
    }

    #[test]
    fn test_close_to_syn_sent() {
        let mut sock = TcpSocket::new(1234);
        assert_eq!(sock.state, TcpState::Closed);
        sock.connect(crate::net::IpAddr::V4([192, 168, 1, 1]), 80);
        assert_eq!(sock.state, TcpState::SynSent);
    }

    #[test]
    fn test_retransmit_queue() {
        let mut sock = TcpSocket::new(1234);
        sock.remote_port = 80;
        sock.remote_ip = crate::net::IpAddr::V4([127, 0, 0, 1]);
        sock.seq_num = 1000;
        sock.state = TcpState::Established;
        sock.send(b"hello");
        assert_eq!(sock.retransmit_queue.len(), 1);
        assert_eq!(sock.retransmit_queue[0].seq, 1000);
        sock.purge_acked(1005);
        assert!(sock.retransmit_queue.is_empty());
    }

    #[test]
    fn test_tcp_header_roundtrip() {
        let hdr = TcpHeader {
            src_port: 1234u16.to_be(),
            dst_port: 80u16.to_be(),
            seq: 0xABCDEF01u32.to_be(),
            ack: 0x12345678u32.to_be(),
            data_offset_flags: ((5u16 << 12) | FLAG_SYN | FLAG_ACK).to_be(),
            window: 64000u16.to_be(),
            checksum: 0xBEEFu16.to_be(),
            urgent: 0u16.to_be(),
        };
        let bytes = hdr.to_bytes();
        let parsed = TcpHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.to_bytes(), bytes);
    }

    #[test]
    fn test_tcp_syn() {
        let idx = socket();
        connect(idx, crate::net::IpAddr::V4([127, 0, 0, 1]), 80);
        let sockets = TCP_SOCKETS.lock();
        let sock = &sockets[idx];
        assert_eq!(sock.state, TcpState::SynSent);
        assert_eq!(sock.retransmit_queue.len(), 1);
        assert_eq!(sock.retransmit_queue[0].flags, FLAG_SYN);
    }

    #[test]
    fn test_tcp_listen_accept() {
        let listener = socket();
        bind(listener, 8080);
        listen(listener);
        assert_eq!(state(listener), TcpState::Listen);

        // Simulate an incoming SYN to port 8080
        let syn_pkt = make_tcp_header(FLAG_SYN, 500, 0, 8080);
        handle_tcp_packet(crate::net::IpAddr::V4([192, 168, 1, 1]), &syn_pkt);

        // Process pending SYNs
        poll();

        let accepted = accept(listener);
        assert!(accepted.is_some());
        let new_sock = accepted.unwrap();
        assert_eq!(state(new_sock), TcpState::SynReceived);
    }

    #[test]
    fn test_accepted_socket_receives_payload_on_listener_port() {
        reset_tcp_for_test();
        let listener = socket();
        bind(listener, 8080);
        listen(listener);

        let remote = crate::net::IpAddr::V4([192, 168, 1, 1]);
        let syn_pkt = make_tcp_header(FLAG_SYN, 500, 0, 8080);
        handle_tcp_packet(remote, &syn_pkt);
        poll();

        let accepted = accept(listener).unwrap();
        let mut data_pkt = make_tcp_header(FLAG_ACK | FLAG_PSH, 501, 1001, 8080);
        data_pkt.extend_from_slice(b"tick");
        handle_tcp_packet(remote, &data_pkt);

        assert_eq!(state(accepted), TcpState::Established);
        let mut buf = [0u8; 8];
        assert_eq!(recv(accepted, &mut buf), 4);
        assert_eq!(&buf[..4], b"tick");
    }

    #[test]
    fn packet_fixture_proof_reports_listener_first_demux() {
        reset_tcp_for_test();

        let proof = packet_fixture_proof();

        assert!(proof.ok);
        assert!(proof.listener_first);
        assert_eq!(proof.payload_bytes, 4);
        assert_eq!(proof.rx_bytes, 4);
        assert_eq!(proof.accepted_state, TcpState::Established);
        assert!(proof.cleanup_ok);
        assert!(TCP_SOCKETS.lock().is_empty());
    }
}
