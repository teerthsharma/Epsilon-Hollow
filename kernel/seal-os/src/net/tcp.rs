// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal TCP state machine.

use alloc::vec::Vec;
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TcpState {
    Closed,
    SynSent,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

#[repr(C, packed)]
struct TcpHeader {
    src_port: u16,
    dst_port: u16,
    seq: u32,
    ack: u32,
    data_offset_flags: u16,
    window: u16,
    checksum: u16,
    urgent: u16,
}

impl TcpHeader {
    fn data_offset(&self) -> usize {
        ((u16::from_be(self.data_offset_flags) >> 12) as usize) * 4
    }
    fn flags(&self) -> u16 {
        u16::from_be(self.data_offset_flags) & 0x3F
    }
}

const FLAG_FIN: u16 = 1;
const FLAG_SYN: u16 = 2;
const _FLAG_RST: u16 = 4;
const FLAG_PSH: u16 = 8;
const FLAG_ACK: u16 = 16;

pub struct TcpSocket {
    state: TcpState,
    local_port: u16,
    remote_port: u16,
    remote_ip: [u8; 4],
    seq_num: u32,
    ack_num: u32,
    rx_buffer: Vec<u8>,
    tx_buffer: Vec<u8>,
}

impl TcpSocket {
    pub fn new(local_port: u16) -> Self {
        Self {
            state: TcpState::Closed,
            local_port,
            remote_port: 0,
            remote_ip: [0; 4],
            seq_num: 1000,
            ack_num: 0,
            rx_buffer: Vec::new(),
            tx_buffer: Vec::new(),
        }
    }

    pub fn connect(&mut self, ip: [u8; 4], port: u16) {
        self.remote_ip = ip;
        self.remote_port = port;
        self.state = TcpState::SynSent;
        self.send_tcp_packet(FLAG_SYN, &[]);
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

    fn send_tcp_packet(&self, flags: u16, payload: &[u8]) {
        let src_ip = crate::net::local_ip();
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
        let mut pseudo = Vec::with_capacity(12 + 20 + payload.len());
        pseudo.extend_from_slice(&src_ip);
        pseudo.extend_from_slice(&self.remote_ip);
        pseudo.push(0);
        pseudo.push(6);
        pseudo.extend_from_slice(&((20 + payload.len()) as u16).to_be_bytes());
        let hdr_bytes = unsafe {
            core::slice::from_raw_parts(&hdr as *const _ as *const u8, 20)
        };
        pseudo.extend_from_slice(hdr_bytes);
        pseudo.extend_from_slice(payload);
        hdr.checksum = crate::net::ipv4::internet_checksum(&pseudo);
        let mut pkt = Vec::with_capacity(20 + payload.len());
        let hdr_bytes = unsafe {
            core::slice::from_raw_parts(&hdr as *const _ as *const u8, 20)
        };
        pkt.extend_from_slice(hdr_bytes);
        pkt.extend_from_slice(payload);
        crate::net::ipv4::send_ipv4_packet(self.remote_ip, 6, &pkt);
    }
}

static TCP_SOCKETS: Mutex<Vec<TcpSocket>> = Mutex::new(Vec::new());
static NEXT_TCP_PORT: Mutex<u16> = Mutex::new(40000);

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

pub fn connect(idx: usize, ip: [u8; 4], port: u16) {
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

pub fn poll() {}

pub fn handle_tcp_packet(_src: [u8; 4], pkt: &[u8]) {
    if pkt.len() < 20 {
        return;
    }
    let hdr = unsafe { &*(pkt.as_ptr() as *const TcpHeader) };
    let data_offset = hdr.data_offset();
    if data_offset < 20 || pkt.len() < data_offset {
        return;
    }
    let flags = hdr.flags();
    let dst_port = u16::from_be(hdr.dst_port);
    let payload = &pkt[data_offset..];

    let mut sockets = TCP_SOCKETS.lock();
    for sock in sockets.iter_mut() {
        if sock.local_port == dst_port {
            match sock.state {
                TcpState::SynSent => {
                    if flags & FLAG_SYN != 0 && flags & FLAG_ACK != 0 {
                        sock.ack_num = u32::from_be(hdr.seq) + 1;
                        sock.seq_num += 1;
                        sock.send_tcp_packet(FLAG_ACK, &[]);
                        sock.state = TcpState::Established;
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
            return;
        }
    }
}
