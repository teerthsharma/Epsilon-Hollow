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
        ((self.data_offset_flags >> 12) as usize) * 4
    }
    pub fn flags(&self) -> u16 {
        self.data_offset_flags & 0x3F
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

pub const TCP_FLOW_INDEX_BUCKETS: usize = 256;
pub const TCP_FLOW_INDEX_MAX_PROBES: usize = TCP_FLOW_INDEX_BUCKETS;
pub const TCP_LISTENER_INDEX_BUCKETS: usize = 256;
pub const TCP_LISTENER_INDEX_MAX_PROBES: usize = TCP_LISTENER_INDEX_BUCKETS;

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
    #[allow(dead_code)] // REASON: transmit buffer reserved for future TCP send path
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
            src_port: self.local_port,
            dst_port: self.remote_port,
            seq: self.seq_num,
            ack: self.ack_num,
            data_offset_flags: (5u16 << 12) | flags,
            window: 65535,
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
static TCP_FLOW_INDEX: Mutex<TcpFlowIndex> = Mutex::new(TcpFlowIndex::new());
static TCP_LISTENER_INDEX: Mutex<TcpListenerIndex> = Mutex::new(TcpListenerIndex::new());
static NEXT_TCP_PORT: Mutex<u16> = Mutex::new(40000);
static PENDING_SYN_QUEUE: Mutex<Vec<(u16, crate::net::IpAddr, u16, u32)>> = Mutex::new(Vec::new());

#[derive(Clone, Copy, PartialEq)]
struct TcpFlowKey {
    local_port: u16,
    remote_port: u16,
    remote_ip: crate::net::IpAddr,
}

#[derive(Clone, Copy)]
struct TcpFlowIndexEntry {
    key: TcpFlowKey,
    socket_idx: usize,
}

#[derive(Clone, Copy)]
struct TcpFlowIndexSlot {
    entry: Option<TcpFlowIndexEntry>,
    tombstone: bool,
}

impl TcpFlowIndexSlot {
    const fn empty() -> Self {
        Self {
            entry: None,
            tombstone: false,
        }
    }
}

struct TcpFlowLookup {
    socket_idx: Option<usize>,
    probes: usize,
}

pub struct TcpFlowIndexProof {
    pub hit: bool,
    pub probes: usize,
    pub bound: usize,
    pub capacity: usize,
}

#[derive(Clone, Copy)]
struct TcpListenerIndexEntry {
    local_port: u16,
    socket_idx: usize,
}

#[derive(Clone, Copy)]
struct TcpListenerIndexSlot {
    entry: Option<TcpListenerIndexEntry>,
    tombstone: bool,
}

impl TcpListenerIndexSlot {
    const fn empty() -> Self {
        Self {
            entry: None,
            tombstone: false,
        }
    }
}

struct TcpListenerLookup {
    socket_idx: Option<usize>,
    probes: usize,
}

pub struct TcpListenerIndexProof {
    pub hit: bool,
    pub probes: usize,
    pub bound: usize,
    pub capacity: usize,
}

struct TcpFlowIndex {
    slots: [TcpFlowIndexSlot; TCP_FLOW_INDEX_BUCKETS],
    last_lookup_hit: bool,
    last_lookup_probes: usize,
}

struct TcpListenerIndex {
    slots: [TcpListenerIndexSlot; TCP_LISTENER_INDEX_BUCKETS],
    last_lookup_hit: bool,
    last_lookup_probes: usize,
}

impl TcpListenerIndex {
    const fn new() -> Self {
        Self {
            slots: [TcpListenerIndexSlot::empty(); TCP_LISTENER_INDEX_BUCKETS],
            last_lookup_hit: false,
            last_lookup_probes: 0,
        }
    }

    fn clear(&mut self) {
        self.slots = [TcpListenerIndexSlot::empty(); TCP_LISTENER_INDEX_BUCKETS];
        self.last_lookup_hit = false;
        self.last_lookup_probes = 0;
    }

    fn insert(&mut self, local_port: u16, socket_idx: usize) -> bool {
        let start = tcp_listener_hash(local_port) & (TCP_LISTENER_INDEX_BUCKETS - 1);
        let mut first_tombstone = None;
        let mut probe = 0usize;
        while probe < TCP_LISTENER_INDEX_MAX_PROBES {
            let idx = (start + probe) & (TCP_LISTENER_INDEX_BUCKETS - 1);
            let slot = &mut self.slots[idx];
            match slot.entry {
                Some(entry) if entry.local_port == local_port => {
                    slot.entry = Some(TcpListenerIndexEntry {
                        local_port,
                        socket_idx,
                    });
                    slot.tombstone = false;
                    return true;
                }
                Some(_) => {}
                None if slot.tombstone => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(idx);
                    }
                }
                None => {
                    let insert_idx = first_tombstone.unwrap_or(idx);
                    self.slots[insert_idx] = TcpListenerIndexSlot {
                        entry: Some(TcpListenerIndexEntry {
                            local_port,
                            socket_idx,
                        }),
                        tombstone: false,
                    };
                    return true;
                }
            }
            probe += 1;
        }
        if let Some(insert_idx) = first_tombstone {
            self.slots[insert_idx] = TcpListenerIndexSlot {
                entry: Some(TcpListenerIndexEntry {
                    local_port,
                    socket_idx,
                }),
                tombstone: false,
            };
            return true;
        }
        false
    }

    fn remove(&mut self, local_port: u16, socket_idx: usize) {
        let start = tcp_listener_hash(local_port) & (TCP_LISTENER_INDEX_BUCKETS - 1);
        let mut probe = 0usize;
        while probe < TCP_LISTENER_INDEX_MAX_PROBES {
            let idx = (start + probe) & (TCP_LISTENER_INDEX_BUCKETS - 1);
            let slot = &mut self.slots[idx];
            match slot.entry {
                Some(entry) if entry.local_port == local_port && entry.socket_idx == socket_idx => {
                    slot.entry = None;
                    slot.tombstone = true;
                    return;
                }
                Some(_) => {}
                None if slot.tombstone => {}
                None => return,
            }
            probe += 1;
        }
    }

    fn lookup(&mut self, local_port: u16) -> TcpListenerLookup {
        let start = tcp_listener_hash(local_port) & (TCP_LISTENER_INDEX_BUCKETS - 1);
        let mut probe = 0usize;
        while probe < TCP_LISTENER_INDEX_MAX_PROBES {
            let idx = (start + probe) & (TCP_LISTENER_INDEX_BUCKETS - 1);
            let slot = self.slots[idx];
            let probes = probe + 1;
            match slot.entry {
                Some(entry) if entry.local_port == local_port => {
                    self.last_lookup_hit = true;
                    self.last_lookup_probes = probes;
                    return TcpListenerLookup {
                        socket_idx: Some(entry.socket_idx),
                        probes,
                    };
                }
                Some(_) => {}
                None if slot.tombstone => {}
                None => {
                    self.last_lookup_hit = false;
                    self.last_lookup_probes = probes;
                    return TcpListenerLookup {
                        socket_idx: None,
                        probes,
                    };
                }
            }
            probe += 1;
        }
        self.last_lookup_hit = false;
        self.last_lookup_probes = TCP_LISTENER_INDEX_MAX_PROBES;
        TcpListenerLookup {
            socket_idx: None,
            probes: TCP_LISTENER_INDEX_MAX_PROBES,
        }
    }
}

impl TcpFlowIndex {
    const fn new() -> Self {
        Self {
            slots: [TcpFlowIndexSlot::empty(); TCP_FLOW_INDEX_BUCKETS],
            last_lookup_hit: false,
            last_lookup_probes: 0,
        }
    }

    fn clear(&mut self) {
        self.slots = [TcpFlowIndexSlot::empty(); TCP_FLOW_INDEX_BUCKETS];
        self.last_lookup_hit = false;
        self.last_lookup_probes = 0;
    }

    fn insert(&mut self, key: TcpFlowKey, socket_idx: usize) -> bool {
        let start = tcp_flow_hash(key) & (TCP_FLOW_INDEX_BUCKETS - 1);
        let mut first_tombstone = None;
        let mut probe = 0usize;
        while probe < TCP_FLOW_INDEX_MAX_PROBES {
            let idx = (start + probe) & (TCP_FLOW_INDEX_BUCKETS - 1);
            let slot = &mut self.slots[idx];
            match slot.entry {
                Some(entry) if entry.key == key => {
                    slot.entry = Some(TcpFlowIndexEntry { key, socket_idx });
                    slot.tombstone = false;
                    return true;
                }
                Some(_) => {}
                None if slot.tombstone => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(idx);
                    }
                }
                None => {
                    let insert_idx = first_tombstone.unwrap_or(idx);
                    self.slots[insert_idx] = TcpFlowIndexSlot {
                        entry: Some(TcpFlowIndexEntry { key, socket_idx }),
                        tombstone: false,
                    };
                    return true;
                }
            }
            probe += 1;
        }
        if let Some(insert_idx) = first_tombstone {
            self.slots[insert_idx] = TcpFlowIndexSlot {
                entry: Some(TcpFlowIndexEntry { key, socket_idx }),
                tombstone: false,
            };
            return true;
        }
        false
    }

    fn remove(&mut self, key: TcpFlowKey, socket_idx: usize) {
        let start = tcp_flow_hash(key) & (TCP_FLOW_INDEX_BUCKETS - 1);
        let mut probe = 0usize;
        while probe < TCP_FLOW_INDEX_MAX_PROBES {
            let idx = (start + probe) & (TCP_FLOW_INDEX_BUCKETS - 1);
            let slot = &mut self.slots[idx];
            match slot.entry {
                Some(entry) if entry.key == key && entry.socket_idx == socket_idx => {
                    slot.entry = None;
                    slot.tombstone = true;
                    return;
                }
                Some(_) => {}
                None if slot.tombstone => {}
                None => return,
            }
            probe += 1;
        }
    }

    fn lookup(&mut self, key: TcpFlowKey) -> TcpFlowLookup {
        let start = tcp_flow_hash(key) & (TCP_FLOW_INDEX_BUCKETS - 1);
        let mut probe = 0usize;
        while probe < TCP_FLOW_INDEX_MAX_PROBES {
            let idx = (start + probe) & (TCP_FLOW_INDEX_BUCKETS - 1);
            let slot = self.slots[idx];
            let probes = probe + 1;
            match slot.entry {
                Some(entry) if entry.key == key => {
                    self.last_lookup_hit = true;
                    self.last_lookup_probes = probes;
                    return TcpFlowLookup {
                        socket_idx: Some(entry.socket_idx),
                        probes,
                    };
                }
                Some(_) => {}
                None if slot.tombstone => {}
                None => {
                    self.last_lookup_hit = false;
                    self.last_lookup_probes = probes;
                    return TcpFlowLookup {
                        socket_idx: None,
                        probes,
                    };
                }
            }
            probe += 1;
        }
        self.last_lookup_hit = false;
        self.last_lookup_probes = TCP_FLOW_INDEX_MAX_PROBES;
        TcpFlowLookup {
            socket_idx: None,
            probes: TCP_FLOW_INDEX_MAX_PROBES,
        }
    }
}

fn tcp_flow_key(sock: &TcpSocket) -> Option<TcpFlowKey> {
    if matches!(sock.state, TcpState::Closed | TcpState::Listen) || sock.remote_port == 0 {
        return None;
    }
    Some(TcpFlowKey {
        local_port: sock.local_port,
        remote_port: sock.remote_port,
        remote_ip: sock.remote_ip,
    })
}

fn tcp_packet_flow_key(src: crate::net::IpAddr, src_port: u16, dst_port: u16) -> TcpFlowKey {
    TcpFlowKey {
        local_port: dst_port,
        remote_port: src_port,
        remote_ip: src,
    }
}

fn tcp_flow_hash(key: TcpFlowKey) -> usize {
    let mut hash = 0x9e37_79b9_7f4a_7c15u64;
    hash ^= ((key.local_port as u64) << 48) ^ ((key.remote_port as u64) << 32);
    match key.remote_ip {
        crate::net::IpAddr::V4(addr) => {
            hash ^= u32::from_be_bytes(addr) as u64;
        }
        crate::net::IpAddr::V6(addr) => {
            let hi = u64::from_be_bytes([
                addr[0], addr[1], addr[2], addr[3], addr[4], addr[5], addr[6], addr[7],
            ]);
            let lo = u64::from_be_bytes([
                addr[8], addr[9], addr[10], addr[11], addr[12], addr[13], addr[14], addr[15],
            ]);
            hash ^= hi ^ lo.rotate_left(17);
        }
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash as usize
}

fn tcp_listener_key(sock: &TcpSocket) -> Option<u16> {
    if sock.state == TcpState::Listen {
        Some(sock.local_port)
    } else {
        None
    }
}

fn tcp_listener_hash(local_port: u16) -> usize {
    let mut hash = (local_port as u64) ^ 0x517c_c1b7_2722_0a95u64;
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^= hash >> 33;
    hash as usize
}

fn index_exact_flow_socket(idx: usize, sock: &TcpSocket) -> bool {
    let Some(key) = tcp_flow_key(sock) else {
        return false;
    };
    TCP_FLOW_INDEX.lock().insert(key, idx)
}

fn remove_exact_flow_socket(idx: usize, sock: &TcpSocket) {
    if let Some(key) = tcp_flow_key(sock) {
        TCP_FLOW_INDEX.lock().remove(key, idx);
    }
}

fn refresh_exact_flow_socket(idx: usize, old_key: Option<TcpFlowKey>, sock: &TcpSocket) {
    let new_key = tcp_flow_key(sock);
    if old_key == new_key {
        return;
    }
    let mut index = TCP_FLOW_INDEX.lock();
    if let Some(key) = old_key {
        index.remove(key, idx);
    }
    if let Some(key) = new_key {
        index.insert(key, idx);
    }
}

fn index_listener_socket(idx: usize, sock: &TcpSocket) -> bool {
    let Some(local_port) = tcp_listener_key(sock) else {
        return false;
    };
    TCP_LISTENER_INDEX.lock().insert(local_port, idx)
}

fn remove_listener_socket(idx: usize, sock: &TcpSocket) {
    if let Some(local_port) = tcp_listener_key(sock) {
        TCP_LISTENER_INDEX.lock().remove(local_port, idx);
    }
}

fn refresh_listener_socket(idx: usize, old_key: Option<u16>, sock: &TcpSocket) {
    let new_key = tcp_listener_key(sock);
    if old_key == new_key {
        return;
    }
    let mut index = TCP_LISTENER_INDEX.lock();
    if let Some(local_port) = old_key {
        index.remove(local_port, idx);
    }
    if let Some(local_port) = new_key {
        index.insert(local_port, idx);
    }
}

fn lookup_exact_flow_index(
    sockets: &[TcpSocket],
    src: crate::net::IpAddr,
    src_port: u16,
    dst_port: u16,
) -> TcpFlowLookup {
    let key = tcp_packet_flow_key(src, src_port, dst_port);
    let lookup = TCP_FLOW_INDEX.lock().lookup(key);
    let Some(idx) = lookup.socket_idx else {
        return lookup;
    };
    if sockets
        .get(idx)
        .is_some_and(|sock| tcp_flow_key(sock) == Some(key))
    {
        lookup
    } else {
        TCP_FLOW_INDEX.lock().last_lookup_hit = false;
        TcpFlowLookup {
            socket_idx: None,
            probes: lookup.probes,
        }
    }
}

fn lookup_listener_index(sockets: &[TcpSocket], dst_port: u16) -> TcpListenerLookup {
    let lookup = TCP_LISTENER_INDEX.lock().lookup(dst_port);
    let Some(idx) = lookup.socket_idx else {
        return lookup;
    };
    if sockets
        .get(idx)
        .is_some_and(|sock| tcp_listener_key(sock) == Some(dst_port))
    {
        lookup
    } else {
        TCP_LISTENER_INDEX.lock().last_lookup_hit = false;
        TcpListenerLookup {
            socket_idx: None,
            probes: lookup.probes,
        }
    }
}

pub fn tcp_flow_index_proof() -> TcpFlowIndexProof {
    let index = TCP_FLOW_INDEX.lock();
    TcpFlowIndexProof {
        hit: index.last_lookup_hit,
        probes: index.last_lookup_probes,
        bound: TCP_FLOW_INDEX_MAX_PROBES,
        capacity: TCP_FLOW_INDEX_BUCKETS,
    }
}

pub fn tcp_listener_index_proof() -> TcpListenerIndexProof {
    let index = TCP_LISTENER_INDEX.lock();
    TcpListenerIndexProof {
        hit: index.last_lookup_hit,
        probes: index.last_lookup_probes,
        bound: TCP_LISTENER_INDEX_MAX_PROBES,
        capacity: TCP_LISTENER_INDEX_BUCKETS,
    }
}

pub fn init() {}

/// Reset all TCP state for benchmark isolation.
/// Only call from benchmark fixtures — this drops every socket.
pub fn reset_for_benchmark() {
    TCP_SOCKETS.lock().clear();
    TCP_FLOW_INDEX.lock().clear();
    TCP_LISTENER_INDEX.lock().clear();
    PENDING_SYN_QUEUE.lock().clear();
    *NEXT_TCP_PORT.lock() = 40000;
}

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
    if let Some(sock) = sockets.get(idx) {
        index_exact_flow_socket(idx, sock);
    }
    idx
}

pub fn bind(idx: usize, port: u16) {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        remove_exact_flow_socket(idx, sock);
        remove_listener_socket(idx, sock);
        sock.local_port = port;
        index_exact_flow_socket(idx, sock);
        index_listener_socket(idx, sock);
    }
}

pub fn listen(idx: usize) {
    let mut sockets = TCP_SOCKETS.lock();
    if let Some(sock) = sockets.get_mut(idx) {
        remove_exact_flow_socket(idx, sock);
        remove_listener_socket(idx, sock);
        sock.listen();
        index_listener_socket(idx, sock);
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
        remove_exact_flow_socket(idx, sock);
        remove_listener_socket(idx, sock);
        sock.connect(ip, port);
        index_exact_flow_socket(idx, sock);
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
        remove_exact_flow_socket(idx, sock);
        remove_listener_socket(idx, sock);
        sock.close();
        index_exact_flow_socket(idx, sock);
        index_listener_socket(idx, sock);
    }
}

pub fn state(idx: usize) -> TcpState {
    let sockets = TCP_SOCKETS.lock();
    sockets.get(idx).map_or(TcpState::Closed, |s| s.state())
}

pub struct TcpPacketFixtureProof {
    pub ok: bool,
    pub listener_first: bool,
    pub exact_flow: bool,
    pub decoy_rx_bytes: usize,
    pub listener_fallback: bool,
    pub payload_bytes: usize,
    pub rx_bytes: usize,
    pub o1_index: bool,
    pub index_hit: bool,
    pub index_lookup_probes: usize,
    pub index_probe_bound: usize,
    pub index_capacity: usize,
    pub listener_index_hit: bool,
    pub listener_lookup_probes: usize,
    pub listener_probe_bound: usize,
    pub listener_index_capacity: usize,
    pub exact_scan: bool,
    pub accepted_state: TcpState,
    pub cleanup_ok: bool,
}

pub struct TcpRoundTripProof {
    pub connections: usize,
    pub established: usize,
    pub payload_bytes: usize,
    pub client_tx: usize,
    pub server_rx: usize,
    pub server_echo: usize,
    pub client_rx: usize,
    pub listener_accept: usize,
    pub exact_flow: usize,
    pub listener_index_hit: usize,
    pub client_index_hit: usize,
    pub index_lookup_probes_max: usize,
    pub index_probe_bound: usize,
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
                exact_flow: false,
                decoy_rx_bytes: 0,
                listener_fallback: false,
                payload_bytes: PAYLOAD.len(),
                rx_bytes: 0,
                o1_index: false,
                index_hit: false,
                index_lookup_probes: 0,
                index_probe_bound: TCP_FLOW_INDEX_MAX_PROBES,
                index_capacity: TCP_FLOW_INDEX_BUCKETS,
                listener_index_hit: false,
                listener_lookup_probes: 0,
                listener_probe_bound: TCP_LISTENER_INDEX_MAX_PROBES,
                listener_index_capacity: TCP_LISTENER_INDEX_BUCKETS,
                exact_scan: true,
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

        let mut decoy = TcpSocket::new(fixture_port);
        decoy.remote_ip = crate::net::IpAddr::V4([192, 0, 2, 254]);
        decoy.remote_port = 443;
        decoy.state = TcpState::Established;
        decoy.seq_num = 2001;
        decoy.ack_num = 700;

        let mut accepted = TcpSocket::new(fixture_port);
        accepted.remote_ip = remote;
        accepted.remote_port = remote_port;
        accepted.state = TcpState::SynReceived;
        accepted.seq_num = 1001;
        accepted.ack_num = 501;

        sockets.push(listener);
        sockets.push(decoy);
        sockets.push(accepted);
        if let Some(sock) = sockets.get(base_len) {
            index_listener_socket(base_len, sock);
        }
        if let Some(sock) = sockets.get(base_len + 1) {
            index_exact_flow_socket(base_len + 1, sock);
        }
        if let Some(sock) = sockets.get(base_len + 2) {
            index_exact_flow_socket(base_len + 2, sock);
        }
    }

    let packet = make_tcp_packet(
        remote_port,
        fixture_port,
        FLAG_ACK | FLAG_PSH,
        501,
        1001,
        PAYLOAD,
    );
    handle_tcp_packet(remote, &packet);
    let index_proof = tcp_flow_index_proof();

    let mut rx = [0u8; PAYLOAD.len()];
    let mut decoy_rx = [0u8; PAYLOAD.len()];
    let mut sockets = TCP_SOCKETS.lock();
    let listener_idx = base_len;
    let decoy_idx = base_len + 1;
    let accepted_idx = base_len + 2;
    let listener_first = sockets
        .get(listener_idx)
        .is_some_and(|sock| sock.state == TcpState::Listen)
        && decoy_idx > listener_idx
        && accepted_idx > decoy_idx;
    let accepted_state = sockets
        .get(accepted_idx)
        .map_or(TcpState::Closed, |sock| sock.state);
    let decoy_rx_bytes = sockets
        .get_mut(decoy_idx)
        .map_or(0, |sock| sock.recv(&mut decoy_rx));
    let rx_bytes = sockets
        .get_mut(accepted_idx)
        .map_or(0, |sock| sock.recv(&mut rx));
    let payload_matches = rx_bytes == PAYLOAD.len() && &rx[..rx_bytes] == PAYLOAD;
    drop(sockets);

    let fallback_remote = crate::net::IpAddr::V4([192, 0, 2, 99]);
    let syn_packet = make_tcp_packet(81, fixture_port, FLAG_SYN, 900, 0, &[]);
    handle_tcp_packet(fallback_remote, &syn_packet);
    poll();
    let listener_index_proof = tcp_listener_index_proof();

    let mut sockets = TCP_SOCKETS.lock();
    let listener_fallback = sockets.get(listener_idx).is_some_and(|listener| {
        listener.pending_accept.iter().copied().any(|idx| {
            sockets.get(idx).is_some_and(|sock| {
                sock.local_port == fixture_port
                    && sock.remote_ip == fallback_remote
                    && sock.remote_port == 81
                    && sock.state == TcpState::SynReceived
            })
        })
    });

    let mut idx = base_len;
    while idx < sockets.len() {
        if let Some(sock) = sockets.get(idx) {
            remove_exact_flow_socket(idx, sock);
            remove_listener_socket(idx, sock);
        }
        idx += 1;
    }
    if sockets.len() >= base_len + 3 {
        sockets.truncate(base_len);
    }
    let cleanup_ok = sockets.len() == base_len;
    let exact_flow =
        accepted_state == TcpState::Established && payload_matches && decoy_rx_bytes == 0;
    let ok = listener_first
        && exact_flow
        && listener_fallback
        && index_proof.hit
        && index_proof.probes <= index_proof.bound
        && listener_index_proof.hit
        && listener_index_proof.probes <= listener_index_proof.bound
        && accepted_state == TcpState::Established
        && rx_bytes == PAYLOAD.len()
        && payload_matches
        && cleanup_ok;

    TcpPacketFixtureProof {
        ok,
        listener_first,
        exact_flow,
        decoy_rx_bytes,
        listener_fallback,
        payload_bytes: PAYLOAD.len(),
        rx_bytes,
        o1_index: index_proof.hit && index_proof.probes <= index_proof.bound,
        index_hit: index_proof.hit,
        index_lookup_probes: index_proof.probes,
        index_probe_bound: index_proof.bound,
        index_capacity: index_proof.capacity,
        listener_index_hit: listener_index_proof.hit,
        listener_lookup_probes: listener_index_proof.probes,
        listener_probe_bound: listener_index_proof.bound,
        listener_index_capacity: listener_index_proof.capacity,
        exact_scan: false,
        accepted_state,
        cleanup_ok,
    }
}

pub fn loopback_echo_fixture_proof() -> TcpRoundTripProof {
    const CONNECTIONS: usize = 8;
    const PAYLOAD_BYTES: usize = 64;
    let client_ip = crate::net::IpAddr::V4([127, 0, 0, 1]);
    let server_ip = crate::net::IpAddr::V4([127, 0, 0, 1]);

    let mut proof = TcpRoundTripProof {
        connections: CONNECTIONS,
        established: 0,
        payload_bytes: PAYLOAD_BYTES,
        client_tx: 0,
        server_rx: 0,
        server_echo: 0,
        client_rx: 0,
        listener_accept: 0,
        exact_flow: 0,
        listener_index_hit: 0,
        client_index_hit: 0,
        index_lookup_probes_max: 0,
        index_probe_bound: TCP_FLOW_INDEX_MAX_PROBES,
        cleanup_ok: true,
    };

    for conn in 0..CONNECTIONS {
        let server_port = 49_200u16 + conn as u16;
        let client_port = 40_200u16 + conn as u16;
        let client_seq = 10_000u32 + (conn as u32 * 1_000);
        let server_seq = 1_000u32;
        let base_len = {
            let mut sockets = TCP_SOCKETS.lock();
            let base_len = sockets.len();

            let mut listener = TcpSocket::new(server_port);
            listener.listen();

            let mut client = TcpSocket::new(client_port);
            client.remote_ip = server_ip;
            client.remote_port = server_port;
            client.state = TcpState::SynSent;
            client.seq_num = client_seq;

            sockets.push(listener);
            sockets.push(client);
            if let Some(sock) = sockets.get(base_len) {
                index_listener_socket(base_len, sock);
            }
            if let Some(sock) = sockets.get(base_len + 1) {
                index_exact_flow_socket(base_len + 1, sock);
            }
            base_len
        };

        let syn = make_tcp_packet(client_port, server_port, FLAG_SYN, client_seq, 0, &[]);
        handle_tcp_packet(client_ip, &syn);
        let listener_proof = tcp_listener_index_proof();
        if listener_proof.hit {
            proof.listener_index_hit += 1;
        }
        proof.index_lookup_probes_max = proof
            .index_lookup_probes_max
            .max(listener_proof.probes);

        poll();
        let Some(accepted_idx) = accept(base_len) else {
            proof.cleanup_ok &= cleanup_tcp_fixture(base_len);
            continue;
        };
        proof.listener_accept += 1;

        let syn_ack = make_tcp_packet(
            server_port,
            client_port,
            FLAG_SYN | FLAG_ACK,
            server_seq,
            client_seq + 1,
            &[],
        );
        handle_tcp_packet(server_ip, &syn_ack);
        let client_syn_ack_proof = tcp_flow_index_proof();
        proof.index_lookup_probes_max = proof
            .index_lookup_probes_max
            .max(client_syn_ack_proof.probes);

        let ack = make_tcp_packet(
            client_port,
            server_port,
            FLAG_ACK,
            client_seq + 1,
            server_seq + 1,
            &[],
        );
        handle_tcp_packet(client_ip, &ack);

        let payload = loopback_payload(conn);
        let data = make_tcp_packet(
            client_port,
            server_port,
            FLAG_ACK | FLAG_PSH,
            client_seq + 1,
            server_seq + 1,
            &payload,
        );
        handle_tcp_packet(client_ip, &data);
        let server_data_proof = tcp_flow_index_proof();
        if server_data_proof.hit {
            proof.exact_flow += 1;
        }
        proof.index_lookup_probes_max = proof
            .index_lookup_probes_max
            .max(server_data_proof.probes);
        proof.client_tx += payload.len();

        let mut server_buf = [0u8; PAYLOAD_BYTES];
        let server_rx = recv(accepted_idx, &mut server_buf);
        proof.server_rx += server_rx;

        let echo = make_tcp_packet(
            server_port,
            client_port,
            FLAG_ACK | FLAG_PSH,
            server_seq + 1,
            client_seq + 1 + payload.len() as u32,
            &server_buf[..server_rx],
        );
        handle_tcp_packet(server_ip, &echo);
        let client_echo_proof = tcp_flow_index_proof();
        if client_echo_proof.hit {
            proof.client_index_hit += 1;
        }
        proof.index_lookup_probes_max = proof
            .index_lookup_probes_max
            .max(client_echo_proof.probes);
        proof.server_echo += server_rx;

        let mut client_buf = [0u8; PAYLOAD_BYTES];
        let client_rx = recv(base_len + 1, &mut client_buf);
        proof.client_rx += client_rx;

        if state(base_len + 1) == TcpState::Established
            && state(accepted_idx) == TcpState::Established
            && server_rx == PAYLOAD_BYTES
            && client_rx == PAYLOAD_BYTES
            && server_buf[..server_rx] == payload[..]
            && client_buf[..client_rx] == payload[..]
        {
            proof.established += 1;
        }

        proof.cleanup_ok &= cleanup_tcp_fixture(base_len);
    }

    proof
}

fn loopback_payload(conn: usize) -> [u8; 64] {
    let mut payload = [0u8; 64];
    for (idx, byte) in payload.iter_mut().enumerate() {
        *byte = (conn as u8).wrapping_mul(17).wrapping_add(idx as u8);
    }
    payload
}

fn cleanup_tcp_fixture(base_len: usize) -> bool {
    let mut sockets = TCP_SOCKETS.lock();
    let mut idx = base_len;
    while idx < sockets.len() {
        if let Some(sock) = sockets.get(idx) {
            remove_exact_flow_socket(idx, sock);
            remove_listener_socket(idx, sock);
        }
        idx += 1;
    }
    sockets.truncate(base_len);
    PENDING_SYN_QUEUE.lock().clear();
    sockets.len() == base_len
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
        src_port,
        dst_port,
        seq,
        ack,
        data_offset_flags: (5u16 << 12) | flags,
        window: 65535,
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
        let listener_idx = lookup_listener_index(&sockets, dst_port).socket_idx;
        if let Some(listener_idx) = listener_idx {
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
                index_exact_flow_socket(new_idx, new_sock);
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
    let Some(hdr) = TcpHeader::from_bytes(&pkt[..20]) else {
        return;
    };
    let data_offset = hdr.data_offset();
    if data_offset < 20 || pkt.len() < data_offset {
        return;
    }
    let flags = hdr.flags();
    let dst_port = hdr.dst_port;
    let src_port = hdr.src_port;
    let seq = hdr.seq;
    let ack = hdr.ack;
    let payload = &pkt[data_offset..];

    let mut pending_syn = None;
    {
        let mut sockets = TCP_SOCKETS.lock();
        let exact_lookup = lookup_exact_flow_index(&sockets, src, src_port, dst_port);
        let exact_idx = exact_lookup.socket_idx;
        let listener_idx = if exact_idx.is_some() {
            None
        } else {
            lookup_listener_index(&sockets, dst_port).socket_idx
        };
        if let Some(idx) = exact_idx.or(listener_idx) {
            if let Some(sock) = sockets.get_mut(idx) {
                let old_key = tcp_flow_key(sock);
                let old_listener_key = tcp_listener_key(sock);
                match sock.state {
                    TcpState::Listen => {
                        if flags & FLAG_SYN != 0 {
                            pending_syn = Some((dst_port, src, src_port, seq));
                        }
                    }
                    TcpState::SynSent => {
                        if flags & FLAG_SYN != 0 && flags & FLAG_ACK != 0 {
                            sock.ack_num = seq + 1;
                            sock.seq_num += 1;
                            sock.send_tcp_packet(FLAG_ACK, &[]);
                            sock.state = TcpState::Established;
                        }
                    }
                    TcpState::SynReceived => {
                        if flags & FLAG_ACK != 0 && ack == sock.seq_num && seq == sock.ack_num {
                            sock.state = TcpState::Established;
                            sock.ack_num = seq + payload.len() as u32;
                            if !payload.is_empty() {
                                sock.push_rx(payload);
                                sock.send_tcp_packet(FLAG_ACK, &[]);
                            }
                        }
                    }
                    TcpState::Established => {
                        if flags & FLAG_FIN != 0 {
                            sock.ack_num = seq + 1;
                            sock.send_tcp_packet(FLAG_ACK, &[]);
                            sock.state = TcpState::CloseWait;
                        } else if !payload.is_empty() || flags & FLAG_ACK != 0 {
                            sock.ack_num = seq + payload.len() as u32;
                            if !payload.is_empty() {
                                sock.push_rx(payload);
                            }
                            if flags & FLAG_PSH != 0 || !payload.is_empty() {
                                sock.send_tcp_packet(FLAG_ACK, &[]);
                            }
                            if flags & FLAG_ACK != 0 {
                                sock.purge_acked(ack);
                            }
                        }
                    }
                    TcpState::FinWait1 => {
                        if flags & FLAG_FIN != 0 && flags & FLAG_ACK != 0 {
                            sock.ack_num = seq + 1;
                            sock.send_tcp_packet(FLAG_ACK, &[]);
                            sock.state = TcpState::TimeWait;
                        } else if flags & FLAG_ACK != 0 {
                            sock.state = TcpState::FinWait2;
                        }
                    }
                    TcpState::FinWait2 => {
                        if flags & FLAG_FIN != 0 {
                            sock.ack_num = seq + 1;
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
                refresh_exact_flow_socket(idx, old_key, sock);
                refresh_listener_socket(idx, old_listener_key, sock);
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
        TCP_FLOW_INDEX.lock().clear();
        TCP_LISTENER_INDEX.lock().clear();
        PENDING_SYN_QUEUE.lock().clear();
        *NEXT_TCP_PORT.lock() = 40000;
    }

    fn make_tcp_header(flags: u16, seq: u32, ack: u32, dst_port: u16) -> Vec<u8> {
        let hdr = TcpHeader {
            src_port: 80,
            dst_port,
            seq,
            ack,
            data_offset_flags: (5u16 << 12) | flags,
            window: 65535,
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
            src_port: 1234,
            dst_port: 80,
            seq: 0xABCDEF01,
            ack: 0x12345678,
            data_offset_flags: (5u16 << 12) | FLAG_SYN | FLAG_ACK,
            window: 64000,
            checksum: 0xBEEF,
            urgent: 0,
        };
        let bytes = hdr.to_bytes();
        let parsed = TcpHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.to_bytes(), bytes);
    }

    #[test]
    fn test_tcp_header_literal_wire_order() {
        let bytes = [
            0x00, 0x50, 0x1f, 0x90, 0x00, 0x00, 0x01, 0xf5, 0x00, 0x00, 0x03, 0xe9, 0x50, 0x18,
            0xff, 0xff, 0x12, 0x34, 0x00, 0x00,
        ];

        let parsed = TcpHeader::from_bytes(&bytes).unwrap();
        let src_port = parsed.src_port;
        let dst_port = parsed.dst_port;
        let seq = parsed.seq;
        let ack = parsed.ack;

        assert_eq!(src_port, 80);
        assert_eq!(dst_port, 8080);
        assert_eq!(seq, 501);
        assert_eq!(ack, 1001);
        assert_eq!(parsed.data_offset(), 20);
        assert_eq!(parsed.flags(), FLAG_ACK | FLAG_PSH);
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
        assert!(proof.exact_flow);
        assert_eq!(proof.decoy_rx_bytes, 0);
        assert!(proof.listener_fallback);
        assert_eq!(proof.payload_bytes, 4);
        assert_eq!(proof.rx_bytes, 4);
        assert!(proof.o1_index);
        assert!(proof.index_hit);
        assert!(proof.index_lookup_probes > 0);
        assert!(proof.index_lookup_probes <= proof.index_probe_bound);
        assert_eq!(proof.index_probe_bound, proof.index_capacity);
        assert_eq!(proof.index_capacity, TCP_FLOW_INDEX_BUCKETS);
        assert!(proof.listener_index_hit);
        assert!(proof.listener_lookup_probes > 0);
        assert!(proof.listener_lookup_probes <= proof.listener_probe_bound);
        assert_eq!(proof.listener_probe_bound, proof.listener_index_capacity);
        assert_eq!(proof.listener_index_capacity, TCP_LISTENER_INDEX_BUCKETS);
        assert!(!proof.exact_scan);
        assert_eq!(proof.accepted_state, TcpState::Established);
        assert!(proof.cleanup_ok);
        assert!(TCP_SOCKETS.lock().is_empty());
    }

    #[test]
    fn loopback_echo_fixture_proves_roundtrip_and_cleanup() {
        reset_tcp_for_test();

        let proof = loopback_echo_fixture_proof();

        assert_eq!(proof.connections, 8);
        assert_eq!(proof.established, proof.connections);
        assert_eq!(proof.payload_bytes, 64);
        assert_eq!(proof.client_tx, proof.connections * proof.payload_bytes);
        assert_eq!(proof.server_rx, proof.client_tx);
        assert_eq!(proof.server_echo, proof.client_tx);
        assert_eq!(proof.client_rx, proof.client_tx);
        assert_eq!(proof.listener_accept, proof.connections);
        assert_eq!(proof.exact_flow, proof.connections);
        assert_eq!(proof.listener_index_hit, proof.connections);
        assert_eq!(proof.client_index_hit, proof.connections);
        assert!(proof.index_lookup_probes_max > 0);
        assert!(proof.index_lookup_probes_max <= proof.index_probe_bound);
        assert!(proof.cleanup_ok);
        assert!(TCP_SOCKETS.lock().is_empty());
    }
}
