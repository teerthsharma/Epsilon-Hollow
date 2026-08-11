// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Minimal TCP state machine with retransmission timer.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
                &core::ptr::addr_of!(self.src_port)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[2..4].copy_from_slice(
                &core::ptr::addr_of!(self.dst_port)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[4..8].copy_from_slice(&core::ptr::addr_of!(self.seq).read_unaligned().to_be_bytes());
            b[8..12].copy_from_slice(&core::ptr::addr_of!(self.ack).read_unaligned().to_be_bytes());
            b[12..14].copy_from_slice(
                &core::ptr::addr_of!(self.data_offset_flags)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[14..16].copy_from_slice(
                &core::ptr::addr_of!(self.window)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[16..18].copy_from_slice(
                &core::ptr::addr_of!(self.checksum)
                    .read_unaligned()
                    .to_be_bytes(),
            );
            b[18..20].copy_from_slice(
                &core::ptr::addr_of!(self.urgent)
                    .read_unaligned()
                    .to_be_bytes(),
            );
        }
        b
    }
}

const FLAG_FIN: u16 = 1;
const FLAG_SYN: u16 = 2;
const FLAG_RST: u16 = 4;
const FLAG_PSH: u16 = 8;
const FLAG_ACK: u16 = 16;

/// Retransmit timeout a fresh socket starts with, in timer ticks.
const RTO_INITIAL: u64 = 1000;

/// Ceiling the exponential backoff stops at, in timer ticks. A peer that never
/// answers is retried at this fixed slow rate rather than forever faster.
const RTO_MAX: u64 = 64_000;

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
            rto: RTO_INITIAL,
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
                // The address `send_ipv4_packet` will stamp, which is 127.0.0.1
                // on the loopback branch and not `local_ip()`. Summing
                // `local_ip()` built a segment that verified against an address
                // the datagram never carried, and `handle_tcp_packet` below now
                // checks the one it does. See `ipv4::source_for`.
                let src_ip = crate::net::ipv4::source_for(remote_ip);
                let mut segment = Vec::with_capacity(20 + payload.len());
                segment.extend_from_slice(&hdr.to_bytes());
                segment.extend_from_slice(payload);
                let cksum = crate::net::ipv4::transport_checksum(
                    crate::net::IpAddr::V4(src_ip),
                    crate::net::IpAddr::V4(remote_ip),
                    6,
                    &segment,
                )
                .unwrap_or(0);
                unsafe {
                    core::ptr::addr_of_mut!(hdr.checksum).write(cksum);
                }
                let mut pkt = Vec::with_capacity(20 + payload.len());
                pkt.extend_from_slice(&hdr.to_bytes());
                pkt.extend_from_slice(payload);
                queue_tx(crate::net::IpAddr::V4(remote_ip), pkt);
            }
            crate::net::IpAddr::V6(remote_ip) => {
                // `send_ipv6_packet` stamps `local_ip_v6()` on every datagram,
                // its `::1` branch included, so no `source_for` counterpart is
                // needed: the address it will stamp is this one.
                let src_ip = crate::net::local_ip_v6();
                let mut segment = Vec::with_capacity(20 + payload.len());
                segment.extend_from_slice(&hdr.to_bytes());
                segment.extend_from_slice(payload);
                let cksum = crate::net::ipv4::transport_checksum(
                    crate::net::IpAddr::V6(src_ip),
                    crate::net::IpAddr::V6(remote_ip),
                    6,
                    &segment,
                )
                .unwrap_or(0);
                unsafe {
                    core::ptr::addr_of_mut!(hdr.checksum).write(cksum);
                }
                let mut pkt = Vec::with_capacity(20 + payload.len());
                pkt.extend_from_slice(&hdr.to_bytes());
                pkt.extend_from_slice(payload);
                queue_tx(crate::net::IpAddr::V6(remote_ip), pkt);
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

    /// Whether an incoming reset is acceptable, per RFC 793 section 3.4.
    ///
    /// In SYN-SENT nothing has been received, so the only test available is
    /// whether the reset acknowledges our SYN: `connect` leaves `seq_num` at the
    /// initial send sequence and the SYN occupies it, so that acknowledgement is
    /// `seq_num + 1`. Every synchronised state instead requires the reset to
    /// land on the next byte expected. A reset that fails either test is
    /// dropped, so guessing the four-tuple is not on its own enough to tear a
    /// connection down.
    fn accepts_reset(&self, flags: u16, seq: u32, ack: u32) -> bool {
        match self.state {
            TcpState::Closed | TcpState::Listen => false,
            TcpState::SynSent => flags & FLAG_ACK != 0 && ack == self.seq_num.wrapping_add(1),
            _ => seq == self.ack_num,
        }
    }

    /// Abort the connection without disturbing the socket table.
    ///
    /// `accept` hands out indices into `TCP_SOCKETS` and both demux indexes
    /// store them, so removing the element would silently retarget every handle
    /// a caller still holds at whatever socket slid into the slot. The socket
    /// becomes `Closed` in place instead, which `tcp_flow_key` and
    /// `tcp_listener_key` report as unkeyable, so the `refresh_*` calls at the
    /// end of `handle_tcp_packet` drop it out of both indexes. A caller that was
    /// polling `state(idx)` sees `Closed` rather than waiting on `SynSent`
    /// forever. Bytes already delivered stay in `rx_buffer` and remain readable.
    fn abort(&mut self) {
        self.state = TcpState::Closed;
        self.retransmit_queue.clear();
        self.rto = RTO_INITIAL;
    }

    fn purge_acked(&mut self, ack: u32) {
        let before = self.retransmit_queue.len();
        self.retransmit_queue.retain(|e| {
            let end = e.seq + e.payload.len() as u32;
            end > ack || (ack < 1000 && end < 1000)
        });
        if self.retransmit_queue.len() != before {
            // Forward progress, so the backoff accumulated while the peer was
            // silent is spent: the next loss is retried promptly.
            self.rto = RTO_INITIAL;
        }
    }

    /// Retransmit the oldest segment whose timer has expired.
    ///
    /// The entry is taken off the queue before `send_tcp_packet` puts it back,
    /// so the queue holds one entry per unacknowledged segment however many
    /// times that segment is retransmitted. It previously grew by one on every
    /// expiry -- the condition that queues a segment is the same one that put it
    /// there, so the resend always queued a second copy -- and the original kept
    /// its old timestamp, so the next `poll` picked it straight back up and the
    /// connection retransmitted at poll rate rather than at the RTO. A
    /// `connect` to a host that never answers grew that queue without bound.
    ///
    /// `seq_num` is restored afterwards. Leaving it at the retransmitted
    /// segment's sequence rewound the connection permanently, and the next
    /// `send` then reused those numbers for different bytes.
    ///
    /// The re-push is unconditional in effect: `send_tcp_packet` queues when the
    /// payload is non-empty or the segment is SYN or FIN, which is exactly what
    /// admitted this entry in the first place.
    fn retransmit_expired(&mut self) {
        let now = crate::drivers::interrupts::ticks();
        let Some(pos) = self
            .retransmit_queue
            .iter()
            .position(|entry| now.wrapping_sub(entry.time) >= self.rto)
        else {
            return;
        };
        // ponytail: remove(pos) is O(n) on a queue that holds single digits,
        // the same ceiling the flush loop already carries.
        let entry = self.retransmit_queue.remove(pos);
        let next_seq = self.seq_num;
        self.seq_num = entry.seq;
        self.send_tcp_packet(entry.flags, &entry.payload);
        self.seq_num = next_seq;
        self.rto = self.rto.saturating_mul(2).min(RTO_MAX);
    }
}

static TCP_SOCKETS: Mutex<Vec<TcpSocket>> = Mutex::new(Vec::new());
static TCP_FLOW_INDEX: Mutex<TcpFlowIndex> = Mutex::new(TcpFlowIndex::new());
static TCP_LISTENER_INDEX: Mutex<TcpListenerIndex> = Mutex::new(TcpListenerIndex::new());
/// Lowest and highest port `socket` hands out, inclusive.
///
/// The counter is confined to this range and wraps from the ceiling back to the
/// floor. It used to be a bare `*p += 1` on a `u16` starting at 40000, so
/// allocation 25,536 left the range entirely: 65535 + 1 is 0 in release and an
/// abort under `overflow-checks`, and the counter then walked 0, 1, 2 ... up
/// through the well-known range -- handing out 22, 80 and 443 to outgoing
/// connections, and colliding with any listener bound there.
const EPHEMERAL_PORT_MIN: u16 = 40_000;
const EPHEMERAL_PORT_MAX: u16 = u16::MAX;

static NEXT_TCP_PORT: Mutex<u16> = Mutex::new(EPHEMERAL_PORT_MIN);
static PENDING_SYN_QUEUE: Mutex<Vec<(u16, crate::net::IpAddr, u16, u32)>> = Mutex::new(Vec::new());

/// Segments TCP has built but not yet handed to the IP layer.
///
/// Every one of `send_tcp_packet`'s callers holds `TCP_SOCKETS` across the
/// call: `poll` takes it before creating the accepted socket and still holds it
/// at the SYN-ACK, all seven state arms of `handle_tcp_packet` hold it, and so
/// do the `connect` / `send` / `close` wrappers. `ipv4::send_ipv4_packet`
/// hands a 127.0.0.1 destination straight to `handle_ipv4_packet`, which
/// dispatches protocol 6 back into `handle_tcp_packet`, which locks
/// `TCP_SOCKETS` a second time. That mutex is a `spin::Mutex` -- non-reentrant,
/// no interrupt masking -- so the second acquire spins forever: no panic, no
/// fault report, the boot thread simply stops while the timer keeps ticking.
///
/// Before the checksum store sites were fixed, the loopback header failed
/// `handle_ipv4_packet`'s verifier and the path returned before reaching TCP,
/// so the stall was masked rather than absent. The loopback branch had never
/// once carried a segment.
///
/// Building the segment here and transmitting it from `flush_tx` once the guard
/// has dropped removes the re-entry at the single chokepoint every send routes
/// through, rather than at each of the ten call sites -- and a call site that
/// forgets to flush loses nothing, because `poll` flushes unconditionally.
static TCP_TX_QUEUE: Mutex<Vec<(crate::net::IpAddr, Vec<u8>)>> = Mutex::new(Vec::new());

/// Set while `flush_tx` owns the drain. A segment delivered over loopback can
/// make `handle_tcp_packet` queue a reply and call `flush_tx` again from inside
/// the outer flush; the flag sends that inner call straight back so the reply
/// is picked up by the outer loop's next iteration. Re-entry depth is therefore
/// one `handle_ipv4_packet` frame by construction, not by how the TCP state
/// machine happens to reply.
static TX_FLUSHING: AtomicBool = AtomicBool::new(false);

static TCP_TX_DROPPED: AtomicUsize = AtomicUsize::new(0);

/// Segments allowed to sit unflushed. Reached only if something queues without
/// ever reaching a flush point, so it is a memory ceiling, not a rate limit.
pub const TCP_TX_QUEUE_LIMIT: usize = 256;

/// Segments one `flush_tx` may transmit before it stops. A loopback exchange
/// that answers every segment with another would otherwise keep the drain loop
/// fed indefinitely; at the budget the remaining segments are dropped and
/// counted rather than transmitted.
pub const TCP_TX_FLUSH_BUDGET: usize = 256;

fn queue_tx(dst: crate::net::IpAddr, pkt: Vec<u8>) {
    let mut queue = TCP_TX_QUEUE.lock();
    if queue.len() >= TCP_TX_QUEUE_LIMIT {
        TCP_TX_DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    queue.push((dst, pkt));
}

/// Hand every queued segment to the IP layer. Callers must not hold
/// `TCP_SOCKETS`: loopback delivery re-enters `handle_tcp_packet`, which takes
/// it.
pub fn flush_tx() {
    if TX_FLUSHING.swap(true, Ordering::Acquire) {
        return;
    }
    let mut budget = TCP_TX_FLUSH_BUDGET;
    loop {
        // ponytail: remove(0) is O(n) on a queue that holds single digits in
        // every observed exchange. A ring buffer if that ever stops being true.
        let next = {
            let mut queue = TCP_TX_QUEUE.lock();
            if queue.is_empty() {
                None
            } else {
                Some(queue.remove(0))
            }
        };
        let Some((dst, pkt)) = next else {
            break;
        };
        if budget == 0 {
            TCP_TX_DROPPED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        budget -= 1;
        match dst {
            crate::net::IpAddr::V4(ip) => crate::net::ipv4::send_ipv4_packet(ip, 6, &pkt),
            crate::net::IpAddr::V6(ip) => crate::net::ipv6::send_ipv6_packet(ip, 6, &pkt),
        }
    }
    TX_FLUSHING.store(false, Ordering::Release);
}

/// Segments dropped by either transmit ceiling since boot.
pub fn tcp_tx_dropped() -> usize {
    TCP_TX_DROPPED.load(Ordering::Relaxed)
}

/// Segments built but not yet transmitted.
pub fn tcp_tx_queued() -> usize {
    TCP_TX_QUEUE.lock().len()
}

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
    TCP_TX_QUEUE.lock().clear();
    *NEXT_TCP_PORT.lock() = EPHEMERAL_PORT_MIN;
}

/// The handle `socket` returns when it could not allocate one.
///
/// No `Vec` can hold `usize::MAX` elements, so this indexes nothing: `bind`,
/// `listen`, `accept`, `connect`, `send`, `recv` and `close` all resolve
/// through `get`/`get_mut` and do nothing with it, and `state` reports
/// `Closed`. A caller that never checks therefore fails closed -- its
/// connection simply never establishes -- and one that wants to say why can
/// compare against this.
pub const TCP_SOCKET_NONE: usize = usize::MAX;

/// The next free port in `EPHEMERAL_PORT_MIN..=EPHEMERAL_PORT_MAX`, or `None`
/// if every port in that range is held by a socket in `sockets`.
///
/// A wrapped counter eventually names a port that is still in use, and nothing
/// else refuses one: `bind` writes any port over a socket's own, both indexes
/// happily hold two sockets on one local port, and the exact-flow lookup then
/// resolves a segment to whichever of them was indexed last. So the counter
/// skips what is taken -- the same "free port" test `packet_fixture_proof`
/// already applies before it seizes one.
///
/// The search is bounded by the width of the range, so a full table returns
/// rather than spinning. The counter is left one past the port handed out,
/// which is why an exhausted range still leaves it inside the range.
///
/// ponytail: the in-use test is a linear scan and the caller holds
/// `TCP_SOCKETS` across it, so allocation is O(sockets) per candidate on a
/// table that holds single digits in every observed exchange -- the same
/// ceiling `packet_fixture_proof`'s own port search carries. A port bitmap if a
/// host ever holds thousands of sockets.
fn alloc_ephemeral_port(sockets: &[TcpSocket]) -> Option<u16> {
    let mut p = NEXT_TCP_PORT.lock();
    for _ in 0..=(EPHEMERAL_PORT_MAX - EPHEMERAL_PORT_MIN) {
        let port = *p;
        // `port + 1` cannot overflow: the branch above takes the ceiling, and
        // the ceiling is the largest port there is.
        *p = if port == EPHEMERAL_PORT_MAX {
            EPHEMERAL_PORT_MIN
        } else {
            port + 1
        };
        if !sockets.iter().any(|sock| sock.local_port == port) {
            return Some(port);
        }
    }
    None
}

/// Open an unbound socket, or `TCP_SOCKET_NONE` if no ephemeral port is free.
pub fn socket() -> usize {
    let mut sockets = TCP_SOCKETS.lock();
    let Some(port) = alloc_ephemeral_port(&sockets) else {
        return TCP_SOCKET_NONE;
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
    {
        let mut sockets = TCP_SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(idx) {
            remove_exact_flow_socket(idx, sock);
            remove_listener_socket(idx, sock);
            sock.connect(ip, port);
            index_exact_flow_socket(idx, sock);
        }
    }
    flush_tx();
}

pub fn send(idx: usize, buf: &[u8]) {
    {
        let mut sockets = TCP_SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(idx) {
            sock.send(buf);
        }
    }
    flush_tx();
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
    {
        let mut sockets = TCP_SOCKETS.lock();
        if let Some(sock) = sockets.get_mut(idx) {
            remove_exact_flow_socket(idx, sock);
            remove_listener_socket(idx, sock);
            sock.close();
            index_exact_flow_socket(idx, sock);
            index_listener_socket(idx, sock);
        }
    }
    flush_tx();
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
    // The address this fixture's segments are addressed to. A literal rather
    // than `local_ip()`: the demux ignores the destination but the checksum does
    // not, and a benchmark that read an address DHCP can change underneath it
    // would not be reproducible.
    let local = crate::net::IpAddr::V4([192, 0, 2, 2]);
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
        remote,
        local,
        remote_port,
        fixture_port,
        FLAG_ACK | FLAG_PSH,
        501,
        1001,
        PAYLOAD,
    );
    handle_tcp_packet(remote, local, &packet);
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
    let syn_packet = make_tcp_packet(
        fallback_remote,
        local,
        81,
        fixture_port,
        FLAG_SYN,
        900,
        0,
        &[],
    );
    handle_tcp_packet(fallback_remote, local, &syn_packet);
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

        let syn = make_tcp_packet(
            client_ip,
            server_ip,
            client_port,
            server_port,
            FLAG_SYN,
            client_seq,
            0,
            &[],
        );
        handle_tcp_packet(client_ip, server_ip, &syn);
        let listener_proof = tcp_listener_index_proof();
        if listener_proof.hit {
            proof.listener_index_hit += 1;
        }
        proof.index_lookup_probes_max = proof.index_lookup_probes_max.max(listener_proof.probes);

        poll();
        let Some(accepted_idx) = accept(base_len) else {
            proof.cleanup_ok &= cleanup_tcp_fixture(base_len);
            continue;
        };
        proof.listener_accept += 1;

        let syn_ack = make_tcp_packet(
            server_ip,
            client_ip,
            server_port,
            client_port,
            FLAG_SYN | FLAG_ACK,
            server_seq,
            client_seq + 1,
            &[],
        );
        handle_tcp_packet(server_ip, client_ip, &syn_ack);
        let client_syn_ack_proof = tcp_flow_index_proof();
        proof.index_lookup_probes_max = proof
            .index_lookup_probes_max
            .max(client_syn_ack_proof.probes);

        let ack = make_tcp_packet(
            client_ip,
            server_ip,
            client_port,
            server_port,
            FLAG_ACK,
            client_seq + 1,
            server_seq + 1,
            &[],
        );
        handle_tcp_packet(client_ip, server_ip, &ack);

        let payload = loopback_payload(conn);
        let data = make_tcp_packet(
            client_ip,
            server_ip,
            client_port,
            server_port,
            FLAG_ACK | FLAG_PSH,
            client_seq + 1,
            server_seq + 1,
            &payload,
        );
        handle_tcp_packet(client_ip, server_ip, &data);
        let server_data_proof = tcp_flow_index_proof();
        if server_data_proof.hit {
            proof.exact_flow += 1;
        }
        proof.index_lookup_probes_max = proof.index_lookup_probes_max.max(server_data_proof.probes);
        proof.client_tx += payload.len();

        let mut server_buf = [0u8; PAYLOAD_BYTES];
        let server_rx = recv(accepted_idx, &mut server_buf);
        proof.server_rx += server_rx;

        let echo = make_tcp_packet(
            server_ip,
            client_ip,
            server_port,
            client_port,
            FLAG_ACK | FLAG_PSH,
            server_seq + 1,
            client_seq + 1 + payload.len() as u32,
            &server_buf[..server_rx],
        );
        handle_tcp_packet(server_ip, client_ip, &echo);
        let client_echo_proof = tcp_flow_index_proof();
        if client_echo_proof.hit {
            proof.client_index_hit += 1;
        }
        proof.index_lookup_probes_max = proof.index_lookup_probes_max.max(client_echo_proof.probes);
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
    // Segments addressed to the sockets just dropped must not reach the next
    // connection's freshly indexed sockets on the same ports.
    TCP_TX_QUEUE.lock().clear();
    sockets.len() == base_len
}

/// A segment as it arrives at `handle_tcp_packet`, carrying the checksum a real
/// sender computes over the pseudo-header for `src` -> `dst`.
///
/// The addresses are parameters rather than `local_ip()` so the fixtures stay
/// deterministic and so each caller checksums against the pair it then delivers
/// under: `handle_tcp_packet` verifies against the addresses it is handed, and a
/// fixture that summed a different pair would be dropped before the demux.
fn make_tcp_packet(
    src: crate::net::IpAddr,
    dst: crate::net::IpAddr,
    src_port: u16,
    dst_port: u16,
    flags: u16,
    seq: u32,
    ack: u32,
    payload: &[u8],
) -> Vec<u8> {
    let mut hdr = TcpHeader {
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
    let cksum = crate::net::ipv4::transport_checksum(src, dst, 6, &packet).unwrap_or(0);
    hdr.checksum = cksum;
    packet[..20].copy_from_slice(&hdr.to_bytes());
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
            // An accepted socket answers on the port the segment was addressed
            // to, so it needs no ephemeral port. It used to take one anyway and
            // overwrite it on the next line: the number went nowhere, but the
            // counter moved once per SYN a remote peer sent, which is how a
            // peer that never opened a connection here drove it to its ceiling.
            let new_idx = sockets.len();
            let mut new_sock = TcpSocket::new(dst_port);
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

    {
        let mut sockets = TCP_SOCKETS.lock();
        for sock in sockets.iter_mut() {
            if !sock.retransmit_queue.is_empty() {
                sock.retransmit_expired();
            }
        }
    }
    flush_tx();
}

/// `dst` is the destination the IP header carried. It is summed into the
/// pseudo-header below, and it is not interchangeable with `local_ip()`: a
/// segment addressed to any other address this host answers for would be
/// refused, and over IPv4 `local_ip()` is 0.0.0.0 until DHCP completes.
pub fn handle_tcp_packet(src: crate::net::IpAddr, dst: crate::net::IpAddr, pkt: &[u8]) {
    if pkt.len() < 20 {
        return;
    }
    // RFC 793 section 3.1: the checksum covers the pseudo-header, the header and
    // the data, and unlike UDP over IPv4 it is mandatory in both families --
    // there is no "not computed" encoding to admit. Nothing else on this path
    // checks a byte of the segment: the IPv4 header checksum covers the header
    // alone and IPv6 has none at all, so without this an attacker who could
    // corrupt a segment in flight, or a link that flipped a bit, drove the state
    // machine below on whatever sequence numbers and flags came out.
    //
    // Checked before the header is parsed for anything but its length, so no
    // field of an unverified segment reaches the demux.
    if crate::net::ipv4::transport_checksum(src, dst, 6, pkt) != Some(0) {
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
                if flags & FLAG_RST != 0 {
                    // A reset is never processed as anything else, acceptable
                    // or not. Before this, a bare reset at ESTABLISHED did
                    // nothing at all -- the data arm below needs an ACK flag or
                    // a payload and a reset carries neither -- while a RST+ACK
                    // fell into that arm and was treated as an ordinary
                    // acknowledgement, moving `ack_num` to wherever it pointed
                    // and leaving the connection up.
                    if sock.accepts_reset(flags, seq, ack) {
                        sock.abort();
                    }
                } else {
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
                                // The SYN this answers is still on the
                                // retransmit queue; only the ESTABLISHED arm
                                // used to purge, so it survived the handshake
                                // and was resent one RTO into a connection that
                                // was already up.
                                sock.purge_acked(ack);
                            }
                        }
                        TcpState::SynReceived => {
                            if flags & FLAG_ACK != 0 && ack == sock.seq_num && seq == sock.ack_num {
                                sock.state = TcpState::Established;
                                sock.ack_num = seq + payload.len() as u32;
                                sock.purge_acked(ack);
                                if !payload.is_empty() {
                                    sock.push_rx(payload);
                                    sock.send_tcp_packet(FLAG_ACK, &[]);
                                }
                            }
                        }
                        TcpState::Established => {
                            // RFC 793 section 3.9, "first check sequence
                            // number". The SYN-RECEIVED arm above already
                            // required this and this one did not, so a
                            // duplicate retransmission from an ordinary peer
                            // appended its payload a second time, and anything
                            // that knew the four-tuple could inject at any
                            // sequence at all.
                            //
                            // The test is strict equality against the next byte
                            // expected, matching that sibling rather than the
                            // full `RCV.NXT <= SEG.SEQ < RCV.NXT + RCV.WND`
                            // window: a segment that arrives out of order is
                            // dropped, not held, because there is no reassembly
                            // queue to hold it in, and the peer retransmits it
                            // in order. Nothing is acknowledged on the drop --
                            // RFC 793 asks for an ACK there, but two peers each
                            // holding a stale sequence would then trade
                            // acknowledgements without end, and it hands an
                            // off-path prober a reply to measure.
                            if seq == sock.ack_num {
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
                        TcpState::Closing => {
                            if flags & FLAG_ACK != 0 {
                                sock.state = TcpState::TimeWait;
                            }
                        }
                        TcpState::TimeWait => {
                            // RFC 793: remain in TIME-WAIT for 2*MSL; ignore further data
                        }
                        TcpState::Closed => {
                            // No-op: closed sockets ignore all incoming packets
                        }
                    }
                }
                refresh_exact_flow_socket(idx, old_key, sock);
                refresh_listener_socket(idx, old_listener_key, sock);
            }
        }
    }
    if let Some(syn) = pending_syn {
        PENDING_SYN_QUEUE.lock().push(syn);
    }
    // The `TCP_SOCKETS` guard above has dropped, so a reply this packet
    // produced can now go out -- including back through loopback, which will
    // re-enter here and need that same lock.
    flush_tx();
}

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// See net/udp.rs's own test-module header for why: `kernel/seal-os` is excluded
// from the workspace, so `cargo test --workspace` never reaches it, and the
// crate does not build under `cfg(test)` on a host either. Every case below
// spent its life in a `#[cfg(test)] mod tests` that had never once executed.
// `testing/runner.rs` calls `crate::net::tcp::tests::register_all()`, so they
// now run under the QEMU boot proof in CI.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// Every case runs against one process-wide socket table, so each starts
    /// from an empty one rather than from whatever its predecessor left behind.
    ///
    /// `test_tcp_listen_accept` is why. It opens a listener on 8080 and drives a
    /// SYN from `REMOTE`:80 at it, and the case before it leaves an ESTABLISHED
    /// socket on exactly that four-tuple in the flow index. The exact-flow
    /// lookup runs before the listener fallback and answered with the stale
    /// socket, so the listener never saw the SYN and `accept` returned `None`.
    /// Run on its own the same case passed.
    fn reset_tcp_for_test() {
        TCP_SOCKETS.lock().clear();
        TCP_FLOW_INDEX.lock().clear();
        TCP_LISTENER_INDEX.lock().clear();
        PENDING_SYN_QUEUE.lock().clear();
        TCP_TX_QUEUE.lock().clear();
        *NEXT_TCP_PORT.lock() = EPHEMERAL_PORT_MIN;
    }

    /// Push an already-configured socket and index its flow, the way `socket`
    /// and `poll` do.
    ///
    /// A socket that is only pushed is unreachable: `handle_tcp_packet` resolves
    /// through `TCP_FLOW_INDEX` and never scans the table. Indexing alone is not
    /// enough either -- the key comes from the socket's own four-tuple, so a
    /// socket still holding `TcpSocket::new`'s default 0.0.0.0 `remote_ip` is
    /// filed under an address no segment carries and the lookup misses just the
    /// same. Both halves were missing from the four cases that failed; setting
    /// `remote_ip` to the address the fixture delivers from is as load-bearing
    /// as this call.
    fn push_indexed(sock: TcpSocket) -> usize {
        let mut sockets = TCP_SOCKETS.lock();
        let idx = sockets.len();
        sockets.push(sock);
        index_exact_flow_socket(idx, &sockets[idx]);
        idx
    }

    /// Age every queued segment by `by` ticks.
    ///
    /// `retransmit_expired` tests `ticks() - entry.time` against the RTO, so
    /// rewinding the entries is the same experiment as advancing the clock --
    /// and the clock cannot be advanced from here:
    /// `interrupts::mock_advance_ticks` is `#[cfg(test)]`, which this module is
    /// not under `test-mode`. It is also the more honest of the two in kernel,
    /// where `ticks()` is a live ~1 kHz counter that no test may rewind.
    fn age_retransmit_queue(sock: &mut TcpSocket, by: u64) {
        for entry in sock.retransmit_queue.iter_mut() {
            entry.time = entry.time.wrapping_sub(by);
        }
    }

    /// Every case below delivers from `REMOTE` to `LOCAL`, and
    /// `handle_tcp_packet` verifies the checksum against the pair it is handed,
    /// so the fixture is checksummed against the same pair.
    const REMOTE: crate::net::IpAddr = crate::net::IpAddr::V4([192, 168, 1, 1]);
    const LOCAL: crate::net::IpAddr = crate::net::IpAddr::V4([192, 168, 1, 2]);

    fn make_tcp_header(flags: u16, seq: u32, ack: u32, dst_port: u16) -> Vec<u8> {
        make_tcp_packet(REMOTE, LOCAL, 80, dst_port, flags, seq, ack, &[])
    }

    /// `make_tcp_header` with a payload. A segment's checksum covers its data,
    /// so appending bytes to an already-stamped header leaves it wrong.
    fn make_tcp_segment(flags: u16, seq: u32, ack: u32, dst_port: u16, payload: &[u8]) -> Vec<u8> {
        make_tcp_packet(REMOTE, LOCAL, 80, dst_port, flags, seq, ack, payload)
    }

    fn test_syn_sent_to_established() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(1234);
        sock.remote_ip = REMOTE;
        sock.remote_port = 80;
        sock.state = TcpState::SynSent;
        let pkt = make_tcp_header(FLAG_SYN | FLAG_ACK, 500, 1001, 1234);
        let idx = push_indexed(sock);

        handle_tcp_packet(REMOTE, LOCAL, &pkt);

        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets[idx].state == TcpState::Established,
            "a SYN-ACK in SYN-SENT did not complete the handshake"
        );
        test_assert!(
            sockets[idx].ack_num == 501,
            "SYN-SENT did not acknowledge the peer's SYN sequence + 1"
        );
        test_assert!(
            sockets[idx].seq_num == 1001,
            "SYN-SENT did not consume the sequence its own SYN occupied"
        );
        TestResult::Pass
    }

    /// `ack_num` is the next byte this socket expects, and the ESTABLISHED arm
    /// requires the segment to land on it (RFC 793 section 3.9). The fixture
    /// therefore expects byte 500, which is the sequence the FIN carries.
    fn test_established_to_closewait() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(1234);
        sock.remote_ip = REMOTE;
        sock.remote_port = 80;
        sock.state = TcpState::Established;
        sock.ack_num = 500;
        let pkt = make_tcp_header(FLAG_FIN | FLAG_ACK, 500, 1000, 1234);
        let idx = push_indexed(sock);

        handle_tcp_packet(REMOTE, LOCAL, &pkt);

        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets[idx].state == TcpState::CloseWait,
            "a FIN at ESTABLISHED did not move the socket to CLOSE-WAIT"
        );
        test_assert!(
            sockets[idx].ack_num == 501,
            "the FIN was not acknowledged: a FIN occupies one sequence number"
        );
        TestResult::Pass
    }

    fn test_fin_wait1_to_timewait() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(1234);
        sock.remote_ip = REMOTE;
        sock.remote_port = 80;
        sock.state = TcpState::FinWait1;
        let pkt = make_tcp_header(FLAG_FIN | FLAG_ACK, 500, 1000, 1234);
        let idx = push_indexed(sock);

        handle_tcp_packet(REMOTE, LOCAL, &pkt);

        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets[idx].state == TcpState::TimeWait,
            "a simultaneous FIN-ACK in FIN-WAIT-1 did not reach TIME-WAIT"
        );
        test_assert!(
            sockets[idx].ack_num == 501,
            "the peer's FIN was not acknowledged in FIN-WAIT-1"
        );
        TestResult::Pass
    }

    fn test_close_to_syn_sent() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(1234);
        test_assert!(
            sock.state == TcpState::Closed,
            "a fresh socket is not CLOSED"
        );
        sock.connect(crate::net::IpAddr::V4([192, 168, 1, 1]), 80);
        test_assert!(
            sock.state == TcpState::SynSent,
            "connect did not leave the socket in SYN-SENT"
        );
        TestResult::Pass
    }

    fn test_retransmit_queue() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(1234);
        sock.remote_port = 80;
        sock.remote_ip = crate::net::IpAddr::V4([127, 0, 0, 1]);
        sock.seq_num = 1000;
        sock.state = TcpState::Established;
        sock.send(b"hello");
        test_assert!(
            sock.retransmit_queue.len() == 1,
            "a sent payload was not queued for retransmission"
        );
        test_assert!(
            sock.retransmit_queue[0].seq == 1000,
            "the queued segment was filed under the wrong sequence"
        );
        sock.purge_acked(1005);
        test_assert!(
            sock.retransmit_queue.is_empty(),
            "an acknowledged segment stayed on the retransmit queue"
        );
        TestResult::Pass
    }

    fn test_tcp_header_roundtrip() -> TestResult {
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
        let Some(parsed) = TcpHeader::from_bytes(&bytes) else {
            return TestResult::Fail("a 20-byte header failed to parse");
        };
        test_assert!(
            parsed.to_bytes() == bytes,
            "TcpHeader did not survive a serialize/parse round trip"
        );
        TestResult::Pass
    }

    fn test_tcp_header_literal_wire_order() -> TestResult {
        let bytes = [
            0x00, 0x50, 0x1f, 0x90, 0x00, 0x00, 0x01, 0xf5, 0x00, 0x00, 0x03, 0xe9, 0x50, 0x18,
            0xff, 0xff, 0x12, 0x34, 0x00, 0x00,
        ];

        let Some(parsed) = TcpHeader::from_bytes(&bytes) else {
            return TestResult::Fail("a 20-byte header failed to parse");
        };
        let src_port = parsed.src_port;
        let dst_port = parsed.dst_port;
        let seq = parsed.seq;
        let ack = parsed.ack;

        test_assert!(src_port == 80, "source port is not big-endian on the wire");
        test_assert!(
            dst_port == 8080,
            "destination port is not big-endian on the wire"
        );
        test_assert!(seq == 501, "sequence number is not big-endian on the wire");
        test_assert!(
            ack == 1001,
            "acknowledgement number is not big-endian on the wire"
        );
        test_assert!(
            parsed.data_offset() == 20,
            "data offset is not the high nibble of byte 12 in 4-byte words"
        );
        test_assert!(
            parsed.flags() == FLAG_ACK | FLAG_PSH,
            "flags are not the low 6 bits of the offset/flags word"
        );
        test_assert!(
            parsed.to_bytes() == bytes,
            "re-serializing a literal wire header changed its bytes"
        );
        TestResult::Pass
    }

    fn test_tcp_syn() -> TestResult {
        reset_tcp_for_test();
        let idx = socket();
        connect(idx, crate::net::IpAddr::V4([127, 0, 0, 1]), 80);
        let sockets = TCP_SOCKETS.lock();
        let sock = &sockets[idx];
        test_assert!(
            sock.state == TcpState::SynSent,
            "connect did not leave the socket in SYN-SENT"
        );
        test_assert!(
            sock.retransmit_queue.len() == 1,
            "the SYN was not queued for retransmission"
        );
        test_assert!(
            sock.retransmit_queue[0].flags == FLAG_SYN,
            "the queued segment is not the SYN"
        );
        TestResult::Pass
    }

    fn test_tcp_listen_accept() -> TestResult {
        reset_tcp_for_test();
        let listener = socket();
        bind(listener, 8080);
        listen(listener);
        test_assert!(
            state(listener) == TcpState::Listen,
            "listen did not put the socket in LISTEN"
        );

        // Simulate an incoming SYN to port 8080
        let syn_pkt = make_tcp_header(FLAG_SYN, 500, 0, 8080);
        handle_tcp_packet(REMOTE, LOCAL, &syn_pkt);

        // Process pending SYNs
        poll();

        let Some(new_sock) = accept(listener) else {
            return TestResult::Fail("a SYN to a listening port produced nothing to accept");
        };
        test_assert!(
            state(new_sock) == TcpState::SynReceived,
            "the accepted socket is not in SYN-RECEIVED"
        );
        TestResult::Pass
    }

    fn test_accepted_socket_receives_payload_on_listener_port() -> TestResult {
        reset_tcp_for_test();
        let listener = socket();
        bind(listener, 8080);
        listen(listener);

        let syn_pkt = make_tcp_header(FLAG_SYN, 500, 0, 8080);
        handle_tcp_packet(REMOTE, LOCAL, &syn_pkt);
        poll();

        let Some(accepted) = accept(listener) else {
            return TestResult::Fail("a SYN to a listening port produced nothing to accept");
        };
        let data_pkt = make_tcp_segment(FLAG_ACK | FLAG_PSH, 501, 1001, 8080, b"tick");
        handle_tcp_packet(REMOTE, LOCAL, &data_pkt);

        test_assert!(
            state(accepted) == TcpState::Established,
            "the ACK completing the handshake did not establish the accepted socket"
        );
        let mut buf = [0u8; 8];
        let got = recv(accepted, &mut buf);
        test_assert!(
            got == 4,
            "the payload did not reach the accepted socket -- the listener took it"
        );
        test_assert!(&buf[..4] == b"tick", "the delivered payload is not the one sent");
        TestResult::Pass
    }

    fn packet_fixture_proof_reports_listener_first_demux() -> TestResult {
        reset_tcp_for_test();

        let proof = packet_fixture_proof();

        test_assert!(proof.ok, "the packet fixture did not report ok");
        test_assert!(
            proof.listener_first,
            "the listener is not ordered ahead of the flow sockets it accepted"
        );
        test_assert!(
            proof.exact_flow,
            "the segment was not demuxed to the exact flow"
        );
        test_assert!(
            proof.decoy_rx_bytes == 0,
            "a socket on the same local port but a different four-tuple got the payload"
        );
        test_assert!(
            proof.listener_fallback,
            "a SYN for an unknown flow did not fall back to the listener"
        );
        test_assert!(proof.payload_bytes == 4, "the fixture payload changed size");
        test_assert!(
            proof.rx_bytes == 4,
            "the accepted socket did not receive the whole payload"
        );
        test_assert!(proof.o1_index, "the flow lookup was not a bounded index hit");
        test_assert!(proof.index_hit, "the flow index missed");
        test_assert!(
            proof.index_lookup_probes > 0,
            "the flow index reported a hit in zero probes"
        );
        test_assert!(
            proof.index_lookup_probes <= proof.index_probe_bound,
            "the flow lookup exceeded its probe bound"
        );
        test_assert!(
            proof.index_probe_bound == proof.index_capacity,
            "the flow probe bound no longer matches the table capacity"
        );
        test_assert!(
            proof.index_capacity == TCP_FLOW_INDEX_BUCKETS,
            "the flow index capacity is not the declared bucket count"
        );
        test_assert!(proof.listener_index_hit, "the listener index missed");
        test_assert!(
            proof.listener_lookup_probes > 0,
            "the listener index reported a hit in zero probes"
        );
        test_assert!(
            proof.listener_lookup_probes <= proof.listener_probe_bound,
            "the listener lookup exceeded its probe bound"
        );
        test_assert!(
            proof.listener_probe_bound == proof.listener_index_capacity,
            "the listener probe bound no longer matches the table capacity"
        );
        test_assert!(
            proof.listener_index_capacity == TCP_LISTENER_INDEX_BUCKETS,
            "the listener index capacity is not the declared bucket count"
        );
        test_assert!(!proof.exact_scan, "the demux fell back to a linear scan");
        test_assert!(
            proof.accepted_state == TcpState::Established,
            "the accepted socket did not reach ESTABLISHED"
        );
        test_assert!(proof.cleanup_ok, "the fixture did not restore the socket table");
        test_assert!(
            TCP_SOCKETS.lock().is_empty(),
            "the fixture left sockets behind"
        );
        TestResult::Pass
    }

    fn loopback_echo_fixture_proves_roundtrip_and_cleanup() -> TestResult {
        reset_tcp_for_test();

        let proof = loopback_echo_fixture_proof();

        test_assert!(proof.connections == 8, "the fixture changed connection count");
        test_assert!(
            proof.established == proof.connections,
            "not every connection reached ESTABLISHED on both ends"
        );
        test_assert!(proof.payload_bytes == 64, "the fixture payload changed size");
        test_assert!(
            proof.client_tx == proof.connections * proof.payload_bytes,
            "the client did not send the whole payload on every connection"
        );
        test_assert!(
            proof.server_rx == proof.client_tx,
            "the server did not receive every byte the client sent"
        );
        test_assert!(
            proof.server_echo == proof.client_tx,
            "the server did not echo every byte it received"
        );
        test_assert!(
            proof.client_rx == proof.client_tx,
            "the client did not receive every echoed byte"
        );
        test_assert!(
            proof.listener_accept == proof.connections,
            "not every SYN produced an accepted socket"
        );
        test_assert!(
            proof.exact_flow == proof.connections,
            "a data segment missed the exact-flow index"
        );
        test_assert!(
            proof.listener_index_hit == proof.connections,
            "a SYN missed the listener index"
        );
        test_assert!(
            proof.client_index_hit == proof.connections,
            "an echo missed the client's flow index entry"
        );
        test_assert!(
            proof.index_lookup_probes_max > 0,
            "the index reported hits in zero probes"
        );
        test_assert!(
            proof.index_lookup_probes_max <= proof.index_probe_bound,
            "a lookup exceeded its probe bound"
        );
        test_assert!(proof.cleanup_ok, "the fixture did not restore the socket table");
        test_assert!(
            TCP_SOCKETS.lock().is_empty(),
            "the fixture left sockets behind"
        );
        TestResult::Pass
    }

    /// A bare reset at ESTABLISHED used to do nothing whatever: the arm needed
    /// an ACK flag or a payload to run, and a reset carries neither. The socket
    /// must abort, stop retransmitting, and leave the flow index with it, so
    /// the four-tuple no longer resolves.
    fn reset_in_window_aborts_established_socket() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(9000);
        sock.remote_ip = REMOTE;
        sock.remote_port = 80;
        sock.state = TcpState::Established;
        sock.seq_num = 5000;
        sock.ack_num = 700;
        sock.send(b"unacked");
        test_assert!(
            sock.retransmit_queue.len() == 1,
            "the sent payload was not queued for retransmission"
        );
        let idx = push_indexed(sock);

        handle_tcp_packet(REMOTE, LOCAL, &make_tcp_header(FLAG_RST, 700, 0, 9000));

        test_assert!(
            state(idx) == TcpState::Closed,
            "an in-window reset did not abort the connection"
        );
        test_assert!(
            TCP_SOCKETS.lock()[idx].retransmit_queue.is_empty(),
            "an aborted socket is still retransmitting"
        );

        // The flow left the index with the socket, so a later segment for the
        // same four-tuple reaches nothing.
        let data = make_tcp_segment(FLAG_ACK | FLAG_PSH, 700, 5008, 9000, b"after");
        handle_tcp_packet(REMOTE, LOCAL, &data);
        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets[idx].state == TcpState::Closed,
            "a segment reopened an aborted socket"
        );
        test_assert!(
            sockets[idx].rx_buffer.is_empty(),
            "an aborted socket accepted data: its flow is still in the index"
        );
        TestResult::Pass
    }

    /// A reset that does not land on the next expected byte is a blind guess at
    /// the sequence space and must be dropped -- and dropped whole, not
    /// half-processed as an ordinary ACK, which is what the ESTABLISHED arm did
    /// with any RST+ACK it was handed.
    fn reset_out_of_window_is_ignored_at_established() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(9010);
        sock.remote_ip = REMOTE;
        sock.remote_port = 80;
        sock.state = TcpState::Established;
        sock.seq_num = 5000;
        sock.ack_num = 700;
        let idx = push_indexed(sock);

        handle_tcp_packet(
            REMOTE,
            LOCAL,
            &make_tcp_header(FLAG_RST | FLAG_ACK, 5700, 5000, 9010),
        );

        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets[idx].state == TcpState::Established,
            "an out-of-window reset tore the connection down"
        );
        test_assert!(
            sockets[idx].ack_num == 700,
            "an out-of-window RST+ACK was processed as an ordinary acknowledgement"
        );
        TestResult::Pass
    }

    /// RFC 793: in SYN-SENT a reset is acceptable only if it acknowledges the
    /// SYN. `connect` leaves `seq_num` at the initial send sequence and the SYN
    /// occupies it, so the acknowledgement of that SYN is `seq_num + 1`. A bare
    /// reset, or one acknowledging anything else, leaves the socket connecting.
    fn reset_at_syn_sent_aborts_only_when_it_acknowledges_the_syn() -> TestResult {
        reset_tcp_for_test();
        let mut idxs = [0usize; 3];
        for (n, slot) in idxs.iter_mut().enumerate() {
            let mut sock = TcpSocket::new(9020 + n as u16);
            sock.connect(REMOTE, 80);
            test_assert!(
                sock.seq_num == 1000,
                "connect no longer starts at the initial send sequence this case assumes"
            );
            *slot = push_indexed(sock);
        }

        handle_tcp_packet(REMOTE, LOCAL, &make_tcp_header(FLAG_RST | FLAG_ACK, 0, 1001, 9020));
        handle_tcp_packet(REMOTE, LOCAL, &make_tcp_header(FLAG_RST, 0, 0, 9021));
        handle_tcp_packet(REMOTE, LOCAL, &make_tcp_header(FLAG_RST | FLAG_ACK, 0, 4242, 9022));

        test_assert!(
            state(idxs[0]) == TcpState::Closed,
            "a reset acknowledging the SYN did not abort the connection attempt"
        );
        test_assert!(
            state(idxs[1]) == TcpState::SynSent,
            "a bare reset aborted a connection attempt it does not acknowledge"
        );
        test_assert!(
            state(idxs[2]) == TcpState::SynSent,
            "a reset acknowledging the wrong sequence aborted the connection attempt"
        );
        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets[idxs[0]].retransmit_queue.is_empty(),
            "an aborted connection attempt is still retransmitting its SYN"
        );
        test_assert!(
            sockets[idxs[1]].retransmit_queue.len() == 1,
            "a dropped reset discarded the SYN still awaiting an answer"
        );
        test_assert!(
            sockets[idxs[2]].retransmit_queue.len() == 1,
            "a dropped reset discarded the SYN still awaiting an answer"
        );
        TestResult::Pass
    }

    /// Retransmitting used to re-push the segment without removing the entry it
    /// came from, so the queue grew by one per expiry and the original kept its
    /// stale timestamp -- retransmitting again on the very next call rather than
    /// one RTO later. It also left `seq_num` pointing at the older segment, so
    /// the next `send` would have reused those numbers for different bytes.
    fn retransmit_rearms_one_entry_and_leaves_seq_num_alone() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(9030);
        sock.remote_ip = crate::net::IpAddr::V4([192, 0, 2, 13]);
        sock.remote_port = 80;
        sock.state = TcpState::Established;
        sock.seq_num = 1000;
        sock.send(b"hello");
        sock.send(b"world");
        test_assert!(
            sock.retransmit_queue.len() == 2,
            "two sends did not queue two segments"
        );
        test_assert!(sock.seq_num == 1010, "seq_num did not advance over both sends");
        let rto0 = sock.rto;

        age_retransmit_queue(&mut sock, rto0 + 1);
        TCP_TX_QUEUE.lock().clear();

        sock.retransmit_expired();
        sock.retransmit_expired();
        sock.retransmit_expired();

        test_assert!(
            sock.retransmit_queue.len() == 2,
            "retransmitting grew the queue: the entry was re-pushed without being taken off"
        );
        test_assert!(
            sock.seq_num == 1010,
            "retransmitting rewound seq_num and left it there"
        );
        test_assert!(
            tcp_tx_queued() == 1,
            "one expiry transmitted more than one segment: the entry kept its stale timestamp"
        );
        test_assert!(sock.rto > rto0, "the retransmit timeout did not back off");

        let queued = TCP_TX_QUEUE.lock();
        let Some((_, first)) = queued.first() else {
            return TestResult::Fail("the expired segment was not transmitted at all");
        };
        if first.len() < 20 {
            return TestResult::Fail("the transmitted segment is shorter than a TCP header");
        }
        let Some(hdr) = TcpHeader::from_bytes(&first[..20]) else {
            return TestResult::Fail("the transmitted segment has no parseable header");
        };
        let seq = hdr.seq;
        test_assert!(seq == 1000, "the wrong segment was retransmitted");
        test_assert!(
            &first[20..] == b"hello",
            "the retransmitted segment carries the wrong payload"
        );
        drop(queued);

        // Re-stamped, so the next attempt is a whole backed-off RTO away.
        let rto1 = sock.rto;
        age_retransmit_queue(&mut sock, rto1 + 1);
        sock.retransmit_expired();
        test_assert!(
            sock.retransmit_queue.len() == 2,
            "the second expiry grew the queue"
        );
        test_assert!(
            tcp_tx_queued() == 2,
            "the second expiry did not transmit exactly one more segment"
        );

        sock.purge_acked(1010);
        test_assert!(
            sock.retransmit_queue.is_empty(),
            "acknowledging both segments did not clear the queue"
        );
        test_assert!(
            sock.rto == rto0,
            "forward progress did not reset the backed-off retransmit timeout"
        );
        TestResult::Pass
    }

    /// The local port `idx` was given, or 0 if it names no socket.
    fn local_port_of(idx: usize) -> u16 {
        TCP_SOCKETS.lock().get(idx).map_or(0, |sock| sock.local_port)
    }

    /// Whether two sockets in the table share a local port.
    fn any_duplicate_local_port() -> bool {
        let sockets = TCP_SOCKETS.lock();
        sockets
            .iter()
            .enumerate()
            .any(|(i, a)| sockets[i + 1..].iter().any(|b| b.local_port == a.local_port))
    }

    /// The counter used to be a bare `*p += 1` on a `u16` that started at 40000,
    /// so the 25,536th allocation left the range: silently 0, 1, 2 ... in
    /// release, and an abort under `overflow-checks`. Ports 0-1023 are the
    /// well-known range, so the wrap walked the allocator straight through 22,
    /// 80 and 443.
    ///
    /// The boundaries are written as literals here on purpose. An assertion
    /// spelled with the constants it is guarding moves whenever they do, and
    /// then nothing at all is pinned.
    fn ephemeral_ports_wrap_inside_the_range() -> TestResult {
        reset_tcp_for_test();
        *NEXT_TCP_PORT.lock() = 65_534;

        let a = socket();
        let b = socket();
        let c = socket();
        let d = socket();

        test_assert!(
            local_port_of(a) == 65_534,
            "the allocator did not hand out the port the counter was left at"
        );
        test_assert!(
            local_port_of(b) == 65_535,
            "the allocator stopped short of the top of the ephemeral range"
        );
        test_assert!(
            local_port_of(c) == 40_000,
            "the counter did not wrap to the floor of the ephemeral range"
        );
        test_assert!(
            local_port_of(d) == 40_001,
            "the counter did not carry on from the floor after wrapping"
        );
        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets.iter().all(|sock| sock.local_port >= 40_000),
            "an allocated port landed below the ephemeral floor -- 0-1023 is the well-known range"
        );
        TestResult::Pass
    }

    /// A wrapped counter names ports that are still held. Nothing anywhere
    /// checked: `socket` handed out whatever the counter said and `bind` writes
    /// any port at all over it, so two live sockets could sit on one local port
    /// and a segment for either one resolved to whichever the index held.
    fn ephemeral_allocator_skips_a_port_already_in_use() -> TestResult {
        reset_tcp_for_test();

        let held = socket();
        test_assert!(
            local_port_of(held) == 40_000,
            "the first allocation is not the floor of the ephemeral range"
        );
        // Moves a live socket onto the port the counter is about to reach.
        bind(held, 40_001);

        let next = socket();
        test_assert!(
            local_port_of(next) == 40_002,
            "the allocator handed out a port a live socket already holds"
        );
        test_assert!(
            !any_duplicate_local_port(),
            "two sockets share a local port"
        );
        TestResult::Pass
    }

    /// The skip has to walk a run of taken ports and still land on the first
    /// free one, not give up at the first collision and not run off the end of
    /// the range.
    ///
    /// 512 consecutive ports rather than the whole 25,536: the in-use test is a
    /// scan of the socket table, so filling the range costs its square, which
    /// is minutes of QEMU. Exhaustion of the full range is proved on the host
    /// harness instead.
    fn ephemeral_allocator_walks_a_run_of_taken_ports() -> TestResult {
        reset_tcp_for_test();
        const RUN: u16 = 512;
        {
            let mut sockets = TCP_SOCKETS.lock();
            for port in 40_000..40_000 + RUN {
                sockets.push(TcpSocket::new(port));
            }
        }

        let idx = socket();

        test_assert!(
            idx != TCP_SOCKET_NONE,
            "the allocator gave up while the range still had free ports"
        );
        test_assert!(
            local_port_of(idx) == 40_000 + RUN,
            "the allocator did not land on the first free port after the run"
        );
        test_assert!(
            !any_duplicate_local_port(),
            "two sockets share a local port"
        );
        TestResult::Pass
    }

    /// `poll` allocated an ephemeral port for every accepted socket and then
    /// overwrote it with the listener's port on the next line, so the number
    /// was never used -- but the counter moved, and it moved once per SYN a
    /// remote peer sent. Reaching the overflow needed no local socket at all.
    fn accepting_a_connection_consumes_no_ephemeral_port() -> TestResult {
        reset_tcp_for_test();
        let listener = socket();
        bind(listener, 8080);
        listen(listener);

        let before = *NEXT_TCP_PORT.lock();
        handle_tcp_packet(REMOTE, LOCAL, &make_tcp_header(FLAG_SYN, 500, 0, 8080));
        poll();
        let after = *NEXT_TCP_PORT.lock();

        test_assert!(
            after == before,
            "accepting a connection burned an ephemeral port: a remote peer drives the counter"
        );
        let Some(accepted) = accept(listener) else {
            return TestResult::Fail("a SYN to a listening port produced nothing to accept");
        };
        test_assert!(
            local_port_of(accepted) == 8080,
            "the accepted socket is not on the port the segment was addressed to"
        );
        TestResult::Pass
    }

    /// The SYN-RECEIVED arm compares `seq` against `ack_num`; the ESTABLISHED
    /// arm did not, so a duplicate retransmission appended its payload twice and
    /// anything that knew the four-tuple could inject at any sequence at all.
    /// The first segment is the control: the check must not turn every segment
    /// into a drop.
    fn established_accepts_only_the_next_expected_sequence() -> TestResult {
        reset_tcp_for_test();
        let mut sock = TcpSocket::new(9040);
        sock.remote_ip = REMOTE;
        sock.remote_port = 80;
        sock.state = TcpState::Established;
        sock.seq_num = 5000;
        sock.ack_num = 700;
        let idx = push_indexed(sock);

        let in_order = make_tcp_segment(FLAG_ACK | FLAG_PSH, 700, 5000, 9040, b"abc");
        handle_tcp_packet(REMOTE, LOCAL, &in_order);
        handle_tcp_packet(REMOTE, LOCAL, &in_order);

        let injected = make_tcp_segment(FLAG_ACK | FLAG_PSH, 60_000, 5000, 9040, b"evil");
        handle_tcp_packet(REMOTE, LOCAL, &injected);

        handle_tcp_packet(
            REMOTE,
            LOCAL,
            &make_tcp_header(FLAG_FIN | FLAG_ACK, 60_000, 5000, 9040),
        );

        let mut buf = [0u8; 16];
        let n = recv(idx, &mut buf);
        test_assert!(
            &buf[..n] == b"abc",
            "a duplicate or out-of-sequence segment was accepted into the receive buffer"
        );
        let sockets = TCP_SOCKETS.lock();
        test_assert!(
            sockets[idx].ack_num == 703,
            "the next expected sequence moved on a segment that was not next"
        );
        test_assert!(
            sockets[idx].state == TcpState::Established,
            "an out-of-sequence FIN closed the connection"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("tcp::syn_sent_to_established", test_syn_sent_to_established);
        crate::testing::register_test("tcp::established_to_closewait", test_established_to_closewait);
        crate::testing::register_test("tcp::fin_wait1_to_timewait", test_fin_wait1_to_timewait);
        crate::testing::register_test("tcp::close_to_syn_sent", test_close_to_syn_sent);
        crate::testing::register_test("tcp::retransmit_queue", test_retransmit_queue);
        crate::testing::register_test("tcp::header_roundtrip", test_tcp_header_roundtrip);
        crate::testing::register_test(
            "tcp::header_literal_wire_order",
            test_tcp_header_literal_wire_order,
        );
        crate::testing::register_test("tcp::syn", test_tcp_syn);
        crate::testing::register_test("tcp::listen_accept", test_tcp_listen_accept);
        crate::testing::register_test(
            "tcp::accepted_socket_receives_payload_on_listener_port",
            test_accepted_socket_receives_payload_on_listener_port,
        );
        crate::testing::register_test(
            "tcp::packet_fixture_proof_reports_listener_first_demux",
            packet_fixture_proof_reports_listener_first_demux,
        );
        crate::testing::register_test(
            "tcp::loopback_echo_fixture_proves_roundtrip_and_cleanup",
            loopback_echo_fixture_proves_roundtrip_and_cleanup,
        );
        crate::testing::register_test(
            "tcp::reset_in_window_aborts_established_socket",
            reset_in_window_aborts_established_socket,
        );
        crate::testing::register_test(
            "tcp::reset_out_of_window_is_ignored_at_established",
            reset_out_of_window_is_ignored_at_established,
        );
        crate::testing::register_test(
            "tcp::reset_at_syn_sent_aborts_only_when_it_acknowledges_the_syn",
            reset_at_syn_sent_aborts_only_when_it_acknowledges_the_syn,
        );
        crate::testing::register_test(
            "tcp::retransmit_rearms_one_entry_and_leaves_seq_num_alone",
            retransmit_rearms_one_entry_and_leaves_seq_num_alone,
        );
        crate::testing::register_test(
            "tcp::established_accepts_only_the_next_expected_sequence",
            established_accepts_only_the_next_expected_sequence,
        );
        crate::testing::register_test(
            "tcp::ephemeral_ports_wrap_inside_the_range",
            ephemeral_ports_wrap_inside_the_range,
        );
        crate::testing::register_test(
            "tcp::ephemeral_allocator_skips_a_port_already_in_use",
            ephemeral_allocator_skips_a_port_already_in_use,
        );
        crate::testing::register_test(
            "tcp::ephemeral_allocator_walks_a_run_of_taken_ports",
            ephemeral_allocator_walks_a_run_of_taken_ports,
        );
        crate::testing::register_test(
            "tcp::accepting_a_connection_consumes_no_ephemeral_port",
            accepting_a_connection_consumes_no_ephemeral_port,
        );
    }
}
