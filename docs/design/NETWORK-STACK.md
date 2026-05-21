# Network Stack — Design Document

> **Phase 5 Task**: NET-001
> **Priority**: MEDIUM-HIGH (required for remote Aether-Link, package manager, SSH)
> **Estimated Effort**: 6 weeks
> **Blocked by**: Interrupt framework, DMA allocator, physical allocator stability, SMP
> **Blocks**: Aether-Link remote inference, distributed topology sync, package updates

---

## Overview

Bare-metal network stack from NIC driver to BSD socket API. Targets virtio-net (QEMU/cloud default) and e1000 (Intel emulated). TCP/IP is required for the Aether-Link remote inference path and for any package manager or SSH access.

---

## Architecture

### 1. NIC Drivers

#### 1.1 Virtio-net

```rust
pub struct VirtioNet {
    dev: VirtioDevice,
    rx_queue: Virtqueue,      // Queue 0: receive
    tx_queue: Virtqueue,      // Queue 1: transmit
    ctrl_queue: Virtqueue,    // Queue 2: control (MAC filter, VLAN, offloads)
    mac: [u8; 6],
    mtu: u16,
}
```

- [x] PCI scan for virtio-net (vendor 0x1AF4, device 0x1000 / 0x1041)
- [x] Read MAC address from device config space
- [x] Initialize RX virtqueue (queue 0)
- [x] Initialize TX virtqueue (queue 1)
- [x] Initialize control virtqueue (queue 2, optional)
- [x] Pre-fill RX queue with empty DMA buffers
- [x] Implement `receive_packet()` — read from RX used ring
- [x] Implement `transmit_packet()` — write to TX avail ring
- [x] Refill RX descriptors after consumption
- [x] Handle `VIRTIO_NET_F_STATUS` for link up/down
- [x] Handle `VIRTIO_NET_F_CSUM` checksum offload (v2)
- [x] Handle `VIRTIO_NET_F_MRG_RXBUF` multi-buffer packets (v2)

#### 1.2 e1000 (Intel 82540EM / 82545EM)

```rust
pub struct E1000 {
    mmio_base: VirtAddr,
    rx_ring: [E1000RxDesc; 256],
    tx_ring: [E1000TxDesc; 256],
    mac: [u8; 6],
}
```

- [x] PCI scan for e1000 (vendor 0x8086, device 0x100E / 0x100F)
- [x] Map MMIO BAR for register access
- [x] Read MAC from EEPROM (registers `RAL`/`RAH`)
- [x] Reset device via `CTRL.RST`
- [x] Allocate RX ring (256 descriptors)
- [x] Allocate TX ring (256 descriptors)
- [x] Program `RDBAL/RDBAH`, `RDLEN`, `RDH/RDT`
- [x] Program `TDBAL/TDBAH`, `TDLEN`, `TDH/TDT`
- [x] Enable RX/TX via `RCTL`/`TCTL`
- [x] Implement `receive_packet()` — poll DD bit on RX descriptors
- [x] Implement `transmit_packet()` — write descriptor, advance TDT
- [x] Handle `RXT0` + `TXDW` interrupts
- [x] Send EOI after interrupt handling

### 2. Network Layer (L2/L3)

#### 2.1 Ethernet + ARP

```rust
#[repr(C, packed)]
struct EthernetHeader {
    dst: [u8; 6],
    src: [u8; 6],
    ether_type: u16,  // 0x0800 = IPv4, 0x0806 = ARP
}
```

- [x] Define `EthernetHeader` struct
- [x] Parse incoming Ethernet frames by ether_type
- [x] Implement ARP packet format (`ArpPacket` struct)
- [x] Implement ARP request generation (broadcast)
- [x] Implement ARP reply handling (cache MAC)
- [x] Implement ARP cache with timeout (600 seconds)
- [x] Queue IP packets pending ARP resolution
- [x] Implement ARP cache eviction (LRU, max 256 entries)
- [x] Handle ARP cache miss by broadcasting request

#### 2.2 IPv4

```rust
#[repr(C)]
struct Ipv4Header {
    version_ihl: u8,      // 0x45
    dscp_ecn: u8,
    total_len: u16,
    id: u16,
    flags_frag: u16,
    ttl: u8,
    protocol: u8,         // 1 = ICMP, 6 = TCP, 17 = UDP
    checksum: u16,
    src: Ipv4Addr,
    dst: Ipv4Addr,
}
```

- [x] Define `Ipv4Header` struct
- [x] Implement IPv4 checksum calculation
- [x] Implement IPv4 header parsing
- [x] Implement IPv4 header construction
- [x] Implement packet forwarding (optional, v2)
- [x] Handle fragmentation-needed ICMP (for PMTU discovery)
- [x] Set DF flag on all outgoing packets
- [x] Implement TTL decrement and drop on 0
- [x] Implement local address check (`dst` matches our interface)

#### 2.3 ICMP

```rust
enum IcmpType {
    EchoReply = 0,
    EchoRequest = 8,
    DestUnreachable = 3,
    TimeExceeded = 11,
}
```

- [x] Define `IcmpHeader` struct
- [x] Implement ICMP checksum
- [x] Handle ICMP Echo Request → reply with Echo Reply
- [x] Handle ICMP Destination Unreachable (for PMTU)
- [x] Handle ICMP Time Exceeded
- [x] Implement `ping` utility (userspace via raw socket)

### 3. Transport Layer (L4)

#### 3.1 UDP

```rust
pub struct UdpSocket {
    local_addr: SocketAddrV4,
    peer_addr: Option<SocketAddrV4>,
    rx_queue: VecDeque<Vec<u8>>,
    bound: bool,
}
```

- [x] Define `UdpHeader` struct
- [x] Implement UDP checksum (optional in IPv4)
- [x] Implement `socket(AF_INET, SOCK_DGRAM)`
- [x] Implement `bind()` for UDP
- [x] Implement `sendto()` — stateless send
- [x] Implement `recvfrom()` — stateless receive
- [x] Implement `connect()` for UDP (sets default peer)
- [x] Implement `send()` / `recv()` for connected UDP
- [x] Route UDP packets to correct socket by (local_ip, local_port)

#### 3.2 TCP

Full RFC 793 + RFC 1122 + RFC 5681 (congestion control) + RFC 6298 (RTT estimation).

```rust
pub struct TcpSocket {
    state: TcpState,
    local: SocketAddrV4,
    remote: Option<SocketAddrV4>,
    
    // Send state
    snd_una: u32,         // unacknowledged
    snd_nxt: u32,         // next seq to send
    snd_wnd: u16,         // peer's receive window
    snd_cwnd: u32,        // congestion window
    snd_ssthresh: u32,    // slow start threshold
    
    // Receive state
    rcv_nxt: u32,         // next expected seq
    rcv_wnd: u16,         // our receive window
    
    // Buffers
    tx_buffer: RingBuffer,
    rx_buffer: RingBuffer,
    
    // Timers
    rto: u64,             // retransmission timeout (ms)
    srtt: u64,            // smoothed RTT
    rttvar: u64,          // RTT variance
    
    // Options
    mss: u16,
    sack_permitted: bool,
    timestamps: bool,
}
```

- [x] Define `TcpState` enum (CLOSED, LISTEN, SYN_SENT, SYN_RECEIVED, ESTABLISHED, FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT, CLOSING, LAST_ACK, TIME_WAIT)
- [x] Define `TcpHeader` struct (with options parsing)
- [x] Implement TCP checksum
- [x] Implement three-way handshake (active + passive open)
- [x] Implement `socket(AF_INET, SOCK_STREAM)`
- [x] Implement `bind()` for TCP
- [x] Implement `listen()` + accept queue
- [x] Implement `accept()` — blocking wait for connection
- [x] Implement `connect()` — send SYN, wait for SYN-ACK
- [x] Implement `send()` — copy to tx_buffer, schedule TX
- [x] Implement `recv()` — copy from rx_buffer, send ACK
- [x] Implement connection teardown (FIN exchange, TIME_WAIT)
- [x] Implement slow start (`cwnd` doubles per RTT)
- [x] Implement congestion avoidance (linear growth after `ssthresh`)
- [x] Implement fast retransmit (3 dup ACKs)
- [x] Implement fast recovery
- [x] Implement timeout retransmission with RTO backoff
- [x] Implement RTT estimation (RFC 6298)
- [x] Implement `TIME_WAIT` aging (60 second 2×MSL)
- [x] Implement timer wheel for retransmission

### 4. Socket API

BSD socket API exposed via syscalls. Internal representation:

```rust
pub enum Socket {
    Tcp(TcpSocket),
    Udp(UdpSocket),
    Raw(Ipv4Protocol),
}
```

- [x] Define `Socket` enum
- [x] Define `FileDescriptor` with `FdKind` (File, Socket, Pipe, Epoll)
- [x] Implement `socket()` syscall
- [x] Implement `bind()` syscall
- [x] Implement `listen()` syscall
- [x] Implement `accept()` syscall
- [x] Implement `connect()` syscall
- [x] Implement `send()` / `sendto()` syscalls
- [x] Implement `recv()` / `recvfrom()` syscalls
- [x] Implement `shutdown()` syscall
- [x] Implement `setsockopt()` / `getsockopt()` syscalls
- [x] Implement `SO_REUSEADDR`, `TCP_NODELAY`, `SO_RCVBUF`, `SO_SNDBUF`
- [x] Implement `getsockname()` / `getpeername()`

### 5. Network Interface Abstraction

```rust
pub trait NetworkInterface: Send + Sync {
    fn name(&self) -> &str;
    fn mac(&self) -> [u8; 6];
    fn mtu(&self) -> usize;
    fn is_up(&self) -> bool;
    fn transmit(&self, packet: &[u8]) -> Result<(), NetError>;
    fn register_rx_handler(&self, handler: RxHandler);
}
```

- [x] Define `NetworkInterface` trait
- [x] Define `NET_IFACES` global registry
- [x] Implement `net_ifaces()` iterator
- [x] Implement packet routing to correct interface
- [x] Implement loopback interface (`lo`, 127.0.0.1)
- [x] Define `Route` struct with longest-prefix match
- [x] Implement routing table
- [x] Implement default route (0.0.0.0/0)
- [x] Implement route lookup on every outgoing packet

### 6. DHCP Client

```rust
fn dhcp_discover(iface: &dyn NetworkInterface) {
    send_udp_broadcast(67, DhcpMessage::Discover { mac: iface.mac(), xid: random() });
}
```

- [x] Implement DHCP message format
- [x] Implement DHCP state machine (INIT → SELECTING → REQUESTING → BOUND → RENEWING → REBINDING)
- [x] Send DHCP Discover on boot
- [x] Handle DHCP Offer
- [x] Send DHCP Request
- [x] Handle DHCP Ack
- [x] Configure IP, netmask, gateway from lease
- [x] Set timer for T1 (renew at 50% of lease)
- [x] Set timer for T2 (rebind at 87.5% of lease)
- [x] Implement lease expiration fallback

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_virtio_net_found` | PCI scan discovers virtio-net, MAC readable | [x] |
| `test_e1000_found` | PCI scan discovers e1000, EEPROM MAC readable | [x] |
| `test_tx_packet` | Transmit raw Ethernet frame, capture in host with `tcpdump` | [x] |
| `test_rx_packet` | Host sends packet, kernel receives, ISR fires | [x] |
| `test_arp_request_reply` | Send ARP request for gateway, receive reply, cache populated | [x] |
| `test_icmp_ping` | `ping 127.0.0.1` works (loopback); `ping <gateway>` works | [x] |
| `test_udp_socket` | `socket(AF_INET, SOCK_DGRAM)` → bind → sendto → recvfrom round-trip | [x] |
| `test_tcp_connect` | `connect()` to host port 1234, three-way handshake completes | [x] |
| `test_tcp_accept` | `listen()` + `accept()`, client connects, data exchanged | [x] |
| `test_tcp_retransmit` | Drop every 5th packet, TCP recovers via RTO | [x] |
| `test_tcp_congestion` | Large transfer, `cwnd` grows correctly, loss handled | [x] |
| `test_dhcp_lease` | DHCP discover → offer → request → ack, IP assigned | [x] |
| `test_http_get` | `curl http://example.com` over TCP, HTTP response received | [x] |

---

*Design document produced by Phase 5 Planning*
*2026-05-19*
