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

- [ ] PCI scan for virtio-net (vendor 0x1AF4, device 0x1000 / 0x1041)
- [ ] Read MAC address from device config space
- [ ] Initialize RX virtqueue (queue 0)
- [ ] Initialize TX virtqueue (queue 1)
- [ ] Initialize control virtqueue (queue 2, optional)
- [ ] Pre-fill RX queue with empty DMA buffers
- [ ] Implement `receive_packet()` — read from RX used ring
- [ ] Implement `transmit_packet()` — write to TX avail ring
- [ ] Refill RX descriptors after consumption
- [ ] Handle `VIRTIO_NET_F_STATUS` for link up/down
- [ ] Handle `VIRTIO_NET_F_CSUM` checksum offload (v2)
- [ ] Handle `VIRTIO_NET_F_MRG_RXBUF` multi-buffer packets (v2)

#### 1.2 e1000 (Intel 82540EM / 82545EM)

```rust
pub struct E1000 {
    mmio_base: VirtAddr,
    rx_ring: [E1000RxDesc; 256],
    tx_ring: [E1000TxDesc; 256],
    mac: [u8; 6],
}
```

- [ ] PCI scan for e1000 (vendor 0x8086, device 0x100E / 0x100F)
- [ ] Map MMIO BAR for register access
- [ ] Read MAC from EEPROM (registers `RAL`/`RAH`)
- [ ] Reset device via `CTRL.RST`
- [ ] Allocate RX ring (256 descriptors)
- [ ] Allocate TX ring (256 descriptors)
- [ ] Program `RDBAL/RDBAH`, `RDLEN`, `RDH/RDT`
- [ ] Program `TDBAL/TDBAH`, `TDLEN`, `TDH/TDT`
- [ ] Enable RX/TX via `RCTL`/`TCTL`
- [ ] Implement `receive_packet()` — poll DD bit on RX descriptors
- [ ] Implement `transmit_packet()` — write descriptor, advance TDT
- [ ] Handle `RXT0` + `TXDW` interrupts
- [ ] Send EOI after interrupt handling

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

- [ ] Define `EthernetHeader` struct
- [ ] Parse incoming Ethernet frames by ether_type
- [ ] Implement ARP packet format (`ArpPacket` struct)
- [ ] Implement ARP request generation (broadcast)
- [ ] Implement ARP reply handling (cache MAC)
- [ ] Implement ARP cache with timeout (600 seconds)
- [ ] Queue IP packets pending ARP resolution
- [ ] Implement ARP cache eviction (LRU, max 256 entries)
- [ ] Handle ARP cache miss by broadcasting request

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

- [ ] Define `Ipv4Header` struct
- [ ] Implement IPv4 checksum calculation
- [ ] Implement IPv4 header parsing
- [ ] Implement IPv4 header construction
- [ ] Implement packet forwarding (optional, v2)
- [ ] Handle fragmentation-needed ICMP (for PMTU discovery)
- [ ] Set DF flag on all outgoing packets
- [ ] Implement TTL decrement and drop on 0
- [ ] Implement local address check (`dst` matches our interface)

#### 2.3 ICMP

```rust
enum IcmpType {
    EchoReply = 0,
    EchoRequest = 8,
    DestUnreachable = 3,
    TimeExceeded = 11,
}
```

- [ ] Define `IcmpHeader` struct
- [ ] Implement ICMP checksum
- [ ] Handle ICMP Echo Request → reply with Echo Reply
- [ ] Handle ICMP Destination Unreachable (for PMTU)
- [ ] Handle ICMP Time Exceeded
- [ ] Implement `ping` utility (userspace via raw socket)

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

- [ ] Define `UdpHeader` struct
- [ ] Implement UDP checksum (optional in IPv4)
- [ ] Implement `socket(AF_INET, SOCK_DGRAM)`
- [ ] Implement `bind()` for UDP
- [ ] Implement `sendto()` — stateless send
- [ ] Implement `recvfrom()` — stateless receive
- [ ] Implement `connect()` for UDP (sets default peer)
- [ ] Implement `send()` / `recv()` for connected UDP
- [ ] Route UDP packets to correct socket by (local_ip, local_port)

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

- [ ] Define `TcpState` enum (CLOSED, LISTEN, SYN_SENT, SYN_RECEIVED, ESTABLISHED, FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT, CLOSING, LAST_ACK, TIME_WAIT)
- [ ] Define `TcpHeader` struct (with options parsing)
- [ ] Implement TCP checksum
- [ ] Implement three-way handshake (active + passive open)
- [ ] Implement `socket(AF_INET, SOCK_STREAM)`
- [ ] Implement `bind()` for TCP
- [ ] Implement `listen()` + accept queue
- [ ] Implement `accept()` — blocking wait for connection
- [ ] Implement `connect()` — send SYN, wait for SYN-ACK
- [ ] Implement `send()` — copy to tx_buffer, schedule TX
- [ ] Implement `recv()` — copy from rx_buffer, send ACK
- [ ] Implement connection teardown (FIN exchange, TIME_WAIT)
- [ ] Implement slow start (`cwnd` doubles per RTT)
- [ ] Implement congestion avoidance (linear growth after `ssthresh`)
- [ ] Implement fast retransmit (3 dup ACKs)
- [ ] Implement fast recovery
- [ ] Implement timeout retransmission with RTO backoff
- [ ] Implement RTT estimation (RFC 6298)
- [ ] Implement `TIME_WAIT` aging (60 second 2×MSL)
- [ ] Implement timer wheel for retransmission

### 4. Socket API

BSD socket API exposed via syscalls. Internal representation:

```rust
pub enum Socket {
    Tcp(TcpSocket),
    Udp(UdpSocket),
    Raw(Ipv4Protocol),
}
```

- [ ] Define `Socket` enum
- [ ] Define `FileDescriptor` with `FdKind` (File, Socket, Pipe, Epoll)
- [ ] Implement `socket()` syscall
- [ ] Implement `bind()` syscall
- [ ] Implement `listen()` syscall
- [ ] Implement `accept()` syscall
- [ ] Implement `connect()` syscall
- [ ] Implement `send()` / `sendto()` syscalls
- [ ] Implement `recv()` / `recvfrom()` syscalls
- [ ] Implement `shutdown()` syscall
- [ ] Implement `setsockopt()` / `getsockopt()` syscalls
- [ ] Implement `SO_REUSEADDR`, `TCP_NODELAY`, `SO_RCVBUF`, `SO_SNDBUF`
- [ ] Implement `getsockname()` / `getpeername()`

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

- [ ] Define `NetworkInterface` trait
- [ ] Define `NET_IFACES` global registry
- [ ] Implement `net_ifaces()` iterator
- [ ] Implement packet routing to correct interface
- [ ] Implement loopback interface (`lo`, 127.0.0.1)
- [ ] Define `Route` struct with longest-prefix match
- [ ] Implement routing table
- [ ] Implement default route (0.0.0.0/0)
- [ ] Implement route lookup on every outgoing packet

### 6. DHCP Client

```rust
fn dhcp_discover(iface: &dyn NetworkInterface) {
    send_udp_broadcast(67, DhcpMessage::Discover { mac: iface.mac(), xid: random() });
}
```

- [ ] Implement DHCP message format
- [ ] Implement DHCP state machine (INIT → SELECTING → REQUESTING → BOUND → RENEWING → REBINDING)
- [ ] Send DHCP Discover on boot
- [ ] Handle DHCP Offer
- [ ] Send DHCP Request
- [ ] Handle DHCP Ack
- [ ] Configure IP, netmask, gateway from lease
- [ ] Set timer for T1 (renew at 50% of lease)
- [ ] Set timer for T2 (rebind at 87.5% of lease)
- [ ] Implement lease expiration fallback

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_virtio_net_found` | PCI scan discovers virtio-net, MAC readable | [ ] |
| `test_e1000_found` | PCI scan discovers e1000, EEPROM MAC readable | [ ] |
| `test_tx_packet` | Transmit raw Ethernet frame, capture in host with `tcpdump` | [ ] |
| `test_rx_packet` | Host sends packet, kernel receives, ISR fires | [ ] |
| `test_arp_request_reply` | Send ARP request for gateway, receive reply, cache populated | [ ] |
| `test_icmp_ping` | `ping 127.0.0.1` works (loopback); `ping <gateway>` works | [ ] |
| `test_udp_socket` | `socket(AF_INET, SOCK_DGRAM)` → bind → sendto → recvfrom round-trip | [ ] |
| `test_tcp_connect` | `connect()` to host port 1234, three-way handshake completes | [ ] |
| `test_tcp_accept` | `listen()` + `accept()`, client connects, data exchanged | [ ] |
| `test_tcp_retransmit` | Drop every 5th packet, TCP recovers via RTO | [ ] |
| `test_tcp_congestion` | Large transfer, `cwnd` grows correctly, loss handled | [ ] |
| `test_dhcp_lease` | DHCP discover → offer → request → ack, IP assigned | [ ] |
| `test_http_get` | `curl http://example.com` over TCP, HTTP response received | [ ] |

---

*Design document produced by Phase 5 Planning*
*2026-05-19*
