# 09 — Network Stack: Real TCP/UDP/DHCP/DNS over VirtIO-Net

## Goal

Replace the entirely simulated network stack with a real implementation: virtio-net NIC driver, ARP, IP, UDP, TCP, DHCP, DNS — enough to send and receive packets on a real (or QEMU virtual) network.

## Current State

All files under `kernel/seal-os/src/drivers/net/`:
- `mod.rs` (45 lines): `NetworkStack` with `configure()` — stores config locally, no NIC programmed
- `tcp.rs` (65 lines): `TcpSocket` with full state machine (`SynSent`, `Established`, `FinWait*`, etc.) — but `connect()` never sends SYN, `send()` buffers locally never transmits, `recv()` always returns empty
- `udp.rs` (27 lines): stub
- `dhcp.rs` (53 lines): stub
- `dns.rs` (26 lines): stub
- `http.rs` (22 lines): stub
- `tls.rs` (45 lines): stub

**Summary: Data structures are correct, but zero bytes ever hit a NIC.**

## Implementation Steps

### Phase A: VirtIO-Net Driver

1. **Implement virtio-net device discovery**
   - Use PCI enumeration from `drivers/pci.rs` (real — does port I/O)
   - Find virtio-net device (vendor 0x1AF4, device 0x1000 or 0x1041)
   - Map BAR0 for virtio registers

2. **Implement virtqueue**
   - Descriptor ring, available ring, used ring (standard virtio spec)
   - Allocate DMA-safe buffers for TX and RX queues
   - Interrupt handler for RX completion (integrate with existing IRQ system)

3. **Implement packet send/receive**
   - `nic_send(frame: &[u8])` — place frame in TX virtqueue, notify device
   - `nic_recv() -> Option<Vec<u8>>` — poll RX virtqueue for completed frames
   - Handle virtio-net header (flags, gso_type, etc.)

### Phase B: Ethernet + IP

4. **Ethernet frame parsing/construction**
   - Parse: dest MAC, src MAC, EtherType, payload
   - Construct: wrap IP packet in Ethernet frame with ARP-resolved dest MAC

5. **ARP**
   - ARP request/reply for IPv4 address resolution
   - ARP cache (`BTreeMap<Ipv4Addr, MacAddr>`) with timeout

6. **IPv4**
   - Parse/construct IPv4 headers (version, IHL, total length, TTL, protocol, checksum, src/dst)
   - IP checksum computation
   - No fragmentation (set DF bit, MTU = 1500)

### Phase C: UDP + DHCP + DNS

7. **UDP**
   - Parse/construct UDP datagrams
   - Socket API: `bind(port)`, `send_to(addr, data)`, `recv_from() -> (addr, data)`

8. **DHCP client**
   - DHCP Discover → Offer → Request → Ack
   - Extract: IP address, subnet mask, gateway, DNS server
   - Configure `NetworkStack` with obtained values

9. **DNS resolver**
   - Construct DNS query (A record)
   - Send via UDP to configured DNS server (port 53)
   - Parse response, extract IP address
   - Cache results with TTL

### Phase D: TCP

10. **TCP implementation** (rewrite existing state machine to actually send packets)
    - 3-way handshake: SYN → SYN-ACK → ACK
    - Data transfer: sequence numbers, acknowledgments, sliding window
    - Connection teardown: FIN → FIN-ACK → ACK
    - Retransmission: basic timeout-based retransmit (no congestion control initially)
    - Checksum: TCP pseudo-header checksum

### Phase E: Integration

11. **Wire WiFi driver to real NIC**
    - If virtio-net is available, `wifi.rs` delegates to real NIC instead of simulation
    - Or keep WiFi as a separate concept and use Ethernet for wired networking

12. **HTTP client (optional)**
    - Basic HTTP/1.1 GET over TCP
    - Enough to fetch a URL and display the response

## Dependencies

- **01-kernel-safety** (IRQ handlers for NIC interrupts)
- **02-memory-allocator** (DMA buffer allocation needs real dealloc for buffer reuse)

## Acceptance Criteria

- [ ] QEMU with `-netdev user,id=net0 -device virtio-net-pci,netdev=net0`: NIC detected and initialized
- [ ] DHCP: obtains IP address from QEMU's built-in DHCP server
- [ ] DNS: resolves a hostname to IP address
- [ ] UDP: send and receive UDP datagrams
- [ ] TCP: complete 3-way handshake, send/receive data, clean close
- [ ] `ping` equivalent: send ICMP echo request, receive reply (optional)
- [ ] No simulated data — all packets are real wire traffic
- [ ] Existing `NetworkStack` API preserved (backward compatible)

## Files to Modify

- `kernel/seal-os/src/drivers/net/mod.rs` (real NIC integration)
- `kernel/seal-os/src/drivers/net/tcp.rs` (wire to real packets)
- `kernel/seal-os/src/drivers/net/udp.rs` (wire to real packets)
- `kernel/seal-os/src/drivers/net/dhcp.rs` (real DHCP client)
- `kernel/seal-os/src/drivers/net/dns.rs` (real DNS resolver)

## Files to Create

- `kernel/seal-os/src/drivers/virtio_net.rs` (virtio-net NIC driver)
- `kernel/seal-os/src/drivers/net/arp.rs` (ARP cache + protocol)
- `kernel/seal-os/src/drivers/net/ip.rs` (IPv4 layer)
- `kernel/seal-os/src/drivers/net/ethernet.rs` (frame construction)
- `kernel/seal-os/src/drivers/net/icmp.rs` (optional: ping)
