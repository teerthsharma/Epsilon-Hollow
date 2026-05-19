# AHCI SATA Driver — Design & Implementation Plan

## Status

**Current**: v1.0-alpha — basic PIO-style DMA reads/writes, single port, single command slot, 4 KB bounce buffer. Registered as block device `0x800`.

**Target**: v2.0 — full NCQ, multi-port, multi-sector, error recovery, topology-aware I/O prefetching via Aether-Link.

---

## 1. What's Already Working

The current driver (`drivers/block/ahci.rs`) implements:

| Feature | Status |
|---------|--------|
| PCI enumeration (class 0x01 / subclass 0x06 / prog-if 0x01) | ✅ Working |
| HBA reset + AHCI enable | ✅ Working |
| Port initialization (CLB / FB / command table) | ✅ Working |
| `IDENTIFY DEVICE` | ✅ Working |
| Read sector (512 B) via `READ DMA EXT` | ✅ Working |
| Write sector (512 B) via `WRITE DMA EXT` | ✅ Working |
| Block device registration (`0x800`) | ✅ Working |
| MBR signature test | ✅ Working |

**Architecture**: single `AhciController` struct with identity-mapped DMA buffers allocated via `alloc_frame()`. MMIO through `read_volatile` / `write_volatile`. No interrupts — pure polling.

---

## 2. What's Missing (The Gap)

### 2.1 Single-Threaded I/O Bottleneck
- Only **one command slot** (slot 0) is used
- Only **one port** (first SATA signature) is brought up
- The driver **stops and starts the port** on every command — ~microseconds of overhead per operation
- **No NCQ**: cannot issue multiple commands and let the drive reorder them

### 2.2 Bounce Buffer Limitation
- Single fixed **4 KB bounce buffer** (`buf: *mut [u8; 4096]`)
- Maximum transfer = 8 sectors (4096 / 512)
- Large reads/writes require looped stop-start-submit per 4 KB chunk
- No support for user-supplied buffers — everything copies through the bounce buffer

### 2.3 No Error Recovery
- If `send_command()` times out, the command slot is dead until reboot
- No port reset sequence on error
- No retry logic for transient failures
- `TFD_BSY` / `TFD_DRQ` checked but not acted upon beyond failing

### 2.4 No Multi-Device / Multi-Port
- Only scans for the first port with `SIG_SATA` (`0x00000101`)
- No support for port multipliers
- No support for ATAPI devices (CD/DVD, `SIG_ATAPI = 0xEB140101`)

### 2.5 No Interrupt-Driven I/O
- Every command polls the `PORT_CI` register for completion
- CPU burns cycles in `spin_loop()` waiting for the drive
- No interrupt coalescing — every command is a synchronous round-trip

### 2.6 No Topology Integration
- Aether-Link I/O prefetching is wired to a **mock telemetry stream** in test mode
- Real AHCI I/O does not feed telemetry into the 6D feature extractor
- No adaptive prefetch based on LBA stream patterns

---

## 3. Design — v2.0 AHCI Driver

### 3.1 Core Data Structures

```rust
/// One per HBA port that has a SATA device attached.
pub struct AhciPort {
    port_idx: u32,
    port_base: u64,

    // DMA structures (identity-mapped, physical addresses)
    cl_phys: u64,          // Command List (1 KB, 32 slots)
    fis_phys: u64,         // Received FIS (256 B)
    ct_phys: [u64; 32],    // Per-slot Command Tables (256 B each)
    buf_phys: [u64; 32],   // Per-slot DMA buffers (4 KB each)

    // Virtual pointers (for kernel access)
    cl: *mut CommandList,
    fis: *mut ReceivedFis,
    ct: [*mut CommandTable; 32],
    buf: [*mut [u8; 4096]; 32],

    // Slot tracking
    slot_busy: [AtomicBool; 32],
    slot_completion: [AtomicU32; 32], // PORT_IS bits captured on completion
}

/// One per AHCI controller (one PCI device).
pub struct AhciHba {
    base: u64,
    ports: Vec<AhciPort>,
    irq_vector: u8,
}

/// Command slot state machine.
enum SlotState {
    Free,
    Submitted { lba: u64, count: u16, write: bool },
    Completed(Result<(), BlockError>),
    Error(BlockError),
}
```

### 3.2 NCQ — Native Command Queuing

AHCI supports up to **32 command slots per port**. NCQ allows the drive to reorder commands for optimal seek patterns.

**Implementation**:
1. Allocate a **1 KB command list** (32 slots × 32 bytes)
2. Allocate **32 command tables** (one per slot, 256 B each, 128-byte aligned)
3. Allocate **32 DMA bounce buffers** (one per slot, 4 KB each)
4. Track slot occupancy with an atomic bitmask (`slot_busy: u32`)
5. When a request arrives, find the first free slot, fill the command table, set `PORT_CI` bit
6. The interrupt handler clears the slot on completion
7. Up to 32 commands can be in flight simultaneously

**Non-NCQ fallback**: If the drive does not support NCQ (check `WORD 76` bit 8 of IDENTIFY), serialize to one slot.

### 3.3 Multi-Sector & Scatter-Gather

**PRDT (Physical Region Descriptor Table)** supports up to 65535 entries per command table. Each PRDT entry describes one contiguous physical region:
- `DBA` — data base address (64-bit)
- `DBC` — byte count (22-bit, 0-based, max 4 MB per entry)
- `I` — interrupt on completion

**Implementation for large transfers**:
1. User requests N sectors
2. If N ≤ 8 sectors and buffer is physically contiguous → single PRDT entry, single slot
3. If N > 8 sectors or buffer is scattered → split into multiple PRDT entries (up to 256 per table)
4. Maximum per-command transfer: 256 PRDT entries × 4 MB = **1 GB per command**

### 3.4 Interrupt-Driven I/O

**Setup**:
1. Register the AHCI HBA IRQ (usually GSI 16+ via PCI INT# or MSI)
2. In `init()`, set `HBA_GHC.IE` (global interrupt enable)
3. Per port: set `PORT_IE` to enable `DHRS` (Device to Host Register FIS), `PSS` (PIO Setup), `DSS` (DMA Setup), `DPS` (Descriptor Processed), `UFS` (Unknown FIS)
4. In the ISR:
   - Read `HBA_IS` to find which port(s) fired
   - For each port, read `PORT_IS`, check `DPS` or `DHRS`
   - Clear `PORT_IS` by writing 1s to acknowledge
   - Wake waiters on completed slots

**Benefit**: CPU does not spin. Command throughput scales with drive parallelism.

### 3.5 Error Recovery

**Port-level reset sequence**:
1. Clear `PORT_CMD.ST` and `PORT_CMD.FRE`
2. Wait for `PORT_CMD.CR` and `PORT_CMD.FR` to clear (500 ms timeout)
3. Clear `PORT_SERR` and `PORT_IS`
4. Reprogram `PORT_CLB` / `PORT_FB`
5. Set `PORT_CMD.FRE`, wait for `PORT_CMD.FR`
6. Set `PORT_CMD.ST`, wait for `PORT_CMD.CR`
7. Retry the failed command on a fresh slot

**Retry policy**:
- Transient errors (`TFD_ERR`, `SERR_DIAG_x`): retry up to 3 times with exponential backoff
- Persistent errors: mark port as failed, return `BlockError::IoError` to caller

### 3.6 Multi-Port & ATAPI

**Port scanning**:
1. Read `HBA_PI` (ports implemented bitmask)
2. For each implemented port, read `PORT_SIG`:
   - `0x00000101` = SATA (hard drive / SSD)
   - `0xEB140101` = ATAPI (CD/DVD)
   - `0xC33C0101` = SEMB (enclosure management)
   - `0x96690101` = Port multiplier
3. Bring up each SATA port independently
4. ATAPI ports use `ATAPI PACKET` command (12/16-byte CDB) instead of register FIS

**Block device numbering**:
- `0x800` = first port
- `0x801` = second port
- etc.

---

## 4. Integration Roadmap

### Phase A: Foundation (Week 1)
- [x] Extend `AhciController` → `AhciPort` + `AhciHba` structs
- [x] Allocate 32-slot command list + 32 command tables + 32 DMA buffers
- [x] Rewrite `setup_command()` to target a specific slot
- [x] Add slot allocation / free tracking (`slot_busy` bitmask)
- [x] Port the existing read/write path to use slot 0 with the new structures
- [x] **Goal**: no functional change, but data structures support NCQ

### Phase B: NCQ + Multi-Sector (Week 2)
- [x] Implement slot allocation (`find_free_slot()`)
- [x] Support multiple in-flight commands (set multiple `PORT_CI` bits)
- [x] Extend PRDT to handle multi-sector transfers (> 8 sectors)
- [x] Add `read_sectors_ncq()` / `write_sectors_ncq()` with up to 32 concurrent commands
- [x] Check `IDENTIFY` word 76 for NCQ support; fall back to serialized mode
- [x] **Goal**: large sequential reads scale with drive queue depth

### Phase C: Interrupts (Week 3)
- [ ] Wire AHCI HBA to IDT (IRQ routing via I/O APIC or MSI)
- [ ] Implement `ahci_interrupt_handler()`
- [ ] Replace polling `send_command()` with interrupt-driven completion
- [ ] Add per-slot completion notification (atomic flag + optional waker)
- [ ] **Goal**: CPU idle during I/O wait, command latency drops

### Phase D: Error Recovery (Week 4)
- [ ] Implement `reset_port()` with full stop-start sequence
- [ ] Add retry logic with exponential backoff (3 retries max)
- [x] Parse `PORT_TFD` error bits and `PORT_SERR` diagnostic info
- [ ] Graceful degradation: if a port fails, other ports continue
- [ ] **Goal**: transient drive errors do not crash the kernel

### Phase E: Topology Integration (Week 5)
- [ ] Feed AHCI LBA stream into Aether-Link telemetry extractor
- [ ] 6D feature vector: `(lba, delta_lba, time_since_last, port_idx, cmd_type, queue_depth)`
- [ ] Angle-map via `fast_atan()` → adaptive prefetch trigger
- [ ] Integrate with `BlockStore::try_mount_ahci()` — when AHCI is present, bypass mock backend
- [ ] **Goal**: real disk I/O drives the ML prefetch path

---

## 5. Testing Strategy

| Test | Method | Expected Result |
|------|--------|-----------------|
| IDENTIFY on all ports | In-kernel test | Model string printed for each SATA device |
| Sequential read 1 MB | In-kernel test | 256 sectors read, no timeout, data checksum matches |
| Random read 4 KB | In-kernel test | 8 sectors read from scattered LBAs, all succeed |
| Write-read-verify 512 B | In-kernel test | Write pattern, read back, compare |
| NCQ stress (32 parallel) | In-kernel test | 32 commands submitted, all complete, no slot collision |
| Interrupt latency | QEMU trace | ISR fires within 10 µs of command completion |
| Port reset recovery | QEMU fault injection | Inject SERR, driver resets port, retries succeed |
| MBR boot sector | Existing test | `0x55AA` signature at offset 510 |

---

## 6. File Layout

```
kernel/seal-os/src/drivers/block/
├── mod.rs          # BlockDevice trait, registry, read_block()/write_block()
├── ahci.rs         # v2.0: AhciHba, AhciPort, NCQ, interrupts, error recovery
└── atapi.rs        # NEW: ATAPI PACKET command support (CD/DVD)
```

---

## 7. Dependencies

- `drivers/pci.rs` — BAR5 (ABAR) lookup, MSI/MSI-X capability parsing (future)
- `drivers/interrupts.rs` — IDT entry for AHCI IRQ
- `drivers/apic.rs` — I/O APIC routing or MSI delivery
- `memory/phys.rs` — DMA buffer allocation (identity-mapped below 4 GiB)
- `aether-link/` — Telemetry feature extraction for adaptive prefetch

---

*Document version: 1.0*
*Target kernel version: Seal OS v1.1.0*
