# Virtio-blk Driver — Design Document

> **Phase 4 Task**: BLK-002
> **Priority**: MEDIUM (alternative to AHCI; required for cloud/QEMU default)
> **Estimated Effort**: 3 weeks
> **Blocked by**: Interrupt framework (MSI/legacy IRQ), physical allocator, DMA allocator
> **Blocks**: Block layer abstraction, buffer cache, root filesystem on virtio

---

## Overview

AHCI requires emulated PCI SATA. QEMU/KVM and cloud providers default to `virtio-blk` for better performance (paravirtualized, no ATA command overhead). This driver provides a virtio-blk backend for ManifoldFS and the block layer.

---

## Architecture

### 1. VirtIO Transport Layer

VirtIO devices sit on PCI (legacy) or MMIO (embedded). We target PCI transport first.

**PCI Discovery:**
- Vendor 0x1AF4, Device IDs: 0x1001 (blk), 0x1045 (modern blk)
- BAR0/1 = device configuration space (MMIO or I/O ports)

- [ ] Scan PCI bus for vendor 0x1AF4
- [ ] Match device ID 0x1001 (legacy virtio-blk) or 0x1045 (modern)
- [ ] Read BAR0/1 for MMIO base address
- [ ] Verify BAR is memory-mapped (not I/O ports)
- [ ] Log discovered device to serial

**Device Init:**
```rust
fn virtio_pci_init(dev: &PciDevice) -> VirtioDevice {
    // 1. Reset device
    write_status(dev, 0);
    
    // 2. Acknowledge and set DRIVER
    write_status(dev, VIRTIO_STATUS_ACKNOWLEDGE | VIRTIO_STATUS_DRIVER);
    
    // 3. Negotiate features
    let features = read_host_features(dev);
    let wanted = VIRTIO_BLK_F_SIZE_MAX | VIRTIO_BLK_F_SEG_MAX | VIRTIO_BLK_F_BLK_SIZE;
    write_guest_features(dev, features & wanted);
    
    // 4. Setup virtqueues
    let vq = setup_virtqueue(dev, queue_idx=0, max_queue_size=256);
    
    // 5. Set DRIVER_OK
    write_status(dev, VIRTIO_STATUS_DRIVER_OK);
    
    VirtioDevice { dev, vq, negotiated_features: features & wanted }
}
```

- [ ] Reset device (write 0 to status)
- [ ] Set `ACKNOWLEDGE` status bit
- [ ] Set `DRIVER` status bit
- [ ] Read host features
- [ ] Negotiate wanted features (SIZE_MAX, SEG_MAX, BLK_SIZE)
- [ ] Write negotiated features back
- [ ] Allocate virtqueue 0 for I/O
- [ ] Set `DRIVER_OK` status bit
- [ ] Verify device status reads back as expected
- [ ] Handle device failure (status != OK after init)

### 2. Virtqueue (Split Virtqueue)

```rust
struct Virtqueue {
    desc: &'static mut [VirtqDesc; 256],      // Descriptor table (DMA)
    avail: &'static mut VirtqAvail,            // Available ring (DMA)
    used: &'static mut VirtqUsed,              // Used ring (DMA)
    free_head: u16,                            // Free descriptor chain head
    num_added: u16,
    last_used_idx: u16,
    notify_offset: u16,
}

#[repr(C)]
struct VirtqDesc {
    addr: u64,
    len: u32,
    flags: u16,     // NEXT, WRITE, INDIRECT
    next: u16,
}
```

- [ ] Define `VirtqDesc`, `VirtqAvail`, `VirtqUsed` structs
- [ ] Allocate descriptor table as DMA-coherent memory
- [ ] Allocate available ring as DMA-coherent memory
- [ ] Allocate used ring as DMA-coherent memory
- [ ] Initialize free-list linked via `next` field
- [ ] Implement `alloc_desc_chain(n)` returning linked descriptors
- [ ] Implement `free_desc_chain(head)` returning chain to free-list
- [ ] Implement `vq_submit(chain)` writing to avail ring
- [ ] Implement memory barrier (`fence(SeqCst)`) before notifying device
- [ ] Implement `vq_notify()` writing to notification register
- [ ] Implement `vq_poll()` checking used ring index
- [ ] Implement `vq_get_completion()` returning completed descriptor chain

### 3. Virtio-blk Protocol

```rust
#[repr(C)]
struct VirtioBlkReq {
    typ: u32,       // VIRTIO_BLK_T_IN = 0, VIRTIO_BLK_T_OUT = 1, FLUSH = 4, DISCARD = 11, WRITE_ZEROES = 13
    reserved: u32,
    sector: u64,    // LBA (512-byte sectors)
}

#[repr(C)]
struct VirtioBlkResp {
    status: u8,     // VIRTIO_BLK_S_OK = 0, IOERR = 1, UNSUPP = 2
}
```

- [ ] Define `VirtioBlkReq` struct
- [ ] Define `VirtioBlkResp` struct
- [ ] Define request type constants (IN, OUT, FLUSH, DISCARD, WRITE_ZEROES)
- [ ] Define response status constants (OK, IOERR, UNSUPP)
- [ ] Build read request descriptor chain (header + data + status)
- [ ] Build write request descriptor chain (header + data + status)
- [ ] Build flush request descriptor chain (header + status)
- [ ] Submit request and wait for completion
- [ ] Verify response status == OK
- [ ] Handle IOERR by returning `BlockError::IoError`
- [ ] Handle UNSUPP by returning `BlockError::Unsupported`

### 4. Block Device Abstraction

```rust
pub trait BlockDevice: Send + Sync {
    fn sector_size(&self) -> u64;
    fn num_sectors(&self) -> u64;
    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError>;
    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError>;
    fn flush(&self) -> Result<(), BlockError>;
}
```

- [x] Define `BlockDevice` trait
- [x] Define `BlockError` enum (IoError, Unsupported, Timeout, NoDevice)
- [x] Implement `BlockDevice` for `AhciController`
- [ ] Implement `BlockDevice` for `VirtioBlk`
- [x] Create `BlockDeviceRegistry` for runtime device lookup
- [ ] Implement device major/minor number assignment
- [ ] Add `BlockDevice` to devtmpfs on registration

### 5. Multi-Queue Support (Modern VirtIO)

Modern virtio-blk exposes multiple virtqueues (one per VCPU). For SMP:
```rust
struct VirtioBlk {
    dev: VirtioDevice,
    queues: Vec<Virtqueue>,  // one per CPU
}

fn read_sectors(&self, lba: u64, buf: &mut [u8]) {
    let cpu = current_cpu_number();
    let vq = &self.queues[cpu % self.queues.len()];
    submit_request(vq, lba, buf, READ);
}
```

- [ ] Detect number of available virtqueues
- [ ] Allocate one virtqueue per CPU (up to device max)
- [ ] Route requests to queue for current CPU
- [ ] Implement queue-per-CPU for v2 (v1 uses single queue)

### 6. Interrupt vs Polling

**Phase 1 (polling):**
- [ ] Submit request, spin on used-ring index
- [ ] Timeout after 1 second → `BlockError::Timeout`
- [x] No IRQ setup required

**Phase 2 (interrupt-driven):**
- [ ] Submit request, return
- [ ] ISR reads used ring, wakes waiters
- [ ] Configure MSI-X or legacy PCI IRQ
- [ ] Implement completion handler in ISR
- [ ] `raise_softirq()` from ISR to process completions
- [ ] Per-request `Waker` in `BTreeMap<req_id, Waker>`

### 7. DMA Allocator

All buffers passed to virtio must be contiguous in physical memory.

```rust
pub fn dma_alloc(size: usize) -> DmaBuffer {
    let pages = (size + 4095) / 4096;
    let frame = alloc_contiguous_frames(pages)?;
    DmaBuffer {
        phys: frame,
        virt: phys_to_virt(frame),
        pages,
    }
}
```

- [ ] Implement `DmaBuffer` struct with phys/virt/pages
- [ ] Implement `dma_alloc(size)` using contiguous frame allocation
- [ ] Verify all DMA frames are below 4 GiB (32-bit DMA constraint)
- [ ] Implement `dma_free(buf)` returning frames to allocator
- [ ] Ensure DMA buffers are cache-coherent (identity-mapped, no cache tricks needed for QEMU)

---

## Verification

| Test | What it proves | Status |
|---|---|---|
| `test_virtio_pci_found` | PCI scan discovers vendor 0x1AF4, device 0x1001 or 0x1045 | [ ] |
| `test_virtio_negotiate` | Feature negotiation succeeds, DRIVER_OK set | [ ] |
| `test_virtio_read_sector_0` | Read LBA 0 returns boot sector magic (0xAA55) | [ ] |
| `test_virtio_write_readback` | Write pattern to sector N, read back, compare | [ ] |
| `test_virtio_flush` | `flush()` returns OK, subsequent read is consistent | [ ] |
| `test_virtio_multi_sector` | Read 8 contiguous sectors (4 KiB) in one request | [ ] |
| `test_virtio_msi_interrupt` | Completion fires MSI interrupt, ISR runs, no polling | [ ] |
| `test_blockstore_over_virtio` | ManifoldFS mkfs, store, readback over virtio-blk | [ ] |

---

*Design document produced by Phase 4 Planning*
*2026-05-19*
