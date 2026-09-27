// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Virtio block driver: legacy (virtio 0.9.5) PCI transport, polled, one
//! request in flight, registered with the block layer as `VIRTIO_BLK_DEV_NUM`.
//!
//! The device DMAs to physical addresses. Every ring and buffer it touches is
//! therefore in physically contiguous frames below 4 GiB, which the kernel
//! identity-maps; caller buffers (heap-virtual, possibly scattered) are copied
//! through a bounce buffer.

use core::ptr::{self, addr_of, addr_of_mut, read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

use alloc::boxed::Box;
use spin::Mutex;
use x86_64::instructions::port::Port;

use super::{register_block_device, BlockDevice, BlockError};
use crate::drivers::pci::{get_devices, pci_read32, pci_write32, PciDevice};
use crate::memory::phys::{alloc_frames_contiguous_in_range, LOW_FRAME_LIMIT};
use crate::serial_println;

/// Block device number of the first virtio disk: Linux's `vda` (253:0).
pub const VIRTIO_BLK_DEV_NUM: u32 = 0xFD00;

/// A split virtqueue descriptor.
#[repr(C, align(16))]
pub struct VirtqDesc {
    pub addr: u64,
    pub len: u32,
    pub flags: u16,
    pub next: u16,
}

/// A split virtqueue available ring.
#[repr(C, align(2))]
pub struct VirtqAvail {
    pub flags: u16,
    pub idx: u16,
    pub ring: [u16; 256], // Assuming a max queue size of 256
    pub used_event: u16,
}

/// A split virtqueue used ring element.
#[repr(C)]
pub struct VirtqUsedElem {
    pub id: u32,
    pub len: u32,
}

/// A split virtqueue used ring.
#[repr(C, align(4))]
pub struct VirtqUsed {
    pub flags: u16,
    pub idx: u16,
    pub ring: [VirtqUsedElem; 256],
    pub avail_event: u16,
}

/// Virtio block request header
#[repr(C)]
pub struct VirtioBlkReq {
    pub type_: u32,
    pub reserved: u32,
    pub sector: u64,
}

/// Represents a split virtqueue.
pub struct SplitVirtqueue {
    pub desc: *mut VirtqDesc,
    pub avail: *mut VirtqAvail,
    pub used: *mut VirtqUsed,
    pub queue_size: u16,
    pub last_used_idx: u16,
    pub desc_avail: [bool; 256],
}

impl SplitVirtqueue {
    pub const fn new() -> Self {
        SplitVirtqueue {
            desc: ptr::null_mut(),
            avail: ptr::null_mut(),
            used: ptr::null_mut(),
            queue_size: 0,
            last_used_idx: 0,
            desc_avail: [true; 256],
        }
    }

    pub fn alloc_desc(&mut self) -> Option<u16> {
        for i in 0..self.queue_size {
            if self.desc_avail[i as usize] {
                self.desc_avail[i as usize] = false;
                return Some(i);
            }
        }
        None
    }

    pub fn free_desc(&mut self, idx: u16) {
        self.desc_avail[idx as usize] = true;
    }

    /// Allocate the three descriptors one block request needs (request
    /// header, data buffer, status byte). Either all three succeed, or every
    /// descriptor already taken is freed before returning `None` —
    /// `do_request` used to call `alloc_desc()` three times in a row with
    /// `?` and leaked whichever descriptors it had already taken when a
    /// later call failed, permanently wedging the queue whenever
    /// `queue_size` was smaller than 3.
    pub fn alloc_request_descs(&mut self) -> Option<(u16, u16, u16)> {
        let req = self.alloc_desc()?;
        let buf = match self.alloc_desc() {
            Some(idx) => idx,
            None => {
                self.free_desc(req);
                return None;
            }
        };
        let stat = match self.alloc_desc() {
            Some(idx) => idx,
            None => {
                self.free_desc(req);
                self.free_desc(buf);
                return None;
            }
        };
        Some((req, buf, stat))
    }
}

impl Default for SplitVirtqueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimum descriptors one in-flight block request needs: request header,
/// data buffer, status byte (see `SplitVirtqueue::alloc_request_descs`). A
/// device that reports fewer than this can never complete a single
/// `do_request` call.
const MIN_QUEUE_SIZE: u16 = 3;

/// Validate a device-reported virtqueue size before it sizes any allocation
/// or descriptor scan. Pulled out of `init_device` so it's testable without
/// real MMIO/PCI hardware.
fn validate_queue_size(q_size: u16) -> Result<(), &'static str> {
    if q_size == 0 {
        return Err("Queue 0 is unavailable");
    }
    if q_size < MIN_QUEUE_SIZE {
        return Err("Queue size too small for a block request");
    }
    if q_size > 256 {
        return Err("Queue size too large");
    }
    Ok(())
}

const SECTOR: usize = 512;

// Legacy virtio-pci registers (virtio 0.9.5, 2.1), offsets into I/O BAR0.
const REG_DEVICE_FEATURES: u16 = 0x00;
const REG_GUEST_FEATURES: u16 = 0x04;
const REG_QUEUE_PFN: u16 = 0x08;
const REG_QUEUE_SIZE: u16 = 0x0C;
const REG_QUEUE_SELECT: u16 = 0x0E;
const REG_QUEUE_NOTIFY: u16 = 0x10;
const REG_STATUS: u16 = 0x12;
const REG_ISR: u16 = 0x13;
/// Device-specific config with MSI-X off: `capacity`, u64, 512-byte sectors.
const REG_CAPACITY: u16 = 0x14;

const STATUS_ACKNOWLEDGE: u8 = 1;
const STATUS_DRIVER: u8 = 2;
const STATUS_DRIVER_OK: u8 = 4;
const STATUS_FAILED: u8 = 0x80;

const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;
const VRING_AVAIL_F_NO_INTERRUPT: u16 = 1;

const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;

/// Transitional virtio-blk. A modern-only device (0x1042) has no legacy I/O
/// BAR and is left alone.
const VIRTIO_VENDOR: u16 = 0x1AF4;
const VIRTIO_BLK_LEGACY_DEVICE: u16 = 0x1001;

/// Data moves in chunks of this many bytes through the bounce buffer.
const BOUNCE_PAGES: usize = 8;
const CHUNK_BYTES: usize = BOUNCE_PAGES * 4096;
/// Bytes into the header page where the device writes the status byte.
const STATUS_OFFSET: u64 = 16;

/// Polls of the used ring before a request is declared lost.
// ponytail: a lost request keeps its three descriptors (the device may still
// own them) and a late completion would desynchronise `last_used_idx`; the
// disk is then unusable until reboot. Upgrade path: interrupt-driven
// completion with per-request tokens.
const REQUEST_SPIN_LIMIT: u64 = 200_000_000;

const fn align_page(bytes: usize) -> usize {
    (bytes + 4095) & !4095
}

/// Byte offsets of the available ring and the used ring, and the total size,
/// of a legacy split virtqueue of `size` entries (virtio 0.9.5, 2.3): the
/// descriptor table, then the available ring, then the used ring starting on
/// the next 4096-byte boundary.
const fn legacy_queue_layout(size: usize) -> (usize, usize, usize) {
    let avail = 16 * size;
    let used = align_page(avail + 6 + 2 * size);
    (avail, used, used + align_page(6 + 8 * size))
}

// The layout the device computes from the PFN, for the two sizes QEMU uses.
const _: () = assert!(matches!(legacy_queue_layout(256), (4096, 8192, 12288)));
const _: () = assert!(matches!(legacy_queue_layout(128), (2048, 4096, 8192)));
const _: () = assert!(core::mem::size_of::<VirtqDesc>() == 16);
const _: () = assert!(core::mem::size_of::<VirtioBlkReq>() == 16);

fn outb(port: u16, value: u8) {
    // SAFETY: `port` is inside this device's legacy I/O BAR.
    unsafe { Port::<u8>::new(port).write(value) }
}

fn inb(port: u16) -> u8 {
    // SAFETY: as `outb`.
    unsafe { Port::<u8>::new(port).read() }
}

fn outw(port: u16, value: u16) {
    // SAFETY: as `outb`.
    unsafe { Port::<u16>::new(port).write(value) }
}

fn inw(port: u16) -> u16 {
    // SAFETY: as `outb`.
    unsafe { Port::<u16>::new(port).read() }
}

fn outl(port: u16, value: u32) {
    // SAFETY: as `outb`.
    unsafe { Port::<u32>::new(port).write(value) }
}

fn inl(port: u16) -> u32 {
    // SAFETY: as `outb`.
    unsafe { Port::<u32>::new(port).read() }
}

/// Queue 0 and its bounce memory. Every pointer is into frames this driver
/// allocated below 4 GiB, where virtual and physical addresses coincide.
struct Queue {
    ring: SplitVirtqueue,
    /// Request header at +0, status byte at +`STATUS_OFFSET`.
    header: u64,
    /// `CHUNK_BYTES` of data bounce.
    data: u64,
}

// SAFETY: the raw pointers name DMA frames owned by this queue; they are only
// dereferenced with the device's mutex held.
unsafe impl Send for Queue {}

impl Queue {
    /// Submit one request of `len` bytes (already in the bounce buffer for a
    /// write) and wait for the device to complete it.
    fn submit(&mut self, io: u16, kind: u32, sector: u64, len: usize) -> Result<(), BlockError> {
        let (req, data, status) = self.ring.alloc_request_descs().ok_or(BlockError::Busy)?;
        let size = self.ring.queue_size;
        let data_flags = if kind == VIRTIO_BLK_T_IN {
            VRING_DESC_F_NEXT | VRING_DESC_F_WRITE
        } else {
            VRING_DESC_F_NEXT
        };
        // SAFETY: header, data and the rings are identity-mapped frames owned
        // by this queue; the device touches them only between the notify
        // below and the used-ring update this function waits for.
        unsafe {
            write_volatile(
                self.header as *mut VirtioBlkReq,
                VirtioBlkReq {
                    type_: kind,
                    reserved: 0,
                    sector,
                },
            );
            write_volatile((self.header + STATUS_OFFSET) as *mut u8, 0xFF);
            let desc = self.ring.desc;
            write_volatile(
                desc.add(req as usize),
                VirtqDesc {
                    addr: self.header,
                    len: core::mem::size_of::<VirtioBlkReq>() as u32,
                    flags: VRING_DESC_F_NEXT,
                    next: data,
                },
            );
            write_volatile(
                desc.add(data as usize),
                VirtqDesc {
                    addr: self.data,
                    len: len as u32,
                    flags: data_flags,
                    next: status,
                },
            );
            write_volatile(
                desc.add(status as usize),
                VirtqDesc {
                    addr: self.header + STATUS_OFFSET,
                    len: 1,
                    flags: VRING_DESC_F_WRITE,
                    next: 0,
                },
            );
            let avail = self.ring.avail;
            let idx = read_volatile(addr_of!((*avail).idx));
            write_volatile(addr_of_mut!((*avail).ring[(idx % size) as usize]), req);
            fence(Ordering::SeqCst);
            write_volatile(addr_of_mut!((*avail).idx), idx.wrapping_add(1));
            fence(Ordering::SeqCst);
        }
        outw(io + REG_QUEUE_NOTIFY, 0);

        let used = self.ring.used;
        let mut spins = 0u64;
        // SAFETY: as above; `idx` is read volatile because the device writes it.
        while unsafe { read_volatile(addr_of!((*used).idx)) } == self.ring.last_used_idx {
            spins += 1;
            if spins > REQUEST_SPIN_LIMIT {
                return Err(BlockError::Timeout);
            }
            core::hint::spin_loop();
        }
        fence(Ordering::SeqCst);
        self.ring.last_used_idx = self.ring.last_used_idx.wrapping_add(1);
        // Reading ISR acknowledges the completion and drops the INTx line.
        let _ = inb(io + REG_ISR);
        self.ring.free_desc(req);
        self.ring.free_desc(data);
        self.ring.free_desc(status);
        // SAFETY: the device has completed the request and written the status.
        match unsafe { read_volatile((self.header + STATUS_OFFSET) as *const u8) } {
            0 => Ok(()),
            _ => Err(BlockError::IoError),
        }
    }
}

/// A virtio-blk disk on the legacy PCI transport.
pub struct VirtioBlk {
    io: u16,
    sectors: u64,
    queue: Mutex<Queue>,
}

impl VirtioBlk {
    /// Bring up a transitional virtio-blk function: reset, negotiate no
    /// optional features, hand queue 0 to the device, DRIVER_OK.
    fn init_device(dev: &PciDevice) -> Result<Self, &'static str> {
        if dev.bar0 & 1 == 0 {
            return Err("BAR0 is not I/O space (modern-only transport is not supported)");
        }
        let io = dev.bar_address(0) as u16;
        // I/O space decode and bus mastering: the device DMAs the rings.
        let cmd = pci_read32(dev.bus, dev.device, dev.function, 0x04);
        pci_write32(dev.bus, dev.device, dev.function, 0x04, cmd | 0x1 | 0x4);

        outb(io + REG_STATUS, 0);
        outb(io + REG_STATUS, STATUS_ACKNOWLEDGE);
        outb(io + REG_STATUS, STATUS_ACKNOWLEDGE | STATUS_DRIVER);
        // No optional features. Without VIRTIO_BLK_F_FLUSH (WCE) the device
        // runs its cache write-through, so a completed write is durable and
        // `flush` has nothing to do.
        let _ = inl(io + REG_DEVICE_FEATURES);
        outl(io + REG_GUEST_FEATURES, 0);

        outw(io + REG_QUEUE_SELECT, 0);
        let size = inw(io + REG_QUEUE_SIZE);
        if let Err(e) = validate_queue_size(size) {
            outb(io + REG_STATUS, STATUS_FAILED);
            return Err(e);
        }
        let (avail_off, used_off, ring_bytes) = legacy_queue_layout(size as usize);
        let (Some(ring), Some(bounce)) = (
            alloc_frames_contiguous_in_range(ring_bytes / 4096, 0, LOW_FRAME_LIMIT),
            alloc_frames_contiguous_in_range(1 + BOUNCE_PAGES, 0, LOW_FRAME_LIMIT),
        ) else {
            outb(io + REG_STATUS, STATUS_FAILED);
            return Err("no contiguous DMA memory below 4 GiB");
        };
        let ring = ring.as_u64();
        let bounce = bounce.as_u64();
        // SAFETY: freshly allocated, identity-mapped frames owned by this queue.
        unsafe {
            ptr::write_bytes(ring as *mut u8, 0, ring_bytes);
            ptr::write_bytes(bounce as *mut u8, 0, 4096);
            write_volatile(
                addr_of_mut!((*((ring + avail_off as u64) as *mut VirtqAvail)).flags),
                VRING_AVAIL_F_NO_INTERRUPT,
            );
        }
        let mut queue = SplitVirtqueue::new();
        queue.queue_size = size;
        queue.desc = ring as *mut VirtqDesc;
        queue.avail = (ring + avail_off as u64) as *mut VirtqAvail;
        queue.used = (ring + used_off as u64) as *mut VirtqUsed;
        outl(io + REG_QUEUE_PFN, (ring / 4096) as u32);
        outb(
            io + REG_STATUS,
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_DRIVER_OK,
        );

        let sectors = inl(io + REG_CAPACITY) as u64 | ((inl(io + REG_CAPACITY + 4) as u64) << 32);
        Ok(Self {
            io,
            sectors,
            queue: Mutex::new(Queue {
                ring: queue,
                header: bounce,
                data: bounce + 4096,
            }),
        })
    }

    fn check(&self, lba: u64, len: usize) -> Result<(), BlockError> {
        let count = (len / SECTOR) as u64;
        let past_end = lba.checked_add(count).is_none_or(|end| end > self.sectors);
        if len == 0 || len % SECTOR != 0 || past_end {
            return Err(BlockError::InvalidLba);
        }
        Ok(())
    }
}

impl BlockDevice for VirtioBlk {
    fn sector_size(&self) -> u64 {
        SECTOR as u64
    }

    fn num_sectors(&self) -> u64 {
        self.sectors
    }

    fn read_sectors(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        self.check(lba, buf.len())?;
        let mut queue = self.queue.lock();
        for (i, chunk) in buf.chunks_mut(CHUNK_BYTES).enumerate() {
            let sector = lba + (i * CHUNK_BYTES / SECTOR) as u64;
            queue.submit(self.io, VIRTIO_BLK_T_IN, sector, chunk.len())?;
            // SAFETY: the bounce holds `CHUNK_BYTES >= chunk.len()` bytes the
            // device just wrote.
            unsafe {
                ptr::copy_nonoverlapping(queue.data as *const u8, chunk.as_mut_ptr(), chunk.len())
            };
        }
        Ok(())
    }

    fn write_sectors(&self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        self.check(lba, buf.len())?;
        let mut queue = self.queue.lock();
        for (i, chunk) in buf.chunks(CHUNK_BYTES).enumerate() {
            let sector = lba + (i * CHUNK_BYTES / SECTOR) as u64;
            // SAFETY: the bounce holds `CHUNK_BYTES >= chunk.len()` bytes and
            // the device is idle between requests.
            unsafe { ptr::copy_nonoverlapping(chunk.as_ptr(), queue.data as *mut u8, chunk.len()) };
            queue.submit(self.io, VIRTIO_BLK_T_OUT, sector, chunk.len())?;
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), BlockError> {
        // Write-through: see `init_device`.
        Ok(())
    }
}

/// Bring up the first transitional virtio-blk function and register it as
/// `VIRTIO_BLK_DEV_NUM`.
pub fn init() {
    let Some(dev) = get_devices()
        .into_iter()
        .find(|d| d.vendor_id == VIRTIO_VENDOR && d.device_id == VIRTIO_BLK_LEGACY_DEVICE)
    else {
        return;
    };
    match VirtioBlk::init_device(&dev) {
        Ok(blk) => {
            serial_println!(
                "[virtio-blk] {:02x}:{:02x}.{} legacy transport, queue={}, capacity={} sectors; registered as block device {:#x}",
                dev.bus,
                dev.device,
                dev.function,
                blk.queue.lock().ring.queue_size,
                blk.sectors,
                VIRTIO_BLK_DEV_NUM
            );
            register_block_device(VIRTIO_BLK_DEV_NUM, Box::leak(Box::new(blk)));
        }
        Err(e) => serial_println!(
            "[virtio-blk] {:02x}:{:02x}.{} not brought up: {}",
            dev.bus,
            dev.device,
            dev.function,
            e
        ),
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    /// Regression for the descriptor leak this module used to have:
    /// `do_request` called `alloc_desc()` three times in a row with `?` and
    /// never freed the descriptors it had already taken when a later call
    /// failed. With `queue_size` below 3 that wedged the queue for the rest
    /// of boot. This exercises the fix directly (`alloc_request_descs`),
    /// since `do_request` itself needs real MMIO/DMA memory and a device to
    /// answer the used ring — unreachable in this environment.
    fn test_alloc_request_descs_rolls_back_on_partial_failure() -> TestResult {
        let mut q = SplitVirtqueue::new();
        q.queue_size = 2; // one short of the three a request needs

        test_assert!(q.alloc_request_descs().is_none());

        // The old bug left both slots permanently marked busy here. Both
        // must still be available.
        test_assert!(q.alloc_desc().is_some());
        test_assert!(q.alloc_desc().is_some());
        test_assert!(q.alloc_desc().is_none());
        TestResult::Pass
    }

    /// With enough descriptors, all three are taken together and freeing
    /// them returns the queue to its starting state.
    fn test_alloc_request_descs_succeeds_with_three_slots() -> TestResult {
        let mut q = SplitVirtqueue::new();
        q.queue_size = 3;

        let (req, buf, stat) = match q.alloc_request_descs() {
            Some(t) => t,
            None => return TestResult::Fail("expected Some with queue_size = 3"),
        };
        test_assert!(q.alloc_desc().is_none());

        q.free_desc(req);
        q.free_desc(buf);
        q.free_desc(stat);
        test_assert!(q.alloc_request_descs().is_some());
        TestResult::Pass
    }

    /// A device-reported queue size below 3 is rejected before it ever
    /// reaches `alloc_request_descs` — a request structurally needs 3
    /// descriptors, so 1 or 2 can never complete one.
    fn test_queue_size_floor() -> TestResult {
        test_assert!(validate_queue_size(0).is_err());
        test_assert!(validate_queue_size(1).is_err());
        test_assert!(validate_queue_size(2).is_err());
        test_assert!(validate_queue_size(3).is_ok());
        test_assert!(validate_queue_size(256).is_ok());
        test_assert!(validate_queue_size(257).is_err());
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "virtio_blk::alloc_request_descs_rolls_back_on_partial_failure",
            test_alloc_request_descs_rolls_back_on_partial_failure,
        );
        crate::testing::register_test(
            "virtio_blk::alloc_request_descs_succeeds_with_three_slots",
            test_alloc_request_descs_succeeds_with_three_slots,
        );
        crate::testing::register_test("virtio_blk::queue_size_floor", test_queue_size_floor);
    }
}
