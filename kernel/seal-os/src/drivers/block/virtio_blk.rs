use crate::drivers::pci::{get_devices, PciDevice};
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr;
use x86_64::instructions::port::Port;

/// Represents a generic block device.
pub trait BlockDevice {
    /// Read blocks from the device into the provided buffer.
    fn read_blocks(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str>;
    /// Write blocks to the device from the provided buffer.
    fn write_blocks(&mut self, lba: u64, buf: &[u8]) -> Result<(), &'static str>;
    /// Flush any pending writes to the device.
    fn flush(&mut self) -> Result<(), &'static str>;
}

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
}

/// Virtio Block Device
pub struct VirtioBlk {
    pub base_addr: u64,
    pub is_mmio: bool,
    pub queue: SplitVirtqueue,
    pub status: u32,
}

impl VirtioBlk {
    pub const fn new(base_addr: u64, is_mmio: bool) -> Self {
        VirtioBlk {
            base_addr,
            is_mmio,
            queue: SplitVirtqueue::new(),
            status: 0,
        }
    }

    fn write_u8(&self, offset: u64, val: u8) {
        if self.is_mmio {
            unsafe { core::ptr::write_volatile((self.base_addr + offset) as *mut u8, val) }
        } else {
            unsafe { Port::<u8>::new((self.base_addr + offset) as u16).write(val) }
        }
    }

    fn read_u8(&self, offset: u64) -> u8 {
        if self.is_mmio {
            unsafe { core::ptr::read_volatile((self.base_addr + offset) as *const u8) }
        } else {
            unsafe { Port::<u8>::new((self.base_addr + offset) as u16).read() }
        }
    }

    fn write_u16(&self, offset: u64, val: u16) {
        if self.is_mmio {
            unsafe { core::ptr::write_volatile((self.base_addr + offset) as *mut u16, val) }
        } else {
            unsafe { Port::<u16>::new((self.base_addr + offset) as u16).write(val) }
        }
    }

    fn read_u16(&self, offset: u64) -> u16 {
        if self.is_mmio {
            unsafe { core::ptr::read_volatile((self.base_addr + offset) as *const u16) }
        } else {
            unsafe { Port::<u16>::new((self.base_addr + offset) as u16).read() }
        }
    }

    fn write_u32(&self, offset: u64, val: u32) {
        if self.is_mmio {
            unsafe { core::ptr::write_volatile((self.base_addr + offset) as *mut u32, val) }
        } else {
            unsafe { Port::<u32>::new((self.base_addr + offset) as u16).write(val) }
        }
    }

    fn read_u32(&self, offset: u64) -> u32 {
        if self.is_mmio {
            unsafe { core::ptr::read_volatile((self.base_addr + offset) as *const u32) }
        } else {
            unsafe { Port::<u32>::new((self.base_addr + offset) as u16).read() }
        }
    }

    /// Real PCI discovery and initialization for Virtio-blk
    /// Finds Vendor ID: 0x1AF4, Device ID: 0x1001 or 0x1045
    pub fn discover_and_init() -> Result<Self, &'static str> {
        let devices = get_devices();
        let mut target_dev = None;
        for d in devices {
            if d.vendor_id == 0x1AF4 && (d.device_id == 0x1001 || d.device_id == 0x1045) {
                target_dev = Some(d);
                break;
            }
        }

        let dev = target_dev.ok_or("Virtio-blk device not found")?;

        let is_mmio = dev.bar0 & 1 == 0;
        let base_addr = dev.bar_address(0);
        let mut blk = VirtioBlk::new(base_addr, is_mmio);

        blk.init_device()?;

        Ok(blk)
    }

    fn init_device(&mut self) -> Result<(), &'static str> {
        // 1. Reset device
        self.write_u8(0x12, 0);

        // 2. Set ACKNOWLEDGE and DRIVER
        let mut status = self.read_u8(0x12);
        status |= 1; // ACKNOWLEDGE
        self.write_u8(0x12, status);
        status |= 2; // DRIVER
        self.write_u8(0x12, status);

        // 3. Negotiate Features (Accept all existing)
        let features = self.read_u32(0x00);
        self.write_u32(0x04, features);

        // 4. Setup Queue 0
        self.write_u16(0x0E, 0); // Queue Select
        let q_size = self.read_u16(0x0C);
        if q_size == 0 {
            return Err("Queue 0 is unavailable");
        }
        if q_size > 256 {
            return Err("Queue size too large");
        }

        // Allocate 3 contiguous pages (12KB) for split virtqueue components
        let layout = Layout::from_size_align(12288, 4096)
            .map_err(|_| "Failed to create virtqueue layout")?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("Failed to allocate virtqueue memory");
        }

        self.queue.queue_size = q_size;
        self.queue.desc = ptr as *mut VirtqDesc;
        self.queue.avail = unsafe { ptr.add(4096) } as *mut VirtqAvail;
        self.queue.used = unsafe { ptr.add(8192) } as *mut VirtqUsed;

        // Give PFN (Page Frame Number) to device
        let pfn = (ptr as u64) / 4096;
        self.write_u32(0x08, pfn as u32);

        // 5. Set DRIVER_OK
        status |= 4; // DRIVER_OK
        self.write_u8(0x12, status);

        self.status = status as u32;
        Ok(())
    }

    fn do_request(
        &mut self,
        lba: u64,
        buf: *mut u8,
        len: u32,
        is_write: bool,
    ) -> Result<(), &'static str> {
        let req_idx = self.queue.alloc_desc().ok_or("No desc available")?;
        let buf_idx = self.queue.alloc_desc().ok_or("No desc available")?;
        let stat_idx = self.queue.alloc_desc().ok_or("No desc available")?;

        let mut req = VirtioBlkReq {
            type_: if is_write { 1 } else { 0 }, // VIRTIO_BLK_T_OUT or VIRTIO_BLK_T_IN
            reserved: 0,
            sector: lba,
        };
        let mut blk_status: u8 = 255;

        unsafe {
            // Setup Request Descriptor
            let desc1 = &mut *self.queue.desc.add(req_idx as usize);
            desc1.addr = &mut req as *mut _ as u64;
            desc1.len = core::mem::size_of::<VirtioBlkReq>() as u32;
            desc1.flags = 1; // VRING_DESC_F_NEXT
            desc1.next = buf_idx;

            // Setup Buffer Descriptor
            let desc2 = &mut *self.queue.desc.add(buf_idx as usize);
            desc2.addr = buf as u64;
            desc2.len = len;
            desc2.flags = 1 | (if is_write { 0 } else { 2 }); // NEXT | (WRITE if reading into buffer)
            desc2.next = stat_idx;

            // Setup Status Descriptor
            let desc3 = &mut *self.queue.desc.add(stat_idx as usize);
            desc3.addr = &mut blk_status as *mut _ as u64;
            desc3.len = 1;
            desc3.flags = 2; // VRING_DESC_F_WRITE
            desc3.next = 0;

            // Add to available ring
            let avail = &mut *self.queue.avail;
            let avail_idx = avail.idx % self.queue.queue_size;
            avail.ring[avail_idx as usize] = req_idx;

            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            avail.idx = avail.idx.wrapping_add(1);
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        }

        // Notify device for Queue 0
        self.write_u16(0x10, 0);

        // Poll used ring for completion
        loop {
            unsafe {
                let used = &*self.queue.used;
                if self.queue.last_used_idx != used.idx {
                    let used_idx = self.queue.last_used_idx % self.queue.queue_size;
                    let elem = &used.ring[used_idx as usize];
                    if elem.id == req_idx as u32 {
                        self.queue.last_used_idx = self.queue.last_used_idx.wrapping_add(1);
                        break;
                    }
                    self.queue.last_used_idx = self.queue.last_used_idx.wrapping_add(1);
                }
            }
            core::hint::spin_loop();
        }

        // Free descriptors
        self.queue.free_desc(req_idx);
        self.queue.free_desc(buf_idx);
        self.queue.free_desc(stat_idx);

        if blk_status == 0 {
            Ok(())
        } else {
            Err("Virtio-blk request failed")
        }
    }
}

impl BlockDevice for VirtioBlk {
    fn read_blocks(&mut self, lba: u64, buf: &mut [u8]) -> Result<(), &'static str> {
        self.do_request(lba, buf.as_mut_ptr(), buf.len() as u32, false)
    }

    fn write_blocks(&mut self, lba: u64, buf: &[u8]) -> Result<(), &'static str> {
        self.do_request(lba, buf.as_ptr() as *mut u8, buf.len() as u32, true)
    }

    fn flush(&mut self) -> Result<(), &'static str> {
        // Optional for basic implementation
        Ok(())
    }
}

/// Initialize the Virtio Block Driver
pub fn init() {
    let _blk = VirtioBlk::discover_and_init();
}
