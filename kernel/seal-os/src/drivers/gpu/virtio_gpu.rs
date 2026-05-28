// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! VirtIO-GPU driver — 2D acceleration support.

use crate::drivers::pci::{get_devices, PciDevice};
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr;
use x86_64::instructions::port::Port;

// VirtIO-GPU Command Types
pub const VIRTIO_GPU_CMD_GET_DISPLAY_INFO: u32 = 0x0100;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
pub const VIRTIO_GPU_CMD_RESOURCE_UNREF: u32 = 0x0102;
pub const VIRTIO_GPU_CMD_SET_SCANOUT: u32 = 0x0103;
pub const VIRTIO_GPU_CMD_RESOURCE_FLUSH: u32 = 0x0104;
pub const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
pub const VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
pub const VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;
pub const VIRTIO_GPU_CMD_GET_CAPSET_INFO: u32 = 0x0108;
pub const VIRTIO_GPU_CMD_GET_CAPSET: u32 = 0x0109;
pub const VIRTIO_GPU_CMD_GET_EDID: u32 = 0x010a;

// VirtIO-GPU Formats
pub const VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM: u32 = 1;

#[repr(C)]
pub struct VirtioGpuCtrlHdr {
    pub type_: u32,
    pub flags: u32,
    pub fence_id: u64,
    pub ctx_id: u32,
    pub padding: u32,
}

#[repr(C)]
pub struct VirtioGpuRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
pub struct VirtioGpuResourceCreate2d {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
pub struct VirtioGpuResourceAttachBacking {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: u32,
    pub nr_entries: u32,
}

#[repr(C)]
pub struct VirtioGpuMemEntry {
    pub addr: u64,
    pub length: u32,
    pub padding: u32,
}

#[repr(C)]
pub struct VirtioGpuSetScanout {
    pub hdr: VirtioGpuCtrlHdr,
    pub r: VirtioGpuRect,
    pub scanout_id: u32,
    pub resource_id: u32,
}

#[repr(C)]
pub struct VirtioGpuTransferToHost2d {
    pub hdr: VirtioGpuCtrlHdr,
    pub r: VirtioGpuRect,
    pub offset: u64,
    pub resource_id: u32,
    pub padding: u32,
}

#[repr(C)]
pub struct VirtioGpuResourceFlush {
    pub hdr: VirtioGpuCtrlHdr,
    pub r: VirtioGpuRect,
    pub resource_id: u32,
    pub padding: u32,
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
    pub ring: [u16; 256],
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

pub struct SplitVirtqueue {
    pub desc: *mut VirtqDesc,
    pub avail: *mut VirtqAvail,
    pub used: *mut VirtqUsed,
    pub queue_size: u16,
    pub last_used_idx: u16,
    pub desc_avail: [bool; 256],
}

unsafe impl Send for SplitVirtqueue {}
unsafe impl Sync for SplitVirtqueue {}

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

pub struct VirtioGpu {
    pub base_addr: u64,
    pub is_mmio: bool,
    pub control_queue: SplitVirtqueue,
    pub cursor_queue: SplitVirtqueue,
    pub resource_id_counter: u32,
    pub fb_resource_id: u32,
    pub width: u32,
    pub height: u32,
}

unsafe impl Send for VirtioGpu {}
unsafe impl Sync for VirtioGpu {}

impl VirtioGpu {
    pub const fn new(base_addr: u64, is_mmio: bool) -> Self {
        VirtioGpu {
            base_addr,
            is_mmio,
            control_queue: SplitVirtqueue::new(),
            cursor_queue: SplitVirtqueue::new(),
            resource_id_counter: 1,
            fb_resource_id: 0,
            width: 1024,
            height: 768,
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

    pub fn discover_and_init() -> Result<Self, &'static str> {
        let devices = get_devices();
        let mut target_dev = None;
        for d in devices {
            if d.vendor_id == 0x1AF4 && (d.device_id == 0x1010 || d.device_id == 0x1050) {
                target_dev = Some(d);
                break;
            }
        }

        let dev = target_dev.ok_or("Virtio-GPU device not found")?;
        let is_mmio = dev.bar0 & 1 == 0;
        let base_addr = dev.bar_address(0);
        let mut gpu = VirtioGpu::new(base_addr, is_mmio);

        gpu.init_device()?;
        gpu.setup_framebuffer()?;

        Ok(gpu)
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

        // 3. Negotiate Features
        let features = self.read_u32(0x00);
        self.write_u32(0x04, features);

        // 4. Setup Control Queue (Queue 0)
        let mut cq = SplitVirtqueue::new();
        self.init_queue_internal(0, &mut cq)?;
        self.control_queue = cq;

        // 5. Setup Cursor Queue (Queue 1)
        let mut curq = SplitVirtqueue::new();
        self.init_queue_internal(1, &mut curq)?;
        self.cursor_queue = curq;

        // 6. Set DRIVER_OK
        status |= 4; // DRIVER_OK
        self.write_u8(0x12, status);

        Ok(())
    }

    fn init_queue_internal(
        &self,
        queue_idx: u16,
        vq: &mut SplitVirtqueue,
    ) -> Result<(), &'static str> {
        self.write_u16(0x0E, queue_idx);
        let q_size = self.read_u16(0x0C);
        if q_size == 0 {
            return Err("Queue unavailable");
        }
        if q_size > 256 {
            return Err("Queue size too large");
        }

        let layout = Layout::from_size_align(12288, 4096).map_err(|_| "Failed layout")?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("Failed alloc");
        }

        vq.queue_size = q_size;
        vq.desc = ptr as *mut VirtqDesc;
        vq.avail = unsafe { ptr.add(4096) } as *mut VirtqAvail;
        vq.used = unsafe { ptr.add(8192) } as *mut VirtqUsed;

        let pfn = (ptr as u64) / 4096;
        self.write_u32(0x08, pfn as u32);

        Ok(())
    }

    fn send_command(
        &mut self,
        cmd: *const u8,
        cmd_len: u32,
        resp: *mut u8,
        resp_len: u32,
    ) -> Result<(), &'static str> {
        let cmd_idx = self.control_queue.alloc_desc().ok_or("No desc")?;
        let resp_idx = self.control_queue.alloc_desc().ok_or("No desc")?;

        unsafe {
            let desc1 = &mut *self.control_queue.desc.add(cmd_idx as usize);
            desc1.addr = cmd as u64;
            desc1.len = cmd_len;
            desc1.flags = 1; // NEXT
            desc1.next = resp_idx;

            let desc2 = &mut *self.control_queue.desc.add(resp_idx as usize);
            desc2.addr = resp as u64;
            desc2.len = resp_len;
            desc2.flags = 2; // WRITE
            desc2.next = 0;

            let avail = &mut *self.control_queue.avail;
            avail.ring[avail.idx as usize % self.control_queue.queue_size as usize] = cmd_idx;
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            avail.idx = avail.idx.wrapping_add(1);
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
        }

        self.write_u16(0x10, 0); // Notify Queue 0

        // Poll
        loop {
            unsafe {
                let used = &*self.control_queue.used;
                if self.control_queue.last_used_idx != used.idx {
                    let used_idx = self.control_queue.last_used_idx % self.control_queue.queue_size;
                    let elem = &used.ring[used_idx as usize];
                    if elem.id == cmd_idx as u32 {
                        self.control_queue.last_used_idx =
                            self.control_queue.last_used_idx.wrapping_add(1);
                        break;
                    }
                    self.control_queue.last_used_idx =
                        self.control_queue.last_used_idx.wrapping_add(1);
                }
            }
            core::hint::spin_loop();
        }

        self.control_queue.free_desc(cmd_idx);
        self.control_queue.free_desc(resp_idx);

        Ok(())
    }

    fn setup_framebuffer(&mut self) -> Result<(), &'static str> {
        let resource_id = self.resource_id_counter;
        self.resource_id_counter += 1;

        let create_2d = VirtioGpuResourceCreate2d {
            hdr: VirtioGpuCtrlHdr {
                type_: VIRTIO_GPU_CMD_RESOURCE_CREATE_2D,
                flags: 0,
                fence_id: 0,
                ctx_id: 0,
                padding: 0,
            },
            resource_id,
            format: VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM,
            width: self.width,
            height: self.height,
        };

        let mut resp = VirtioGpuCtrlHdr {
            type_: 0,
            flags: 0,
            fence_id: 0,
            ctx_id: 0,
            padding: 0,
        };
        self.send_command(
            &create_2d as *const _ as *const u8,
            core::mem::size_of_val(&create_2d) as u32,
            &mut resp as *mut _ as *mut u8,
            core::mem::size_of_val(&resp) as u32,
        )?;

        // Attach backing
        let fb_size = self.width * self.height * 4;
        let layout =
            Layout::from_size_align(fb_size as usize, 4096).map_err(|_| "FB layout fail")?;
        let fb_ptr = unsafe { alloc_zeroed(layout) };

        let attach = VirtioGpuResourceAttachBacking {
            hdr: VirtioGpuCtrlHdr {
                type_: VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING,
                flags: 0,
                fence_id: 0,
                ctx_id: 0,
                padding: 0,
            },
            resource_id,
            nr_entries: 1,
        };

        #[repr(C)]
        struct AttachBackingFull {
            attach: VirtioGpuResourceAttachBacking,
            entry: VirtioGpuMemEntry,
        }

        let attach_full = AttachBackingFull {
            attach,
            entry: VirtioGpuMemEntry {
                addr: fb_ptr as u64,
                length: fb_size,
                padding: 0,
            },
        };

        self.send_command(
            &attach_full as *const _ as *const u8,
            core::mem::size_of_val(&attach_full) as u32,
            &mut resp as *mut _ as *mut u8,
            core::mem::size_of_val(&resp) as u32,
        )?;

        // Set scanout
        let set_scanout = VirtioGpuSetScanout {
            hdr: VirtioGpuCtrlHdr {
                type_: VIRTIO_GPU_CMD_SET_SCANOUT,
                flags: 0,
                fence_id: 0,
                ctx_id: 0,
                padding: 0,
            },
            r: VirtioGpuRect {
                x: 0,
                y: 0,
                width: self.width,
                height: self.height,
            },
            scanout_id: 0,
            resource_id,
        };

        self.send_command(
            &set_scanout as *const _ as *const u8,
            core::mem::size_of_val(&set_scanout) as u32,
            &mut resp as *mut _ as *mut u8,
            core::mem::size_of_val(&resp) as u32,
        )?;

        self.fb_resource_id = resource_id;
        Ok(())
    }

    pub fn transfer_to_host_2d(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), &'static str> {
        let transfer = VirtioGpuTransferToHost2d {
            hdr: VirtioGpuCtrlHdr {
                type_: VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D,
                flags: 0,
                fence_id: 0,
                ctx_id: 0,
                padding: 0,
            },
            r: VirtioGpuRect {
                x,
                y,
                width: w,
                height: h,
            },
            offset: (y * self.width + x) as u64 * 4,
            resource_id: self.fb_resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr {
            type_: 0,
            flags: 0,
            fence_id: 0,
            ctx_id: 0,
            padding: 0,
        };
        self.send_command(
            &transfer as *const _ as *const u8,
            core::mem::size_of_val(&transfer) as u32,
            &mut resp as *mut _ as *mut u8,
            core::mem::size_of_val(&resp) as u32,
        )
    }

    pub fn flush(&mut self, x: u32, y: u32, w: u32, h: u32) -> Result<(), &'static str> {
        let flush = VirtioGpuResourceFlush {
            hdr: VirtioGpuCtrlHdr {
                type_: VIRTIO_GPU_CMD_RESOURCE_FLUSH,
                flags: 0,
                fence_id: 0,
                ctx_id: 0,
                padding: 0,
            },
            r: VirtioGpuRect {
                x,
                y,
                width: w,
                height: h,
            },
            resource_id: self.fb_resource_id,
            padding: 0,
        };
        let mut resp = VirtioGpuCtrlHdr {
            type_: 0,
            flags: 0,
            fence_id: 0,
            ctx_id: 0,
            padding: 0,
        };
        self.send_command(
            &flush as *const _ as *const u8,
            core::mem::size_of_val(&flush) as u32,
            &mut resp as *mut _ as *mut u8,
            core::mem::size_of_val(&resp) as u32,
        )
    }
}
