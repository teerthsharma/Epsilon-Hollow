#![allow(dead_code)] // REASON: virtio descriptor flags for future virtqueue completion

//! Virtio-net NIC driver
//! Phase 3.1: NIC Drivers implementation

use crate::drivers::pci::get_devices;
use crate::memory::phys::alloc_frames_contiguous;
use core::ptr;
use x86_64::instructions::port::Port;

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;

/// Virtio Legacy Port I/O Interface
struct VirtioPort {
    base: u16,
}

impl VirtioPort {
    unsafe fn read_u8(&self, offset: u16) -> u8 {
        Port::<u8>::new(self.base + offset).read()
    }
    unsafe fn read_u16(&self, offset: u16) -> u16 {
        Port::<u16>::new(self.base + offset).read()
    }
    unsafe fn read_u32(&self, offset: u16) -> u32 {
        Port::<u32>::new(self.base + offset).read()
    }
    unsafe fn write_u8(&self, offset: u16, value: u8) {
        Port::<u8>::new(self.base + offset).write(value)
    }
    unsafe fn write_u16(&self, offset: u16, value: u16) {
        Port::<u16>::new(self.base + offset).write(value)
    }
    unsafe fn write_u32(&self, offset: u16, value: u32) {
        Port::<u32>::new(self.base + offset).write(value)
    }
}

/// Virtio-net packet header
#[repr(C)]
pub struct VirtioNetHdr {
    pub flags: u8,
    pub gso_type: u8,
    pub hdr_len: u16,
    pub gso_size: u16,
    pub csum_start: u16,
    pub csum_offset: u16,
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

/// Represents a split virtqueue for network RX/TX.
pub struct SplitVirtqueue {
    pub desc: *mut VirtqDesc,
    pub avail: *mut VirtqAvail,
    pub used: *mut VirtqUsed,
    pub queue_size: u16,
    pub last_used_idx: u16,
    pub phys_addr: u64,
}

// Safety: The virtqueue pointers will be initialized with properly allocated memory.
unsafe impl Send for SplitVirtqueue {}
unsafe impl Sync for SplitVirtqueue {}

impl SplitVirtqueue {
    pub const fn new() -> Self {
        SplitVirtqueue {
            desc: ptr::null_mut(),
            avail: ptr::null_mut(),
            used: ptr::null_mut(),
            queue_size: 256,
            last_used_idx: 0,
            phys_addr: 0,
        }
    }
}

/// Virtio-net device
pub struct VirtioNet {
    pub base_addr: u64,
    pub rx_queue: SplitVirtqueue,
    pub tx_queue: SplitVirtqueue,
    pub mac_address: [u8; 6],
    pub link_up: bool,
}

impl VirtioNet {
    pub const fn new(base_addr: u64) -> Self {
        VirtioNet {
            base_addr,
            rx_queue: SplitVirtqueue::new(),
            tx_queue: SplitVirtqueue::new(),
            mac_address: [0; 6],
            link_up: false,
        }
    }

    /// PCI discovery and initialization for Virtio-net
    pub fn discover_and_init() -> Result<Self, &'static str> {
        let devices = get_devices();

        // Discover Virtio-net legacy device: Vendor 0x1AF4, Device 0x1000
        let dev = devices
            .iter()
            .find(|d| d.vendor_id == 0x1AF4 && d.device_id == 0x1000)
            .ok_or("Legacy Virtio-net PCI device not found")?;

        let bar0 = dev.bar0;
        let is_port_io = (bar0 & 1) != 0;
        if !is_port_io {
            return Err("Virtio-net device BAR0 is not Port I/O (we only support legacy port I/O)");
        }
        let base_addr = (bar0 & !0x03) as u64;

        dev.enable_bus_mastering();

        let mut net = VirtioNet::new(base_addr);

        net.reset_and_negotiate();
        net.read_mac_address();
        net.setup_virtqueues();
        net.update_link_status();

        Ok(net)
    }

    pub fn reset_and_negotiate(&mut self) {
        let port = VirtioPort {
            base: self.base_addr as u16,
        };
        unsafe {
            // 1. Reset
            port.write_u8(0x12, 0); // Device status = 0

            // 2. Acknowledge
            port.write_u8(0x12, 1);

            // 3. Driver
            port.write_u8(0x12, 1 | 2); // ACK | DRIVER

            // 4. Read features
            let mut features = port.read_u32(0x00);

            // 5. Negotiate features (MAC and Status)
            let virtio_net_f_mac = 1 << 5;
            let virtio_net_f_status = 1 << 16;

            let wanted_features = virtio_net_f_mac | virtio_net_f_status;
            features &= wanted_features;

            port.write_u32(0x04, features);
        }
    }

    /// Read MAC address from device config space
    pub fn read_mac_address(&mut self) {
        let port = VirtioPort {
            base: self.base_addr as u16,
        };
        unsafe {
            for i in 0..6 {
                self.mac_address[i] = port.read_u8(0x14 + i as u16);
            }
        }
    }

    /// Setup RX and TX virtqueues
    pub fn setup_virtqueues(&mut self) {
        let port = VirtioPort {
            base: self.base_addr as u16,
        };

        // Setup RX queue (Queue 0)
        let rx_phys = alloc_frames_contiguous(3).expect("Virtio RX queue alloc");
        unsafe { core::ptr::write_bytes(rx_phys.as_u64() as *mut u8, 0, 3 * 4096) };
        self.rx_queue.phys_addr = rx_phys.as_u64();
        self.rx_queue.desc = rx_phys.as_u64() as *mut VirtqDesc;
        self.rx_queue.avail = (rx_phys.as_u64() + 4096) as *mut VirtqAvail;
        self.rx_queue.used = (rx_phys.as_u64() + 8192) as *mut VirtqUsed;
        self.rx_queue.queue_size = 256;

        unsafe {
            port.write_u16(0x0E, 0); // Queue Select = 0 (RX)
            port.write_u32(0x08, (rx_phys.as_u64() / 4096) as u32); // Queue Address
        }

        // Setup TX queue (Queue 1)
        let tx_phys = alloc_frames_contiguous(3).expect("Virtio TX queue alloc");
        unsafe { core::ptr::write_bytes(tx_phys.as_u64() as *mut u8, 0, 3 * 4096) };
        self.tx_queue.phys_addr = tx_phys.as_u64();
        self.tx_queue.desc = tx_phys.as_u64() as *mut VirtqDesc;
        self.tx_queue.avail = (tx_phys.as_u64() + 4096) as *mut VirtqAvail;
        self.tx_queue.used = (tx_phys.as_u64() + 8192) as *mut VirtqUsed;
        self.tx_queue.queue_size = 256;

        unsafe {
            port.write_u16(0x0E, 1); // Queue Select = 1 (TX)
            port.write_u32(0x08, (tx_phys.as_u64() / 4096) as u32); // Queue Address
        }

        // Populate RX queue with empty buffers so device can receive packets
        for i in 0..self.rx_queue.queue_size {
            let frame = alloc_frames_contiguous(1).expect("Virtio RX buffer alloc");
            unsafe {
                let desc = self.rx_queue.desc.add(i as usize);
                (*desc).addr = frame.as_u64();
                (*desc).len = 4096;
                (*desc).flags = VIRTQ_DESC_F_WRITE;
                (*desc).next = 0;

                (*self.rx_queue.avail).ring[i as usize] = i;
            }
        }
        unsafe {
            (*self.rx_queue.avail).idx = self.rx_queue.queue_size;
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Release);
            port.write_u16(0x10, 0); // Notify RX queue
        }

        // Set DRIVER_OK to complete initialization
        unsafe {
            port.write_u8(0x12, port.read_u8(0x12) | 4); // DRIVER_OK = 4
        }
    }

    /// Read link status from device and update internal state
    pub fn update_link_status(&mut self) {
        let port = VirtioPort {
            base: self.base_addr as u16,
        };
        unsafe {
            // For Virtio-net, link status is at offset 0x1A if VIRTIO_NET_F_STATUS was negotiated
            let link_status = port.read_u16(0x1A);
            self.link_up = (link_status & 1) != 0;
        }
    }

    /// Return the current link status
    pub fn is_link_up(&self) -> bool {
        self.link_up
    }

    /// Transmit a packet
    pub fn transmit(&mut self, packet: &[u8]) {
        let q = &mut self.tx_queue;
        let head = (unsafe { (*q.avail).idx } % q.queue_size) as usize;

        // Allocate a DMA-capable buffer for the packet + header.
        let frame = alloc_frames_contiguous(1).expect("TX buffer alloc");
        let phys_addr = frame.as_u64();
        let virt_addr = phys_addr as *mut u8;

        unsafe {
            let hdr = virt_addr as *mut VirtioNetHdr;
            core::ptr::write_bytes(hdr, 0, 1);

            let data = virt_addr.add(core::mem::size_of::<VirtioNetHdr>());
            core::ptr::copy_nonoverlapping(packet.as_ptr(), data, packet.len());

            let desc = q.desc.add(head);
            (*desc).addr = phys_addr;
            (*desc).len = (core::mem::size_of::<VirtioNetHdr>() + packet.len()) as u32;
            (*desc).flags = 0; // No next desc
            (*desc).next = 0;

            let avail_ring_index = ((*q.avail).idx % q.queue_size) as usize;
            (*q.avail).ring[avail_ring_index] = head as u16;

            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Release);

            (*q.avail).idx = (*q.avail).idx.wrapping_add(1);

            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Release);
        }

        // Notify
        let port = VirtioPort {
            base: self.base_addr as u16,
        };
        unsafe { port.write_u16(0x10, 1) }; // Queue Notify = 1 (TX)
    }
}

/// Initialization function to be called from the network subsystem
pub fn init() {
    if let Ok(mut net) = VirtioNet::discover_and_init() {
        net.update_link_status();
        crate::serial_println!(
            "[VirtioNet] Initialized with MAC: {:02x?}, Link Up: {}",
            net.mac_address,
            net.is_link_up()
        );
        // Register the NIC to the network stack here
    } else {
        crate::serial_println!("[VirtioNet] Initialization failed");
    }
}
