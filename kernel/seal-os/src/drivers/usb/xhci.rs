// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

#![allow(dead_code)] // REASON: xHCI TRB types, descriptor types, and context structs for future enumeration completion

//! xHCI (USB 3.x) host controller driver — MMIO register interface + enumeration + HID.

use crate::drivers::pci::get_device_by_class;
use crate::memory::phys::alloc_frame;
use crate::serial_println;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

pub const XHCI_CLASS: u8 = 0x0C;
pub const XHCI_SUBCLASS: u8 = 0x03;
pub const XHCI_PROG_IF: u8 = 0x30;

// ── Capability Register Offsets ──────────────────────────────────────────────
pub const CAP_CAPLENGTH: u64 = 0x00;
pub const CAP_HCIVERSION: u64 = 0x02;
pub const CAP_HCSPARAMS1: u64 = 0x04;
pub const CAP_HCSPARAMS2: u64 = 0x08;
pub const CAP_HCSPARAMS3: u64 = 0x0C;
pub const CAP_HCCPARAMS1: u64 = 0x10;
pub const CAP_DBOFF: u64 = 0x14;
pub const CAP_RTSOFF: u64 = 0x18;
pub const CAP_HCCPARAMS2: u64 = 0x1C;

// ── Operational Register Offsets (from OP_BASE) ──────────────────────────────
pub const OP_USBCMD: u64 = 0x00;
pub const OP_USBSTS: u64 = 0x04;
pub const OP_PAGESIZE: u64 = 0x08;
pub const OP_DNCTRL: u64 = 0x14;
pub const OP_CRCR: u64 = 0x18;
pub const OP_DCBAAP: u64 = 0x30;
pub const OP_CONFIG: u64 = 0x38;

// ── Register Bits ────────────────────────────────────────────────────────────
pub const USBCMD_RS: u32 = 1 << 0;
pub const USBCMD_HCRST: u32 = 1 << 1;
pub const USBSTS_HCH: u32 = 1 << 0;
pub const USBSTS_CNR: u32 = 1 << 11;

// ── TRB Types ────────────────────────────────────────────────────────────────
const TRB_TYPE_NORMAL: u32 = 1;
const TRB_TYPE_SETUP_STAGE: u32 = 2;
const TRB_TYPE_DATA_STAGE: u32 = 3;
const TRB_TYPE_STATUS_STAGE: u32 = 4;
const TRB_TYPE_LINK: u32 = 6;
const TRB_TYPE_NOOP_CMD: u32 = 8;
const TRB_TYPE_ENABLE_SLOT: u32 = 9;
const TRB_TYPE_ADDRESS_DEVICE: u32 = 11;
const TRB_TYPE_CONFIGURE_EP: u32 = 12;

// ── USB Descriptors ──────────────────────────────────────────────────────────
const DESC_DEVICE: u8 = 0x01;
const DESC_CONFIGURATION: u8 = 0x02;
const DESC_STRING: u8 = 0x03;
const DESC_INTERFACE: u8 = 0x04;
const DESC_ENDPOINT: u8 = 0x05;
const DESC_HID: u8 = 0x21;
const DESC_REPORT: u8 = 0x22;

// ── Request types ────────────────────────────────────────────────────────────
const REQ_GET_DESCRIPTOR: u8 = 0x06;
const REQ_SET_CONFIGURATION: u8 = 0x09;
const REQ_SET_INTERFACE: u8 = 0x0B;
const REQ_SET_PROTOCOL: u8 = 0x0B; // HID class

#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct Trb {
    data: [u32; 4],
}

impl Trb {
    fn zero() -> Self {
        Self { data: [0; 4] }
    }

    fn set_type(&mut self, t: u32) {
        self.data[3] = (self.data[3] & !0xFC00) | ((t & 0x3F) << 10);
    }

    fn set_cycle(&mut self, c: bool) {
        if c {
            self.data[3] |= 1;
        } else {
            self.data[3] &= !1;
        }
    }

    fn cycle(&self) -> bool {
        (self.data[3] & 1) != 0
    }

    fn completion_code(&self) -> u8 {
        ((self.data[2] >> 24) & 0xFF) as u8
    }

    fn slot_id(&self) -> u8 {
        ((self.data[3] >> 24) & 0xFF) as u8
    }
}

#[repr(C, align(64))]
struct DeviceContext {
    slot: [u32; 8],
    ep0: [u32; 8],
    ep1_out: [u32; 8],
    ep1_in: [u32; 8],
    _pad: [u32; 8 * 26],
}

#[repr(C, align(64))]
struct InputContext {
    ctrl: [u32; 8],
    slot: [u32; 8],
    ep0: [u32; 8],
    ep1_out: [u32; 8],
    ep1_in: [u32; 8],
    _pad: [u32; 8 * 26],
}

#[derive(Clone, Copy)]
struct EndpointRing {
    phys: u64,
    enqueue: u16,
    cycle: bool,
}

pub struct XhciController {
    base_addr: u64,
    op_base: u64,
    rt_base: u64,
    db_base: u64,
    max_slots: u8,
    max_ports: u8,
    initialized: bool,

    cmd_ring: u64,
    cmd_enqueue: u16,
    cmd_cycle: bool,

    event_ring: u64,
    event_dequeue: u16,
    event_cycle: bool,
    erst: u64,

    dcbaap: u64,
    dev_ctx: Vec<u64>,
    endpoint_rings: Vec<(u8, u8, EndpointRing)>,
}

impl XhciController {
    pub fn new(bar0: u64) -> Self {
        let caplength = unsafe { read_volatile(bar0 as *const u8) } as u64;
        let rts_off = unsafe { read_volatile((bar0 + CAP_RTSOFF) as *const u32) as u64 };
        let db_off = unsafe { read_volatile((bar0 + CAP_DBOFF) as *const u32) as u64 };
        Self {
            base_addr: bar0,
            op_base: bar0 + caplength,
            rt_base: bar0 + rts_off,
            db_base: bar0 + db_off,
            max_slots: 0,
            max_ports: 0,
            initialized: false,
            cmd_ring: 0,
            cmd_enqueue: 0,
            cmd_cycle: true,
            event_ring: 0,
            event_dequeue: 0,
            event_cycle: true,
            erst: 0,
            dcbaap: 0,
            dev_ctx: Vec::new(),
            endpoint_rings: Vec::new(),
        }
    }

    #[inline(always)]
    unsafe fn read_cap_32(&self, offset: u64) -> u32 {
        read_volatile((self.base_addr + offset) as *const u32)
    }

    #[inline(always)]
    unsafe fn read_op_32(&self, offset: u64) -> u32 {
        read_volatile((self.op_base + offset) as *const u32)
    }

    #[inline(always)]
    unsafe fn write_op_32(&self, offset: u64, val: u32) {
        write_volatile((self.op_base + offset) as *mut u32, val);
    }

    #[inline(always)]
    unsafe fn read_op_64(&self, offset: u64) -> u64 {
        read_volatile((self.op_base + offset) as *const u64)
    }

    #[inline(always)]
    unsafe fn write_op_64(&self, offset: u64, val: u64) {
        write_volatile((self.op_base + offset) as *mut u64, val);
    }

    #[inline(always)]
    unsafe fn read_rt_32(&self, offset: u64) -> u32 {
        read_volatile((self.rt_base + offset) as *const u32)
    }

    #[inline(always)]
    unsafe fn write_rt_32(&self, offset: u64, val: u32) {
        write_volatile((self.rt_base + offset) as *mut u32, val);
    }

    #[inline(always)]
    unsafe fn write_rt_64(&self, offset: u64, val: u64) {
        write_volatile((self.rt_base + offset) as *mut u64, val);
    }

    #[inline(always)]
    unsafe fn write_db_32(&self, slot: u8, val: u32) {
        write_volatile((self.db_base + (slot as u64) * 4) as *mut u32, val);
    }

    unsafe fn wait_for_bit(&self, offset: u64, bit: u32, target: bool, timeout: usize) -> bool {
        for _ in 0..timeout {
            let val = self.read_op_32(offset);
            if ((val & bit) != 0) == target {
                return true;
            }
            core::hint::spin_loop();
        }
        false
    }

    pub fn reset(&mut self) -> Result<(), &'static str> {
        unsafe {
            let mut cmd = self.read_op_32(OP_USBCMD);
            cmd &= !USBCMD_RS;
            self.write_op_32(OP_USBCMD, cmd);
            if !self.wait_for_bit(OP_USBSTS, USBSTS_HCH, true, 1_000_000) {
                return Err("Timeout waiting for xHCI halt");
            }
            self.write_op_32(OP_USBCMD, USBCMD_HCRST);
            if !self.wait_for_bit(OP_USBCMD, USBCMD_HCRST, false, 1_000_000) {
                return Err("Timeout waiting for xHCI reset to clear");
            }
            if !self.wait_for_bit(OP_USBSTS, USBSTS_CNR, false, 1_000_000) {
                return Err("Timeout waiting for xHCI CNR to clear");
            }
        }
        Ok(())
    }

    pub fn init_registers(&mut self) -> Result<(), &'static str> {
        unsafe {
            let hcsparams1 = self.read_cap_32(CAP_HCSPARAMS1);
            self.max_slots = (hcsparams1 & 0xFF) as u8;
            self.max_ports = ((hcsparams1 >> 24) & 0xFF) as u8;
            serial_println!(
                "[xHCI] Max Slots: {}, Max Ports: {}",
                self.max_slots,
                self.max_ports
            );

            let mut config = self.read_op_32(OP_CONFIG);
            config &= !0xFF;
            config |= self.max_slots as u32;
            self.write_op_32(OP_CONFIG, config);

            // DCBAAP
            let dcbaap_frame = alloc_frame().ok_or("xHCI DCBAAP alloc failed")?;
            self.dcbaap = dcbaap_frame.as_u64();
            for i in 0..512 {
                (self.dcbaap as *mut u64).add(i).write_volatile(0);
            }
            self.write_op_64(OP_DCBAAP, self.dcbaap);

            // Command ring
            let cmd_frame = alloc_frame().ok_or("xHCI cmd ring alloc failed")?;
            self.cmd_ring = cmd_frame.as_u64();
            for i in 0..256 {
                (self.cmd_ring as *mut Trb)
                    .add(i)
                    .write_volatile(Trb::zero());
            }
            self.cmd_enqueue = 0;
            self.cmd_cycle = true;
            self.write_op_64(OP_CRCR, self.cmd_ring | 1);

            // Event ring
            let evt_frame = alloc_frame().ok_or("xHCI event ring alloc failed")?;
            self.event_ring = evt_frame.as_u64();
            for i in 0..256 {
                (self.event_ring as *mut Trb)
                    .add(i)
                    .write_volatile(Trb::zero());
            }
            self.event_dequeue = 0;
            self.event_cycle = true;

            let erst_frame = alloc_frame().ok_or("xHCI ERST alloc failed")?;
            self.erst = erst_frame.as_u64();
            (self.erst as *mut u64).write_volatile(self.event_ring);
            (self.erst as *mut u64).add(1).write_volatile(256);

            let intr_base = self.rt_base + 0x20;
            // ERSTSZ
            write_volatile((intr_base + 0x08) as *mut u32, 1);
            // ERSTBA
            write_volatile((intr_base + 0x10) as *mut u64, self.erst);
            // ERDP
            write_volatile((intr_base + 0x18) as *mut u64, self.event_ring);
            // IMAN: enable interrupt
            let iman = read_volatile((intr_base + 0x00) as *const u32);
            write_volatile((intr_base + 0x00) as *mut u32, iman | (1 << 1) | (1 << 0));
        }
        Ok(())
    }

    pub fn start(&mut self) -> Result<(), &'static str> {
        unsafe {
            let mut cmd = self.read_op_32(OP_USBCMD);
            cmd |= USBCMD_RS;
            self.write_op_32(OP_USBCMD, cmd);
            if !self.wait_for_bit(OP_USBSTS, USBSTS_HCH, false, 1_000_000) {
                return Err("Timeout waiting for xHCI to start");
            }
        }
        self.initialized = true;
        Ok(())
    }

    fn push_cmd(&mut self, trb: Trb) {
        unsafe {
            let addr = (self.cmd_ring + self.cmd_enqueue as u64 * 16) as *mut Trb;
            addr.write_volatile(trb);
        }
        self.cmd_enqueue += 1;
        if self.cmd_enqueue >= 256 {
            self.cmd_enqueue = 0;
            self.cmd_cycle = !self.cmd_cycle;
        }
    }

    fn ring_doorbell(&self, slot: u8) {
        unsafe {
            self.write_db_32(slot, 0);
        }
    }

    fn ring_doorbell_ep(&self, slot: u8, ep: u8) {
        unsafe {
            self.write_db_32(slot, ep as u32);
        }
    }

    fn get_ep_ring(&mut self, slot_id: u8, ep_addr: u8) -> Option<&mut EndpointRing> {
        for (sid, ea, ring) in self.endpoint_rings.iter_mut() {
            if *sid == slot_id && *ea == ep_addr {
                return Some(ring);
            }
        }
        None
    }

    fn push_ep_trb(&mut self, slot_id: u8, ep_addr: u8, mut trb: Trb) -> Result<(), &'static str> {
        let ring = self
            .get_ep_ring(slot_id, ep_addr)
            .ok_or("endpoint ring not found")?;
        if ring.enqueue >= 255 {
            ring.enqueue = 0;
            ring.cycle = !ring.cycle;
        }
        trb.set_cycle(ring.cycle);
        unsafe {
            let addr = (ring.phys + ring.enqueue as u64 * 16) as *mut Trb;
            addr.write_volatile(trb);
        }
        ring.enqueue += 1;
        Ok(())
    }

    fn poll_event(&mut self) -> Option<Trb> {
        unsafe {
            let addr = (self.event_ring + self.event_dequeue as u64 * 16) as *const Trb;
            let trb = addr.read_volatile();
            if trb.cycle() == self.event_cycle {
                self.event_dequeue += 1;
                if self.event_dequeue >= 256 {
                    self.event_dequeue = 0;
                    self.event_cycle = !self.event_cycle;
                    // Update ERDP
                    let intr_base = self.rt_base + 0x20;
                    write_volatile(
                        (intr_base + 0x18) as *mut u64,
                        self.event_ring + self.event_dequeue as u64 * 16,
                    );
                }
                return Some(trb);
            }
        }
        None
    }

    fn wait_cmd_complete(&mut self) -> Option<Trb> {
        for _ in 0..1_000_000 {
            if let Some(evt) = self.poll_event() {
                let ty = (evt.data[3] >> 10) & 0x3F;
                if ty == 33 {
                    // Command Completion Event
                    return Some(evt);
                }
            }
            core::hint::spin_loop();
        }
        None
    }

    fn enable_slot(&mut self) -> Result<u8, &'static str> {
        let mut trb = Trb::zero();
        trb.set_type(TRB_TYPE_ENABLE_SLOT);
        trb.set_cycle(self.cmd_cycle);
        self.push_cmd(trb);
        self.ring_doorbell(0);

        let evt = self.wait_cmd_complete().ok_or("Enable Slot timeout")?;
        let code = evt.completion_code();
        if code != 1 {
            return Err("Enable Slot failed");
        }
        Ok(evt.slot_id())
    }

    fn address_device(&mut self, slot_id: u8, port_id: u8, speed: u8) -> Result<(), &'static str> {
        // Allocate device context
        let ctx_frame = alloc_frame().ok_or("xHCI dev ctx alloc failed")?;
        let ctx_phys = ctx_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(ctx_phys as *mut u8, 0, 4096);
            (self.dcbaap as *mut u64)
                .add(slot_id as usize)
                .write_volatile(ctx_phys);
        }
        self.dev_ctx.push(ctx_phys);

        // Allocate input context
        let in_frame = alloc_frame().ok_or("xHCI input ctx alloc failed")?;
        let in_phys = in_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(in_phys as *mut u8, 0, 4096);
        }

        // Setup input context for control EP0
        let ic = in_phys as *mut u32;
        unsafe {
            *ic.add(0) = 0x03; // A0 + A1 (slot + ep0 valid)
                               // Slot context
            *ic.add(8) = ((speed as u32) << 20) | ((port_id as u32) << 16) | 1;
            // EP0 context (offset 16 dwords = 64 bytes)
            let ep0 = ic.add(16);
            *ep0.add(1) = 8; // max packet size
            *ep0.add(4) = 0; // TR dequeue pointer lo (set below)
            *ep0.add(5) = 0; // TR dequeue pointer hi
            *ep0.add(6) = (1 << 16) | (4 << 3) | 1; // CErr=1, TYPE=Control, MAXBURST=0
        }

        // Allocate EP0 transfer ring
        let ep0_ring_frame = alloc_frame().ok_or("xHCI EP0 ring alloc failed")?;
        let ep0_ring = ep0_ring_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(ep0_ring as *mut u8, 0, 4096);
            let ep0_ctx = (ctx_phys + 64) as *mut u32; // slot ctx = 64 bytes
            *ep0_ctx.add(4) = (ep0_ring | 1) as u32;
            *ep0_ctx.add(5) = (ep0_ring >> 32) as u32;
        }

        let mut trb = Trb::zero();
        trb.data[0] = in_phys as u32;
        trb.data[1] = (in_phys >> 32) as u32;
        trb.data[3] =
            ((slot_id as u32) << 24) | (TRB_TYPE_ADDRESS_DEVICE << 10) | (self.cmd_cycle as u32);
        self.push_cmd(trb);
        self.ring_doorbell(0);

        let evt = self.wait_cmd_complete().ok_or("Address Device timeout")?;
        if evt.completion_code() != 1 {
            return Err("Address Device failed");
        }
        Ok(())
    }

    fn get_descriptor(
        &mut self,
        slot_id: u8,
        desc_type: u8,
        desc_index: u8,
        buf: &mut [u8],
    ) -> Result<(), &'static str> {
        let ctx_phys = unsafe {
            (self.dcbaap as *mut u64)
                .add(slot_id as usize)
                .read_volatile()
        };
        let ep0_ring = unsafe { ((ctx_phys + 64) as *mut u64).add(4).read_volatile() & !0x3F };

        // Setup Stage TRB
        let req_type = 0x80; // Device-to-host, standard, device
        let req = REQ_GET_DESCRIPTOR;
        let wvalue = ((desc_type as u16) << 8) | (desc_index as u16);
        let wlen = buf.len() as u16;
        let setup_data = ((wlen as u64) << 48)
            | ((wvalue as u64) << 32)
            | ((req as u64) << 24)
            | ((req_type as u64) << 16)
            | 8;

        let mut setup_trb = Trb::zero();
        setup_trb.data[0] = setup_data as u32;
        setup_trb.data[1] = (setup_data >> 32) as u32;
        setup_trb.data[2] = 8;
        setup_trb.data[3] = (3 << 16) | (TRB_TYPE_SETUP_STAGE << 10) | (self.cmd_cycle as u32);

        // Data Stage TRB
        let data_phys = buf.as_ptr() as u64;
        let mut data_trb = Trb::zero();
        data_trb.data[0] = data_phys as u32;
        data_trb.data[1] = (data_phys >> 32) as u32;
        data_trb.data[2] = buf.len() as u32;
        data_trb.data[3] = (1 << 16) | (TRB_TYPE_DATA_STAGE << 10) | (self.cmd_cycle as u32);

        // Status Stage TRB
        let mut status_trb = Trb::zero();
        status_trb.data[3] = (TRB_TYPE_STATUS_STAGE << 10) | (self.cmd_cycle as u32);

        unsafe {
            let ring = ep0_ring as *mut Trb;
            ring.add(0).write_volatile(setup_trb);
            ring.add(1).write_volatile(data_trb);
            ring.add(2).write_volatile(status_trb);
        }
        self.ring_doorbell(slot_id);

        // Wait for transfer event on event ring
        for _ in 0..1_000_000 {
            if let Some(evt) = self.poll_event() {
                let ty = (evt.data[3] >> 10) & 0x3F;
                if ty == 32 {
                    // Transfer Event
                    let code = evt.completion_code();
                    if code == 1 || code == 13 {
                        // Success or Short Packet
                        return Ok(());
                    } else {
                        return Err("GET_DESCRIPTOR transfer failed");
                    }
                }
            }
            core::hint::spin_loop();
        }
        Err("GET_DESCRIPTOR timeout")
    }

    fn set_configuration(&mut self, slot_id: u8, config: u8) -> Result<(), &'static str> {
        let ctx_phys = unsafe {
            (self.dcbaap as *mut u64)
                .add(slot_id as usize)
                .read_volatile()
        };
        let ep0_ring = unsafe { ((ctx_phys + 64) as *mut u64).add(4).read_volatile() & !0x3F };

        let req_type = 0x00;
        let req = REQ_SET_CONFIGURATION;
        let setup_data =
            ((config as u64) << 32) | ((req as u64) << 24) | ((req_type as u64) << 16) | 0;

        let mut setup_trb = Trb::zero();
        setup_trb.data[0] = setup_data as u32;
        setup_trb.data[1] = (setup_data >> 32) as u32;
        setup_trb.data[2] = 0;
        setup_trb.data[3] = (3 << 16) | (TRB_TYPE_SETUP_STAGE << 10) | (self.cmd_cycle as u32);

        let mut status_trb = Trb::zero();
        status_trb.data[3] = (TRB_TYPE_STATUS_STAGE << 10) | (self.cmd_cycle as u32);

        unsafe {
            let ring = ep0_ring as *mut Trb;
            ring.add(0).write_volatile(setup_trb);
            ring.add(1).write_volatile(status_trb);
        }
        self.ring_doorbell(slot_id);

        for _ in 0..1_000_000 {
            if let Some(evt) = self.poll_event() {
                let ty = (evt.data[3] >> 10) & 0x3F;
                if ty == 32 {
                    return if evt.completion_code() == 1 {
                        Ok(())
                    } else {
                        Err("SET_CONFIGURATION failed")
                    };
                }
            }
            core::hint::spin_loop();
        }
        Err("SET_CONFIGURATION timeout")
    }

    pub fn enumerate_port(
        &mut self,
        port: u8,
        hid_devices: &mut Vec<super::hid::HidDevice>,
    ) -> Option<super::UsbDevice> {
        serial_println!("[xHCI] Enumerating port {}", port);

        let slot_id = match self.enable_slot() {
            Ok(id) => id,
            Err(e) => {
                serial_println!("[xHCI] Enable slot failed: {}", e);
                return None;
            }
        };

        let portsc = unsafe { self.read_portsc(port) };
        let speed = ((portsc >> 10) & 0xF) as u8;

        if let Err(e) = self.address_device(slot_id, port, speed) {
            serial_println!("[xHCI] Address device failed: {}", e);
            return None;
        }

        let mut dev_desc = [0u8; 18];
        if let Err(e) = self.get_descriptor(slot_id, DESC_DEVICE, 0, &mut dev_desc) {
            serial_println!("[xHCI] Get device descriptor failed: {}", e);
            return None;
        }

        let vid = u16::from_le_bytes([dev_desc[8], dev_desc[9]]);
        let pid = u16::from_le_bytes([dev_desc[10], dev_desc[11]]);
        let class = dev_desc[4];
        let subclass = dev_desc[5];
        let protocol = dev_desc[6];

        serial_println!(
            "[xHCI] Port {} device: {:04X}:{:04X} class={:02X}/{:02X}/{:02X}",
            port,
            vid,
            pid,
            class,
            subclass,
            protocol
        );

        // Set configuration (first config)
        let _ = self.set_configuration(slot_id, 1);

        // If HID, setup interrupt endpoint
        if class == 3 || (class == 0 && subclass == 0 && protocol == 0) {
            if let Some((ep_addr, max_packet)) = self.find_interrupt_endpoint(slot_id) {
                if let Err(e) = self.setup_interrupt_endpoint(slot_id, ep_addr, max_packet) {
                    serial_println!("[xHCI] Interrupt endpoint setup failed: {}", e);
                } else {
                    serial_println!(
                        "[xHCI] HID interrupt endpoint {} ready (max_packet={})",
                        ep_addr,
                        max_packet
                    );
                    hid_devices.push(super::hid::HidDevice::new(
                        protocol, slot_id, ep_addr, max_packet,
                    ));
                }
            }
        }

        // If MSC (Bulk-Only), setup bulk endpoints
        if class == 8 && subclass == 6 && protocol == 0x50 {
            if let Some((bulk_in, bulk_out, max_packet)) = self.find_bulk_endpoints(slot_id) {
                serial_println!(
                    "[xHCI] MSC detected: bulk_in={:02X}, bulk_out={:02X}, max_packet={}",
                    bulk_in,
                    bulk_out,
                    max_packet
                );
                let mut ok = true;
                if let Err(e) = self.setup_bulk_endpoint(slot_id, bulk_in, max_packet) {
                    serial_println!("[xHCI] Bulk IN setup failed: {}", e);
                    ok = false;
                } else if let Err(e) = self.setup_bulk_endpoint(slot_id, bulk_out, max_packet) {
                    serial_println!("[xHCI] Bulk OUT setup failed: {}", e);
                    ok = false;
                }
                if ok {
                    serial_println!("[xHCI] MSC bulk endpoints configured");
                    super::mass_storage::register_device(slot_id, bulk_in, bulk_out, max_packet);
                }
            }
        }

        let speed_enum = match speed {
            1 => super::UsbSpeed::Full,
            2 => super::UsbSpeed::Low,
            3 => super::UsbSpeed::High,
            4 => super::UsbSpeed::Super,
            _ => super::UsbSpeed::Full,
        };

        Some(super::UsbDevice {
            address: slot_id,
            speed: speed_enum,
            vendor_id: vid,
            product_id: pid,
            class,
            subclass,
            protocol,
            port,
        })
    }

    fn find_interrupt_endpoint(&mut self, slot_id: u8) -> Option<(u8, u16)> {
        // Read config descriptor (9 bytes first)
        let mut cfg_header = [0u8; 9];
        if self
            .get_descriptor(slot_id, DESC_CONFIGURATION, 0, &mut cfg_header)
            .is_err()
        {
            return None;
        }
        let total_len = u16::from_le_bytes([cfg_header[2], cfg_header[3]]) as usize;
        if total_len > 4096 {
            return None;
        }
        let mut cfg_buf = alloc::vec![0u8; total_len];
        if self
            .get_descriptor(slot_id, DESC_CONFIGURATION, 0, &mut cfg_buf)
            .is_err()
        {
            return None;
        }

        let mut off = 9;
        while off + 2 <= cfg_buf.len() {
            let len = cfg_buf[off] as usize;
            let dtype = cfg_buf[off + 1];
            if len == 0 || off + len > cfg_buf.len() {
                break;
            }
            if dtype == 0x05 && len >= 7 {
                // Endpoint descriptor
                let ep_addr = cfg_buf[off + 2];
                let attr = cfg_buf[off + 3];
                let max_pkt = u16::from_le_bytes([cfg_buf[off + 4], cfg_buf[off + 5]]);
                if (ep_addr & 0x80) != 0 && (attr & 0x03) == 0x03 {
                    // Interrupt IN
                    return Some((ep_addr, max_pkt));
                }
            }
            off += len;
        }
        None
    }

    pub fn setup_bulk_endpoint(
        &mut self,
        slot_id: u8,
        ep_addr: u8,
        max_packet: u16,
    ) -> Result<(), &'static str> {
        let ep_num = (ep_addr & 0x0F) as u64;
        let ep_idx = if (ep_addr & 0x80) != 0 {
            ep_num * 2 + 1
        } else {
            ep_num * 2
        };
        let ep_type = if (ep_addr & 0x80) != 0 { 6u32 } else { 2u32 }; // Bulk IN / OUT

        let ring_frame = alloc_frame().ok_or("xHCI bulk ring alloc failed")?;
        let ring_phys = ring_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(ring_phys as *mut u8, 0, 4096);
            // Link TRB at index 255 wraps to 0 and toggles cycle
            let mut link = Trb::zero();
            link.data[0] = ring_phys as u32;
            link.data[1] = (ring_phys >> 32) as u32;
            link.data[3] = (1 << 1) | (TRB_TYPE_LINK << 10) | 1;
            ((ring_phys + 255 * 16) as *mut Trb).write_volatile(link);
        }

        let in_frame = alloc_frame().ok_or("xHCI input ctx alloc failed")?;
        let in_phys = in_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(in_phys as *mut u8, 0, 4096);
        }

        let ic = in_phys as *mut u32;
        unsafe {
            *ic.add(0) = 1 << (ep_idx as u32);
            let ep_ctx = ic.add((ep_idx as usize) * 8);
            *ep_ctx.add(1) = (ep_type << 3) | ((max_packet as u32) << 16);
            *ep_ctx.add(4) = (ring_phys | 1) as u32;
            *ep_ctx.add(5) = (ring_phys >> 32) as u32;
            *ep_ctx.add(6) = max_packet as u32;
        }

        let mut trb = Trb::zero();
        trb.data[0] = in_phys as u32;
        trb.data[1] = (in_phys >> 32) as u32;
        trb.data[3] =
            ((slot_id as u32) << 24) | (TRB_TYPE_CONFIGURE_EP << 10) | (self.cmd_cycle as u32);
        self.push_cmd(trb);
        self.ring_doorbell(0);

        let evt = self
            .wait_cmd_complete()
            .ok_or("Configure Endpoint timeout")?;
        if evt.completion_code() != 1 {
            return Err("Configure Endpoint failed");
        }

        self.endpoint_rings.push((
            slot_id,
            ep_addr,
            EndpointRing {
                phys: ring_phys,
                enqueue: 0,
                cycle: true,
            },
        ));

        Ok(())
    }

    pub fn send_bulk_out(
        &mut self,
        slot_id: u8,
        ep_addr: u8,
        data: &[u8],
    ) -> Result<(), &'static str> {
        if data.is_empty() {
            return Ok(());
        }
        let ep_idx = if (ep_addr & 0x80) != 0 {
            (ep_addr & 0x0F) as u64 * 2 + 1
        } else {
            (ep_addr & 0x0F) as u64 * 2
        };

        let mut trb = Trb::zero();
        trb.data[0] = data.as_ptr() as u32;
        trb.data[1] = (data.as_ptr() as u64 >> 32) as u32;
        trb.data[2] = data.len() as u32;
        trb.data[3] = (1 << 5) | (TRB_TYPE_NORMAL << 10);
        self.push_ep_trb(slot_id, ep_addr, trb)?;
        self.ring_doorbell_ep(slot_id, ep_idx as u8);

        for _ in 0..1_000_000 {
            if let Some(evt) = self.poll_event() {
                let ty = (evt.data[3] >> 10) & 0x3F;
                if ty == 32 {
                    // Transfer Event
                    let code = evt.completion_code();
                    if code == 1 || code == 13 {
                        return Ok(());
                    } else {
                        return Err("bulk OUT transfer failed");
                    }
                }
            }
            core::hint::spin_loop();
        }
        Err("bulk OUT timeout")
    }

    pub fn recv_bulk_in(
        &mut self,
        slot_id: u8,
        ep_addr: u8,
        buf: &mut [u8],
    ) -> Result<(), &'static str> {
        if buf.is_empty() {
            return Ok(());
        }
        let ep_idx = if (ep_addr & 0x80) != 0 {
            (ep_addr & 0x0F) as u64 * 2 + 1
        } else {
            (ep_addr & 0x0F) as u64 * 2
        };

        let mut trb = Trb::zero();
        trb.data[0] = buf.as_ptr() as u32;
        trb.data[1] = (buf.as_ptr() as u64 >> 32) as u32;
        trb.data[2] = buf.len() as u32;
        trb.data[3] = (1 << 5) | (TRB_TYPE_NORMAL << 10);
        self.push_ep_trb(slot_id, ep_addr, trb)?;
        self.ring_doorbell_ep(slot_id, ep_idx as u8);

        for _ in 0..1_000_000 {
            if let Some(evt) = self.poll_event() {
                let ty = (evt.data[3] >> 10) & 0x3F;
                if ty == 32 {
                    // Transfer Event
                    let code = evt.completion_code();
                    if code == 1 || code == 13 {
                        return Ok(());
                    } else {
                        return Err("bulk IN transfer failed");
                    }
                }
            }
            core::hint::spin_loop();
        }
        Err("bulk IN timeout")
    }

    fn find_bulk_endpoints(&mut self, slot_id: u8) -> Option<(u8, u8, u16)> {
        let mut cfg_header = [0u8; 9];
        if self
            .get_descriptor(slot_id, DESC_CONFIGURATION, 0, &mut cfg_header)
            .is_err()
        {
            return None;
        }
        let total_len = u16::from_le_bytes([cfg_header[2], cfg_header[3]]) as usize;
        if total_len > 4096 {
            return None;
        }
        let mut cfg_buf = alloc::vec![0u8; total_len];
        if self
            .get_descriptor(slot_id, DESC_CONFIGURATION, 0, &mut cfg_buf)
            .is_err()
        {
            return None;
        }

        let mut off = 9;
        let mut in_msc = false;
        let mut bulk_in: Option<(u8, u16)> = None;
        let mut bulk_out: Option<(u8, u16)> = None;

        while off + 2 <= cfg_buf.len() {
            let len = cfg_buf[off] as usize;
            let dtype = cfg_buf[off + 1];
            if len == 0 || off + len > cfg_buf.len() {
                break;
            }
            match dtype {
                0x04 if len >= 9 => {
                    let if_class = cfg_buf[off + 5];
                    let if_subclass = cfg_buf[off + 6];
                    let if_protocol = cfg_buf[off + 7];
                    in_msc = if_class == 8 && if_subclass == 6 && if_protocol == 0x50;
                }
                0x05 if len >= 7 && in_msc => {
                    let ep_addr = cfg_buf[off + 2];
                    let attr = cfg_buf[off + 3];
                    let max_pkt = u16::from_le_bytes([cfg_buf[off + 4], cfg_buf[off + 5]]);
                    if (attr & 0x03) == 0x02 {
                        // Bulk
                        if (ep_addr & 0x80) != 0 {
                            bulk_in = Some((ep_addr, max_pkt));
                        } else {
                            bulk_out = Some((ep_addr, max_pkt));
                        }
                    }
                }
                _ => {
                    // Non-bulk endpoint type; skip for mass-storage
                }
            }
            off += len;
        }

        match (bulk_in, bulk_out) {
            (Some((in_addr, in_max)), Some((out_addr, out_max))) => {
                Some((in_addr, out_addr, in_max.max(out_max)))
            }
            _ => None,
        }
    }

    fn setup_interrupt_endpoint(
        &mut self,
        slot_id: u8,
        ep_addr: u8,
        max_packet: u16,
    ) -> Result<(), &'static str> {
        let _ctx_phys = unsafe {
            (self.dcbaap as *mut u64)
                .add(slot_id as usize)
                .read_volatile()
        };
        let ep_num = (ep_addr & 0x0F) as u64;
        let ep_idx = if (ep_addr & 0x80) != 0 {
            ep_num * 2 + 1
        } else {
            ep_num * 2
        };

        // Allocate transfer ring
        let ring_frame = alloc_frame().ok_or("xHCI int ring alloc failed")?;
        let ring_phys = ring_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(ring_phys as *mut u8, 0, 4096);
        }

        // Build Input Context
        let in_frame = alloc_frame().ok_or("xHCI input ctx alloc failed")?;
        let in_phys = in_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(in_phys as *mut u8, 0, 4096);
        }

        let ic = in_phys as *mut u32;
        unsafe {
            // Input Control Context: add EP
            *ic.add(0) = 1 << (ep_idx as u32);
            // Slot context (already valid)
            // Endpoint context at offset (ep_idx * 8 dwords)
            let ep_ctx = ic.add((ep_idx as usize) * 8);
            // EP Type = 4 (Interrupt IN), Max Packet Size
            *ep_ctx.add(1) = ((4u32) << 3) | ((max_packet as u32) << 16);
            // TR Dequeue Pointer
            *ep_ctx.add(4) = (ring_phys | 1) as u32;
            *ep_ctx.add(5) = (ring_phys >> 32) as u32;
            // Average TRB Length = max_packet
            *ep_ctx.add(6) = max_packet as u32;
        }

        let mut trb = Trb::zero();
        trb.data[0] = in_phys as u32;
        trb.data[1] = (in_phys >> 32) as u32;
        trb.data[3] =
            ((slot_id as u32) << 24) | (TRB_TYPE_CONFIGURE_EP << 10) | (self.cmd_cycle as u32);
        self.push_cmd(trb);
        self.ring_doorbell(0);

        let evt = self
            .wait_cmd_complete()
            .ok_or("Configure Endpoint timeout")?;
        if evt.completion_code() != 1 {
            return Err("Configure Endpoint failed");
        }

        // Queue a Normal TRB on the interrupt endpoint ring
        let buf_frame = alloc_frame().ok_or("xHCI int buf alloc failed")?;
        let buf_phys = buf_frame.as_u64();
        unsafe {
            core::ptr::write_bytes(buf_phys as *mut u8, 0, 4096);
        }

        let mut norm_trb = Trb::zero();
        norm_trb.data[0] = buf_phys as u32;
        norm_trb.data[1] = (buf_phys >> 32) as u32;
        norm_trb.data[2] = max_packet as u32;
        norm_trb.data[3] = (1 << 5) | (TRB_TYPE_NORMAL << 10) | 1; // IOC=1, cycle=1

        unsafe {
            let ring = ring_phys as *mut Trb;
            ring.add(0).write_volatile(norm_trb);
        }
        // Ring endpoint doorbell (ep = ep_idx)
        unsafe {
            self.write_db_32(slot_id, ep_idx as u32);
        }

        Ok(())
    }

    pub fn poll_ports(
        &mut self,
        devices: &mut Vec<super::UsbDevice>,
        hid_devices: &mut Vec<super::hid::HidDevice>,
    ) {
        for port in 1..=self.max_ports {
            unsafe {
                let portsc = self.read_portsc(port);
                let ccs = (portsc >> 0) & 1;
                let csc = (portsc >> 17) & 1;
                let ped = (portsc >> 1) & 1;
                let pr = (portsc >> 4) & 1;

                if csc != 0 {
                    self.write_portsc(port, portsc | (1 << 17));
                    if ccs != 0 && ped == 0 && pr == 0 {
                        serial_println!("[xHCI] Port {} connected, resetting...", port);
                        let mut w = self.read_portsc(port);
                        w |= 1 << 4;
                        self.write_portsc(port, w);
                    } else if ccs == 0 {
                        serial_println!("[xHCI] Port {} disconnected", port);
                        devices.retain(|d| d.port != port);
                        hid_devices.retain(|_d| {
                            // Find slot_id from devices to know which HID to remove
                            true // Simplified: keep all for now
                        });
                    }
                }

                if pr == 0 && ped != 0 && ccs != 0 {
                    if !devices.iter().any(|d| d.port == port) {
                        if let Some(dev) = self.enumerate_port(port, hid_devices) {
                            devices.push(dev);
                        }
                    }
                }
            }
        }
    }

    pub fn poll_hid(&mut self, hid_devices: &mut [super::hid::HidDevice]) {
        for hid in hid_devices.iter_mut() {
            // Poll event ring for transfer completion on this device's interrupt endpoint
            // We check if a Transfer Event matches this slot_id / ep
            // For simplicity, drain all events and match by slot
            while let Some(evt) = self.poll_event() {
                let ty = (evt.data[3] >> 10) & 0x3F;
                if ty == 32 {
                    // Transfer Event
                    let slot = ((evt.data[3] >> 24) & 0xFF) as u8;
                    if slot == hid.slot_id {
                        let code = evt.completion_code();
                        if code == 1 || code == 13 {
                            // Find the buffer for this endpoint
                            let ctx_phys = unsafe {
                                (self.dcbaap as *mut u64).add(slot as usize).read_volatile()
                            };
                            let ep_num = (hid.ep_addr & 0x0F) as u64;
                            let ep_idx = if (hid.ep_addr & 0x80) != 0 {
                                ep_num * 2 + 1
                            } else {
                                ep_num * 2
                            };
                            let ep_ctx = (ctx_phys + 64 + ep_idx * 32) as *mut u32;
                            let tr_lo = unsafe { ep_ctx.add(4).read_volatile() } as u64;
                            let tr_hi = unsafe { ep_ctx.add(5).read_volatile() } as u64;
                            let ring_phys = (tr_lo & !0x3F) | (tr_hi << 32);
                            // First TRB in ring points to data buffer
                            let buf_phys =
                                unsafe { ((ring_phys as *mut u64).add(0).read_volatile()) & !0x3F };
                            let report = unsafe {
                                core::slice::from_raw_parts(
                                    buf_phys as *const u8,
                                    hid.max_packet.min(8) as usize,
                                )
                            };
                            hid.process_report(report);
                            // Re-queue the TRB
                            let mut norm_trb = Trb::zero();
                            norm_trb.data[0] = buf_phys as u32;
                            norm_trb.data[1] = (buf_phys >> 32) as u32;
                            norm_trb.data[2] = hid.max_packet as u32;
                            norm_trb.data[3] = (1 << 5) | (TRB_TYPE_NORMAL << 10) | 1;
                            unsafe {
                                let ring = ring_phys as *mut Trb;
                                ring.add(0).write_volatile(norm_trb);
                            }
                            unsafe {
                                self.write_db_32(slot, ep_idx as u32);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn max_ports(&self) -> u8 {
        self.max_ports
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn port_base(&self, port: u8) -> u64 {
        self.op_base + 0x400 + ((port - 1) as u64) * 16
    }

    unsafe fn read_portsc(&self, port: u8) -> u32 {
        read_volatile(self.port_base(port) as *const u32)
    }

    unsafe fn write_portsc(&self, port: u8, val: u32) {
        write_volatile(self.port_base(port) as *mut u32, val);
    }

    pub fn probe() -> Option<Self> {
        serial_println!("[xHCI] Probing PCI class 0x0C/0x03 (USB xHCI host controller)...");
        let dev = match get_device_by_class(XHCI_CLASS, XHCI_SUBCLASS, XHCI_PROG_IF) {
            Some(d) => d,
            None => {
                serial_println!("[xHCI] No xHCI controller detected");
                return None;
            }
        };
        serial_println!(
            "[xHCI] Found controller at {:02x}:{:02x}.{} — {:04X}:{:04X}",
            dev.bus,
            dev.device,
            dev.function,
            dev.vendor_id,
            dev.device_id
        );
        dev.enable_bus_mastering();
        Some(Self::new(dev.bar_address(0)))
    }
}

pub fn init() -> Option<()> {
    let mut ctrl = XhciController::probe()?;
    serial_println!("[xHCI] Initializing controller...");
    if let Err(e) = ctrl.reset() {
        serial_println!("[xHCI] Reset failed: {}", e);
        return None;
    }
    if let Err(e) = ctrl.init_registers() {
        serial_println!("[xHCI] Register initialization failed: {}", e);
        return None;
    }
    if let Err(e) = ctrl.start() {
        serial_println!("[xHCI] Start failed: {}", e);
        return None;
    }
    serial_println!("[xHCI] Controller initialized and running");
    // Store controller in USB subsystem for polling
    // For now, return success; usb/mod.rs will store it globally
    *XHCI_GLOBAL.lock() = Some(ctrl);
    Some(())
}

static XHCI_GLOBAL: spin::Mutex<Option<XhciController>> = spin::Mutex::new(None);

pub fn with_xhci<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut XhciController) -> R,
{
    XHCI_GLOBAL.lock().as_mut().map(f)
}
