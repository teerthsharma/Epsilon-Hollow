// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Intel 8254x (e1000) Ethernet driver -- MMIO, legacy descriptors.

use alloc::alloc::{alloc_zeroed, dealloc, Layout};
use core::ptr::{read_volatile, write_volatile};

const NUM_TX_DESC: usize = 256;
const NUM_RX_DESC: usize = 256;
const RX_BUFFER_SIZE: usize = 2048;

// Register offsets
const REG_CTRL: usize = 0x00000;
const REG_STATUS: usize = 0x00008;
const REG_ICR: usize = 0x000C0;
const REG_IMS: usize = 0x000D0;
const REG_IMC: usize = 0x000D8;
const REG_RCTL: usize = 0x00100;
const REG_TCTL: usize = 0x00400;
const REG_TIPG: usize = 0x00410;
const REG_RDBAL: usize = 0x02800;
const REG_RDBAH: usize = 0x02804;
const REG_RDLEN: usize = 0x02808;
const REG_RDH: usize = 0x02810;
const REG_RDT: usize = 0x02818;
const REG_TDBAL: usize = 0x03800;
const REG_TDBAH: usize = 0x03804;
const REG_TDLEN: usize = 0x03808;
const REG_TDH: usize = 0x03810;
const REG_TDT: usize = 0x03818;
const REG_RAL0: usize = 0x05400;
const REG_RAH0: usize = 0x05404;

const RCTL_EN: u32 = 1 << 1;
const RCTL_BAM: u32 = 1 << 15;
const RCTL_BSIZE_2048: u32 = 0 << 16;
const RCTL_SECRC: u32 = 1 << 26;

const TCTL_EN: u32 = 1 << 1;
const TCTL_PSP: u32 = 1 << 3;
const TCTL_CT: u32 = 0x0F << 4;
const TCTL_COLD: u32 = 0x40 << 12;

const TX_CMD_EOP: u8 = 1 << 0;
const TX_CMD_IFCS: u8 = 1 << 1;
const TX_CMD_RS: u8 = 1 << 3;
const TX_STATUS_DD: u8 = 1 << 0;

const RX_STATUS_DD: u8 = 1 << 0;
const RX_STATUS_EOP: u8 = 1 << 1;

#[repr(C, align(16))]
struct TxDesc {
    buffer_addr: u64,
    length: u16,
    cso: u8,
    cmd: u8,
    status: u8,
    css: u8,
    special: u16,
}

#[repr(C, align(16))]
struct RxDesc {
    buffer_addr: u64,
    length: u16,
    packet_checksum: u16,
    status: u8,
    errors: u8,
    special: u16,
}

pub struct E1000 {
    mmio_base: usize,
    tx_ring: *mut TxDesc,
    tx_buffers: [*mut u8; NUM_TX_DESC],
    rx_ring: *mut RxDesc,
    rx_buffers: [*mut u8; NUM_RX_DESC],
    rx_tail: usize,
    mac: [u8; 6],
}

unsafe impl Send for E1000 {}

impl E1000 {
    pub unsafe fn new(mmio_base: usize) -> Option<Self> {
        let mut nic = Self {
            mmio_base,
            tx_ring: core::ptr::null_mut(),
            tx_buffers: [core::ptr::null_mut(); NUM_TX_DESC],
            rx_ring: core::ptr::null_mut(),
            rx_buffers: [core::ptr::null_mut(); NUM_RX_DESC],
            rx_tail: 0,
            mac: [0; 6],
        };

        let tx_layout = Layout::from_size_align(NUM_TX_DESC * core::mem::size_of::<TxDesc>(), 16).ok()?;
        nic.tx_ring = alloc_zeroed(tx_layout) as *mut TxDesc;
        if nic.tx_ring.is_null() {
            return None;
        }

        for i in 0..NUM_TX_DESC {
            let buf_layout = Layout::from_size_align(2048, 16).ok()?;
            let buf = alloc_zeroed(buf_layout);
            if buf.is_null() {
                return None;
            }
            nic.tx_buffers[i] = buf;
        }

        let rx_layout = Layout::from_size_align(NUM_RX_DESC * core::mem::size_of::<RxDesc>(), 16).ok()?;
        nic.rx_ring = alloc_zeroed(rx_layout) as *mut RxDesc;
        if nic.rx_ring.is_null() {
            return None;
        }

        for i in 0..NUM_RX_DESC {
            let buf_layout = Layout::from_size_align(RX_BUFFER_SIZE, 16).ok()?;
            let buf = alloc_zeroed(buf_layout);
            if buf.is_null() {
                return None;
            }
            nic.rx_buffers[i] = buf;
        }

        Some(nic)
    }

    unsafe fn read_reg(&self, reg: usize) -> u32 {
        read_volatile((self.mmio_base + reg) as *const u32)
    }

    unsafe fn write_reg(&self, reg: usize, value: u32) {
        write_volatile((self.mmio_base + reg) as *mut u32, value);
    }

    fn phys_addr(v: *mut u8) -> u64 {
        crate::memory::virt::translate(x86_64::VirtAddr::new(v as u64))
            .map(|p| p.as_u64())
            .unwrap_or(v as u64)
    }

    pub fn init(&mut self) -> bool {
        unsafe {
            let ctrl = self.read_reg(REG_CTRL);
            self.write_reg(REG_CTRL, ctrl | (1 << 26));
            for _ in 0..100_000 {
                if self.read_reg(REG_CTRL) & (1 << 26) == 0 {
                    break;
                }
            }

            self.write_reg(REG_CTRL, self.read_reg(REG_CTRL) | (1 << 6));
            self.write_reg(REG_IMC, 0xFFFFFFFF);
            let _ = self.read_reg(REG_ICR);

            let ral = self.read_reg(REG_RAL0);
            let rah = self.read_reg(REG_RAH0);
            if rah & (1 << 31) != 0 {
                self.mac[0] = (ral & 0xFF) as u8;
                self.mac[1] = ((ral >> 8) & 0xFF) as u8;
                self.mac[2] = ((ral >> 16) & 0xFF) as u8;
                self.mac[3] = ((ral >> 24) & 0xFF) as u8;
                self.mac[4] = (rah & 0xFF) as u8;
                self.mac[5] = ((rah >> 8) & 0xFF) as u8;
            } else {
                self.mac = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
            }

            let tx_phys = Self::phys_addr(self.tx_ring as *mut u8);
            self.write_reg(REG_TDBAL, tx_phys as u32);
            self.write_reg(REG_TDBAH, (tx_phys >> 32) as u32);
            self.write_reg(REG_TDLEN, (NUM_TX_DESC * core::mem::size_of::<TxDesc>()) as u32);
            self.write_reg(REG_TDH, 0);
            self.write_reg(REG_TDT, 0);

            for i in 0..NUM_TX_DESC {
                let buf_phys = Self::phys_addr(self.tx_buffers[i]);
                (*self.tx_ring.add(i)).buffer_addr = buf_phys;
                (*self.tx_ring.add(i)).status = TX_STATUS_DD;
            }

            self.write_reg(REG_TCTL, TCTL_EN | TCTL_PSP | TCTL_CT | TCTL_COLD);
            self.write_reg(REG_TIPG, 0x0060200A);

            let rx_phys = Self::phys_addr(self.rx_ring as *mut u8);
            self.write_reg(REG_RDBAL, rx_phys as u32);
            self.write_reg(REG_RDBAH, (rx_phys >> 32) as u32);
            self.write_reg(REG_RDLEN, (NUM_RX_DESC * core::mem::size_of::<RxDesc>()) as u32);
            self.write_reg(REG_RDH, 0);
            self.write_reg(REG_RDT, (NUM_RX_DESC - 1) as u32);

            for i in 0..NUM_RX_DESC {
                let buf_phys = Self::phys_addr(self.rx_buffers[i]);
                (*self.rx_ring.add(i)).buffer_addr = buf_phys;
                (*self.rx_ring.add(i)).status = 0;
            }

            self.write_reg(REG_RCTL, RCTL_EN | RCTL_BAM | RCTL_BSIZE_2048 | RCTL_SECRC);

            true
        }
    }

    pub fn mac_address(&self) -> [u8; 6] {
        self.mac
    }

    pub fn send_packet(&mut self, buf: &[u8]) -> bool {
        if buf.len() > 2048 {
            return false;
        }
        unsafe {
            let tail = self.read_reg(REG_TDT) as usize;
            let desc = &mut *self.tx_ring.add(tail);
            if desc.status & TX_STATUS_DD == 0 {
                return false;
            }
            core::ptr::copy_nonoverlapping(buf.as_ptr(), self.tx_buffers[tail], buf.len());
            desc.length = buf.len() as u16;
            desc.cmd = TX_CMD_EOP | TX_CMD_IFCS | TX_CMD_RS;
            desc.status = 0;
            let new_tail = (tail + 1) % NUM_TX_DESC;
            self.write_reg(REG_TDT, new_tail as u32);

            for _ in 0..100_000 {
                if desc.status & TX_STATUS_DD != 0 {
                    break;
                }
            }
            true
        }
    }

    pub fn recv_packet(&mut self, buf: &mut [u8]) -> Option<usize> {
        unsafe {
            let tail = self.rx_tail;
            let desc = &mut *self.rx_ring.add(tail);
            if desc.status & RX_STATUS_DD == 0 {
                return None;
            }
            if desc.status & RX_STATUS_EOP == 0 {
                desc.status = 0;
                self.write_reg(REG_RDT, tail as u32);
                self.rx_tail = (tail + 1) % NUM_RX_DESC;
                return None;
            }
            let len = desc.length as usize;
            let to_copy = len.min(buf.len());
            core::ptr::copy_nonoverlapping(self.rx_buffers[tail], buf.as_mut_ptr(), to_copy);
            desc.status = 0;
            self.write_reg(REG_RDT, tail as u32);
            self.rx_tail = (tail + 1) % NUM_RX_DESC;
            Some(to_copy)
        }
    }
}

impl Drop for E1000 {
    fn drop(&mut self) {
        unsafe {
            if !self.tx_ring.is_null() {
                let layout = Layout::from_size_align(NUM_TX_DESC * core::mem::size_of::<TxDesc>(), 16).unwrap();
                dealloc(self.tx_ring as *mut u8, layout);
            }
            for i in 0..NUM_TX_DESC {
                if !self.tx_buffers[i].is_null() {
                    let layout = Layout::from_size_align(2048, 16).unwrap();
                    dealloc(self.tx_buffers[i], layout);
                }
            }
            if !self.rx_ring.is_null() {
                let layout = Layout::from_size_align(NUM_RX_DESC * core::mem::size_of::<RxDesc>(), 16).unwrap();
                dealloc(self.rx_ring as *mut u8, layout);
            }
            for i in 0..NUM_RX_DESC {
                if !self.rx_buffers[i].is_null() {
                    let layout = Layout::from_size_align(RX_BUFFER_SIZE, 16).unwrap();
                    dealloc(self.rx_buffers[i], layout);
                }
            }
        }
    }
}
