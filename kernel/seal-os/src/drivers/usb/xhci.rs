// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! xHCI (USB 3.x) host controller driver — MMIO register interface.

pub const XHCI_CLASS: u8 = 0x0C;
pub const XHCI_SUBCLASS: u8 = 0x03;
pub const XHCI_PROG_IF: u8 = 0x30;

#[repr(C)]
pub struct XhciCapRegs {
    pub caplength: u8,
    pub _rsvd: u8,
    pub hci_version: u16,
    pub hcsparams1: u32,
    pub hcsparams2: u32,
    pub hcsparams3: u32,
    pub hccparams1: u32,
    pub dboff: u32,
    pub rtsoff: u32,
    pub hccparams2: u32,
}

pub struct XhciController {
    base_addr: u64,
    max_slots: u8,
    max_ports: u8,
    initialized: bool,
}

impl XhciController {
    pub fn new(bar0: u64) -> Self {
        Self {
            base_addr: bar0,
            max_slots: 0,
            max_ports: 0,
            initialized: false,
        }
    }

    pub fn reset(&mut self) -> Result<(), &'static str> {
        self.initialized = true;
        Ok(())
    }

    pub fn max_ports(&self) -> u8 {
        self.max_ports
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}
