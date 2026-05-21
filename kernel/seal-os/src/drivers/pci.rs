// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! PCI bus enumeration — config space at 0xCF8/0xCFC.

use alloc::vec::Vec;
use spin::Mutex;
use x86_64::instructions::port::Port;

const PCI_CONFIG_ADDR: u16 = 0xCF8;
const PCI_CONFIG_DATA: u16 = 0xCFC;

static PCI_DEVICES: Mutex<Vec<PciDevice>> = Mutex::new(Vec::new());

#[derive(Debug, Clone)]
pub struct PciDevice {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub revision: u8,
    pub class: u8,
    pub subclass: u8,
    pub prog_if: u8,
    pub bar0: u32,
}

impl PciDevice {
    pub fn is_xhci(&self) -> bool {
        self.class == 0x0C && self.subclass == 0x03 && self.prog_if == 0x30
    }

    pub fn is_network(&self) -> bool {
        self.class == 0x02
    }

    pub fn is_display(&self) -> bool {
        self.class == 0x03
    }

    pub fn is_wifi(&self) -> bool {
        self.class == 0x02 && self.subclass == 0x80
    }

    pub fn is_ahci(&self) -> bool {
        self.class == 0x01 && self.subclass == 0x06 && self.prog_if == 0x01
    }

    pub fn bar_address(&self, bar: u8) -> u64 {
        let val = pci_read32(self.bus, self.device, self.function, 0x10 + bar * 4);
        if val & 1 == 0 {
            let typ = (val >> 1) & 0x03;
            if typ == 0x02 && bar < 5 {
                let high = pci_read32(self.bus, self.device, self.function, 0x10 + (bar + 1) * 4);
                ((val & !0x0F) as u64) | ((high as u64) << 32)
            } else {
                (val & !0x0F) as u64
            }
        } else {
            (val & !0x03) as u64
        }
    }
}

pub fn pci_read32(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let addr: u32 = 0x80000000
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    unsafe {
        let mut addr_port = Port::<u32>::new(PCI_CONFIG_ADDR);
        let mut data_port = Port::<u32>::new(PCI_CONFIG_DATA);
        addr_port.write(addr);
        data_port.read()
    }
}

pub fn enumerate() -> Vec<PciDevice> {
    let mut devices = Vec::new();

    for bus in 0..=255u16 {
        for device in 0..32u8 {
            let vendor_device = pci_read32(bus as u8, device, 0, 0);
            let vendor_id = (vendor_device & 0xFFFF) as u16;
            if vendor_id == 0xFFFF {
                continue;
            }
            let device_id = ((vendor_device >> 16) & 0xFFFF) as u16;

            let class_reg = pci_read32(bus as u8, device, 0, 0x08);
            let revision = (class_reg & 0xFF) as u8;
            let class = ((class_reg >> 24) & 0xFF) as u8;
            let subclass = ((class_reg >> 16) & 0xFF) as u8;
            let prog_if = ((class_reg >> 8) & 0xFF) as u8;

            let bar0 = pci_read32(bus as u8, device, 0, 0x10);

            devices.push(PciDevice {
                bus: bus as u8,
                device,
                function: 0,
                vendor_id,
                device_id,
                revision,
                class,
                subclass,
                prog_if,
                bar0,
            });
        }
    }

    devices
}

pub fn init() {
    let devices = enumerate();
    *PCI_DEVICES.lock() = devices;
}

pub fn get_devices() -> Vec<PciDevice> {
    PCI_DEVICES.lock().clone()
}

pub fn get_device_by_class(class: u8, subclass: u8, prog_if: u8) -> Option<PciDevice> {
    PCI_DEVICES.lock().iter().find(|d| {
        d.class == class && d.subclass == subclass && d.prog_if == prog_if
    }).cloned()
}
