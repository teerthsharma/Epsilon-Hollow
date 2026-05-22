// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! AHCI disk driver — probes PCI for SATA controllers and reports honest status.

use crate::drivers::pci::get_device_by_class;
use crate::serial_println;
use core::ptr::read_volatile;

pub struct AhciController {
    pub bar5: usize,
    pub ports: [Option<AhciPort>; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AhciPort {
    pub index: usize,
    pub signature: u32,
}

const SIG_SATA: u32 = 0x0000_0101;
const SIG_ATAPI: u32 = 0xEB14_0101;
const SIG_SEMB: u32 = 0x9669_0101;

pub fn probe() -> Option<AhciController> {
    let device = get_device_by_class(0x01, 0x06, 0x01)?;
    let bar5 = device.bar_address(5) as usize;

    serial_println!(
        "[disk::ahci] Probing controller at {:02x}:{:02x}.{} — ABAR={:#x}",
        device.bus, device.device, device.function, bar5
    );

    let mut ports: [Option<AhciPort>; 32] = [
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
    ];

    unsafe {
        let pi = read_volatile((bar5 + 0x0C) as *const u32);
        for i in 0..32 {
            if pi & (1 << i) == 0 {
                continue;
            }
            let sig = read_volatile((bar5 + 0x100 + i * 0x80 + 0x24) as *const u32);
            serial_println!("[disk::ahci] Port {} signature: {:#x}", i, sig);
            if sig == SIG_SATA || sig == SIG_ATAPI || sig == SIG_SEMB {
                ports[i] = Some(AhciPort { index: i, signature: sig });
                let sig_name = match sig {
                    SIG_SATA => "SATA",
                    SIG_ATAPI => "ATAPI",
                    SIG_SEMB => "SEMB",
                    _ => "unknown",
                };
                serial_println!("[disk::ahci] Port {} device present ({})", i, sig_name);
            }
        }
    }

    let present = ports.iter().filter(|p| p.is_some()).count();
    if present == 0 {
        serial_println!("[disk::ahci] No AHCI device present on any port");
        return None;
    }

    serial_println!("[disk::ahci] {} port(s) with device present", present);
    Some(AhciController { bar5, ports })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskError {
    NoDisk,
}

pub fn first_disk() -> Result<(), DiskError> {
    let _controller = probe().ok_or(DiskError::NoDisk)?;

    // Honest check: can we actually read sector 0 via the real block layer?
    let mut buf = [0u8; 512];
    match crate::drivers::block::read_block(0x800, 0, &mut buf) {
        Ok(()) => {
            serial_println!("[disk::ahci] First disk readable (sector 0 OK)");
            Ok(())
        }
        Err(e) => {
            serial_println!("[disk::ahci] First disk unreadable: {:?}", e);
            Err(DiskError::NoDisk)
        }
    }
}
