// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! NVIDIA GPU driver -- PCI probe, BAR0 mapping, register readout.
//! Full NVIDIA kernel driver (nouveau equivalent) not yet implemented.
//! This stub maps BAR0 and reads architecture / framebuffer info honestly.

use crate::drivers::pci::PciDevice;
use crate::serial_println;

const NV_PMC_BOOT_0: usize = 0x000000;
const NV_PMC_ENABLE: usize = 0x000200;
const NV_PFB_CFG0: usize = 0x100000;

/// NVIDIA GPU instance.
pub struct NvidiaGpu {
    pub bar0_phys: u64,
    pub arch: u32,
    pub fb_size_mb: u32,
}

impl NvidiaGpu {
    /// Probe a PCI device.  If it is an NVIDIA GPU, map BAR0 and read
    /// the architecture register.  Returns `None` if the device is not
    /// NVIDIA or if BAR0 is unusable.
    pub unsafe fn probe(dev: &PciDevice) -> Option<Self> {
        if dev.vendor_id != 0x10DE {
            return None;
        }

        let bar0 = dev.bar_address(0);
        if bar0 == 0 {
            serial_println!("[NVGPU] BAR0 is zero -- cannot map registers");
            return None;
        }

        // Identity map covers first 16 GiB; typical BAR0 is below 4 GiB.
        let boot0 = core::ptr::read_volatile((bar0 + NV_PMC_BOOT_0 as u64) as *const u32);
        let enable = core::ptr::read_volatile((bar0 + NV_PMC_ENABLE as u64) as *const u32);
        let fb_cfg = core::ptr::read_volatile((bar0 + NV_PFB_CFG0 as u64) as *const u32);

        // Heuristic: framebuffer size from PFB_CFG0 bits 20:12 (in 1 MiB units)
        let fb_size_mb = ((fb_cfg >> 12) & 0x1FF) as u32;
        let fb_size_mb = if fb_size_mb == 0 { 256 } else { fb_size_mb };

        serial_println!(
            "[NVGPU] Detected {:04X}:{:04X} -- arch={:08X} enable={:08X} fb={} MiB",
            dev.vendor_id,
            dev.device_id,
            boot0,
            enable,
            fb_size_mb
        );

        Some(Self {
            bar0_phys: bar0,
            arch: boot0,
            fb_size_mb,
        })
    }
}

pub fn init() -> Option<NvidiaGpu> {
    let devices = crate::drivers::pci::enumerate();
    for dev in &devices {
        if dev.class == 0x03 && dev.vendor_id == 0x10DE {
            return unsafe { NvidiaGpu::probe(dev) };
        }
    }
    None
}
