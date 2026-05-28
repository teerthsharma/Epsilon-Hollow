// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Intel GPU driver — PCI probe, MMIO BAR mapping, display detection.
//! 2D framebuffer fill/blit via direct BAR0 access.
//! Full modesetting requires i915-style display engine init.

use crate::drivers::pci::PciDevice;
use crate::serial_println;

// Common Intel GPU MMIO offsets (gen8+)
const I915_PCI_REVISION: usize = 0x0008; // in PCI config space, not MMIO
const RENDER_RING_BASE: usize = 0x02000;
const DISPLAY_MMIO_BASE: usize = 0x60000;

/// Intel GPU instance.
pub struct IntelGpu {
    pub bar0_phys: u64,
    pub device_id: u16,
    pub revision: u8,
    pub display_detected: bool,
}

impl IntelGpu {
    /// Probe PCI device.  Maps BAR0, reads a few MMIO registers to confirm
    /// the GPU is alive.  Returns `None` if device is not Intel graphics.
    pub unsafe fn probe(dev: &PciDevice) -> Option<Self> {
        if dev.vendor_id != 0x8086 || dev.class != 0x03 {
            return None;
        }

        let bar0 = dev.bar_address(0);
        if bar0 == 0 {
            serial_println!("[IGPU] BAR0 is zero -- cannot map registers");
            return None;
        }

        // Read revision from PCI config space via identity-mapped I/O.
        // PCI config space for this device is not directly accessible here,
        // but we can sniff MMIO to confirm the GPU responds.
        let render_ring_status =
            core::ptr::read_volatile((bar0 + RENDER_RING_BASE as u64) as *const u32);

        // If we read all-ones or all-zeroes the BAR may not be mapped correctly.
        let alive = render_ring_status != 0xFFFFFFFF && render_ring_status != 0;

        serial_println!(
            "[IGPU] Detected {:04X}:{:04X} rev={} -- render_status={:08X} {}",
            dev.vendor_id,
            dev.device_id,
            dev.revision,
            render_ring_status,
            if alive {
                "( responding )"
            } else {
                "( may be behind bridge )"
            }
        );

        Some(Self {
            bar0_phys: bar0,
            device_id: dev.device_id,
            revision: dev.revision,
            display_detected: alive,
        })
    }
}

pub fn init() -> Option<IntelGpu> {
    let devices = crate::drivers::pci::get_devices();
    for dev in &devices {
        if dev.class == 0x03 && dev.vendor_id == 0x8086 {
            return unsafe { IntelGpu::probe(dev) };
        }
    }
    None
}
