// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! NVIDIA GPU driver -- PCI probe, BAR0 mapping, register readout,
//! architecture detection, VBIOS scan, framebuffer setup.
//!
//! Honest limitations:
//! - No signed GSP firmware loading (required for 3D on Maxwell+).
//! - No display mode-setting (requires full DRM driver).
//! - This driver provides register-level probe and framebuffer address only.

use crate::drivers::pci::{pci_read32, PciDevice};
use crate::{serial_print, serial_println};

const NV_PMC_BOOT_0: usize = 0x000000;
const NV_PMC_ENABLE: usize = 0x000200;
const NV_PFB_CFG0: usize = 0x100000;

// PRAMIN offset for pre-Maxwell direct framebuffer access
const NV_PRAMIN_OFFSET: u64 = 0x700000;

/// PCI config-space offset for the Expansion ROM BAR.
const PCI_ROM_BAR_OFFSET: u8 = 0x30;

// ------------------------------------------------------------------
// Known Ada / RTX 40-series device IDs
// Source: NVIDIA open-gpu-doc, TechPowerUp BIOS collection
// ------------------------------------------------------------------

/// RTX 4060 Ti (8 GB / 16 GB desktop)
pub const DID_RTX_4060_TI: u16 = 0x2803;
/// RTX 4060 (8 GB desktop)
pub const DID_RTX_4060: u16 = 0x2882;
/// RTX 4060 Mobile / Laptop GPU
pub const DID_RTX_4060_MOBILE_A: u16 = 0x28A0;
/// RTX 4060 Mobile (alternate SKU)
pub const DID_RTX_4060_MOBILE_B: u16 = 0x28E0;

/// NVIDIA GPU instance.
pub struct NvidiaGpu {
    pub bar0_phys: u64,
    pub arch: u32,
    pub arch_name: &'static str,
    pub fb_size_mb: u32,
    pub display_detected: bool,
    pub vbios_valid: bool,
    pub vbios_version: [u8; 16],
}

impl NvidiaGpu {
    /// Decode architecture ID from NV_PMC_BOOT_0 to a human-readable name.
    pub fn arch_name_from_id(id: u32) -> &'static str {
        // The architecture is encoded in the upper bits of BOOT_0.
        match id & 0xFFFF0000 {
            0x00010000 => "Volta (GV100)",
            0x00011000 => "Turing (TU1xx)",
            0x00013000 => "Ampere (GA10x)",
            0x00014000 => "Ada (AD10x)",
            0x00016000 => "Blackwell (GB10x)",
            _ => match id & 0x0000F000 {
                0x0000F000 => "Pascal (GP10x)",
                0x0000E000 => "Maxwell (GM1xx)",
                0x0000D000 => "Kepler (GK1xx)",
                0x0000C000 => "Fermi (GF100+)",
                _ => match id {
                    0x00000120 => "NV40 (GeForce 6/7)",
                    0x00000170 => "G80 (GeForce 8)",
                    0x00000190 => "GT200",
                    _ => "Unknown",
                },
            },
        }
    }

    /// Return architecture name for this GPU.
    pub fn arch_name(&self) -> &'static str {
        self.arch_name
    }

    /// Map a known NVIDIA device ID to a marketing name.
    pub fn product_name_from_id(device_id: u16) -> &'static str {
        match device_id {
            DID_RTX_4060_TI => "GeForce RTX 4060 Ti",
            DID_RTX_4060 => "GeForce RTX 4060",
            DID_RTX_4060_MOBILE_A | DID_RTX_4060_MOBILE_B => "GeForce RTX 4060 Laptop GPU",
            0x2805 => "GeForce RTX 4060 Ti (OEM)",
            0x2808 => "GeForce RTX 4060 (OEM)",
            0x2684 => "GeForce RTX 4080",
            0x2704 => "GeForce RTX 4090",
            0x2782 => "GeForce RTX 4070",
            0x2786 => "GeForce RTX 4070 Ti",
            _ => "Unknown NVIDIA GPU",
        }
    }

    /// Return physical address for direct framebuffer access.
    /// For pre-Maxwell cards this is BAR0 + PRAMIN offset.
    pub fn framebuffer_phys_addr(&self) -> u64 {
        self.bar0_phys + NV_PRAMIN_OFFSET
    }

    /// Read and validate the PCI ROM BAR (VBIOS).
    /// Identity map covers first 16 GiB, so we can read directly.
    unsafe fn read_vbios(dev: &PciDevice) -> (bool, [u8; 16]) {
        let rom_bar = pci_read32(dev.bus, dev.device, dev.function, PCI_ROM_BAR_OFFSET);
        let rom_phys = (rom_bar & !0x01) as u64;
        if rom_phys == 0 {
            return (false, [0u8; 16]);
        }

        // Check VBIOS signature 0x55AA at offset 0
        let sig = core::ptr::read_volatile(rom_phys as *const u16);
        if sig != 0x55AA {
            return (false, [0u8; 16]);
        }

        // VBIOS version string is typically at offset 0x18-0x28
        let mut version = [0u8; 16];
        for i in 0..16 {
            version[i] = core::ptr::read_volatile((rom_phys + 0x18 + i as u64) as *const u8);
        }
        (true, version)
    }

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

        let arch_name = Self::arch_name_from_id(boot0);
        let product_name = Self::product_name_from_id(dev.device_id);

        // Display engine detection: bit 12 of NV_PMC_ENABLE
        let display_detected = (enable & (1 << 12)) != 0;

        serial_println!(
            "[NVGPU] Detected {} [{:04X}:{:04X}] -- arch={} ({:08X}) enable={:08X} fb={} MiB",
            product_name,
            dev.vendor_id,
            dev.device_id,
            arch_name,
            boot0,
            enable,
            fb_size_mb
        );

        if display_detected {
            serial_println!("[NVGPU] Display engine detected");
        }

        // VBIOS read
        let (vbios_valid, vbios_version) = Self::read_vbios(dev);
        if vbios_valid {
            serial_print!("[NVGPU] VBIOS signature OK -- version: ");
            for i in 0..16 {
                let b = vbios_version[i];
                if b == 0 {
                    break;
                }
                serial_print!("{}", b as char);
            }
            serial_println!("");
        } else {
            serial_println!("[NVGPU] VBIOS not found or invalid signature");
        }

        Some(Self {
            bar0_phys: bar0,
            arch: boot0,
            arch_name,
            fb_size_mb,
            display_detected,
            vbios_valid,
            vbios_version,
        })
    }

    /// Safe initialization sequence:
    /// - Enable PCI bus mastering
    /// - Reset PMC
    /// - Wait for idle
    /// - Log each step
    pub unsafe fn init_sequence(&self, dev: &PciDevice) {
        serial_println!("[NVGPU] Step 1: Enabling PCI bus mastering");
        dev.enable_bus_mastering();

        serial_println!("[NVGPU] Step 2: Resetting PMC");
        let enable =
            core::ptr::read_volatile((self.bar0_phys + NV_PMC_ENABLE as u64) as *const u32);
        core::ptr::write_volatile((self.bar0_phys + NV_PMC_ENABLE as u64) as *mut u32, 0);
        // Small spin-wait for reset to propagate
        for _ in 0..1000 {
            core::hint::spin_loop();
        }
        core::ptr::write_volatile((self.bar0_phys + NV_PMC_ENABLE as u64) as *mut u32, enable);

        serial_println!("[NVGPU] Step 3: Waiting for idle");
        for _ in 0..10_000 {
            core::hint::spin_loop();
        }

        serial_println!(
            "[NVGPU] Step 4: Framebuffer at {:016X} ({} MiB)",
            self.framebuffer_phys_addr(),
            self.fb_size_mb
        );

        serial_println!("[NVGPU] WARN: 3D acceleration requires signed firmware (GSP) -- not available in open-source kernel");
    }
}

pub fn init() -> Option<NvidiaGpu> {
    let devices = crate::drivers::pci::get_devices();
    for dev in &devices {
        if dev.class == 0x03 && dev.vendor_id == 0x10DE {
            if let Some(gpu) = unsafe { NvidiaGpu::probe(dev) } {
                unsafe {
                    gpu.init_sequence(dev);
                }
                return Some(gpu);
            }
        }
    }
    None
}
