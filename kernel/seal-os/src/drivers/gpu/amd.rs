// Seal OS -- Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! AMD GPU driver -- PCI probe, BAR mapping, architecture detection, VBIOS readout.
//! AMD GPU init driver — ATOMBIOS, VRAM/GART detect, safe init sequence.
//! 3D acceleration requires signed SMU firmware (not included).
//! This driver performs safe initialization and identification only.
//! No firmware is loaded; 3D acceleration and display output are NOT available.
//!
//! Honest limitations:
//! - DCN display engine requires SMU firmware binaries -- skipped.
//! - UVD/VCE/VCN video engines require firmware -- skipped.
//! - GART setup requires CP/SDMA firmware -- only aperture size is logged.
//! - 3D/compute pipes require shader firmware -- not initialized.

use crate::drivers::pci::{self, PciDevice};
use crate::serial_println;

// MMIO register offsets (used for identification and status only).
const MM_GRBM_STATUS: u64 = 0x8010;
const MM_GRBM_SOFT_RESET: u64 = 0x8020;
const MM_CONFIG_MEMSIZE: u64 = 0x5428;
const MM_MC_VM_FB_OFFSET: u64 = 0x81;

// PCI config space offsets.
const PCI_ROM_BAR_OFFSET: u8 = 0x30;
const PCI_COMMAND_OFFSET: u8 = 0x04;

/// AMD GPU architecture family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmdArchitecture {
    SouthernIslands, // HD 7xxx
    Rdna2,           // RX 6000
    Rdna3,           // RX 7000
    Unknown,
}

impl AmdArchitecture {
    pub fn from_device_id(device_id: u16) -> Self {
        match device_id {
            0x6700..=0x68FF => Self::SouthernIslands,
            0x2600..=0x26FF => Self::Rdna2,
            0x7400..=0x75FF => Self::Rdna3,
            _ => Self::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::SouthernIslands => "Southern Islands (HD 7xxx)",
            Self::Rdna2 => "RDNA2 (RX 6000)",
            Self::Rdna3 => "RDNA3 (RX 7000)",
            Self::Unknown => "Unknown",
        }
    }

    /// Returns true for generations that need the legacy soft-reset path.
    pub fn needs_soft_reset(&self) -> bool {
        matches!(self, Self::SouthernIslands)
    }

    /// Returns true if this generation is expected to carry Display Core Next.
    pub fn has_dcn(&self) -> bool {
        // RX Vega and newer have DCN. Heuristic: RDNA2+ in our tables.
        matches!(self, Self::Rdna2 | Self::Rdna3)
    }
}

/// AMD GPU instance.
pub struct AmdGpu {
    pub bar0_phys: u64,
    pub bar2_phys: u64,
    pub bar5_phys: u64,
    pub device_id: u16,
    pub revision: u8,
    pub arch: AmdArchitecture,
    pub vram_mb: u32,
    pub gart_mb: u32,
    pub bios_version: [u8; 32],
    pub bios_version_len: usize,
}

impl AmdGpu {
    /// Probe a PCI device. If it is an AMD GPU, map BARs and read
    /// architecture / framebuffer / BIOS info. Returns `None` if the
    /// device is not AMD or if BAR0 is unusable.
    pub unsafe fn probe(dev: &PciDevice) -> Option<Self> {
        if dev.vendor_id != 0x1002 || dev.class != 0x03 {
            return None;
        }

        serial_println!("[AMDGPU] Probing device {:04X}:{:04X} rev={}", dev.vendor_id, dev.device_id, dev.revision);

        // -----------------------------------------------------------------
        // 1. Enable PCI bus mastering.
        // -----------------------------------------------------------------
        dev.enable_bus_mastering();
        serial_println!("[AMDGPU] PCI bus mastering enabled");

        // -----------------------------------------------------------------
        // 2. Map BAR addresses.
        // -----------------------------------------------------------------
        let bar0 = dev.bar_address(0);
        if bar0 == 0 {
            serial_println!("[AMDGPU] BAR0 is zero -- cannot map MMIO registers");
            return None;
        }
        let bar2 = dev.bar_address(2);
        let bar5 = dev.bar_address(5);

        serial_println!("[AMDGPU] BAR0 (MMIO)      = {:016X}", bar0);
        serial_println!("[AMDGPU] BAR2 (VRAM/GART) = {:016X}", bar2);
        serial_println!("[AMDGPU] BAR5 (aperture)  = {:016X}", bar5);

        // -----------------------------------------------------------------
        // 3. Architecture detection from device ID.
        // -----------------------------------------------------------------
        let arch = AmdArchitecture::from_device_id(dev.device_id);
        serial_println!("[AMDGPU] Architecture: {}", arch.name());

        // -----------------------------------------------------------------
        // 4. Basic MMIO health check.
        // -----------------------------------------------------------------
        let grbm_status = core::ptr::read_volatile((bar0 + MM_GRBM_STATUS) as *const u32);
        serial_println!("[AMDGPU] GRBM_STATUS = {:08X}", grbm_status);

        // Log MC_VM_FB_OFFSET as a hint (GCN+); this is an offset, not a size.
        let fb_offset = core::ptr::read_volatile((bar0 + MM_MC_VM_FB_OFFSET) as *const u32);
        serial_println!("[AMDGPU] MC_VM_FB_OFFSET = {:08X} (hint only)", fb_offset);

        // -----------------------------------------------------------------
        // 5. VRAM detection.
        // -----------------------------------------------------------------
        let vram_mb = Self::detect_vram_size(dev, bar0, bar2);
        serial_println!("[AMDGPU] VRAM detected: {} MiB", vram_mb);

        // -----------------------------------------------------------------
        // 6. GART aperture detection.
        // -----------------------------------------------------------------
        let gart_mb = Self::detect_gart_size(dev);
        if gart_mb > 0 {
            serial_println!("[AMDGPU] GART aperture detected: {} MiB", gart_mb);
        } else {
            serial_println!("[AMDGPU] GART aperture size not detectable via PCI BAR heuristic");
        }

        // -----------------------------------------------------------------
        // 7. VBIOS / ATOMBIOS readout.
        // -----------------------------------------------------------------
        let (bios_version, bios_version_len) = Self::read_atombios(dev);
        if bios_version_len > 0 {
            let ver_str = core::str::from_utf8(&bios_version[..bios_version_len]).unwrap_or("???");
            serial_println!("[AMDGPU] BIOS version: {}", ver_str);
        } else {
            serial_println!("[AMDGPU] BIOS version not readable (ROM BAR may be disabled or unmapped)");
        }

        // -----------------------------------------------------------------
        // 8. DCN detection.
        // -----------------------------------------------------------------
        if arch.has_dcn() {
            serial_println!("[AMDGPU] Display Core Next (DCN) present on this generation");
            serial_println!("[AMDGPU] NOTE: DCN display engine init requires SMU firmware -- skipping");
        } else {
            serial_println!("[AMDGPU] DCN not present on this generation (or unknown)");
        }

        // -----------------------------------------------------------------
        // 9. Safe init: soft reset for pre-GCN / Southern Islands.
        // -----------------------------------------------------------------
        if arch.needs_soft_reset() {
            serial_println!("[AMDGPU] Performing soft reset via GRBM_SOFT_RESET...");
            core::ptr::write_volatile((bar0 + MM_GRBM_SOFT_RESET) as *mut u32, 0x00000001);

            // Poll until reset bit clears or we time out.
            let mut completed = false;
            for i in 0..10_000 {
                let status = core::ptr::read_volatile((bar0 + MM_GRBM_SOFT_RESET) as *const u32);
                if status & 0x1 == 0 {
                    serial_println!("[AMDGPU] Soft reset complete after {} iterations", i);
                    completed = true;
                    break;
                }
                core::arch::x86_64::_mm_pause();
            }
            if !completed {
                serial_println!("[AMDGPU] WARNING: Soft reset did not complete in time");
            }
        }

        // -----------------------------------------------------------------
        // 10. Final idle status.
        // -----------------------------------------------------------------
        let final_grbm = core::ptr::read_volatile((bar0 + MM_GRBM_STATUS) as *const u32);
        let idle_hint = (final_grbm >> 15) & 1;
        serial_println!(
            "[AMDGPU] Final GRBM_STATUS = {:08X} -- {}",
            final_grbm,
            if idle_hint == 1 { "IDLE" } else { "BUSY/UNKNOWN" }
        );

        serial_println!(
            "[AMDGPU] Initialized {:04X}:{:04X} -- BAR0={:08X} BAR2={:08X} BAR5={:08X}",
            dev.vendor_id,
            dev.device_id,
            bar0,
            bar2,
            bar5
        );

        Some(Self {
            bar0_phys: bar0,
            bar2_phys: bar2,
            bar5_phys: bar5,
            device_id: dev.device_id,
            revision: dev.revision,
            arch,
            vram_mb,
            gart_mb,
            bios_version,
            bios_version_len,
        })
    }

    // -----------------------------------------------------------------
    // VRAM size heuristics
    // -----------------------------------------------------------------

    /// Detect VRAM size using MMIO register first, then PCI BAR2 probing.
    unsafe fn detect_vram_size(dev: &PciDevice, bar0: u64, bar2: u64) -> u32 {
        // Try mmCONFIG_MEMSIZE first (valid on some generations).
        let mmio_size = core::ptr::read_volatile((bar0 + MM_CONFIG_MEMSIZE) as *const u32);
        if mmio_size != 0 && mmio_size != 0xFFFFFFFF {
            let mb = mmio_size / (1024 * 1024);
            if mb > 0 && mb <= 65_536 {
                serial_println!("[AMDGPU] VRAM size from CONFIG_MEMSIZE: {} MiB", mb);
                return mb;
            }
        }

        // Fallback: probe BAR2 size.
        let bar2_size = Self::probe_bar_size(dev, 2);
        if bar2_size > 0 {
            let mb = (bar2_size / (1024 * 1024)) as u32;
            serial_println!(
                "[AMDGPU] VRAM size from BAR2 probe: {} MiB (heuristic; may include GART aperture)",
                mb
            );
            return mb;
        }

        // If BAR2 is mapped but probe failed, use a safe fallback.
        if bar2 != 0 {
            serial_println!("[AMDGPU] VRAM size fallback: using fixed 256 MiB guess");
            return 256;
        }

        serial_println!("[AMDGPU] VRAM size could not be detected; defaulting to 0");
        0
    }

    // -----------------------------------------------------------------
    // GART aperture heuristics
    // -----------------------------------------------------------------

    /// Detect GART aperture size from BAR4 or a small BAR2.
    fn detect_gart_size(dev: &PciDevice) -> u32 {
        // BAR4 is often the GART table on older generations.
        let bar4 = dev.bar_address(4);
        if bar4 != 0 {
            let size = Self::probe_bar_size(dev, 4);
            if size > 0 {
                return (size / (1024 * 1024)) as u32;
            }
        }

        // Some APUs or integrated GPUs expose a small BAR2 for GART.
        let bar2 = dev.bar_address(2);
        if bar2 != 0 {
            let size = Self::probe_bar_size(dev, 2);
            if size > 0 && size < 512 * 1024 * 1024 {
                return (size / (1024 * 1024)) as u32;
            }
        }

        0
    }

    // -----------------------------------------------------------------
    // PCI BAR size probing
    // -----------------------------------------------------------------

    /// Probe the size of a PCI BAR by writing all-ones and reading back.
    /// This temporarily modifies the BAR; the original value is restored.
    fn probe_bar_size(dev: &PciDevice, bar: u8) -> u64 {
        let offset = 0x10 + bar * 4;
        let original = pci::pci_read32(dev.bus, dev.device, dev.function, offset);

        // BAR not implemented.
        if original == 0 {
            return 0;
        }

        // Write all-ones and read back the mask.
        pci::pci_write32(dev.bus, dev.device, dev.function, offset, 0xFFFFFFFF);
        let mask = pci::pci_read32(dev.bus, dev.device, dev.function, offset);

        // Restore original value.
        pci::pci_write32(dev.bus, dev.device, dev.function, offset, original);

        if mask == 0 {
            return 0;
        }

        let size_mask = mask & !0x0F; // clear type/prefetch bits
        let size = (!size_mask).wrapping_add(1) as u64;

        size
    }

    // -----------------------------------------------------------------
    // VBIOS / ATOMBIOS readout
    // -----------------------------------------------------------------

    /// Read PCI ROM BAR, verify 0x55AA signature, check ATOMBIOS magic,
    /// and extract the BIOS version string.
    unsafe fn read_atombios(dev: &PciDevice) -> ([u8; 32], usize) {
        let mut version = [0u8; 32];
        let rom_bar_raw = pci::pci_read32(dev.bus, dev.device, dev.function, PCI_ROM_BAR_OFFSET);

        // ROM BAR address is in upper bits; bit 0 is enable.
        let rom_enabled = (rom_bar_raw & 1) != 0;
        let rom_addr = (rom_bar_raw & !0x7FF) as u64;

        if rom_addr == 0 {
            serial_println!("[AMDGPU] ROM BAR address is zero");
            return (version, 0);
        }

        // Enable ROM BAR if it was disabled.
        if !rom_enabled {
            pci::pci_write32(dev.bus, dev.device, dev.function, PCI_ROM_BAR_OFFSET, rom_bar_raw | 1);
        }

        // Read signature at offset 0.
        let sig = core::ptr::read_volatile(rom_addr as *const u16);
        if sig != 0x55AA {
            serial_println!("[AMDGPU] ROM signature {:04X} != 0x55AA -- BIOS not found", sig);
            // Restore ROM BAR state before returning.
            if !rom_enabled {
                pci::pci_write32(dev.bus, dev.device, dev.function, PCI_ROM_BAR_OFFSET, rom_bar_raw);
            }
            return (version, 0);
        }

        // ATOMBIOS magic "ATOM" at offset 0x30 inside ROM.
        let atom_magic = core::ptr::read_volatile((rom_addr + 0x30) as *const u32);
        let atom_bytes = [
            (atom_magic & 0xFF) as u8,
            ((atom_magic >> 8) & 0xFF) as u8,
            ((atom_magic >> 16) & 0xFF) as u8,
            ((atom_magic >> 24) & 0xFF) as u8,
        ];
        let is_atombios = &atom_bytes == b"ATOM" || &atom_bytes == b"atom";

        serial_println!(
            "[AMDGPU] ROM found at {:08X}, sig={:04X}, ATOM magic={:08X} ({})",
            rom_addr,
            sig,
            atom_magic,
            if is_atombios { "ATOMBIOS" } else { "legacy/unknown" }
        );

        // BIOS version string typically at offset 0x50.
        let mut len = 0usize;
        for i in 0..32usize {
            let byte = core::ptr::read_volatile((rom_addr + 0x50 + i as u64) as *const u8);
            if byte == 0 {
                break;
            }
            version[i] = byte;
            len += 1;
        }

        // Restore ROM BAR if we enabled it.
        if !rom_enabled {
            pci::pci_write32(dev.bus, dev.device, dev.function, PCI_ROM_BAR_OFFSET, rom_bar_raw);
        }

        (version, len)
    }

    /// Return the detected VRAM size in MiB.
    pub fn vram_size_mb(&self) -> u32 {
        self.vram_mb
    }

    /// Return the detected GART aperture size in MiB.
    pub fn gart_size_mb(&self) -> u32 {
        self.gart_mb
    }
}

/// Scan PCI bus for an AMD display controller and initialize it.
/// Returns `Some(AmdGpu)` on success, `None` if no AMD GPU is found.
pub fn init() -> Option<AmdGpu> {
    let devices = crate::drivers::pci::enumerate();
    for dev in &devices {
        if dev.class == 0x03 && dev.vendor_id == 0x1002 {
            return unsafe { AmdGpu::probe(dev) };
        }
    }
    None
}
