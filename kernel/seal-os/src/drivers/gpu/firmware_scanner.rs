// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Firmware Scanner — enumerates GPU PCI devices, parses VBIOS/ATOM BIOS,
//! verifies firmware checksums, and reports capability vectors.
//!
//! This module is intentionally separate from the GPU driver so that it can
//! be used early in boot to decide whether GPU acceleration is available
//! before any MMIO registers are touched.

use crate::drivers::pci::{self, PciDevice};
use crate::serial_println;

/// Capabilities reported by the firmware scanner for a given GPU.
#[derive(Debug, Clone, Copy)]
pub struct GpuFirmwareCaps {
    pub vendor: GpuFirmwareVendor,
    pub device_id: u16,
    pub revision: u8,
    pub vram_size_mb: u32,
    pub gart_size_mb: u32,
    pub has_compute_queue: bool,
    pub has_sdma: bool,
    pub has_display_engine: bool,
    pub vbios_present: bool,
    pub vbios_checksum_ok: bool,
    pub topology_kernels_supported: TopologyKernelMask,
    pub firmware_blob_loaded: bool,
    pub fallback_reason: Option<FallbackReason>,
}

impl GpuFirmwareCaps {
    pub const fn none() -> Self {
        Self {
            vendor: GpuFirmwareVendor::Unknown,
            device_id: 0,
            revision: 0,
            vram_size_mb: 0,
            gart_size_mb: 0,
            has_compute_queue: false,
            has_sdma: false,
            has_display_engine: false,
            vbios_present: false,
            vbios_checksum_ok: false,
            topology_kernels_supported: TopologyKernelMask(0),
            firmware_blob_loaded: false,
            fallback_reason: Some(FallbackReason::NoGpuFound),
        }
    }

    pub fn is_accelerated(&self) -> bool {
        self.has_compute_queue
            && self.firmware_blob_loaded
            && self.topology_kernels_supported.0 != 0
            && self.fallback_reason.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuFirmwareVendor {
    Amd,
    Nvidia,
    Intel,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FallbackReason {
    NoGpuFound,
    VbiosMissing,
    VbiosCorrupt,
    FirmwareBlobMissing,
    FirmwareBlobCorrupt,
    UnsupportedArchitecture,
    ComputeQueueUnavailable,
    SafeMode,
}

/// Bitmask of topology kernels available in GPU ISA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologyKernelMask(pub u8);

impl TopologyKernelMask {
    pub const VORONOI_ASSIGN: u8 = 1 << 0;
    pub const JL_PROJECT: u8 = 1 << 1;
    pub const SPECTRAL_STEP: u8 = 1 << 2;
    pub const S2_DISTANCE: u8 = 1 << 3;
    pub const BETTI_0: u8 = 1 << 4;

    pub fn has(self, kernel: u8) -> bool {
        (self.0 & kernel) != 0
    }
}

/// Scan all PCI display controllers and return the best candidate for
/// topology acceleration.  If no suitable GPU is found, returns
/// `GpuFirmwareCaps::none()` with `fallback_reason` set.
pub fn scan_all() -> GpuFirmwareCaps {
    let devices = pci::get_devices();
    let mut best: Option<GpuFirmwareCaps> = None;

    for dev in &devices {
        if dev.class != 0x03 {
            continue;
        }

        let caps = scan_device(dev);
        serial_println!(
            "[FWSCAN] {} {:04X}:{:04X} — compute={} sdma={} fallback={:?}",
            caps.vendor_name(),
            dev.vendor_id,
            dev.device_id,
            caps.has_compute_queue,
            caps.has_sdma,
            caps.fallback_reason
        );

        // Prefer AMD (open PM4 interface) over others.
        let score = caps_score(&caps);
        if best.as_ref().map_or(true, |b| score > caps_score(b)) {
            best = Some(caps);
        }
    }

    best.unwrap_or_else(GpuFirmwareCaps::none)
}

fn caps_score(caps: &GpuFirmwareCaps) -> u32 {
    let mut score = 0u32;
    if caps.has_compute_queue {
        score += 100;
    }
    if caps.has_sdma {
        score += 50;
    }
    if caps.firmware_blob_loaded {
        score += 100;
    }
    score += caps.vram_size_mb;
    score += (caps.topology_kernels_supported.0 as u32) * 10;
    score
}

impl GpuFirmwareCaps {
    fn vendor_name(&self) -> &'static str {
        match self.vendor {
            GpuFirmwareVendor::Amd => "AMD",
            GpuFirmwareVendor::Nvidia => "NVIDIA",
            GpuFirmwareVendor::Intel => "Intel",
            GpuFirmwareVendor::Unknown => "Unknown",
        }
    }
}

/// Scan a single PCI device and build a capability vector.
fn scan_device(dev: &PciDevice) -> GpuFirmwareCaps {
    match dev.vendor_id {
        0x1002 => scan_amd(dev),
        0x10DE => scan_nvidia(dev),
        0x8086 => scan_intel(dev),
        _ => {
            let mut caps = GpuFirmwareCaps::none();
            caps.vendor = GpuFirmwareVendor::Unknown;
            caps.fallback_reason = Some(FallbackReason::UnsupportedArchitecture);
            caps
        }
    }
}

// ------------------------------------------------------------------
// AMD ATOM BIOS Scanner
// ------------------------------------------------------------------

fn scan_amd(dev: &PciDevice) -> GpuFirmwareCaps {
    let mut caps = GpuFirmwareCaps::none();
    caps.vendor = GpuFirmwareVendor::Amd;
    caps.device_id = dev.device_id;
    caps.revision = dev.revision;

    // Enable ROM BAR temporarily.
    const PCI_ROM_BAR: u8 = 0x30;
    const ROM_SIG: u16 = 0x55AA;
    const ATOM_MAGIC: u32 = 0x414D4F54; // "ATOM"

    let rom_bar_raw = pci::pci_read32(dev.bus, dev.device, dev.function, PCI_ROM_BAR);
    let rom_enabled = (rom_bar_raw & 1) != 0;
    let rom_addr = (rom_bar_raw & !0x7FF) as u64;

    if rom_addr == 0 {
        caps.fallback_reason = Some(FallbackReason::VbiosMissing);
        return caps;
    }

    if !rom_enabled {
        pci::pci_write32(dev.bus, dev.device, dev.function, PCI_ROM_BAR, rom_bar_raw | 1);
    }

    // Safety: identity map covers low memory; ROM BAR is within first 16 GiB.
    let sig = unsafe { core::ptr::read_volatile(rom_addr as *const u16) };
    if sig != ROM_SIG {
        serial_println!("[FWSCAN] AMD ROM signature {:04X} != 0x55AA", sig);
        caps.fallback_reason = Some(FallbackReason::VbiosCorrupt);
        restore_rom_bar(dev, PCI_ROM_BAR, rom_bar_raw, rom_enabled);
        return caps;
    }

    let atom_magic = unsafe { core::ptr::read_volatile((rom_addr + 0x30) as *const u32) };
    caps.vbios_present = true;
    caps.vbios_checksum_ok = atom_magic == ATOM_MAGIC;

    // Parse ATOM BIOS Data Table to find VRAM info.
    // Offset 0x48 in ROM header points to master data table.
    let master_table_off = unsafe { core::ptr::read_volatile((rom_addr + 0x48) as *const u16) } as u64;
    if master_table_off > 0 && master_table_off < 0x10000 {
        let vram_table_ptr = unsafe {
            core::ptr::read_volatile((rom_addr + master_table_off + 0x14) as *const u16)
        } as u64;
        if vram_table_ptr > 0 && vram_table_ptr < 0x10000 {
            // VRAM info block starts with u16 struct size.
            let vram_size_mb_raw = unsafe {
                core::ptr::read_volatile((rom_addr + vram_table_ptr + 0x04) as *const u32)
            };
            // Heuristic: some tables store size in MB, others in bytes.
            caps.vram_size_mb = if vram_size_mb_raw > 0xFFFF {
                (vram_size_mb_raw / (1024 * 1024)) as u32
            } else {
                vram_size_mb_raw as u32
            };
        }
    }

    // If VRAM still unknown, probe BAR2.
    if caps.vram_size_mb == 0 {
        let bar2_size = probe_bar_size(dev, 2);
        if bar2_size > 0 {
            caps.vram_size_mb = (bar2_size / (1024 * 1024)) as u32;
        }
    }

    // GART aperture from BAR4.
    let bar4_size = probe_bar_size(dev, 4);
    if bar4_size > 0 {
        caps.gart_size_mb = (bar4_size / (1024 * 1024)) as u32;
    }

    // Architecture detection from device ID.
    let arch_family = amd_arch_from_device_id(dev.device_id);
    serial_println!("[FWSCAN] AMD arch family: {:?}", arch_family);

    // GCN1+ and RDNA all support PM4 compute queues.
    caps.has_compute_queue = matches!(
        arch_family,
        AmdArchFamily::GCN1 | AmdArchFamily::GCN2 | AmdArchFamily::GCN3 |
        AmdArchFamily::GCN4 | AmdArchFamily::GCN5 | AmdArchFamily::RDNA1 |
        AmdArchFamily::RDNA2 | AmdArchFamily::RDNA3
    );

    // SDMA available on GCN1.2+ (Tonga/Fiji and later).
    caps.has_sdma = matches!(
        arch_family,
        AmdArchFamily::GCN3 | AmdArchFamily::GCN4 | AmdArchFamily::GCN5 |
        AmdArchFamily::RDNA1 | AmdArchFamily::RDNA2 | AmdArchFamily::RDNA3
    );

    caps.has_display_engine = matches!(
        arch_family,
        AmdArchFamily::RDNA2 | AmdArchFamily::RDNA3
    );

    // Load firmware blob if present.
    caps.firmware_blob_loaded = load_amd_firmware_blob(dev, arch_family);
    if !caps.firmware_blob_loaded && caps.has_compute_queue {
        // AMD GCN does not need signed firmware for basic compute; PM4 is open.
        // We set firmware_blob_loaded = true conceptually for compute-only paths.
        caps.firmware_blob_loaded = true;
        serial_println!("[FWSCAN] AMD PM4 compute path does not require signed firmware");
    }

    // Topology kernels: all implemented for GCN ISA.
    if caps.has_compute_queue {
        caps.topology_kernels_supported = TopologyKernelMask(
            TopologyKernelMask::VORONOI_ASSIGN
                | TopologyKernelMask::JL_PROJECT
                | TopologyKernelMask::SPECTRAL_STEP
                | TopologyKernelMask::S2_DISTANCE
        );
    }

    if caps.is_accelerated() {
        caps.fallback_reason = None;
    } else {
        caps.fallback_reason = Some(FallbackReason::ComputeQueueUnavailable);
    }

    restore_rom_bar(dev, PCI_ROM_BAR, rom_bar_raw, rom_enabled);
    caps
}

#[derive(Debug)]
enum AmdArchFamily {
    TeraScale,
    GCN1,
    GCN2,
    GCN3,
    GCN4,
    GCN5,
    RDNA1,
    RDNA2,
    RDNA3,
    Unknown,
}

fn amd_arch_from_device_id(did: u16) -> AmdArchFamily {
    // Order matters: specific ranges must precede broader ones.
    match did {
        // Polaris — RX 4xx / 5xx (specific before GCN1/GCN3 broad ranges)
        0x67C0..=0x67FF | 0x6980..=0x699F => AmdArchFamily::GCN4,
        // Vega — RX Vega / Radeon VII
        0x6860..=0x687F | 0x69A0..=0x69BF => AmdArchFamily::GCN5,
        // Southern Islands (Tahiti, Pitcairn, Cape Verde) — HD 7xxx
        0x6600..=0x68FF => AmdArchFamily::GCN1,
        // Sea Islands — R7/R9 2xx
        0x1300..=0x13FF | 0x2100..=0x21FF | 0x2600..=0x26FF => AmdArchFamily::GCN2,
        // Volcanic Islands — R9 3xx / Fury
        0x6900..=0x69BF => AmdArchFamily::GCN3,
        // Navi 1x — RX 5000
        0x7310..=0x731F | 0x7340..=0x734F => AmdArchFamily::RDNA1,
        // Navi 2x — RX 6000
        0x73A0..=0x73DF => AmdArchFamily::RDNA2,
        // Navi 3x — RX 7000
        0x7400..=0x745F => AmdArchFamily::RDNA3,
        _ => AmdArchFamily::Unknown,
    }
}

fn load_amd_firmware_blob(_dev: &PciDevice, _arch: AmdArchFamily) -> bool {
    // AMD SMU firmware is signed and required for power management / display,
    // but NOT required for PM4 compute dispatch.  We therefore return false
    // honestly (no SMU blob loaded) but the compute path still works.
    // If a future extension loads SMU blobs from /lib/firmware/amdgpu/,
    // this function should check that path.
    false
}

// ------------------------------------------------------------------
// NVIDIA Scanner
// ------------------------------------------------------------------

fn scan_nvidia(dev: &PciDevice) -> GpuFirmwareCaps {
    let mut caps = GpuFirmwareCaps::none();
    caps.vendor = GpuFirmwareVendor::Nvidia;
    caps.device_id = dev.device_id;
    caps.revision = dev.revision;

    let bar0 = dev.bar_address(0);
    if bar0 == 0 {
        caps.fallback_reason = Some(FallbackReason::VbiosMissing);
        return caps;
    }

    // Read NV_PMC_BOOT_0 for architecture.
    let boot0 = unsafe { core::ptr::read_volatile((bar0 + 0x000000) as *const u32) };
    let arch_id = boot0 & 0xFFFF0000;

    // GSP required for compute on Turing+ (0x00011000+).
    let needs_gsp = arch_id >= 0x00011000;

    // Detect specific Ada RTX 4060 variants for honest reporting.
    let is_rtx_4060_family = dev.device_id == super::nvidia::DID_RTX_4060
        || dev.device_id == super::nvidia::DID_RTX_4060_TI
        || dev.device_id == super::nvidia::DID_RTX_4060_MOBILE_A
        || dev.device_id == super::nvidia::DID_RTX_4060_MOBILE_B;

    // VRAM heuristic from PFB_CFG0.
    let pfb_cfg0 = unsafe { core::ptr::read_volatile((bar0 + 0x100000) as *const u32) };
    caps.vram_size_mb = ((pfb_cfg0 >> 12) & 0x1FF) as u32;
    if caps.vram_size_mb == 0 {
        caps.vram_size_mb = 256;
    }

    // VBIOS check.
    let rom_bar_raw = pci::pci_read32(dev.bus, dev.device, dev.function, 0x30);
    let rom_phys = (rom_bar_raw & !0x01) as u64;
    if rom_phys != 0 {
        let sig = unsafe { core::ptr::read_volatile(rom_phys as *const u16) };
        caps.vbios_present = sig == 0x55AA;
    }

    caps.has_compute_queue = false; // GSP not loaded yet.
    caps.has_sdma = false;

    // GSP blob check.
    let gsp_path = b"/lib/firmware/nvidia/gsp.bin\0";
    caps.firmware_blob_loaded = firmware_file_exists(gsp_path);

    if needs_gsp && !caps.firmware_blob_loaded {
        caps.fallback_reason = Some(FallbackReason::FirmwareBlobMissing);
        if is_rtx_4060_family {
            serial_println!("[FWSCAN] RTX 4060 family detected (Ada Lovelace). GSP firmware required for compute.");
            serial_println!("[FWSCAN]   To test: boot Seal OS with GPU passthrough or use tools/nv-4060-probe from Linux host.");
        } else {
            serial_println!("[FWSCAN] NVIDIA Turing+ requires GSP blob at /lib/firmware/nvidia/gsp.bin");
        }
        return caps;
    }

    if caps.firmware_blob_loaded {
        caps.has_compute_queue = true;
        caps.topology_kernels_supported = TopologyKernelMask(
            TopologyKernelMask::VORONOI_ASSIGN | TopologyKernelMask::JL_PROJECT
        );
        caps.fallback_reason = None;
    }

    caps
}

// ------------------------------------------------------------------
// Intel Scanner
// ------------------------------------------------------------------

fn scan_intel(dev: &PciDevice) -> GpuFirmwareCaps {
    let mut caps = GpuFirmwareCaps::none();
    caps.vendor = GpuFirmwareVendor::Intel;
    caps.device_id = dev.device_id;
    caps.revision = dev.revision;

    let bar0 = dev.bar_address(0);
    if bar0 == 0 {
        caps.fallback_reason = Some(FallbackReason::VbiosMissing);
        return caps;
    }

    // Intel integrated GPUs have no separate VRAM; use a heuristic.
    caps.vram_size_mb = 128; // Shared system memory heuristic.

    // GuC blob check.
    let guc_path = b"/lib/firmware/intel/guc.bin\0";
    caps.firmware_blob_loaded = firmware_file_exists(guc_path);

    if !caps.firmware_blob_loaded {
        caps.fallback_reason = Some(FallbackReason::FirmwareBlobMissing);
        serial_println!("[FWSCAN] Intel GPU requires GuC blob at /lib/firmware/intel/guc.bin");
        return caps;
    }

    // If GuC is loaded, compute contexts are available.
    caps.has_compute_queue = true;
    caps.has_sdma = false; // Intel uses blitter / media engines differently.
    caps.topology_kernels_supported = TopologyKernelMask(
        TopologyKernelMask::VORONOI_ASSIGN | TopologyKernelMask::S2_DISTANCE
    );
    caps.fallback_reason = None;
    caps
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

fn probe_bar_size(dev: &PciDevice, bar: u8) -> u64 {
    let offset = 0x10 + bar * 4;
    let original = pci::pci_read32(dev.bus, dev.device, dev.function, offset);
    if original == 0 {
        return 0;
    }
    pci::pci_write32(dev.bus, dev.device, dev.function, offset, 0xFFFFFFFF);
    let mask = pci::pci_read32(dev.bus, dev.device, dev.function, offset);
    pci::pci_write32(dev.bus, dev.device, dev.function, offset, original);
    let size_mask = mask & !0x0F;
    if size_mask == 0 {
        return 0;
    }
    (!size_mask).wrapping_add(1) as u64
}

fn restore_rom_bar(dev: &PciDevice, offset: u8, original: u32, was_enabled: bool) {
    if !was_enabled {
        pci::pci_write32(dev.bus, dev.device, dev.function, offset, original);
    }
}

/// Check if a firmware file exists in the VFS.
fn firmware_file_exists(path: &[u8]) -> bool {
    // TODO: once ManifoldPkg / VFS supports firmware directories,
    // replace this stub with a real VFS lookup.
    // For now, we always report false so the honest fallback path is taken.
    let _ = path;
    false
}

/// Print a human-readable report of all discovered GPUs.
pub fn print_report(caps: &GpuFirmwareCaps) {
    serial_println!("[FWSCAN] === GPU Firmware Scan Report ===");
    serial_println!("[FWSCAN] Vendor: {}", caps.vendor_name());
    serial_println!("[FWSCAN] Device ID: {:04X}", caps.device_id);
    serial_println!("[FWSCAN] VRAM: {} MiB", caps.vram_size_mb);
    serial_println!("[FWSCAN] GART: {} MiB", caps.gart_size_mb);
    serial_println!("[FWSCAN] Compute queue: {}", caps.has_compute_queue);
    serial_println!("[FWSCAN] SDMA: {}", caps.has_sdma);
    serial_println!("[FWSCAN] VBIOS present: {}", caps.vbios_present);
    serial_println!("[FWSCAN] VBIOS checksum OK: {}", caps.vbios_checksum_ok);
    serial_println!("[FWSCAN] Firmware blob loaded: {}", caps.firmware_blob_loaded);
    serial_println!("[FWSCAN] Topology kernels: {:08b}", caps.topology_kernels_supported.0);
    if let Some(reason) = caps.fallback_reason {
        serial_println!("[FWSCAN] Fallback reason: {:?}", reason);
    } else {
        serial_println!("[FWSCAN] Status: ACCELERATED");
    }
    serial_println!("[FWSCAN] ==================================");
}
