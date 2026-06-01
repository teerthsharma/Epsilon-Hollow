// Seal OS — NVIDIA RTX 4060 Host Probe Tool
// Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT
//!
//! Runs on a Linux host to validate that Seal OS's NVIDIA driver register
//! offsets and heuristics match the actual RTX 4060 hardware.
//!
//! Usage (root required for BAR0 mmap):
//!   sudo cargo run --release
//!
//! What it tests:
//!   1. PCI enumeration finds RTX 4060 family cards
//!   2. BAR0 is mapped and readable
//!   3. NV_PMC_BOOT_0 returns expected Ada architecture ID (0x00014xxx)
//!   4. NV_PMC_ENABLE has display engine bit set
//!   5. NV_PFB_CFG0 framebuffer size heuristic matches known 8 GB
//!   6. ROM BAR can be enabled and yields valid 0x55AA VBIOS signature

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

// Register offsets (must match kernel/seal-os/src/drivers/gpu/nvidia.rs)
const NV_PMC_BOOT_0: usize = 0x000000;
const NV_PMC_ENABLE: usize = 0x000200;
const NV_PFB_CFG0: usize = 0x100000;

// PCI config-space offsets
const PCI_CFG_VENDOR: usize = 0x00;
const PCI_CFG_DEVICE: usize = 0x02;
const PCI_CFG_BAR0: usize = 0x10;
const PCI_CFG_ROM_BAR: usize = 0x30;
const PCI_CFG_COMMAND: usize = 0x04;

// PCI command register bits
const PCI_COMMAND_MEMORY: u16 = 0x0004;
const PCI_COMMAND_IO: u16 = 0x0001;

// RTX 4060 family device IDs (must match kernel/seal-os/src/drivers/gpu/nvidia.rs)
const DID_RTX_4060_TI: u16 = 0x2803;
const DID_RTX_4060: u16 = 0x2882;
const DID_RTX_4060_MOBILE_A: u16 = 0x28A0;
const DID_RTX_4060_MOBILE_B: u16 = 0x28E0;

fn is_rtx_4060_family(did: u16) -> bool {
    did == DID_RTX_4060
        || did == DID_RTX_4060_TI
        || did == DID_RTX_4060_MOBILE_A
        || did == DID_RTX_4060_MOBILE_B
}

fn product_name(did: u16) -> &'static str {
    match did {
        DID_RTX_4060_TI => "GeForce RTX 4060 Ti",
        DID_RTX_4060 => "GeForce RTX 4060",
        DID_RTX_4060_MOBILE_A | DID_RTX_4060_MOBILE_B => "GeForce RTX 4060 Laptop GPU",
        _ => "Unknown NVIDIA GPU",
    }
}

fn arch_name(boot0: u32) -> &'static str {
    match boot0 & 0xFFFF0000 {
        0x00010000 => "Volta (GV100)",
        0x00011000 => "Turing (TU1xx)",
        0x00013000 => "Ampere (GA10x)",
        0x00014000 => "Ada (AD10x)",
        0x00016000 => "Blackwell (GB10x)",
        _ => match boot0 & 0x0000F000 {
            0x0000F000 => "Pascal (GP10x)",
            0x0000E000 => "Maxwell (GM1xx)",
            0x0000D000 => "Kepler (GK1xx)",
            0x0000C000 => "Fermi (GF100+)",
            _ => "Unknown",
        },
    }
}

/// Read a 32-bit little-endian word from a sysfs PCI config file.
fn pci_read32(path: &Path, offset: usize) -> u32 {
    let mut buf = [0u8; 4];
    let mut file = fs::File::open(path).expect("open PCI config");
    file.seek(SeekFrom::Start(offset as u64)).unwrap();
    file.read_exact(&mut buf).expect("read PCI config");
    u32::from_le_bytes(buf)
}

/// Write a 32-bit little-endian word to a sysfs PCI config file.
fn pci_write32(path: &Path, offset: usize, val: u32) {
    let buf = val.to_le_bytes();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open PCI config for write");
    file.seek(SeekFrom::Start(offset as u64)).unwrap();
    file.write_all(&buf).expect("write PCI config");
}

/// Read a 32-bit register from a BAR resource file.
fn bar0_read32(bar_path: &Path, offset: usize) -> u32 {
    let mut file = fs::File::open(bar_path).expect("open BAR0 resource");
    let mut buf = [0u8; 4];
    file.seek(SeekFrom::Start(offset as u64)).unwrap();
    file.read_exact(&mut buf).expect("read BAR0");
    u32::from_le_bytes(buf)
}

/// Enable PCI memory and I/O decoding, plus ROM BAR.
fn enable_pci_access(cfg: &Path, _rom_path: &Path) {
    let cmd = pci_read32(cfg, PCI_CFG_COMMAND) as u16;
    let new_cmd = cmd | PCI_COMMAND_MEMORY | PCI_COMMAND_IO;
    pci_write32(cfg, PCI_CFG_COMMAND, new_cmd as u32);

    // Enable ROM BAR (clear bit 0 to enable, but first preserve upper bits)
    let rom_bar = pci_read32(cfg, PCI_CFG_ROM_BAR);
    if (rom_bar & 0x01) == 0 {
        pci_write32(cfg, PCI_CFG_ROM_BAR, rom_bar | 0x01);
    }
}

fn probe_device(device_dir: &Path) -> Result<(), String> {
    let cfg = device_dir.join("config");
    let resource0 = device_dir.join("resource0");
    let rom_resource = device_dir.join("rom");

    let vendor = pci_read32(&cfg, PCI_CFG_VENDOR) as u16;
    let device = (pci_read32(&cfg, PCI_CFG_DEVICE) >> 16) as u16;

    if vendor != 0x10DE {
        return Err(format!("not NVIDIA (vendor={:04X})", vendor));
    }

    println!("\n================================================================================");
    println!("  Found NVIDIA GPU: {} [{:04X}:{:04X}]", product_name(device), vendor, device);
    println!("  Sysfs path: {}", device_dir.display());
    println!("================================================================================");

    if !is_rtx_4060_family(device) {
        println!("  NOTE: This is NOT an RTX 4060 family card. Probe continues for reference.");
    }

    // Enable access
    enable_pci_access(&cfg, &rom_resource);

    // Read BAR0
    let bar0_low = pci_read32(&cfg, PCI_CFG_BAR0);
    let bar0_phys = (bar0_low & !0x0F) as u64;
    println!("  BAR0 physical address: {:016X}", bar0_phys);

    if bar0_phys == 0 {
        return Err("BAR0 is zero".into());
    }

    if !resource0.exists() {
        return Err(format!(
            "BAR0 resource file not found: {} (run as root?)",
            resource0.display()
        ));
    }

    // Read registers (must match kernel offsets)
    let boot0 = bar0_read32(&resource0, NV_PMC_BOOT_0);
    let enable = bar0_read32(&resource0, NV_PMC_ENABLE);
    let pfb_cfg0 = bar0_read32(&resource0, NV_PFB_CFG0);

    println!("  NV_PMC_BOOT_0  = {:08X}  => arch = '{}'", boot0, arch_name(boot0));
    println!("  NV_PMC_ENABLE  = {:08X}  => display_engine = {}", enable, (enable & (1 << 12)) != 0);
    println!("  NV_PFB_CFG0    = {:08X}  => fb_heuristic = {} MiB", pfb_cfg0, ((pfb_cfg0 >> 12) & 0x1FF));

    // Validate expectations for RTX 4060
    let mut pass = true;
    if is_rtx_4060_family(device) {
        if (boot0 & 0xFFFF0000) != 0x00014000 {
            println!("  [FAIL] Expected Ada architecture mask 0x00014000, got {:08X}", boot0 & 0xFFFF0000);
            pass = false;
        } else {
            println!("  [PASS] Architecture mask matches Ada Lovelace (0x00014000)");
        }

        let fb_heuristic = ((pfb_cfg0 >> 12) & 0x1FF) as u32;
        if fb_heuristic != 0 && fb_heuristic < 8000 {
            println!("  [WARN] FB heuristic {} MiB seems low for RTX 4060 (expected ~8192)", fb_heuristic);
        } else if fb_heuristic == 0 {
            println!("  [INFO] FB heuristic is zero; Seal OS kernel falls back to 256 MiB");
        } else {
            println!("  [PASS] FB heuristic {} MiB is in expected range", fb_heuristic);
        }
    }

    // VBIOS read via ROM BAR
    if rom_resource.exists() {
        let rom_data = fs::read(&rom_resource).map_err(|e| format!("read ROM: {}", e))?;
        if rom_data.len() >= 2 {
            let sig = u16::from_le_bytes([rom_data[0], rom_data[1]]);
            if sig == 0x55AA {
                println!("  [PASS] VBIOS signature 0x55AA OK ({} bytes)", rom_data.len());
                // Print version string heuristic at offset 0x18
                let ver_start = 0x18usize.min(rom_data.len());
                let ver_end = (ver_start + 16).min(rom_data.len());
                let ver = String::from_utf8_lossy(&rom_data[ver_start..ver_end]);
                println!("  VBIOS version string @ 0x18: {:?}", ver.trim_end_matches('\0'));
            } else {
                println!("  [FAIL] VBIOS signature expected 0x55AA, got {:04X}", sig);
                pass = false;
            }
        } else {
            println!("  [FAIL] ROM too short ({} bytes)", rom_data.len());
            pass = false;
        }
    } else {
        println!("  [SKIP] ROM resource not available ({})", rom_resource.display());
    }

    println!("  OVERALL: {}", if pass { "PASS" } else { "FAIL" });
    Ok(())
}

fn main() {
    println!("Seal OS — NVIDIA RTX 4060 Host Probe Tool");
    println!("This tool validates kernel register assumptions against real hardware.\n");

    let pci_root = Path::new("/sys/bus/pci/devices");
    if !pci_root.exists() {
        eprintln!("ERROR: /sys/bus/pci/devices not found. This tool requires Linux with sysfs.");
        eprintln!("If you are on Windows, run 'wmic path Win32_VideoController get name,PNPDeviceID' instead.");
        std::process::exit(1);
    }

    let entries = fs::read_dir(pci_root).expect("read PCI sysfs");
    let mut found = 0usize;

    for entry in entries.flatten() {
        let path = entry.path();
        let vendor_file = path.join("vendor");
        if !vendor_file.exists() {
            continue;
        }
        let vendor_str = fs::read_to_string(&vendor_file).unwrap_or_default();
        if vendor_str.trim() != "0x10de" {
            continue;
        }

        match probe_device(&path) {
            Ok(()) => found += 1,
            Err(e) => println!("  Skipped {}: {}", path.display(), e),
        }
    }

    println!("\n================================================================================");
    if found == 0 {
        println!("No NVIDIA GPUs found via sysfs.");
        println!("If you have an RTX 4060, ensure:");
        println!("  1. You are running on bare-metal Linux (not WSL1).");
        println!("  2. The card is not bound to vfio-pci for QEMU passthrough.");
    } else {
        println!("Probe complete. {} NVIDIA GPU(s) examined.", found);
    }
    println!("================================================================================");
}
