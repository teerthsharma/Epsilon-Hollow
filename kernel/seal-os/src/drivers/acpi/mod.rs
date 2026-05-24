// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! ACPI subsystem — parses RSDP, RSDT/XSDT, and MADT for SMP discovery.

pub mod fadt;
pub mod madt;
pub mod rsdp;
pub mod topological_power;

use crate::boot::boot_info::BootInfo;
use crate::serial_println;

/// Initialise the ACPI subsystem from the UEFI-provided RSDP (if any).
pub fn init(boot_info: &BootInfo) {
    if boot_info.acpi_rsdp == 0 {
        serial_println!("[ACPI] No RSDP found — skipping ACPI init");
        return;
    }
    serial_println!("[ACPI] Using RSDP from UEFI @ {:#X}", boot_info.acpi_rsdp);
    let rsdp = match rsdp::find_rsdp(boot_info.acpi_rsdp) {
        Some(addr) => addr,
        None => {
            serial_println!("[ACPI] RSDP not found");
            return;
        }
    };
    serial_println!("[ACPI] RSDP validated @ {:#X}", rsdp);

    let madt_phys = match rsdp::walk_sdt(rsdp, b"APIC") {
        Some(addr) => addr,
        None => {
            serial_println!("[ACPI] MADT not found");
            return;
        }
    };
    serial_println!("[ACPI] MADT found @ {:#X}", madt_phys);

    unsafe {
        madt::parse_madt(madt_phys);
    }

    fadt::init(rsdp);
    topological_power::init();
}

/// Number of enabled CPUs found in the MADT.
pub fn cpu_count() -> usize {
    madt::cpu_count()
}

/// APIC ID of the Bootstrap Processor (BSP).
pub fn bsp_apic_id() -> u32 {
    madt::bsp_apic_id()
}

/// Slice of all enabled APIC IDs (including BSP).
pub fn apic_ids() -> &'static [u32] {
    madt::apic_ids()
}

/// Physical base address of the first I/O APIC.
pub fn ioapic_base() -> u64 {
    madt::ioapic_base()
}

/// Attempt ACPI power-off.
/// Returns `true` if a shutdown command was issued.
pub fn power_off() -> bool {
    fadt::enter_soft_off();
    true
}

/// Quick sanity print of ACPI discovery results.
pub fn test_acpi() {
    serial_println!(
        "[test_acpi] CPUs: {}  IOAPIC base: {:#X}",
        cpu_count(),
        ioapic_base()
    );
}
