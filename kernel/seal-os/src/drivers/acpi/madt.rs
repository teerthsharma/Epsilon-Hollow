// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! MADT parser — collects Local APIC IDs and IO APIC base address.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use super::rsdp::SdtHeader;

const MAX_CPUS: usize = 32;

// APIC ID arrays are written exactly once during early boot and read-only
// thereafter.  We use UnsafeCell so the static is Sync; all writes happen
// before any other CPU reads the data.
struct ApicIdArray(UnsafeCell<[u32; MAX_CPUS]>);
unsafe impl Sync for ApicIdArray {}

static APIC_IDS: ApicIdArray = ApicIdArray(UnsafeCell::new([0; MAX_CPUS]));
static APIC_COUNT: AtomicUsize = AtomicUsize::new(0);

static AP_APIC_IDS: ApicIdArray = ApicIdArray(UnsafeCell::new([0; MAX_CPUS]));
static AP_APIC_COUNT: AtomicUsize = AtomicUsize::new(0);

static IOAPIC_BASE: AtomicU64 = AtomicU64::new(0);
static BSP_APIC_ID: AtomicU32 = AtomicU32::new(0);

#[repr(C, packed)]
struct Madt {
    header: SdtHeader,
    local_apic_addr: u32,
    flags: u32,
}

#[repr(C, packed)]
struct MadtEntryHeader {
    entry_type: u8,
    length: u8,
}

#[repr(C, packed)]
struct LocalApicEntry {
    header: MadtEntryHeader,
    acpi_processor_id: u8,
    apic_id: u8,
    flags: u32,
}

#[repr(C, packed)]
struct IoApicEntry {
    header: MadtEntryHeader,
    ioapic_id: u8,
    reserved: u8,
    ioapic_addr: u32,
    global_system_interrupt_base: u32,
}

#[allow(dead_code)] // REASON: MADT entry type for future interrupt source override parsing
#[repr(C, packed)]
struct IntSrcOverrideEntry {
    header: MadtEntryHeader,
    bus_source: u8,
    irq_source: u8,
    global_system_interrupt: u32,
    flags: u16,
}

/// Read the local APIC ID of the current processor via MSR 0x1B + MMIO.
fn current_lapic_id() -> u32 {
    let apic_base = unsafe { x86_64::registers::model_specific::Msr::new(0x1B).read() };
    let base_addr = apic_base & 0xFFFF_FFFF_FFFF_F000;
    let reg = unsafe { core::ptr::read_volatile((base_addr + 0x20) as *const u32) };
    (reg >> 24) & 0xFF
}

/// Parse the MADT at `madt_phys`.
///
/// # Safety
/// Caller must ensure `madt_phys` points to a valid MADT in identity-mapped memory.
pub unsafe fn parse_madt(madt_phys: u64) {
    let madt = madt_phys as *const Madt;
    let header = unsafe { core::ptr::addr_of!((*madt).header).read_unaligned() };
    if !header.is_valid() {
        return;
    }

    let this_lapic = current_lapic_id();
    let mut ids: [u32; MAX_CPUS] = [0; MAX_CPUS];
    let mut count = 0;

    let entries_start = madt_phys + core::mem::size_of::<Madt>() as u64;
    let entries_end = madt_phys + header.length as u64;
    let mut offset = entries_start;

    while offset < entries_end {
        if offset + core::mem::size_of::<MadtEntryHeader>() as u64 > entries_end {
            crate::serial_println!(
                "[ACPI/MADT] MADT truncated: cannot read entry header at offset {:#X}",
                offset - madt_phys
            );
            break;
        }
        let hdr = offset as *const MadtEntryHeader;
        let entry_type = unsafe { core::ptr::addr_of!((*hdr).entry_type).read_unaligned() };
        let entry_len = unsafe { core::ptr::addr_of!((*hdr).length).read_unaligned() as u64 };
        if entry_len == 0 {
            crate::serial_println!("[ACPI/MADT] MADT entry claims length 0, stopping parse");
            break;
        }

        match entry_type {
            0 => {
                if offset + core::mem::size_of::<LocalApicEntry>() as u64 <= entries_end {
                    let entry = offset as *const LocalApicEntry;
                    let flags = unsafe { core::ptr::addr_of!((*entry).flags).read_unaligned() }; // copy out of packed struct
                    if (flags & 1) != 0 && count < MAX_CPUS {
                        let apic_id = unsafe {
                            core::ptr::addr_of!((*entry).apic_id).read_unaligned() as u32
                        };
                        ids[count] = apic_id;
                        count += 1;
                        if apic_id == this_lapic {
                            BSP_APIC_ID.store(apic_id, Ordering::SeqCst);
                        }
                    }
                } else {
                    crate::serial_println!(
                        "[ACPI/MADT] Local APIC entry at offset {:#X} truncated, skipping",
                        offset - madt_phys
                    );
                }
            }
            1 => {
                if offset + core::mem::size_of::<IoApicEntry>() as u64 <= entries_end {
                    let entry = offset as *const IoApicEntry;
                    let ioapic_addr =
                        unsafe { core::ptr::addr_of!((*entry).ioapic_addr).read_unaligned() };
                    IOAPIC_BASE.store(ioapic_addr as u64, Ordering::SeqCst);
                } else {
                    crate::serial_println!(
                        "[ACPI/MADT] I/O APIC entry at offset {:#X} truncated, skipping",
                        offset - madt_phys
                    );
                }
            }
            2 => {
                // Interrupt Source Override — parsed but not stored.
            }
            _ => {}
        }

        offset += entry_len;
    }
    // Commit discovered IDs to the global arrays.
    unsafe {
        let ptr = APIC_IDS.0.get();
        for i in 0..count {
            (*ptr)[i] = ids[i];
        }
    }
    APIC_COUNT.store(count, Ordering::SeqCst);

    // If BSP wasn't matched, default to the first ID.
    if BSP_APIC_ID.load(Ordering::SeqCst) == 0 && count > 0 {
        BSP_APIC_ID.store(ids[0], Ordering::SeqCst);
    }

    // Build AP-only list.
    let bsp = BSP_APIC_ID.load(Ordering::SeqCst);
    let mut ap_count = 0;
    unsafe {
        let ptr = AP_APIC_IDS.0.get();
        for i in 0..count {
            if ids[i] != bsp {
                (*ptr)[ap_count] = ids[i];
                ap_count += 1;
            }
        }
    }
    AP_APIC_COUNT.store(ap_count, Ordering::SeqCst);
}

/// Number of enabled CPUs discovered.
pub fn cpu_count() -> usize {
    APIC_COUNT.load(Ordering::SeqCst)
}

/// APIC ID of the Bootstrap Processor.
pub fn bsp_apic_id() -> u32 {
    BSP_APIC_ID.load(Ordering::SeqCst)
}

/// All enabled APIC IDs, including the BSP.
pub fn apic_ids() -> &'static [u32] {
    let count = APIC_COUNT.load(Ordering::SeqCst);
    let array = unsafe { &*APIC_IDS.0.get() };
    &array[..count]
}

/// APIC IDs of application processors (all except BSP).
pub fn ap_apic_ids() -> &'static [u32] {
    let count = AP_APIC_COUNT.load(Ordering::SeqCst);
    let array = unsafe { &*AP_APIC_IDS.0.get() };
    &array[..count]
}

/// Physical base address of the first I/O APIC.
pub fn ioapic_base() -> u64 {
    IOAPIC_BASE.load(Ordering::SeqCst)
}
