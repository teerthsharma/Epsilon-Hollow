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
    // Borrow the header in place rather than copying it out: `is_valid`'s
    // checksum walk and length ceiling are computed relative to `self`'s own
    // address, which must be the table's real physical location. A
    // `read_unaligned()` copy onto the stack (the old code) would checksum
    // stack garbage instead of the actual MADT bytes. `SdtHeader` is
    // `#[repr(C, packed)]` (align 1), so dereferencing this raw pointer to
    // form a reference is sound regardless of `madt_phys`'s alignment.
    let header = unsafe { &*core::ptr::addr_of!((*madt).header) };
    if !header.is_valid() {
        crate::serial_println!(
            "[ACPI/MADT] MADT header at {:#X} failed validation (bad length or checksum), aborting parse",
            madt_phys
        );
        return;
    }
    let header_length = unsafe { core::ptr::addr_of!(header.length).read_unaligned() } as u64;

    let this_lapic = current_lapic_id();
    let mut ids: [u32; MAX_CPUS] = [0; MAX_CPUS];
    let mut count = 0;

    let entries_start = madt_phys + core::mem::size_of::<Madt>() as u64;
    // `header_length` is now bounded by `is_valid` (floor: size_of::<SdtHeader>(),
    // ceiling: MAX_TABLE_LEN and the identity-mapped region — see rsdp.rs),
    // so this can no longer wander past mapped memory.
    let entries_end = madt_phys + header_length;
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
            _ => {
                // Unknown MADT entry type; skip
            }
        }

        offset += entry_len;
    }
    // Commit discovered IDs to the global arrays.
    unsafe {
        let slots = &mut *APIC_IDS.0.get();
        slots[..count].copy_from_slice(&ids[..count]);
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

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// A synthetic MADT with exactly one Local APIC entry immediately after
    /// the fixed `Madt` header — real memory, laid out exactly like a real
    /// table, so `parse_madt` (which takes a raw physical-style address) can
    /// run against it unmodified.
    #[repr(C, packed)]
    struct TestMadtOneEntry {
        madt: Madt,
        entry: LocalApicEntry,
    }

    /// Recompute the header checksum over exactly `len` bytes (matching what
    /// `SdtHeader::is_valid` will checksum once `length` is set to `len`).
    fn fix_checksum(buf: &mut TestMadtOneEntry, len: usize) {
        unsafe {
            core::ptr::addr_of_mut!(buf.madt.header.checksum).write_unaligned(0);
            let bytes = core::slice::from_raw_parts(buf as *const _ as *const u8, len);
            let sum = bytes.iter().fold(0u8, |s, &b| s.wrapping_add(b));
            core::ptr::addr_of_mut!(buf.madt.header.checksum)
                .write_unaligned(0u8.wrapping_sub(sum));
        }
    }

    fn build_test_madt(declared_length: u32, apic_id: u8) -> TestMadtOneEntry {
        let mut m = TestMadtOneEntry {
            madt: Madt {
                header: SdtHeader {
                    signature: *b"APIC",
                    length: declared_length,
                    revision: 4,
                    checksum: 0,
                    oem_id: *b"SEALOS",
                    oem_table_id: *b"TESTMADT",
                    oem_revision: 1,
                    creator_id: 0,
                    creator_revision: 1,
                },
                local_apic_addr: 0xFEE0_0000,
                flags: 0,
            },
            entry: LocalApicEntry {
                header: MadtEntryHeader {
                    entry_type: 0,
                    length: core::mem::size_of::<LocalApicEntry>() as u8,
                },
                acpi_processor_id: 0,
                apic_id,
                flags: 1, // enabled
            },
        };
        fix_checksum(&mut m, declared_length as usize);
        m
    }

    /// RED (was): `header` was `read_unaligned()`-copied onto the stack,
    /// then `header.is_valid()` checksummed `header.length` bytes starting
    /// at *that stack copy's* address — reading past its 36 real bytes into
    /// unrelated stack memory instead of the table. GREEN: `header` now
    /// borrows `(*madt).header` in place (see `parse_madt` above), so the
    /// checksum below is computed over the real table and a corrupted byte
    /// is reliably caught.
    fn test_madt_bad_checksum_rejected() -> TestResult {
        let mut table = build_test_madt(core::mem::size_of::<TestMadtOneEntry>() as u32, 0xAA);
        // Corrupt one byte after the checksum was already made self-consistent.
        unsafe {
            core::ptr::addr_of_mut!(table.madt.header.checksum).write_unaligned(
                core::ptr::addr_of!(table.madt.header.checksum)
                    .read_unaligned()
                    .wrapping_add(1),
            );
        }
        let before = cpu_count();
        unsafe {
            parse_madt(&table as *const _ as u64);
        }
        // A rejected header must leave the previously-discovered CPU set
        // untouched — `parse_madt` returns before storing anything.
        test_assert_eq!(cpu_count(), before);
        TestResult::Pass
    }

    /// RED (was): `entries_end = madt_phys + header.length as u64` trusted
    /// an unvalidated `length`; here `length` is deliberately short (covers
    /// only the fixed `Madt` header, 44 bytes) while a real, well-formed
    /// Local APIC entry (apic_id 0xAA, enabled) sits in memory immediately
    /// after it. GREEN: the entry walk bounds every read against
    /// `entries_end`, so an entry beyond the declared length is never read.
    fn test_madt_entries_beyond_length_not_parsed() -> TestResult {
        let table = build_test_madt(core::mem::size_of::<Madt>() as u32, 0xAA);
        unsafe {
            parse_madt(&table as *const _ as u64);
        }
        test_assert_eq!(cpu_count(), 0);
        test_assert!(
            !apic_ids().contains(&0xAA),
            "entry beyond declared MADT length was parsed"
        );
        TestResult::Pass
    }

    /// Positive control for the two tests above: the exact same entry
    /// bytes, but with `length` correctly covering them, must be parsed.
    /// Without this, `test_madt_entries_beyond_length_not_parsed` could pass
    /// vacuously (e.g. if entry parsing were silently broken).
    fn test_madt_entry_within_length_is_parsed() -> TestResult {
        let table = build_test_madt(core::mem::size_of::<TestMadtOneEntry>() as u32, 0xAA);
        unsafe {
            parse_madt(&table as *const _ as u64);
        }
        test_assert_eq!(cpu_count(), 1);
        test_assert!(
            apic_ids().contains(&0xAA),
            "in-bounds entry was not parsed"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("acpi::madt::bad_checksum_rejected", test_madt_bad_checksum_rejected);
        crate::testing::register_test(
            "acpi::madt::entries_beyond_length_not_parsed",
            test_madt_entries_beyond_length_not_parsed,
        );
        crate::testing::register_test(
            "acpi::madt::entry_within_length_is_parsed",
            test_madt_entry_within_length_is_parsed,
        );
    }
}
