// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! RSDP/RSDT/XSDT walker.

use core::mem;

/// Root System Description Pointer (ACPI 2.0, 36 bytes).
#[repr(C, packed)]
pub struct Rsdp {
    pub signature: [u8; 8],
    pub checksum: u8,
    pub oem_id: [u8; 6],
    pub revision: u8,
    pub rsdt_addr: u32,
    pub length: u32,
    pub xsdt_addr: u64,
    pub ext_checksum: u8,
    pub reserved: [u8; 3],
}

impl Rsdp {
    /// Validate signature and checksum(s).
    pub fn is_valid(&self) -> bool {
        let sig = unsafe { core::ptr::addr_of!(self.signature).read_unaligned() };
        if &sig != b"RSD PTR " {
            return false;
        }

        // Validate v1 checksum over first 20 bytes.
        let bytes = unsafe { core::slice::from_raw_parts(self as *const _ as *const u8, 20) };
        if bytes.iter().fold(0u8, |sum, &b| sum.wrapping_add(b)) != 0 {
            return false;
        }

        let revision = unsafe { core::ptr::addr_of!(self.revision).read_unaligned() };
        if revision >= 2 {
            let len = unsafe { core::ptr::addr_of!(self.length).read_unaligned() } as usize;
            // The ACPI 2.0+ RSDP is a fixed, fully-declared 36-byte structure
            // (this `struct Rsdp` above) with no trailing/variable data — the
            // spec defines `Length` to be exactly `size_of::<Rsdp>()`. Sanity
            // check BEFORE the checksum walk below, and require an exact
            // match rather than just a floor: a `Length` that's merely large
            // (e.g. 0xFFFF_FFF0) must not be allowed to size the scan.
            if len != mem::size_of::<Rsdp>() {
                return false;
            }
            let ext_bytes =
                unsafe { core::slice::from_raw_parts(self as *const _ as *const u8, len) };
            if ext_bytes.iter().fold(0u8, |sum, &b| sum.wrapping_add(b)) != 0 {
                return false;
            }
        }
        true
    }
}

/// Search for a valid RSDP.
///
/// `uefi_rsdp` is the physical address passed by the UEFI config table (0 if unknown).
pub fn find_rsdp(uefi_rsdp: u64) -> Option<u64> {
    if uefi_rsdp != 0 {
        let rsdp = unsafe { &*(uefi_rsdp as *const Rsdp) };
        if rsdp.is_valid() {
            return Some(uefi_rsdp);
        }
    }

    // Scan EBDA: segment stored at 0x40E, multiplied by 16.
    let ebda_seg = unsafe { core::ptr::read_volatile(0x40E as *const u16) };
    let ebda_base = (ebda_seg as u64) * 16;
    if let Some(addr) = scan_region(ebda_base, ebda_base + 1024) {
        return Some(addr);
    }

    // Scan BIOS region 0xE0000-0xFFFFF.
    scan_region(0xE0000, 0x100000)
}

fn scan_region(start: u64, end: u64) -> Option<u64> {
    for addr in (start..end).step_by(16) {
        if addr + 8 > end {
            break;
        }
        let sig = unsafe { core::slice::from_raw_parts(addr as *const u8, 8) };
        if sig == b"RSD PTR " {
            if addr + mem::size_of::<Rsdp>() as u64 > end {
                crate::serial_println!(
                    "[ACPI/RSDP] RSDP candidate at {:#X} truncated, skipping",
                    addr
                );
                continue;
            }
            let rsdp = unsafe { &*(addr as *const Rsdp) };
            if rsdp.is_valid() {
                return Some(addr);
            }
        }
    }
    None
}

/// Common ACPI System Description Table header (36 bytes).
#[repr(C, packed)]
pub struct SdtHeader {
    pub signature: [u8; 4],
    pub length: u32,
    pub revision: u8,
    pub checksum: u8,
    pub oem_id: [u8; 6],
    pub oem_table_id: [u8; 8],
    pub oem_revision: u32,
    pub creator_id: u32,
    pub creator_revision: u32,
}

/// Heuristic ceiling on a single ACPI table's declared `Length`.
///
/// ACPI itself does not cap `Length` below the field's own 32-bit range, and
/// this driver has no access to the firmware memory map (it lives in
/// `BootInfo`, threaded only as far as `drivers::acpi::mod::init` — outside
/// this file's scope) to look up the real backing region for a table. In its
/// absence this is a heuristic, not a spec value: every real-world ACPI
/// table (DSDT/SSDT included) stays well under it by multiple orders of
/// magnitude.
/// ponytail: heuristic byte-count ceiling, not derived from the firmware
/// memory map. Tighten by threading BootInfo's ACPI-reclaim region length
/// through drivers::acpi::mod::init if a real machine ever needs more.
const MAX_TABLE_LEN: u64 = 16 * 1024 * 1024; // 16 MiB

impl SdtHeader {
    /// Validate checksum: sum of all bytes in the table must be 0 (mod 256).
    ///
    /// `self` must be a reference into the table's real location in
    /// identity-mapped physical memory — the length ceiling below is
    /// computed relative to `self`'s own address, which only means anything
    /// if it *is* the table's physical base (never call this on a copy).
    pub fn is_valid(&self) -> bool {
        let len = unsafe { core::ptr::addr_of!(self.length).read_unaligned() } as u64;
        let min_len = mem::size_of::<SdtHeader>() as u64;

        // Sanity check the declared length BEFORE it ever sizes the checksum
        // scan below — a floor (this is the smallest a real header can be)
        // and two independent ceilings:
        //   1. MAX_TABLE_LEN: heuristic real-world table size (see above).
        //   2. IDENTITY_MAP_SIZE: the kernel only identity-maps the first
        //      16 GiB of physical memory with PRESENT huge pages
        //      (memory/virt.rs:27,92-96) — a read past that point is
        //      guaranteed to fault, and `page_fault_handler` does not
        //      recover boot-time ACPI faults. `checked_add` also closes the
        //      address-wraparound case.
        if len < min_len || len > MAX_TABLE_LEN {
            return false;
        }
        let base = self as *const _ as u64;
        match base.checked_add(len) {
            Some(end) if end <= crate::memory::virt::IDENTITY_MAP_SIZE => {}
            _ => return false,
        }

        let bytes =
            unsafe { core::slice::from_raw_parts(self as *const _ as *const u8, len as usize) };
        bytes.iter().fold(0u8, |sum, &b| sum.wrapping_add(b)) == 0
    }
}

/// Walk the RSDT or XSDT pointed to by `rsdp_addr` and return the physical
/// address of the table matching `signature`.
pub fn walk_sdt(rsdp_addr: u64, signature: &[u8; 4]) -> Option<u64> {
    let rsdp = unsafe { &*(rsdp_addr as *const Rsdp) };
    let rsdp_revision = unsafe { core::ptr::addr_of!(rsdp.revision).read_unaligned() };
    let rsdp_xsdt_addr = unsafe { core::ptr::addr_of!(rsdp.xsdt_addr).read_unaligned() };
    let rsdp_rsdt_addr = unsafe { core::ptr::addr_of!(rsdp.rsdt_addr).read_unaligned() };
    let use_xsdt = rsdp_revision >= 2 && rsdp_xsdt_addr != 0;
    let root_phys = if use_xsdt {
        rsdp_xsdt_addr
    } else {
        rsdp_rsdt_addr as u64
    };

    let header = unsafe { &*(root_phys as *const SdtHeader) };
    if !header.is_valid() {
        return None;
    }

    let entry_size = if use_xsdt { 8usize } else { 4usize };
    let header_size = mem::size_of::<SdtHeader>();
    let header_len = unsafe { core::ptr::addr_of!(header.length).read_unaligned() } as usize;
    if header_len < header_size {
        crate::serial_println!(
            "[ACPI/RSDP] SDT at {:#X} claims length {} < header size {}",
            root_phys,
            header_len,
            header_size
        );
        return None;
    }
    let num_entries = (header_len - header_size) / entry_size;
    let entries_base = root_phys + header_size as u64;
    let table_end = root_phys + header_len as u64;

    for i in 0..num_entries {
        let entry_addr = entries_base + i as u64 * entry_size as u64;
        if entry_addr + entry_size as u64 > table_end {
            crate::serial_println!(
                "[ACPI/RSDP] SDT entry {} exceeds table bounds, stopping walk",
                i
            );
            break;
        }
        let entry_phys = if use_xsdt {
            unsafe { core::ptr::read_volatile(entry_addr as *const u64) }
        } else {
            unsafe { core::ptr::read_volatile(entry_addr as *const u32) as u64 }
        };

        if entry_phys == 0 {
            continue;
        }

        let tbl_header = unsafe { &*(entry_phys as *const SdtHeader) };
        let tbl_sig = unsafe { core::ptr::addr_of!(tbl_header.signature).read_unaligned() };
        if &tbl_sig == signature && tbl_header.is_valid() {
            return Some(entry_phys);
        }
    }

    None
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;

    fn checksum_of(bytes: &[u8]) -> u8 {
        bytes.iter().fold(0u8, |s, &b| s.wrapping_add(b))
    }

    fn build_valid_rsdp() -> Rsdp {
        let mut r = Rsdp {
            signature: *b"RSD PTR ",
            checksum: 0,
            oem_id: *b"SEALOS",
            revision: 2,
            rsdt_addr: 0,
            length: mem::size_of::<Rsdp>() as u32,
            xsdt_addr: 0,
            ext_checksum: 0,
            reserved: [0; 3],
        };
        unsafe {
            let v1 = core::slice::from_raw_parts(&r as *const _ as *const u8, 20);
            let sum = checksum_of(v1);
            core::ptr::addr_of_mut!(r.checksum).write_unaligned(0u8.wrapping_sub(sum));

            let full = core::slice::from_raw_parts(&r as *const _ as *const u8, mem::size_of::<Rsdp>());
            let sum2 = checksum_of(full);
            core::ptr::addr_of_mut!(r.ext_checksum).write_unaligned(0u8.wrapping_sub(sum2));
        }
        r
    }

    fn build_valid_sdt_header() -> SdtHeader {
        let mut h = SdtHeader {
            signature: *b"TEST",
            length: mem::size_of::<SdtHeader>() as u32,
            revision: 1,
            checksum: 0,
            oem_id: *b"SEALOS",
            oem_table_id: *b"TESTTABL",
            oem_revision: 1,
            creator_id: 0,
            creator_revision: 1,
        };
        unsafe {
            let bytes =
                core::slice::from_raw_parts(&h as *const _ as *const u8, mem::size_of::<SdtHeader>());
            let sum = checksum_of(bytes);
            core::ptr::addr_of_mut!(h.checksum).write_unaligned(0u8.wrapping_sub(sum));
        }
        h
    }

    fn test_rsdp_valid_v2_accepted() -> TestResult {
        let r = build_valid_rsdp();
        test_assert!(r.is_valid(), "well-formed ACPI 2.0 RSDP must validate");
        TestResult::Pass
    }

    /// RED (was): the `revision >= 2` branch only enforced `len <
    /// size_of::<Rsdp>()` — a floor with no ceiling — so `length =
    /// 0xFFFF_FFF0` passed straight through into a checksum scan sized off
    /// that value. GREEN: the RSDP is a fixed 36-byte structure with no
    /// trailing data, so `length` must equal `size_of::<Rsdp>()` exactly.
    fn test_rsdp_rejects_oversized_length() -> TestResult {
        let mut r = build_valid_rsdp();
        unsafe {
            core::ptr::addr_of_mut!(r.length).write_unaligned(0xFFFF_FFF0u32);
        }
        test_assert!(
            !r.is_valid(),
            "RSDP with length 0xFFFF_FFF0 must be rejected, not scanned"
        );
        TestResult::Pass
    }

    fn test_rsdp_rejects_undersized_length() -> TestResult {
        let mut r = build_valid_rsdp();
        unsafe {
            core::ptr::addr_of_mut!(r.length).write_unaligned(20u32);
        }
        test_assert!(
            !r.is_valid(),
            "RSDP length below size_of::<Rsdp>() must be rejected"
        );
        TestResult::Pass
    }

    fn test_sdt_header_valid_accepted() -> TestResult {
        let h = build_valid_sdt_header();
        test_assert!(h.is_valid(), "well-formed 36-byte SDT header must validate");
        TestResult::Pass
    }

    fn test_sdt_header_rejects_undersized_length() -> TestResult {
        let mut h = build_valid_sdt_header();
        unsafe {
            core::ptr::addr_of_mut!(h.length).write_unaligned(10u32);
        }
        test_assert!(
            !h.is_valid(),
            "SDT header length below size_of::<SdtHeader>() must be rejected"
        );
        TestResult::Pass
    }

    /// RED (was): `SdtHeader::is_valid` sized `slice::from_raw_parts(self,
    /// len)` directly off the attacker-controlled `length` field with no
    /// upper bound. A header declaring `length = 0xFFFF_FFF0` triggered a
    /// ~4 GiB linear byte scan from the table's address — reached from the
    /// root RSDT/XSDT header (old rsdp.rs:135, *before* the separate
    /// `header_len < header_size` sanity check that would have caught it),
    /// every signature-matched sub-table during the entry walk (old
    /// rsdp.rs:176, no length check at all), and the MADT's own header
    /// (madt.rs:83). GREEN: `is_valid` now rejects any length over
    /// `MAX_TABLE_LEN`, checked before the checksum scan, so all three call
    /// sites reject it via this one shared function.
    fn test_sdt_header_rejects_oversized_length() -> TestResult {
        let mut h = build_valid_sdt_header();
        unsafe {
            core::ptr::addr_of_mut!(h.length).write_unaligned(0xFFFF_FFF0u32);
        }
        test_assert!(
            !h.is_valid(),
            "SDT header with length 0xFFFF_FFF0 must be rejected, not scanned"
        );
        TestResult::Pass
    }

    fn test_sdt_header_rejects_bad_checksum() -> TestResult {
        let mut h = build_valid_sdt_header();
        unsafe {
            let c = core::ptr::addr_of!(h.checksum).read_unaligned();
            core::ptr::addr_of_mut!(h.checksum).write_unaligned(c.wrapping_add(1));
        }
        test_assert!(!h.is_valid(), "SDT header with corrupted checksum must be rejected");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("acpi::rsdp::rsdp_valid_v2_accepted", test_rsdp_valid_v2_accepted);
        crate::testing::register_test(
            "acpi::rsdp::rsdp_rejects_oversized_length",
            test_rsdp_rejects_oversized_length,
        );
        crate::testing::register_test(
            "acpi::rsdp::rsdp_rejects_undersized_length",
            test_rsdp_rejects_undersized_length,
        );
        crate::testing::register_test(
            "acpi::rsdp::sdt_header_valid_accepted",
            test_sdt_header_valid_accepted,
        );
        crate::testing::register_test(
            "acpi::rsdp::sdt_header_rejects_undersized_length",
            test_sdt_header_rejects_undersized_length,
        );
        crate::testing::register_test(
            "acpi::rsdp::sdt_header_rejects_oversized_length",
            test_sdt_header_rejects_oversized_length,
        );
        crate::testing::register_test(
            "acpi::rsdp::sdt_header_rejects_bad_checksum",
            test_sdt_header_rejects_bad_checksum,
        );
    }
}
