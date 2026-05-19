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
        let sig = unsafe { core::ptr::addr_of!((*self).signature).read_unaligned() };
        if &sig != b"RSD PTR " {
            return false;
        }

        // Validate v1 checksum over first 20 bytes.
        let bytes = unsafe { core::slice::from_raw_parts(self as *const _ as *const u8, 20) };
        if bytes.iter().fold(0u8, |sum, &b| sum.wrapping_add(b)) != 0 {
            return false;
        }

        let revision = unsafe { core::ptr::addr_of!((*self).revision).read_unaligned() };
        if revision >= 2 {
            let len = unsafe { core::ptr::addr_of!((*self).length).read_unaligned() } as usize;
            if len < mem::size_of::<Rsdp>() {
                return false;
            }
            let ext_bytes = unsafe {
                core::slice::from_raw_parts(self as *const _ as *const u8, len)
            };
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

impl SdtHeader {
    /// Validate checksum: sum of all bytes in the table must be 0 (mod 256).
    pub fn is_valid(&self) -> bool {
        let len = unsafe { core::ptr::addr_of!((*self).length).read_unaligned() } as usize;
        let bytes = unsafe {
            core::slice::from_raw_parts(self as *const _ as *const u8, len)
        };
        bytes.iter().fold(0u8, |sum, &b| sum.wrapping_add(b)) == 0
    }
}

/// Walk the RSDT or XSDT pointed to by `rsdp_addr` and return the physical
/// address of the table matching `signature`.
pub fn walk_sdt(rsdp_addr: u64, signature: &[u8; 4]) -> Option<u64> {
    let rsdp = unsafe { &*(rsdp_addr as *const Rsdp) };
    let rsdp_revision = unsafe { core::ptr::addr_of!((*rsdp).revision).read_unaligned() };
    let rsdp_xsdt_addr = unsafe { core::ptr::addr_of!((*rsdp).xsdt_addr).read_unaligned() };
    let rsdp_rsdt_addr = unsafe { core::ptr::addr_of!((*rsdp).rsdt_addr).read_unaligned() };
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
    let header_len = unsafe { core::ptr::addr_of!((*header).length).read_unaligned() } as usize;
    if header_len < header_size {
        crate::serial_println!(
            "[ACPI/RSDP] SDT at {:#X} claims length {} < header size {}",
            root_phys, header_len, header_size
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
        let tbl_sig = unsafe { core::ptr::addr_of!((*tbl_header).signature).read_unaligned() };
        if &tbl_sig == signature {
            if tbl_header.is_valid() {
                return Some(entry_phys);
            }
        }
    }

    None
}
