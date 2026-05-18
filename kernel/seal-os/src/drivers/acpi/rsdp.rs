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

        if self.revision >= 2 {
            let len = self.length as usize;
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
        let sig = unsafe { core::slice::from_raw_parts(addr as *const u8, 8) };
        if sig == b"RSD PTR " {
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
        let len = self.length as usize;
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
    let use_xsdt = rsdp.revision >= 2 && rsdp.xsdt_addr != 0;
    let root_phys = if use_xsdt {
        rsdp.xsdt_addr
    } else {
        rsdp.rsdt_addr as u64
    };

    let header = unsafe { &*(root_phys as *const SdtHeader) };
    if !header.is_valid() {
        return None;
    }

    let entry_size = if use_xsdt { 8usize } else { 4usize };
    let header_size = mem::size_of::<SdtHeader>();
    let num_entries = (header.length as usize - header_size) / entry_size;
    let entries_base = root_phys + header_size as u64;

    for i in 0..num_entries {
        let entry_phys = if use_xsdt {
            unsafe { core::ptr::read_volatile((entries_base + i as u64 * 8) as *const u64) }
        } else {
            unsafe { core::ptr::read_volatile((entries_base + i as u64 * 4) as *const u32) as u64 }
        };

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
