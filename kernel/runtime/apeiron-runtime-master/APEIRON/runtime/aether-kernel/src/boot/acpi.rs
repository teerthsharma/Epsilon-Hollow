// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

use core::slice;
use core::mem;
use alloc::vec::Vec;

#[repr(C, packed)]
struct Rsdp {
    signature: [u8; 8],
    checksum: u8,
    oemid: [u8; 6],
    revision: u8,
    rsdt_addr: u32,
    length: u32,
    xsdt_addr: u64,
    ext_checksum: u8,
    reserved: [u8; 3],
}

#[repr(C, packed)]
struct SdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oemid: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: u32,
    creator_revision: u32,
}

// Helper to validate checksum
unsafe fn validate_checksum(ptr: *const u8, len: usize) -> bool {
    let mut sum: u8 = 0;
    for i in 0..len {
        sum = sum.wrapping_add(*ptr.add(i));
    }
    sum == 0
}

/// Parse the ACPI tables to count the number of NUMA nodes from the SRAT.
/// Returns 1 if SRAT is not found or no NUMA domains are defined.
pub unsafe fn parse_numa_nodes(rsdp_addr: u64) -> usize {
    if rsdp_addr == 0 { return 1; }

    let rsdp = &*(rsdp_addr as *const Rsdp);
    if &rsdp.signature != b"RSD PTR " { return 1; }

    // Prefer XSDT if revision >= 2, otherwise RSDT
    // XSDT address is valid if revision >= 2.
    // However, some systems might provide XSDT even if revision is low or vice versa, but standard says rev >= 2 for XSDT.
    let (sdt_addr, entry_size) = if rsdp.revision >= 2 && rsdp.xsdt_addr != 0 {
        (rsdp.xsdt_addr, 8) // XSDT uses 64-bit pointers
    } else {
        (rsdp.rsdt_addr as u64, 4) // RSDT uses 32-bit pointers
    };

    if sdt_addr == 0 { return 1; }

    let header = &*(sdt_addr as *const SdtHeader);
    // Optional: validate checksum of XSDT/RSDT
    // if !validate_checksum(sdt_addr as *const u8, header.length as usize) { return 1; }

    let entries_len = (header.length as usize - mem::size_of::<SdtHeader>()) / entry_size;
    let entries_ptr = (sdt_addr as *const u8).add(mem::size_of::<SdtHeader>());

    for i in 0..entries_len {
        let entry_addr = if entry_size == 8 {
             // 64-bit address need to be read carefully if unaligned, but x86 handles unaligned loads usually.
             // Given packed structs, we should be careful.
             // entries_ptr is u8, so .add() is byte offset.
             // We cast to *const u64 / *const u32.
             // If XSDT/RSDT are packed, these might be unaligned.
             // Ideally we use read_unaligned but core::ptr::read_unaligned is available.
             (entries_ptr.add(i * 8) as *const u64).read_unaligned()
        } else {
             (entries_ptr.add(i * 4) as *const u32).read_unaligned() as u64
        };

        if entry_addr == 0 { continue; }

        let entry_header = &*(entry_addr as *const SdtHeader);
        if &entry_header.signature == b"SRAT" {
             return parse_srat(entry_addr);
        }
    }

    1
}

unsafe fn parse_srat(addr: u64) -> usize {
     let header = &*(addr as *const SdtHeader);
     let data_len = header.length as usize;
     // SRAT Header: SDT Header (36 bytes) + Reserved (4 bytes) + Reserved (8 bytes) = 48 bytes
     let mut offset = mem::size_of::<SdtHeader>() + 4 + 8;

     let mut proximity_domains = Vec::new();

     while offset < data_len {
         let type_ptr = (addr as *const u8).add(offset);
         // Ensure we don't read past end
         if offset + 2 > data_len { break; }

         let type_val = *type_ptr;
         let length = *type_ptr.add(1) as usize;

         if length == 0 || offset + length > data_len { break; }

         match type_val {
             0 => { // Processor Local APIC/SAPIC Affinity
                 // Offset 2: Proximity Domain [7:0]
                 // Offset 3: APIC ID
                 // Offset 4: Flags (4 bytes)
                 // Offset 8: Local SAPIC EID
                 // Offset 9: Proximity Domain [31:8]
                 if length >= 16 {
                     let flags = (type_ptr.add(4) as *const u32).read_unaligned();
                     if (flags & 1) != 0 { // Enabled
                         let domain_low = *type_ptr.add(2);
                         let domain_high_ptr = type_ptr.add(9);
                         let domain_high = slice::from_raw_parts(domain_high_ptr, 3);

                         let domain = (domain_low as u32) |
                                      ((domain_high[0] as u32) << 8) |
                                      ((domain_high[1] as u32) << 16) |
                                      ((domain_high[2] as u32) << 24);

                         if !proximity_domains.contains(&domain) {
                             proximity_domains.push(domain);
                         }
                     }
                 }
             }
             1 => { // Memory Affinity
                 // Offset 2: Proximity Domain (4 bytes)
                 // ...
                 // Offset 28: Flags (4 bytes)
                 if length >= 40 { // Length is 40 for Type 1
                     let flags = (type_ptr.add(28) as *const u32).read_unaligned();
                     if (flags & 1) != 0 { // Enabled
                         let domain = (type_ptr.add(2) as *const u32).read_unaligned();
                         if !proximity_domains.contains(&domain) {
                             proximity_domains.push(domain);
                         }
                     }
                 }
             }
             2 => { // Processor Local x2APIC Affinity
                 // Offset 4: Proximity Domain (4 bytes)
                 // Offset 12: Flags (4 bytes)
                 if length >= 24 {
                     let flags = (type_ptr.add(12) as *const u32).read_unaligned();
                     if (flags & 1) != 0 { // Enabled
                         let domain = (type_ptr.add(4) as *const u32).read_unaligned();
                         if !proximity_domains.contains(&domain) {
                             proximity_domains.push(domain);
                         }
                     }
                 }
             }
             _ => {}
         }

         offset += length;
     }

     if proximity_domains.is_empty() { 1 } else { proximity_domains.len() }
}
