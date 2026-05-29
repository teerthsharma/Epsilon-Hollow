// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! PGTABLE-ASM integration — x86_64 page-table primitives in assembly.
//!
//! Implements recursive mapping, physical page-table walking, CR3+PCID switch,
//! INVPCID TLB invalidation, and KPTI CR3 swap.  All routines are real
//! hand-written x86_64 assembly embedded via `global_asm!`.

use core::arch::global_asm;
use x86_64::{PhysAddr, VirtAddr};

// ---------------------------------------------------------------------------
// Assembly definitions
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .global setup_recursive_mapping
    .p2align 4
setup_recursive_mapping:
    // rdi = pml4_phys
    mov rax, rdi
    or  rax, 0x003          // present + writable + supervisor
    mov [rdi + 511*8], rax  // PML4[511] -> PML4
    ret

    .global walk_page_table
    .p2align 4
walk_page_table:
    // rdi = pml4_phys, rsi = vaddr
    // returns physical address in rax, 0 if unmapped

    // --- PML4 ---
    mov rax, rsi
    shr rax, 39
    and rax, 0x1FF
    mov r8, [rdi + rax*8]       // PML4E
    test r8b, 1
    jz .Lunmapped

    // --- PDPT ---
    mov rax, rsi
    shr rax, 30
    and rax, 0x1FF
    mov r9, r8
    and r9, ~0xFFF              // PDPT physical address
    mov r10, [r9 + rax*8]       // PDPTE
    test r10b, 1
    jz .Lunmapped
    test r10, 0x80              // 1 GiB huge page?
    jnz .Lhuge_1g

    // --- PD ---
    mov rax, rsi
    shr rax, 21
    and rax, 0x1FF
    mov r9, r10
    and r9, ~0xFFF              // PD physical address
    mov r10, [r9 + rax*8]       // PDE
    test r10b, 1
    jz .Lunmapped
    test r10, 0x80              // 2 MiB huge page?
    jnz .Lhuge_2m

    // --- PT ---
    mov rax, rsi
    shr rax, 12
    and rax, 0x1FF
    mov r9, r10
    and r9, ~0xFFF              // PT physical address
    mov r10, [r9 + rax*8]       // PTE
    test r10b, 1
    jz .Lunmapped

    // 4 KiB page
    mov rax, r10
    and rax, ~0xFFF
    mov r9, rsi
    and r9, 0xFFF
    or  rax, r9
    ret

.Lhuge_1g:
    mov rax, r10
    and rax, ~0x3FFFFFFF
    mov r9, rsi
    and r9, 0x3FFFFFFF
    or  rax, r9
    ret

.Lhuge_2m:
    mov rax, r10
    and rax, ~0x1FFFFF
    mov r9, rsi
    and r9, 0x1FFFFF
    or  rax, r9
    ret

.Lunmapped:
    xor eax, eax
    ret

    .global switch_cr3_pcid
    .p2align 4
switch_cr3_pcid:
    // rdi = pml4_phys, rsi = pcid (12-bit)
    mov rax, rsi
    and rax, 0xFFF
    or  rax, rdi
    mov cr3, rax
    ret

    .global invpcid_all
    .p2align 4
invpcid_all:
    // rdi = pcid
    sub rsp, 16
    mov [rsp], rdi
    mov qword ptr [rsp+8], 0
    mov rax, 1                  // type 1 = single PCID, all non-global
    // invpcid rax, [rsp]  — byte-encoded for maximum assembler compatibility
    .byte 0x66, 0x0F, 0x38, 0xF2, 0x04, 0x24
    add rsp, 16
    ret

    .global kpti_swap_cr3
    .p2align 4
kpti_swap_cr3:
    // rdi = new_cr3
    mov rax, rdi
    mov cr3, rax
    ret
"#
);

extern "C" {
    fn setup_recursive_mapping(pml4_phys: u64);
    fn walk_page_table(pml4_phys: u64, vaddr: u64) -> u64;
    fn switch_cr3_pcid(pml4_phys: u64, pcid: u16);
    fn invpcid_all(pcid: u16);
    fn kpti_swap_cr3(new_cr3: u64);
}

// ---------------------------------------------------------------------------
// Safe / unsafe Rust wrappers
// ---------------------------------------------------------------------------

/// Install a recursive self-map at PML4\[511\].
///
/// # Safety
/// `pml4_phys` must be the physical address of a valid, active PML4.
/// Must be called once per address space before `translate()` is used.
pub unsafe fn install_recursive_mapping(pml4_phys: PhysAddr) {
    unsafe { setup_recursive_mapping(pml4_phys.as_u64()) }
}

/// Walk the page tables starting from `pml4_phys` to resolve `vaddr`.
///
/// Returns `Some(phys)` if the page is mapped, `None` otherwise.
/// Handles 4 KiB, 2 MiB, and 1 GiB pages.
pub fn translate_phys(pml4_phys: PhysAddr, vaddr: VirtAddr) -> Option<PhysAddr> {
    let result = unsafe { walk_page_table(pml4_phys.as_u64(), vaddr.as_u64()) };
    if result == 0 {
        None
    } else {
        Some(PhysAddr::new(result))
    }
}

/// Write CR3 with an embedded 12-bit PCID.
///
/// # Safety
/// The PCID must be valid (≤ 4095) and the PML4 physical address must be
/// correctly aligned.
pub unsafe fn write_cr3_pcid(pml4_phys: PhysAddr, pcid: u16) {
    unsafe { switch_cr3_pcid(pml4_phys.as_u64(), pcid & 0xFFF) }
}

/// Flush all non-global TLB entries for the given PCID.
///
/// # Safety
/// Must be called with interrupts disabled to avoid races on the stack
/// scratch space used by the assembly stub.
pub unsafe fn flush_pcid(pcid: u16) {
    unsafe { invpcid_all(pcid & 0xFFF) }
}

/// Raw CR3 swap for KPTI entry/exit.
///
/// # Safety
/// `new_cr3` must be the physical address of a valid page-table root.
/// Caller must ensure the switch is atomic with respect to interrupts.
pub unsafe fn swap_cr3(new_cr3: u64) {
    unsafe { kpti_swap_cr3(new_cr3) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;

    fn test_walk_page_table_zero_is_unmapped() -> TestResult {
        // VA 0 with a dummy PML4 of all zeros should return 0
        let dummy_pml4 = [0u64; 512];
        let phys = translate_phys(PhysAddr::new(dummy_pml4.as_ptr() as u64), VirtAddr::new(0));
        if phys.is_some() {
            return TestResult::Fail("Expected None for zeroed PML4");
        }
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "pgtable_asm::walk_zero_unmapped",
            test_walk_page_table_zero_is_unmapped,
        );
    }
}
