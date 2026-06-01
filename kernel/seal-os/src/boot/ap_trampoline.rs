// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! AP real-to-long mode trampoline.
//!
//! The BSP copies the raw bytes of `ap_trampoline()` to a low-memory page
//! (0x8000).  The AP starts executing there in 16-bit real mode after the
//! INIT-SIPI-SIPI sequence.

use core::arch::naked_asm;

use crate::cpu::PerCpu;
use crate::process::context_switch::KERNEL_STACK_SIZE;

// ---------------------------------------------------------------------------
// Trampoline page layout
// ---------------------------------------------------------------------------

pub const TRAMPOLINE_PAGE: u64 = 0x8000;

pub const OFF_GDTR: u64 = 0x100;
pub const OFF_PROT32_PTR: u64 = 0x110;
pub const OFF_LONG64_PTR: u64 = 0x114;
pub const OFF_GDT_START: u64 = 0x120;
pub const OFF_BSP_PML4: u64 = 0x180;
pub const OFF_AP_PER_CPU_PTR: u64 = 0x188;
pub const OFF_AP_MAIN_ADDR: u64 = 0x190;
pub const OFF_32BIT_CODE: u64 = 0x40;
pub const OFF_64BIT_CODE: u64 = 0xC0;

// ---------------------------------------------------------------------------
// Naked trampoline function
// ---------------------------------------------------------------------------

/// AP real-to-long mode trampoline code.
///
/// # Safety
/// This function contains raw assembly and must never be called directly from
/// Rust. The BSP must copy its machine-code bytes to `TRAMPOLINE_PAGE` (0x8000)
/// and ensure the GDTR, GDT, PML4, per-CPU pointer, and `ap_main` address are
/// correctly populated at the fixed offsets before issuing INIT-SIPI-SIPI.
/// Incorrect setup causes the AP to triple-fault or corrupt memory.
#[unsafe(naked)]
#[no_mangle]
#[link_section = ".ap_trampoline"]
pub unsafe extern "C" fn ap_trampoline() -> ! {
    naked_asm!(
        // ---------- 16-bit real mode (offset 0x00) ----------
        ".code16",
        "cli",
        "cld",
        "xor ax, ax",
        "mov ds, ax",
        "mov es, ax",
        "mov ss, ax",
        "mov sp, 0x7C00",

        // Enable A20 via fast A20 gate
        "in al, 0x92",
        "or al, 2",
        "out 0x92, al",

        // Load temporary GDT
        "lgdt [{gdtr_addr}]",

        // Enable protected mode
        "mov eax, cr0",
        "or eax, 1",
        "mov cr0, eax",

        // Far jump to 32-bit code at fixed offset 0x40
        "jmp fword ptr [{prot32_ptr}]",

        // Pad to ensure 32-bit code starts exactly at offset 0x40
        ".org 0x40",

        // ---------- 32-bit protected mode (offset 0x40) ----------
        ".code32",
        "mov ax, 0x10",
        "mov ds, ax",
        "mov es, ax",
        "mov fs, ax",
        "mov gs, ax",
        "mov ss, ax",
        "mov esp, 0x7C00",

        // Enable PAE
        "mov eax, cr4",
        "or eax, 0x20",
        "mov cr4, eax",

        // Load PML4
        "mov eax, [{pml4_addr}]",
        "mov cr3, eax",

        // Enable long mode (EFER.LME)
        "mov ecx, 0xC0000080",
        "rdmsr",
        "or eax, 0x100",
        "wrmsr",

        // Enable paging
        "mov eax, cr0",
        "or eax, 0x80000000",
        "mov cr0, eax",

        // Far jump to 64-bit code at fixed offset 0x80
        "jmp fword ptr [{long64_ptr}]",

        // Pad to ensure 64-bit code starts exactly at offset 0xC0
        ".org 0xC0",

        // ---------- 64-bit long mode (offset 0x80) ----------
        ".code64",
        "mov ax, 0x20",
        "mov ds, ax",
        "mov es, ax",
        "mov fs, ax",
        "mov gs, ax",
        "mov ss, ax",

        // Load per-cpu pointer and set stack
        "mov rax, [{per_cpu_ptr_addr}]",
        "lea rsp, [rax + {kernel_stack_off} + {kernel_stack_size}]",

        // Set GS base (IA32_GS_BASE = 0xC0000101)
        "mov rcx, 0xC0000101",
        "mov rdx, rax",
        "shr rdx, 32",
        "and eax, 0xFFFFFFFF",
        "wrmsr",

        // Call ap_main
        "mov rax, [{ap_main_addr}]",
        "call rax",

        // Halt forever if ap_main returns
        "1:",
        "hlt",
        "jmp 1b",

        gdtr_addr = const (TRAMPOLINE_PAGE + OFF_GDTR),
        prot32_ptr = const (TRAMPOLINE_PAGE + OFF_PROT32_PTR),
        pml4_addr = const (TRAMPOLINE_PAGE + OFF_BSP_PML4),
        long64_ptr = const (TRAMPOLINE_PAGE + OFF_LONG64_PTR),
        per_cpu_ptr_addr = const (TRAMPOLINE_PAGE + OFF_AP_PER_CPU_PTR),
        ap_main_addr = const (TRAMPOLINE_PAGE + OFF_AP_MAIN_ADDR),
        kernel_stack_off = const core::mem::offset_of!(PerCpu, kernel_stack),
        kernel_stack_size = const KERNEL_STACK_SIZE,
    );
}
