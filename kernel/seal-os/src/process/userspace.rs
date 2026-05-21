// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Ring-3 entry and syscall infrastructure.

use core::arch::{asm, global_asm};
use core::sync::atomic::Ordering;
use x86_64::registers::model_specific::Msr;
use x86_64::PrivilegeLevel;

use crate::memory::gdt;

/// Saved CPU state for a userspace task.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct UserContext {
    pub page_table: u64, // physical address of PML4
    pub rsp: u64,
    pub rip: u64,
    pub rflags: u64,
    pub rbx: u64,
    pub rbp: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

impl UserContext {
    pub const fn zero() -> Self {
        Self {
            page_table: 0,
            rsp: 0,
            rip: 0,
            rflags: 0x202,
            rbx: 0,
            rbp: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
        }
    }
}

/// Enter ring 3 using `iretq`.
///
/// # Safety
/// Must be called with interrupts disabled.  The `context` must describe a
/// fully initialised userspace task (valid page table, stack, entry point).
pub unsafe fn enter_userspace(context: &UserContext) -> ! {
    // Load user page tables.
    asm!(
        "mov cr3, {cr3}",
        cr3 = in(reg) context.page_table,
        options(nomem, nostack, preserves_flags)
    );

    // Build the interrupt-return stack frame.
    let user_ss = ((gdt::USER_DATA_SELECTOR.load(Ordering::Relaxed) >> 3) as u64) * 8 | (PrivilegeLevel::Ring3 as u16 as u64);
    let user_cs = ((gdt::USER_CODE_SELECTOR.load(Ordering::Relaxed) >> 3) as u64) * 8 | (PrivilegeLevel::Ring3 as u16 as u64);

    asm!(
        "push {user_ss}",
        "push {user_rsp}",
        "push {user_rflags}",
        "push {user_cs}",
        "push {user_rip}",
        "iretq",
        user_ss = in(reg) user_ss,
        user_rsp = in(reg) context.rsp,
        user_rflags = in(reg) context.rflags | 0x202, // ensure IF is set
        user_cs = in(reg) user_cs,
        user_rip = in(reg) context.rip,
        options(noreturn)
    );
}

/// Trampoline called from the kernel context-switch path to drop into
/// userspace for the first time.
#[no_mangle]
pub extern "C" fn enter_userspace_trampoline(entry: u64, stack: u64, pt: u64) -> ! {
    let ctx = UserContext {
        page_table: pt,
        rsp: stack,
        rip: entry,
        rflags: 0x202,
        ..UserContext::zero()
    };
    unsafe { enter_userspace(&ctx) }
}

// ---------------------------------------------------------------------------
// Syscall handling via `syscall` / `sysret`
// ---------------------------------------------------------------------------

/// MSR indices for fast system calls.
const MSR_STAR: u32 = 0xC000_0081;
const MSR_LSTAR: u32 = 0xC000_0082;
const MSR_CSTAR: u32 = 0xC000_0083;
const MSR_SFMASK: u32 = 0xC000_0084;

/// Frame pushed by `syscall_entry` before calling into Rust.
#[repr(C)]
pub struct SyscallFrame {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64, // saved RIP
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64, // saved RFLAGS
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

global_asm!(
    ".global syscall_entry",
    ".align 16",
    "syscall_entry:",
    "push r15",
    "push r14",
    "push r13",
    "push r12",
    "push r11",
    "push r10",
    "push r9",
    "push r8",
    "push rbp",
    "push rdi",
    "push rsi",
    "push rdx",
    "push rcx",
    "push rbx",
    "push rax",
    "mov rdi, rsp",
    "call do_syscall",
    "pop rax",
    "pop rbx",
    "pop rcx",
    "pop rdx",
    "pop rsi",
    "pop rdi",
    "pop rbp",
    "pop r8",
    "pop r9",
    "pop r10",
    "pop r11",
    "pop r12",
    "pop r13",
    "pop r14",
    "pop r15",
    "sysretq",
);

extern "C" {
    fn syscall_entry();
}

/// Rust side of the syscall handler.
///
/// Reads the syscall number and arguments from the saved frame, dispatches
/// via the syscall table, and writes the return value back into RAX.
#[no_mangle]
pub unsafe extern "C" fn do_syscall(frame: *mut SyscallFrame) {
    unsafe {
        let f = &mut *frame;
        let num = f.rax;
        let arg0 = f.rdi;
        let arg1 = f.rsi;
        let arg2 = f.rdx;

        let result = crate::syscall::table::dispatch(num, arg0, arg1, arg2);
        f.rax = result.code as u64;
    }
}

/// Program the SYSCALL/SYSRET MSRs so that `syscall` from ring 3 lands in
/// `syscall_entry`.
pub fn init_syscall_msrs() {
    unsafe {
        let star = (((gdt::USER_CODE32_SELECTOR.load(Ordering::Relaxed) >> 3) as u64) << 48)
                 | (((gdt::KERNEL_CODE_SELECTOR.load(Ordering::Relaxed) >> 3) as u64) << 32);

        Msr::new(MSR_STAR).write(star);
        Msr::new(MSR_LSTAR).write(syscall_entry as *const () as u64);
        Msr::new(MSR_CSTAR).write(syscall_entry as *const () as u64);
        // Clear IF on syscall entry; sysret restores it from saved R11.
        Msr::new(MSR_SFMASK).write(0x200);
    }
}

/// A tiny embedded ELF binary for the emergency shell.
/// Real userspace init that calls SYS_WRITE then SYS_EXIT.
pub const EMERGENCY_SHELL_ELF: &[u8] = &[
    0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x3e, 0x00, 0x01, 0x00, 0x00, 0x00, 0x78, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x38, 0x00, 0x01, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xd0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, 0x48,
    0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc6, 0xa6, 0x00, 0x40, 0x00, 0x48, 0xc7, 0xc2,
    0x2a, 0x00, 0x00, 0x00, 0x0f, 0x05, 0x48, 0xc7, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x48, 0xc7, 0xc7,
    0x00, 0x00, 0x00, 0x00, 0x0f, 0x05, 0x5b, 0x75, 0x73, 0x65, 0x72, 0x73, 0x70, 0x61, 0x63, 0x65,
    0x5d, 0x20, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x66, 0x72, 0x6f, 0x6d, 0x20, 0x53, 0x65, 0x61,
    0x6c, 0x20, 0x4f, 0x53, 0x20, 0x75, 0x73, 0x65, 0x72, 0x73, 0x70, 0x61, 0x63, 0x65, 0x21, 0x0a,
];
