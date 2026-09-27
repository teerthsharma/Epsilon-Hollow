// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Ring-3 entry and syscall infrastructure.

use core::arch::{asm, global_asm};
use core::mem::offset_of;
use core::sync::atomic::Ordering;
use x86_64::registers::model_specific::{Efer, EferFlags, GsBase, KernelGsBase, Msr};
use x86_64::PrivilegeLevel;

use crate::cpu::PerCpu;
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
/// CR3 must already hold `context.page_table`: `schedule()` loads it before
/// switching to the task, and a syscall runs on it throughout.
pub unsafe fn enter_userspace(context: &mut UserContext) -> ! {
    // Deliver any pending signals before returning to ring 3.
    crate::process::signal::check_and_handle_signals(context);

    // Build the interrupt-return stack frame.
    // Parenthesised explicitly: `*` binds tighter than `|`, so the selector
    // index is scaled to a byte offset first and the RPL bits are OR-ed in
    // afterwards. Relied on to build ring-3 selectors, so the grouping is
    // spelled out rather than left to precedence.
    let user_ss = (((gdt::USER_DATA_SELECTOR.load(Ordering::Relaxed) >> 3) as u64) * 8)
        | (PrivilegeLevel::Ring3 as u16 as u64);
    let user_cs = (((gdt::USER_CODE_SELECTOR.load(Ordering::Relaxed) >> 3) as u64) * 8)
        | (PrivilegeLevel::Ring3 as u16 as u64);

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
/// userspace for the first time. `switch_context` enters it with its
/// arguments in rdi/rsi/rdx, hence `sysv64`: `extern "C"` is the Microsoft x64
/// convention on this target and would read them from rcx/rdx/r8.
#[no_mangle]
pub extern "sysv64" fn enter_userspace_trampoline(entry: u64, stack: u64, pt: u64) -> ! {
    x86_64::instructions::interrupts::disable();
    let mut ctx = UserContext {
        page_table: pt,
        rsp: stack,
        rip: entry,
        rflags: 0x202,
        ..UserContext::zero()
    };
    unsafe { enter_userspace(&mut ctx) }
}

// ---------------------------------------------------------------------------
// Syscall handling via `syscall` / `sysret`
// ---------------------------------------------------------------------------

/// MSR indices for fast system calls.
const MSR_STAR: u32 = 0xC000_0081;
const MSR_LSTAR: u32 = 0xC000_0082;
const MSR_CSTAR: u32 = 0xC000_0083;
const MSR_SFMASK: u32 = 0xC000_0084;

/// RFLAGS bits IA32_FMASK must clear on every `syscall`, so ring 0 never
/// inherits them from ring 3: TF (0x100, single-step into the kernel), IF
/// (0x200, no interrupt before the stack switch), DF (0x400, Rust and memcpy
/// assume DF=0), NT (0x4000, a later `iretq` would take a task return) and AC
/// (0x40000, a set AC disables SMAP for the whole handler). Linux masks the
/// same set plus IOPL.
const FMASK_MUST_CLEAR: u64 = 0x100 | 0x200 | 0x400 | 0x4000 | 0x40000;

/// Write IA32_FMASK, and fail the build unless the value written clears every
/// bit in `FMASK_MUST_CLEAR`. The invocation keeps the literal inside the
/// `Msr::new(MSR_SFMASK).write(..)` call so the value checked is the value
/// written.
macro_rules! write_checked_fmask {
    (Msr::new(MSR_SFMASK).write($mask:literal)) => {{
        const _: () = assert!(
            $mask & FMASK_MUST_CLEAR == FMASK_MUST_CLEAR,
            "IA32_FMASK must clear TF, IF, DF, NT and AC"
        );
        Msr::new(MSR_SFMASK).write($mask)
    }};
}

/// Byte offsets into `PerCpu` that `syscall_entry` reads through GS.
const PERCPU_SYSCALL_USER_RSP: usize = offset_of!(PerCpu, syscall_user_rsp);
/// This CPU's TSS RSP0: the current task's kernel stack top, which
/// `schedule()` writes whenever it switches to a userspace task.
const PERCPU_KERNEL_RSP: usize = offset_of!(PerCpu, tss.privilege_stack_table);

/// Frame pushed by `syscall_entry` before calling into Rust.
///
/// All six argument registers of the x86_64 syscall convention (rdi, rsi,
/// rdx, r10, r8, r9) are in it; `dispatch` reads the first three.
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
    pub rsp: u64, // user RSP
}

// `syscall` changes neither RSP nor, without `swapgs`, the GS base, so the
// entry touches no memory until it has swapped GS and loaded this task's
// kernel stack: the only store before that is the user RSP into a per-CPU
// scratch slot. Interrupts stay off until `do_syscall` (IF is in the FMASK).
//
// GS: `init_syscall_msrs` sets IA32_KERNEL_GS_BASE to this CPU's `PerCpu`, the
// value IA32_GS_BASE already holds, and ring 3 has no way to change its own GS
// base (CR4.FSGSBASE is clear and there is no ARCH_SET_GS). Both `swapgs`
// therefore leave GS on `PerCpu`, which is also why the IDT handlers, which
// never `swapgs`, find `PerCpu` when they interrupt ring 3.
// ponytail: user GS is pinned to the kernel's. Upgrade path, before any
// user-settable GS base (ARCH_SET_GS or FSGSBASE): user GS 0 in ring 3, and a
// `swapgs` on every IDT entry and exit whose saved CS has RPL 3.
//
// Return: `sysretq` loads RIP from RCX, and on Intel CPUs a non-canonical RCX
// raises #GP in ring 0 after RSP is already the user's, so a return address
// outside the user half never reaches `sysretq`; the task is killed instead.
global_asm!(
    ".global syscall_entry",
    ".p2align 4",
    "syscall_entry:",
    "swapgs",
    "mov qword ptr gs:[{user_rsp}], rsp",
    "mov rsp, qword ptr gs:[{kernel_rsp}]",
    "push qword ptr gs:[{user_rsp}]",
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
    "call {do_syscall}",
    // A handler may have switched tasks and come back with IF=1; nothing may
    // interrupt once RSP is the user's again.
    "cli",
    "mov rcx, qword ptr [rsp + 16]",
    "shr rcx, 47",
    "jnz 2f",
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
    "pop rsp",
    "swapgs",
    "lfence",
    "sysretq",
    "2:",
    "call {bad_return}",
    "ud2",
    user_rsp = const PERCPU_SYSCALL_USER_RSP,
    kernel_rsp = const PERCPU_KERNEL_RSP,
    do_syscall = sym do_syscall,
    bad_return = sym syscall_return_outside_user_half,
);

extern "C" {
    fn syscall_entry();
}

/// Rust side of the syscall handler.
///
/// Reads the syscall number and arguments from the saved frame, dispatches
/// via the syscall table, and writes the return value back into RAX.
///
/// Runs on the calling task's own page table, which maps the kernel as well
/// as the task (see `elf::map_user_page`), so `copy_from_user` and
/// `copy_to_user` reach the caller's memory directly.
///
/// # Safety
/// Called only from `syscall_entry`, on the kernel stack, with `frame`
/// pointing at the register block that stub just pushed: it must be a valid,
/// aligned, uniquely-owned `SyscallFrame` that stays live until this returns,
/// because the frame is mutated in place and then popped back into the
/// registers. Not callable from ordinary Rust code. `sysv64` because the stub
/// passes `frame` in rdi.
#[no_mangle]
pub unsafe extern "sysv64" fn do_syscall(frame: *mut SyscallFrame) {
    unsafe {
        let f = &mut *frame;
        let num = f.rax;
        let arg0 = f.rdi;
        let arg1 = f.rsi;
        let arg2 = f.rdx;

        let result = crate::syscall::table::dispatch(num, arg0, arg1, arg2);
        if !crate::process::signal::prepare_syscall_restart(f, num, result.code) {
            f.rax = result.code as u64;
        }
    }
}

/// Reached from `syscall_entry` instead of `sysretq` when the return address
/// is not a user address. Kills the calling task; never returns.
extern "sysv64" fn syscall_return_outside_user_half() -> ! {
    crate::serial_println!(
        "[userspace] task {} syscall return address is outside the user half; killing it",
        crate::process::scheduler::current_task_id()
    );
    crate::process::scheduler::mark_current_dead();
    loop {
        crate::process::scheduler::yield_current();
        x86_64::instructions::interrupts::enable_and_hlt();
    }
}

/// Program the SYSCALL/SYSRET MSRs so that `syscall` from ring 3 lands in
/// `syscall_entry`. Per CPU: every CPU that runs user code calls it, after
/// its GS base points at its `PerCpu`.
pub fn init_syscall_msrs() {
    unsafe {
        // STAR holds selectors, not GDT indices. `syscall` loads CS from
        // STAR[47:32] and SS from that + 8; `sysretq` loads CS from
        // STAR[63:48] + 16 and SS from STAR[63:48] + 8. With the GDT of
        // `init_gdt` that is kernel CS 0x08 / SS 0x10 on entry and user CS
        // 0x2b / SS 0x23 on return.
        let star = ((gdt::USER_CODE32_SELECTOR.load(Ordering::Relaxed) as u64) << 48)
            | ((gdt::KERNEL_CODE_SELECTOR.load(Ordering::Relaxed) as u64) << 32);

        Msr::new(MSR_STAR).write(star);
        Msr::new(MSR_LSTAR).write(syscall_entry as *const () as u64);
        Msr::new(MSR_CSTAR).write(syscall_entry as *const () as u64);
        write_checked_fmask!(Msr::new(MSR_SFMASK).write(0x4_7700));
        // See `syscall_entry` for why the two GS bases are equal.
        KernelGsBase::write(GsBase::read());
        // Without SCE, `syscall` is #UD. Firmware leaves it clear (EFER 0xd00).
        Efer::update(|flags| flags.insert(EferFlags::SYSTEM_CALL_EXTENSIONS));
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
