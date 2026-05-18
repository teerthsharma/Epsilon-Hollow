// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Low-level x86_64 context switch.
//!
//! `switch_context` saves the current CPU state into `old` and restores from `new`.
//! This is the core primitive that makes preemptive multitasking possible.

use core::arch::asm;

/// Size of the FXSAVE area (512 bytes for x87/MMX/SSE).
pub const FXSAVE_SIZE: usize = 512;

/// Size of each kernel stack (64 KiB).
pub const KERNEL_STACK_SIZE: usize = 65536;

/// Full CPU context for a kernel task.
///
/// Layout must match the assembly in `switch_context`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TaskContext {
    // General-purpose registers
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rbx: u64,
    pub rax: u64,

    // Control registers / state
    pub rip: u64,
    pub rsp: u64,
    pub rflags: u64,

    // FPU / SSE state
    pub fxsave: [u8; FXSAVE_SIZE],
}

impl TaskContext {
    pub const fn zero() -> Self {
        Self {
            r15: 0,
            r14: 0,
            r13: 0,
            r12: 0,
            r11: 0,
            r10: 0,
            r9: 0,
            r8: 0,
            rbp: 0,
            rdi: 0,
            rsi: 0,
            rdx: 0,
            rcx: 0,
            rbx: 0,
            rax: 0,
            rip: 0,
            rsp: 0,
            rflags: 0,
            fxsave: [0; FXSAVE_SIZE],
        }
    }
}

/// Switch from `old` context to `new` context.
///
/// # Safety
/// - `old` and `new` must point to valid, aligned `TaskContext` structs.
/// - Interrupts must be disabled by the caller.
/// - This function returns into the `new` context's execution flow.
pub unsafe fn switch_context(old: *mut TaskContext, new: *const TaskContext) {
    asm!(
        // Save current GPRs into `old`
        // old pointer is in {old} (rdi)
        "mov [{old} + 0x00], r15",
        "mov [{old} + 0x08], r14",
        "mov [{old} + 0x10], r13",
        "mov [{old} + 0x18], r12",
        "mov [{old} + 0x20], r11",
        "mov [{old} + 0x28], r10",
        "mov [{old} + 0x30], r9",
        "mov [{old} + 0x38], r8",
        "mov [{old} + 0x40], rbp",
        // Save rdi into a scratch register first, then store
        "mov r15, rdi",
        "mov [{old} + 0x48], r15",
        "mov r15, rsi",
        "mov [{old} + 0x50], r15",
        "mov r15, rdx",
        "mov [{old} + 0x58], r15",
        "mov r15, rcx",
        "mov [{old} + 0x60], r15",
        "mov r15, rbx",
        "mov [{old} + 0x68], r15",
        "mov r15, rax",
        "mov [{old} + 0x70], r15",

        // Save RIP (return address is on stack)
        "mov r15, [rsp]",
        "mov [{old} + 0x78], r15",

        // Save RSP (stack pointer after return address)
        "lea r15, [rsp + 8]",
        "mov [{old} + 0x80], r15",

        // Save RFLAGS
        "pushfq",
        "pop r15",
        "mov [{old} + 0x88], r15",

        // Save FPU state
        "fxsave [{old} + 0x90]",

        // Restore FPU state from new
        "fxrstor [{new} + 0x90]",

        // Restore GPRs from `new`
        "mov r15, [{new} + 0x00]",
        "mov r14, [{new} + 0x08]",
        "mov r13, [{new} + 0x10]",
        "mov r12, [{new} + 0x18]",
        "mov r11, [{new} + 0x20]",
        "mov r10, [{new} + 0x28]",
        "mov r9,  [{new} + 0x30]",
        "mov r8,  [{new} + 0x38]",
        "mov rbp, [{new} + 0x40]",
        "mov rdx, [{new} + 0x58]",
        "mov rcx, [{new} + 0x60]",
        "mov rbx, [{new} + 0x68]",
        "mov rax, [{new} + 0x70]",

        // Restore RFLAGS
        "mov r11, [{new} + 0x88]",
        "push r11",
        "popfq",

        // Set up new stack
        "mov rsp, [{new} + 0x80]",
        // Push return address (rip) onto new stack
        "push qword ptr [{new} + 0x78]",

        // Restore rsi and rdi last
        "mov rsi, [{new} + 0x50]",
        "mov rdi, [{new} + 0x48]",

        // Return into new task
        "ret",
        old = in(reg) old,
        new = in(reg) new,
        clobber_abi("system"),
    );
}

/// Wrapper function called when a newly created task is first scheduled.
///
/// The task's `rip` is set to this function, and its `rdi` contains the
/// user entry function pointer.
extern "C" fn kernel_task_wrapper(entry: fn()) {
    entry();
    // When entry returns, mark task as dead and yield
    super::scheduler::mark_current_dead();
    super::scheduler::yield_current();
}

/// Prepare the initial context for a kernel task.
///
/// `stack` must be a mutable slice of at least `KERNEL_STACK_SIZE` bytes.
/// `entry` is the function the task will start executing.
pub fn init_task_context(stack: &mut [u8], entry: fn()) -> TaskContext {
    let stack_top = stack.as_mut_ptr() as u64 + stack.len() as u64;
    // Align stack to 16 bytes as required by SysV AMD64 ABI
    let stack_top = stack_top & !0xF;

    let mut ctx = TaskContext::zero();
    ctx.rip = kernel_task_wrapper as u64;
    ctx.rdi = entry as u64;
    ctx.rsp = stack_top;
    ctx.rflags = 0x202; // Interrupt enable (IF) bit set

    ctx
}
