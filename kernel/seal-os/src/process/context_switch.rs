// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Low-level x86_64 context switch.
//!
//! `switch_context` saves the current CPU state into `old` and restores from `new`.
//! This is the core primitive that makes preemptive multitasking possible.

use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Legacy FXSAVE area size (512 bytes for x87/MMX/SSE).
pub const FXSAVE_SIZE: usize = 512;

/// Generous upper bound for the XSAVE area to accommodate AVX-512 and future
/// extensions. The actual used size is queried via CPUID at boot.
pub const XSAVE_MAX_SIZE: usize = 4096;

/// Size of each kernel stack (256 KiB).
pub const KERNEL_STACK_SIZE: usize = 256 * 1024;

#[derive(Clone, Copy)]
#[repr(align(64))]
struct AlignedXsaveArea([u8; XSAVE_MAX_SIZE + 64]);

static mut EMERGENCY_XSAVE_AREAS: [AlignedXsaveArea; 2] =
    [AlignedXsaveArea([0; XSAVE_MAX_SIZE + 64]); 2];

static XSAVE_SUPPORTED: AtomicBool = AtomicBool::new(false);
static XSAVE_AREA_SIZE: AtomicUsize = AtomicUsize::new(FXSAVE_SIZE);
static XSAVE_MASK_EAX: AtomicUsize = AtomicUsize::new(0);
static XSAVE_MASK_EDX: AtomicUsize = AtomicUsize::new(0);

/// Return whether XSAVE is supported and enabled on this CPU.
pub fn xsave_supported() -> bool {
    XSAVE_SUPPORTED.load(Ordering::Relaxed)
}

/// Return the detected XSAVE area size (defaults to 512 before detection).
pub fn xsave_area_size() -> usize {
    XSAVE_AREA_SIZE.load(Ordering::Relaxed)
}

/// Detect XSAVE support via CPUID leaf 0x0D and initialize globals.
///
/// # Safety
/// Must be called once per CPU during early boot.
pub unsafe fn detect_xsave() {
    let leaf1 = core::arch::x86_64::__cpuid(1);
    let has_xsave = (leaf1.ecx & (1 << 26)) != 0;
    let osxsave = (leaf1.ecx & (1 << 27)) != 0;

    if !has_xsave || !osxsave {
        XSAVE_SUPPORTED.store(false, Ordering::Relaxed);
        XSAVE_AREA_SIZE.store(FXSAVE_SIZE, Ordering::Relaxed);
        XSAVE_MASK_EAX.store(0, Ordering::Relaxed);
        XSAVE_MASK_EDX.store(0, Ordering::Relaxed);
        return;
    }

    // Query max XSAVE area size for all valid XCR0 bits (leaf 0x0D sub-leaf 0, ECX)
    let leaf_d = core::arch::x86_64::__cpuid_count(0x0D, 0);
    let size = leaf_d.ecx as usize;
    let size = size.max(FXSAVE_SIZE).min(XSAVE_MAX_SIZE);

    XSAVE_SUPPORTED.store(true, Ordering::Relaxed);
    XSAVE_AREA_SIZE.store(size, Ordering::Relaxed);

    // Read XCR0 to use as the xsave/xrstor state-component mask
    let xcr0_low: u32;
    let xcr0_high: u32;
    core::arch::asm!("xgetbv", in("ecx") 0u32, out("eax") xcr0_low, out("edx") xcr0_high);
    XSAVE_MASK_EAX.store(xcr0_low as usize, Ordering::Relaxed);
    XSAVE_MASK_EDX.store(xcr0_high as usize, Ordering::Relaxed);
}

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

    // Pointer to the XSAVE/FXSAVE area (must be 64-byte aligned for XSAVE).
    pub xsave_ptr: *mut u8,
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
            xsave_ptr: core::ptr::null_mut(),
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
    sanitize_xsave_ptr(old, 0);
    sanitize_xsave_ptr(new as *mut TaskContext, 1);
    if XSAVE_SUPPORTED.load(Ordering::Relaxed) {
        switch_context_xsave(old, new);
    } else {
        switch_context_fxsave(old, new);
    }
}

unsafe fn sanitize_xsave_ptr(ctx: *mut TaskContext, slot: usize) {
    if ctx.is_null() {
        return;
    }

    let ptr = (*ctx).xsave_ptr as usize;
    if ptr != 0 && ptr & 63 == 0 {
        return;
    }

    let fallback = core::ptr::addr_of_mut!(EMERGENCY_XSAVE_AREAS[slot].0).cast::<u8>();
    (*ctx).xsave_ptr = fallback;
    crate::serial_println!(
        "[scheduler] repaired invalid xsave_ptr: slot={} old_ptr={:#x} new_ptr={:#x}",
        slot,
        ptr,
        fallback as usize
    );
}

unsafe fn switch_context_xsave(old: *mut TaskContext, new: *const TaskContext) {
    let mask_low = XSAVE_MASK_EAX.load(Ordering::Relaxed) as u32;
    let mask_high = XSAVE_MASK_EDX.load(Ordering::Relaxed) as u32;
    asm!(
        // Save current GPRs into `old`
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

        // Save extended state via XSAVE
        "mov r15, [{old} + 0x90]",
        "xsave [r15]",

        // Restore extended state via XRSTOR
        "mov r15, [{new} + 0x90]",
        "xrstor [r15]",

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
        in("eax") mask_low,
        in("edx") mask_high,
        out("r11") _,
        out("r15") _,
        clobber_abi("system"),
    );
}

unsafe fn switch_context_fxsave(old: *mut TaskContext, new: *const TaskContext) {
    asm!(
        // Save current GPRs into `old`
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

        // Save FPU state via FXSAVE
        "mov r15, [{old} + 0x90]",
        "fxsave [r15]",

        // Restore FPU state via FXRSTOR
        "mov r15, [{new} + 0x90]",
        "fxrstor [r15]",

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
        out("r11") _,
        out("r15") _,
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
/// `xsave_ptr` must point to a valid, aligned XSAVE/FXSAVE area.
pub fn init_task_context(stack: &mut [u8], entry: fn(), xsave_ptr: *mut u8) -> TaskContext {
    let stack_top = stack.as_mut_ptr() as u64 + stack.len() as u64;
    // Align stack to 16 bytes as required by SysV AMD64 ABI
    let stack_top = stack_top & !0xF;

    let mut ctx = TaskContext::zero();
    ctx.rip = kernel_task_wrapper as *const () as u64;
    ctx.rdi = entry as u64;
    ctx.rsp = stack_top;
    ctx.rflags = 0x202; // Interrupt enable (IF) bit set
    ctx.xsave_ptr = xsave_ptr;

    ctx
}
