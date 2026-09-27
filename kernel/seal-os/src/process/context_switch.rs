// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Low-level x86_64 context switch.
//!
//! `switch_context` saves the current CPU state into `old` and restores from `new`.
//! This is the core primitive that makes preemptive multitasking possible.

use core::arch::naked_asm;
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

struct SyncUnsafeCell<T>(core::cell::UnsafeCell<T>);
unsafe impl<T> Sync for SyncUnsafeCell<T> {}

static EMERGENCY_XSAVE_AREAS: SyncUnsafeCell<[AlignedXsaveArea; 2]> = SyncUnsafeCell(
    core::cell::UnsafeCell::new([AlignedXsaveArea([0; XSAVE_MAX_SIZE + 64]); 2]),
);

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
    let size = size.clamp(FXSAVE_SIZE, XSAVE_MAX_SIZE);

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
        switch_context_xsave(
            old,
            new,
            XSAVE_MASK_EAX.load(Ordering::Relaxed) as u32,
            XSAVE_MASK_EDX.load(Ordering::Relaxed) as u32,
        );
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

    let fallback = unsafe { (*EMERGENCY_XSAVE_AREAS.0.get())[slot].0.as_mut_ptr() };
    (*ctx).xsave_ptr = fallback;
    crate::serial_println!(
        "[scheduler] repaired invalid xsave_ptr: slot={} old_ptr={:#x} new_ptr={:#x}",
        slot,
        ptr,
        fallback as usize
    );
}

// The switch primitives are naked `sysv64` functions so that on entry `[rsp]`
// is exactly their own return address: the saved `rip`/`rsp` pair resumes
// `old` as if this call had returned. Written as inline `asm!` inside ordinary
// functions they were inlined into `schedule()`, where `[rsp]` is a slot of
// `schedule()`'s own frame, so a switched-out task could never resume.
// `sysv64` is spelled out because `extern "C"` is the Microsoft x64
// convention on x86_64-unknown-uefi.
//
// Only the SysV callee-saved registers (rbx, rbp, r12-r15) carry meaning across
// a switch, so only they are saved. Every register is restored from `new`,
// because a task that has never run takes its first arguments in rdi/rsi/rdx.

#[unsafe(naked)]
unsafe extern "sysv64" fn switch_context_xsave(
    old: *mut TaskContext,
    new: *const TaskContext,
    mask_low: u32,
    mask_high: u32,
) {
    naked_asm!(
        "mov [rdi + 0x00], r15",
        "mov [rdi + 0x08], r14",
        "mov [rdi + 0x10], r13",
        "mov [rdi + 0x18], r12",
        "mov [rdi + 0x40], rbp",
        "mov [rdi + 0x68], rbx",
        "mov rax, [rsp]",
        "mov [rdi + 0x78], rax",
        "lea rax, [rsp + 8]",
        "mov [rdi + 0x80], rax",
        "pushfq",
        "pop qword ptr [rdi + 0x88]",
        // XSAVE/XRSTOR take the state-component mask in edx:eax.
        "mov eax, edx",
        "mov edx, ecx",
        "mov r8, [rdi + 0x90]",
        "xsave [r8]",
        "mov r8, [rsi + 0x90]",
        "xrstor [r8]",
        "jmp {restore}",
        restore = sym restore_context,
    );
}

#[unsafe(naked)]
unsafe extern "sysv64" fn switch_context_fxsave(old: *mut TaskContext, new: *const TaskContext) {
    naked_asm!(
        "mov [rdi + 0x00], r15",
        "mov [rdi + 0x08], r14",
        "mov [rdi + 0x10], r13",
        "mov [rdi + 0x18], r12",
        "mov [rdi + 0x40], rbp",
        "mov [rdi + 0x68], rbx",
        "mov rax, [rsp]",
        "mov [rdi + 0x78], rax",
        "lea rax, [rsp + 8]",
        "mov [rdi + 0x80], rax",
        "pushfq",
        "pop qword ptr [rdi + 0x88]",
        "mov r8, [rdi + 0x90]",
        "fxsave [r8]",
        "mov r8, [rsi + 0x90]",
        "fxrstor [r8]",
        "jmp {restore}",
        restore = sym restore_context,
    );
}

/// Shared tail of both switch primitives: load `new` (in rsi) and resume it.
/// RFLAGS is restored last, once RSP already points at `new`'s stack, so an
/// IF=1 in `new` cannot open an interrupt window on `old`'s stack.
#[unsafe(naked)]
unsafe extern "sysv64" fn restore_context() {
    naked_asm!(
        "mov r15, [rsi + 0x00]",
        "mov r14, [rsi + 0x08]",
        "mov r13, [rsi + 0x10]",
        "mov r12, [rsi + 0x18]",
        "mov r11, [rsi + 0x20]",
        "mov r10, [rsi + 0x28]",
        "mov r9,  [rsi + 0x30]",
        "mov r8,  [rsi + 0x38]",
        "mov rbp, [rsi + 0x40]",
        "mov rdx, [rsi + 0x58]",
        "mov rcx, [rsi + 0x60]",
        "mov rbx, [rsi + 0x68]",
        "mov rax, [rsi + 0x70]",
        "mov rsp, [rsi + 0x80]",
        "push qword ptr [rsi + 0x78]",
        "push qword ptr [rsi + 0x88]",
        "mov rdi, [rsi + 0x48]",
        "mov rsi, [rsi + 0x50]",
        "popfq",
        "ret",
    );
}

/// Give a fresh XSAVE/FXSAVE area the power-on FPU control state: x87 FCW
/// 0x037F and MXCSR 0x1F80, every exception masked. The first switch into a
/// task restores its area before anything was ever saved there, and an
/// all-zero image unmasks every SSE exception.
///
/// # Safety
/// `area` must be the 64-byte-aligned start of a writable XSAVE/FXSAVE area
/// at least `FXSAVE_SIZE` bytes long (FCW is bytes 0..2, MXCSR bytes 24..28).
pub unsafe fn init_fpu_area(area: *mut u8) {
    // SAFETY: the caller guarantees `area` as stated under # Safety.
    unsafe {
        area.cast::<u16>().write(0x037F);
        area.add(24).cast::<u32>().write(0x1F80);
    }
}

/// First code a new kernel task runs; `switch_context` enters it with
/// `entry` in rdi.
#[allow(improper_ctypes_definitions)] // REASON: kernel_task_wrapper is an internal kernel ABI boundary, not user FFI
extern "sysv64" fn kernel_task_wrapper(entry: fn()) -> ! {
    entry();
    // A dead task is never requeued, so the yield below returns only while no
    // other task is ready; keep yielding until one is.
    super::scheduler::mark_current_dead();
    loop {
        super::scheduler::yield_current();
        x86_64::instructions::hlt();
    }
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
    ctx.rdi = entry as *const () as u64;
    // `switch_context` enters through `ret`, so the entry must see the RSP a
    // `call` would have left: 8 below a 16-byte boundary.
    ctx.rsp = stack_top - 8;
    ctx.rflags = 0x202; // Interrupt enable (IF) bit set
    ctx.xsave_ptr = xsave_ptr;

    ctx
}
