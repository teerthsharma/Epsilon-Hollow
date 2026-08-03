// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Spectre-v2 (Branch Target Injection) retpoline mitigation.
//!
//! Provides naked thunk functions for all general-purpose registers. Each has
//! the canonical retpoline shape — a `call` over a `pause; lfence` capture loop
//! into a tail that overwrites the return slot with the real target:
//!
//! ```text
//!     call 2f
//! 1:  pause            ; speculation following the call lands here
//!     lfence           ; ...and is trapped spinning, not running ahead
//!     jmp 1b
//! 2:  mov [rsp], rax   ; architectural path: real target into the return slot
//!     ret
//! ```
//!
//! The capture loop is the mitigation. Without it — `call 2f; 2: mov [rsp],
//! rax; ret`, which is what these thunks were until the loop was added — the
//! capture target is the next instruction, nothing is trapped, and the sequence
//! is architecturally equivalent to a bare indirect call.
//!
//! SCOPE: nothing in this kernel currently routes an indirect branch through
//! these thunks, and kernel-wide `-Zretpoline` codegen is off (see
//! `.cargo/config.toml`). They are a correct mitigation that is not yet wired,
//! and `security::features::retpoline()` reports exactly that and no more.

use core::arch::naked_asm;

/// Initialize retpoline reporting.
pub fn init() {
    crate::serial_println!(
        "[Retpoline] Spectre-v2: syscall exit uses lfence + sysretq; thunks for rax-r15 present and well-formed, but no indirect branch is routed through them"
    );
}

/// Minimal retpoline indirect branch thunk for RAX.
///
/// Replaces an indirect jump/call via RAX.  The caller must load the target
/// into RAX and then `jmp` or `call` this thunk.  Speculative execution that
/// reaches the internal `call` will spin at the pause/lfence trap, while the
/// architectural path overwrites the stack slot with the real target and
/// returns to it.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RAX already holding the
/// branch target. Control never returns to the immediate caller: the thunk
/// overwrites its own return slot with RAX and `ret`s into it, leaving the
/// original caller's return address on top of the stack. RAX must therefore
/// hold the address of executable code entered under the SysV ABI, with that
/// target's argument registers already set up. Invoking this from Rust as an
/// ordinary `extern "C"` function jumps to whatever RAX happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], rax",
        "ret",
    );
}

/// Retpoline indirect branch thunk for RBX.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RBX already holding the
/// branch target. The thunk overwrites its own return slot with RBX and `ret`s
/// into it, so it never returns to its immediate caller and RBX must hold the
/// address of executable code entered under the SysV ABI with that target's
/// argument registers set up. Calling it from Rust as an ordinary `extern "C"`
/// function jumps to whatever RBX happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rbx() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], rbx",
        "ret",
    );
}

/// Retpoline indirect branch thunk for RCX.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RCX already holding the
/// branch target. The thunk overwrites its own return slot with RCX and `ret`s
/// into it, so it never returns to its immediate caller and RCX must hold the
/// address of executable code entered under the SysV ABI with that target's
/// argument registers set up. Calling it from Rust as an ordinary `extern "C"`
/// function jumps to whatever RCX happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rcx() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], rcx",
        "ret",
    );
}

/// Retpoline indirect branch thunk for RDX.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RDX already holding the
/// branch target. The thunk overwrites its own return slot with RDX and `ret`s
/// into it, so it never returns to its immediate caller and RDX must hold the
/// address of executable code entered under the SysV ABI with that target's
/// argument registers set up. Calling it from Rust as an ordinary `extern "C"`
/// function jumps to whatever RDX happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rdx() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], rdx",
        "ret",
    );
}

/// Retpoline indirect branch thunk for RSI.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RSI already holding the
/// branch target. The thunk overwrites its own return slot with RSI and `ret`s
/// into it, so it never returns to its immediate caller and RSI must hold the
/// address of executable code entered under the SysV ABI. Note that RSI is the
/// second SysV argument register, so the target cannot expect an argument
/// there. Calling this from Rust jumps to whatever RSI happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rsi() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], rsi",
        "ret",
    );
}

/// Retpoline indirect branch thunk for RDI.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RDI already holding the
/// branch target. The thunk overwrites its own return slot with RDI and `ret`s
/// into it, so it never returns to its immediate caller and RDI must hold the
/// address of executable code entered under the SysV ABI. Note that RDI is the
/// first SysV argument register, so the target cannot expect an argument
/// there. Calling this from Rust jumps to whatever RDI happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rdi() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], rdi",
        "ret",
    );
}

/// Retpoline indirect branch thunk for RBP.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RBP already holding the
/// branch target. The thunk overwrites its own return slot with RBP and `ret`s
/// into it, so it never returns to its immediate caller and RBP must hold the
/// address of executable code entered under the SysV ABI. Because RBP is
/// callee-saved and normally the frame pointer, the caller must have finished
/// with its frame pointer before loading a branch target into it.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rbp() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], rbp",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R8.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R8 already holding the
/// branch target. The thunk overwrites its own return slot with R8 and `ret`s
/// into it, so it never returns to its immediate caller and R8 must hold the
/// address of executable code entered under the SysV ABI. R8 is the fifth SysV
/// argument register, so the target cannot expect an argument there. Calling
/// this from Rust jumps to whatever R8 happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r8() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r8",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R9.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R9 already holding the
/// branch target. The thunk overwrites its own return slot with R9 and `ret`s
/// into it, so it never returns to its immediate caller and R9 must hold the
/// address of executable code entered under the SysV ABI. R9 is the sixth SysV
/// argument register, so the target cannot expect an argument there. Calling
/// this from Rust jumps to whatever R9 happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r9() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r9",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R10.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R10 already holding the
/// branch target. The thunk overwrites its own return slot with R10 and `ret`s
/// into it, so it never returns to its immediate caller and R10 must hold the
/// address of executable code entered under the SysV ABI with that target's
/// argument registers set up. Calling it from Rust as an ordinary `extern "C"`
/// function jumps to whatever R10 happens to contain.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r10() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r10",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R11.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R11 already holding the
/// branch target. The thunk overwrites its own return slot with R11 and `ret`s
/// into it, so it never returns to its immediate caller and R11 must hold the
/// address of executable code entered under the SysV ABI. R11 holds the saved
/// RFLAGS on the `syscall` path, so it must not be used as the branch register
/// between `syscall` entry and the matching `sysretq`.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r11() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r11",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R12.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R12 already holding the
/// branch target. The thunk overwrites its own return slot with R12 and `ret`s
/// into it, so it never returns to its immediate caller and R12 must hold the
/// address of executable code entered under the SysV ABI. R12 is callee-saved,
/// so the caller must have spilled any live value before loading a branch
/// target into it.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r12() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r12",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R13.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R13 already holding the
/// branch target. The thunk overwrites its own return slot with R13 and `ret`s
/// into it, so it never returns to its immediate caller and R13 must hold the
/// address of executable code entered under the SysV ABI. R13 is callee-saved,
/// so the caller must have spilled any live value before loading a branch
/// target into it.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r13() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r13",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R14.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R14 already holding the
/// branch target. The thunk overwrites its own return slot with R14 and `ret`s
/// into it, so it never returns to its immediate caller and R14 must hold the
/// address of executable code entered under the SysV ABI. R14 is callee-saved,
/// so the caller must have spilled any live value before loading a branch
/// target into it.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r14() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r14",
        "ret",
    );
}

/// Retpoline indirect branch thunk for R15.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with R15 already holding the
/// branch target. The thunk overwrites its own return slot with R15 and `ret`s
/// into it, so it never returns to its immediate caller and R15 must hold the
/// address of executable code entered under the SysV ABI. R15 is callee-saved,
/// so the caller must have spilled any live value before loading a branch
/// target into it.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r15() {
    naked_asm!(
        "call 2f",
        "1:",
        "pause",
        "lfence",
        "jmp 1b",
        "2:",
        "mov [rsp], r15",
        "ret",
    );
}

/// AMD-style retpoline barrier for indirect branches.
///
/// Use this before an indirect `jmp rax` to block speculation.
///
/// # Safety
/// Entered only from assembly via `call`/`jmp`, with RAX already holding the
/// branch target. After the `lfence` the thunk tail-jumps to RAX without
/// touching the stack, so it never returns to its immediate caller: RAX must
/// hold the address of executable code entered under the SysV ABI, and the
/// target inherits the stack unchanged (when reached by `call`, the target's
/// `ret` therefore returns to this thunk's caller).
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn lfence_jmp_rax() {
    naked_asm!("lfence", "jmp rax",);
}

/// Inline LFENCE barrier for Spectre-v2 mitigation.
pub fn lfence_barrier() {
    unsafe {
        core::arch::x86_64::_mm_lfence();
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use crate::testing::TestResult;

    pub fn register_all() {
        // Retpoline thunks are verified by assembly inspection.
    }
}
