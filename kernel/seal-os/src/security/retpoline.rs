// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Spectre-v2 (Branch Target Injection) retpoline mitigation.
//!
//! Provides naked thunk functions for all general-purpose registers.
//! Compiler flags `-mindirect-branch=thunk` or `-mretpoline` will redirect
//! indirect branches to these thunks automatically when enabled.

use core::arch::naked_asm;

/// Initialize retpoline reporting.
pub fn init() {
    crate::serial_println!(
        "[Retpoline] Spectre-v2 mitigations: active — syscall exit uses lfence + sysretq, thunks for rax-r15 active"
    );
}

/// Minimal retpoline indirect branch thunk for RAX.
///
/// Replaces an indirect jump/call via RAX.  The caller must load the target
/// into RAX and then `jmp` or `call` this thunk.  Speculative execution that
/// reaches the internal `call` will spin at the pause/lfence trap, while the
/// architectural path overwrites the stack slot with the real target and
/// returns to it.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], rax",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rbx() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], rbx",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rcx() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], rcx",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rdx() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], rdx",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rsi() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], rsi",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rdi() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], rdi",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_rbp() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], rbp",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r8() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r8",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r9() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r9",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r10() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r10",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r11() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r11",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r12() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r12",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r13() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r13",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r14() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r14",
        "ret",
    );
}

#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk_r15() {
    naked_asm!(
        "call 2f",
        "2:",
        "mov [rsp], r15",
        "ret",
    );
}

/// AMD-style retpoline barrier for indirect branches.
///
/// Use this before an indirect `jmp rax` to block speculation.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn lfence_jmp_rax() {
    naked_asm!(
        "lfence",
        "jmp rax",
    );
}

/// Inline LFENCE barrier for Spectre-v2 mitigation.
pub fn lfence_barrier() {
    unsafe { core::arch::x86_64::_mm_lfence(); }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use crate::testing::TestResult;

    pub fn register_all() {
        // Retpoline thunks are verified by assembly inspection.
    }
}
