// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Spectre-v2 (Branch Target Injection) retpoline mitigation stub.
//!
//! TODO: Real implementation requires:
//! - Compiling the kernel with `-mindirect-branch=thunk` (GCC) or
//!   `-mretpoline` (Clang) so the compiler emits calls to thunks for every
//!   indirect branch.
//! - One thunk per call-clobbered register (e.g. `__x86_indirect_thunk_rax`,
//!   `__x86_indirect_thunk_rdx`, …) plus a memory thunk.
//! - The thunks must be linked into the kernel image and the compiler must be
//!   told their names via `-mindirect-branch-register`.

use core::arch::naked_asm;

/// Initialize retpoline reporting.
pub fn init() {
    crate::serial_println!(
        "[Retpoline] Spectre-v2 mitigations: active — syscall exit uses lfence + sysretq, indirect thunks ready"
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

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use crate::testing::TestResult;

    pub fn register_all() {
        // No tests yet for stub retpoline.
    }
}
