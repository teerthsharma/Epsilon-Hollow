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
        "[Retpoline] Spectre-v2 mitigations: stub — needs compiler retpoline flags + indirect branch thunk"
    );
}

/// Minimal retpoline indirect branch thunk.
///
/// This is a capture thunk: speculative execution that reaches it will spin
/// forever at the `pause`/`lfence` loop instead of being redirected to a
/// gadget.  Architecturally it is a no-op placeholder.
///
/// TODO: Replace with a full set of register-specific thunks and compiler
/// integration.
#[unsafe(naked)]
#[no_mangle]
pub unsafe extern "C" fn indirect_branch_thunk() {
    naked_asm!(
        "call 2f",
        "2:",
        "pause",
        "lfence",
        "jmp [rsp]",
    );
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use crate::testing::TestResult;

    pub fn register_all() {
        // No tests yet for stub retpoline.
    }
}
