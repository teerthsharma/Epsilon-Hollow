// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! HOLLOW-ASM integration — bare-metal assembly runtime components.
//!
//! Implements the stack-switch primitive, scalar I/O telemetry extraction,
//! and TSC read routines from the hollow-asm design in hand-written x86_64
//! assembly.  AVX-512 paths are future work; the scalar baseline is honest
//! and always available.

use core::arch::global_asm;

// ---------------------------------------------------------------------------
// Assembly definitions
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .global switch_stack_asm
    .align 16
switch_stack_asm:
    // rdi = old_sp (pointer to u64), rsi = new_sp
    // Symmetric stack switch: save callee-saved regs on old stack,
    // store old RSP, load new RSP, restore callee-saved regs from new stack.
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov [rdi], rsp          // *old_sp = rsp (points to saved regs)
    mov rsp, rsi            // rsp = new_sp

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

    .global extract_telemetry_asm
    .align 16
extract_telemetry_asm:
    // rdi = io_ring (pointer to IOToken array)
    // esi = depth (number of tokens)
    // Returns packed u64: (device_id << 48) | (available << 32) | (best_index << 16) | best_priority
    // Token layout (16 bytes): [device_id:u64, status:u32, priority:u16, pending:u16]

    xor eax, eax            // best_priority = 0
    xor ecx, ecx            // best_index = 0
    xor edx, edx            // available_count = 0
    xor r8d, r8d            // i = 0
    test esi, esi
    jz .Ldone

.Lloop:
    cmp r8d, esi
    jae .Ldone

    // Load token at offset i*16
    mov r9, r8
    shl r9, 4
    mov r10, [rdi + r9]           // device_id
    mov r10d, [rdi + r9 + 8]      // status
    movzx r11d, word ptr [rdi + r9 + 12] // priority

    test r10d, r10d
    jz .Lnext                       // status == 0 -> not available

    inc edx                         // available_count++
    cmp r11d, eax
    jbe .Lnext
    mov eax, r11d                   // best_priority = priority
    mov ecx, r8d                    // best_index = i

.Lnext:
    inc r8
    jmp .Lloop

.Ldone:
    // Pack result
    mov r9, rcx
    shl r9, 4
    mov r9, [rdi + r9]          // device_id of best entry
    shl r9, 48
    shl rdx, 32
    shl rcx, 16
    or  rax, rcx
    or  rax, rdx
    or  rax, r9
    ret

    .global read_tsc
    .align 16
read_tsc:
    rdtsc
    shl rdx, 32
    or  rax, rdx
    ret
"#
);

extern "C" {
    fn switch_stack_asm(old_sp: *mut u64, new_sp: u64);
    fn extract_telemetry_asm(io_ring: *const u64, depth: u32) -> u64;
    fn read_tsc() -> u64;
}

// ---------------------------------------------------------------------------
// Rust wrappers
// ---------------------------------------------------------------------------

/// Symmetric stack switch.
///
/// Saves callee-saved registers on the current stack, writes the current
/// RSP (which points at the saved registers) into `*old_sp`, then loads
/// `new_sp` and restores callee-saved registers from the new stack.
///
/// # Safety
/// - `old_sp` must be valid and writable.
/// - `new_sp` must point to a previously saved register frame in the same
///   format (six pushes: rbp, rbx, r12, r13, r14, r15).
/// - This function returns normally into the new stack's return address.
pub unsafe fn switch_stack(old_sp: *mut u64, new_sp: u64) {
    unsafe { switch_stack_asm(old_sp, new_sp) }
}

/// Packed telemetry result from `extract_telemetry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TelemetryWord(pub u64);

impl TelemetryWord {
    pub fn device_id(self) -> u64 {
        self.0 >> 48
    }
    pub fn available(self) -> u32 {
        ((self.0 >> 32) & 0xFFFF) as u32
    }
    pub fn best_index(self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }
    pub fn best_priority(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }
}

/// Scan an I/O ring and return the highest-priority ready entry.
///
/// # Safety
/// `io_ring` must point to at least `depth` contiguous 16-byte tokens.
pub unsafe fn extract_telemetry(io_ring: *const u64, depth: u32) -> TelemetryWord {
    TelemetryWord(unsafe { extract_telemetry_asm(io_ring, depth) })
}

/// Read the 64-bit Time Stamp Counter.
///
/// # Safety
/// Caller must handle TSC sync across CPUs if measurements are compared.
pub unsafe fn rdtsc() -> u64 {
    unsafe { read_tsc() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;

    fn test_telemetry_empty_ring() -> TestResult {
        let word = unsafe { extract_telemetry(core::ptr::null(), 0) };
        if word.0 != 0 {
            return TestResult::Fail("Empty telemetry should be zero");
        }
        TestResult::Pass
    }

    fn test_telemetry_word_packing() -> TestResult {
        let w = TelemetryWord((0xABCDu64 << 48) | (5u64 << 32) | (3u64 << 16) | 7);
        if w.device_id() != 0xABCD {
            return TestResult::Fail("device_id mismatch");
        }
        if w.available() != 5 {
            return TestResult::Fail("available mismatch");
        }
        if w.best_index() != 3 {
            return TestResult::Fail("best_index mismatch");
        }
        if w.best_priority() != 7 {
            return TestResult::Fail("best_priority mismatch");
        }
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("hollow_asm::telemetry_empty", test_telemetry_empty_ring);
        crate::testing::register_test("hollow_asm::word_packing", test_telemetry_word_packing);
    }
}
