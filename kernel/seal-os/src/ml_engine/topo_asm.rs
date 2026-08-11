// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TOPO-ASM integration — topological computation primitives in x86_64 assembly.
//!
//! Implements scalar versions of:
//!   - L^∞ landscape distance: max_i |a\[i\] - b\[i\]|
//!   - Betti accumulator: count finite birth/death pairs
//!
//! # Calling convention
//!
//! `x86_64-unknown-uefi` is a Windows-like target: `extern "C"` here is the
//! Win64 convention, so integer arguments arrive in RCX, RDX, R8, R9 — not in
//! the System V RDI, RSI, RDX. The target also builds `-sse,+soft-float`, so an
//! `extern "C" fn(..) -> f64` returns its bit pattern in RAX rather than in
//! XMM0; the emitted code for a UEFI-target caller feeds the returned RAX
//! straight into `__gtdf2`. Both routines below follow that convention, and
//! `simd_landscape_distance` additionally leaves the result in XMM0 so it stays
//! correct if the crate is ever built for a target with SSE enabled.
//!
//! # Removed AVX-512 path
//!
//! An AVX-512 variant used to live here. It is gone rather than repaired: the
//! kernel never executes `xsetbv`, so XCR0 keeps whatever state mask the
//! firmware left, and UEFI only guarantees the SSE state components. Issuing
//! ZMM instructions with the AVX-512 state bits clear raises #UD, which this
//! kernel cannot report. Its guard, `ml_engine::detect_avx512()`, reads CPUID
//! feature bits only and checks neither OSXSAVE nor XCR0, so it cannot tell the
//! difference. Restore the vector path only alongside XCR0 enablement and a
//! boot proof from AVX-512 hardware.

use core::arch::global_asm;

// ---------------------------------------------------------------------------
// Scalar assembly baseline
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .global simd_landscape_distance
    .p2align 4
simd_landscape_distance:
    // Win64: rcx = a (f64*), rdx = b (f64*), r8 = len.
    // Returns max_i |a[i] - b[i]| in rax (soft-float f64) and in xmm0 (SSE f64).
    xor rax, rax
    xorpd xmm0, xmm0            // max_diff = +0.0 in both return registers
    test r8, r8
    jz .Ldist_done

    pcmpeqd xmm2, xmm2
    psrlq xmm2, 1               // xmm2 = 0x7FFFFFFFFFFFFFFF, sign-clearing mask
    xor r9, r9

.Ldist_loop:
    movsd xmm1, [rcx + r9*8]
    subsd xmm1, [rdx + r9*8]
    andpd xmm1, xmm2            // |a[i] - b[i]|
    maxsd xmm0, xmm1            // MAXSD yields its source when either is NaN,
                                // so a NaN difference propagates instead of
                                // reading as 0.0
    inc r9
    cmp r9, r8
    jb .Ldist_loop

    movq rax, xmm0

.Ldist_done:
    ret

    .global simd_betti_accumulate
    .p2align 4
simd_betti_accumulate:
    // Win64: rcx = barcode ([birth: f64, death: f64]*), rdx = pairs,
    //        r8b = dim (part of the ABI; the scalar count ignores it).
    // Returns the number of finite pairs in rax.
    xor rax, rax
    test rdx, rdx
    jz .Lbetti_done
    xor r9, r9

.Lbetti_loop:
    mov r10, r9
    shl r10, 4
    movsd xmm0, [rcx + r10]         // birth
    movsd xmm1, [rcx + r10 + 8]     // death
    ucomisd xmm1, xmm0
    jbe .Lbetti_next                // death <= birth, or either is NaN, so the
                                    // pair is not finite -> skip
    inc rax

.Lbetti_next:
    inc r9
    cmp r9, rdx
    jb .Lbetti_loop

.Lbetti_done:
    ret
"#
);

extern "C" {
    fn simd_landscape_distance(a: *const f64, b: *const f64, len: usize) -> f64;
    fn simd_betti_accumulate(barcode: *const f64, pairs: usize, dim: u8) -> u64;
}

// ---------------------------------------------------------------------------
// Rust wrappers
// ---------------------------------------------------------------------------

/// Compute the L^∞ distance between two f64 landscapes.
///
/// Reads `min(a.len(), b.len())` elements; a length mismatch truncates rather
/// than reading past the shorter slice.
pub fn landscape_distance(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    // SAFETY: `len` is the shorter of the two lengths and the callee reads
    // exactly `len` f64 from each pointer, so every load is in bounds of a live
    // slice. It touches no other memory and returns by value.
    unsafe { simd_landscape_distance(a.as_ptr(), b.as_ptr(), len) }
}

/// Compute the L^∞ distance between two fixed-size 64-D landscapes.
///
/// This is the canonical size used by the Hollow-ASM topological scheduler.
pub fn landscape_distance_64(a: &[f64; 64], b: &[f64; 64]) -> f64 {
    landscape_distance(a.as_slice(), b.as_slice())
}

/// Count finite persistence pairs (birth < death) in a barcode.
///
/// The `dim` parameter is part of the topo-asm ABI; the scalar accumulator
/// counts all finite pairs regardless of dimension.
pub fn betti_count(barcode: &[(f64, f64)], _dim: u8) -> u64 {
    if barcode.is_empty() {
        return 0;
    }
    // Barcode is [(birth, death)]; we pass it as a flat f64 array. The pair is
    // two f64, so `(birth, death)` occupies 16 bytes with birth first.
    let flat = barcode.as_ptr() as *const f64;
    // SAFETY: the callee reads exactly `barcode.len()` 16-byte pairs starting at
    // the slice base, which is the slice's own extent, and writes nothing.
    unsafe { simd_betti_accumulate(flat, barcode.len(), _dim) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;

    fn test_distance_identity() -> TestResult {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let d = landscape_distance(&a, &a);
        if d != 0.0 {
            return TestResult::Fail("Distance of identical arrays must be 0");
        }
        TestResult::Pass
    }

    fn test_distance_basic() -> TestResult {
        let a = [0.0f64, 0.0, 0.0];
        let b = [3.0f64, -1.0, 2.0];
        let d = landscape_distance(&a, &b);
        // max(|0-3|, |0+1|, |0-2|) = max(3, 1, 2) = 3
        if (d - 3.0).abs() > 1e-12 {
            return TestResult::Fail("Expected distance 3.0");
        }
        TestResult::Pass
    }

    fn test_distance_peak_at_end() -> TestResult {
        let a = [0.0f64, 0.0, 0.0, 0.0];
        let b = [1.0f64, 0.5, 2.0, 10.0];
        let d = landscape_distance(&a, &b);
        // The peak is the last element, so an accumulator that keeps the first
        // candidate instead of the larger one, or a loop that stops short of
        // the final index, cannot reach 10.0.
        if (d - 10.0).abs() > 1e-12 {
            return TestResult::Fail("Expected distance 10.0");
        }
        TestResult::Pass
    }

    fn test_betti_count() -> TestResult {
        let barcode = [
            (0.0, 1.0), // finite -> count
            (1.0, 1.0), // infinite (birth == death) -> skip
            (2.0, 5.0), // finite -> count
        ];
        let c = betti_count(&barcode, 0);
        if c != 2 {
            return TestResult::Fail("Expected Betti count 2");
        }
        TestResult::Pass
    }

    fn test_betti_count_rejects_nan() -> TestResult {
        // A NaN endpoint orders against nothing, so it is not a finite pair.
        let barcode = [(0.0f64, f64::NAN), (f64::NAN, 1.0f64)];
        if betti_count(&barcode, 0) != 0 {
            return TestResult::Fail("NaN endpoints must not count as finite");
        }
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("topo_asm::distance_identity", test_distance_identity);
        crate::testing::register_test("topo_asm::distance_basic", test_distance_basic);
        crate::testing::register_test("topo_asm::distance_peak_at_end", test_distance_peak_at_end);
        crate::testing::register_test("topo_asm::betti_count", test_betti_count);
        crate::testing::register_test(
            "topo_asm::betti_count_rejects_nan",
            test_betti_count_rejects_nan,
        );
    }
}
