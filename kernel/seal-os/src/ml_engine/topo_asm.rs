// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! TOPO-ASM integration — topological computation primitives in x86_64 assembly.
//!
//! Implements scalar (and AVX-512) versions of:
//!   - L^∞ landscape distance: max_i |a\[i\] - b\[i\]|
//!   - Betti accumulator: count finite birth/death pairs
//!
//! The AVX-512 path is compiled unconditionally but must only be called after
//! a successful CPUID check (see `ml_engine::detect_avx512()`).

use core::arch::global_asm;

// ---------------------------------------------------------------------------
// Scalar assembly baseline
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .global simd_landscape_distance
    .align 16
simd_landscape_distance:
    // rdi = a (f64*), rsi = b (f64*), rdx = len
    test rdx, rdx
    jz .Lzero_dist

    xorpd xmm0, xmm0        // max_diff = 0.0
    xor rcx, rcx

.Ldist_loop:
    cmp rcx, rdx
    jae .Ldist_done

    movsd xmm1, [rdi + rcx*8]
    subsd xmm1, [rsi + rcx*8]
    // abs(xmm1) via integer masking
    movq rax, xmm1
    shl rax, 1
    shr rax, 1              // clear sign bit = abs for finite f64
    movq xmm1, rax
    ucomisd xmm0, xmm1
    jbe .Ldist_next
    movsd xmm0, xmm1

.Ldist_next:
    inc rcx
    jmp .Ldist_loop

.Ldist_done:
    ret

.Lzero_dist:
    xorpd xmm0, xmm0
    ret

    .global simd_betti_accumulate
    .align 16
simd_betti_accumulate:
    // rdi = barcode ([birth: f64, death: f64]*)
    // rsi = pairs
    // dl  = dim (part of ABI; scalar count ignores it)
    xor rax, rax
    xor rcx, rcx
    test rsi, rsi
    jz .Lbetti_done

.Lbetti_loop:
    cmp rcx, rsi
    jae .Lbetti_done

    mov r9, rcx
    shl r9, 4
    movsd xmm0, [rdi + r9]      // birth
    movsd xmm1, [rdi + r9 + 8]  // death
    ucomisd xmm0, xmm1
    jae .Lbetti_next                // birth >= death -> infinite, skip

    inc rax

.Lbetti_next:
    inc rcx
    jmp .Lbetti_loop

.Lbetti_done:
    ret
"#
);

// ---------------------------------------------------------------------------
// AVX-512 vectorized path (compiled unconditionally; runtime-guarded)
// ---------------------------------------------------------------------------

global_asm!(
    r#"
    .global simd_landscape_distance_avx512
    .align 16
simd_landscape_distance_avx512:
    // rdi = a (f64*), rsi = b (f64*), rdx = len
    test rdx, rdx
    jz .Lzero_avx

    // Build abs mask in zmm2
    xor eax, eax
    dec rax
    shr rax, 1              // rax = 0x7FFFFFFFFFFFFFFF
    movq xmm2, rax
    vpbroadcastq zmm2, xmm2

    vpxorq zmm0, zmm0, zmm0     // max_diff vector = 0

    mov r8, rdx
    and r8, ~7                  // r8 = len rounded down to multiple of 8
    xor rcx, rcx

.Lvec_loop:
    cmp rcx, r8
    jae .Lscalar_tail

    vmovupd zmm3, [rdi + rcx*8]
    vsubpd zmm3, zmm3, [rsi + rcx*8]
    vpandq zmm3, zmm3, zmm2
    vmaxpd zmm0, zmm0, zmm3

    add rcx, 8
    jmp .Lvec_loop

.Lscalar_tail:
    cmp rcx, rdx
    jae .Lreduce

    movsd xmm3, [rdi + rcx*8]
    subsd xmm3, [rsi + rcx*8]
    movq rax, xmm3
    shl rax, 1
    shr rax, 1
    movq xmm3, rax
    vmaxsd xmm0, xmm0, xmm3

    inc rcx
    jmp .Lscalar_tail

.Lreduce:
    // Horizontal max of zmm0 -> xmm0
    vextractf64x4 ymm3, zmm0, 1
    vmaxpd ymm0, ymm0, ymm3
    vextractf128 xmm3, ymm0, 1
    vmaxpd xmm0, xmm0, xmm3
    movsd xmm3, xmm0
    shufpd xmm0, xmm0, 1
    maxsd xmm0, xmm3
    ret

.Lzero_avx:
    vpxorq xmm0, xmm0, xmm0
    ret
"#
);

extern "C" {
    fn simd_landscape_distance(a: *const f64, b: *const f64, len: usize) -> f64;
    fn simd_betti_accumulate(barcode: *const f64, pairs: usize, dim: u8) -> u64;
    fn simd_landscape_distance_avx512(a: *const f64, b: *const f64, len: usize) -> f64;
}

// ---------------------------------------------------------------------------
// Rust wrappers with runtime dispatch
// ---------------------------------------------------------------------------

/// Compute the L^∞ distance between two f64 landscapes.
///
/// Dispatches to AVX-512 if the CPU supports it, otherwise scalar assembly.
pub fn landscape_distance(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let avx512 = crate::ml_engine::detect_avx512();
    unsafe {
        if avx512 {
            simd_landscape_distance_avx512(a.as_ptr(), b.as_ptr(), len)
        } else {
            simd_landscape_distance(a.as_ptr(), b.as_ptr(), len)
        }
    }
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
    // Barcode is [(birth, death)]; we pass it as a flat f64 array.
    let flat = barcode.as_ptr() as *const f64;
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

    pub fn register_all() {
        crate::testing::register_test("topo_asm::distance_identity", test_distance_identity);
        crate::testing::register_test("topo_asm::distance_basic", test_distance_basic);
        crate::testing::register_test("topo_asm::betti_count", test_betti_count);
    }
}
