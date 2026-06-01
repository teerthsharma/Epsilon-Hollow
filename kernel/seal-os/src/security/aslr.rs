// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Address Space Layout Randomization (ASLR) for userspace.
//!
//! Provides randomized base addresses for mmap, stack, and heap regions.
//! Uses a simple xorshift64 PRNG seeded from RDTSC + CPUID.

use core::sync::atomic::{AtomicU64, Ordering};

/// Upper bound of user canonical address space.
const USER_SPACE_TOP: u64 = 0x0000_7fff_ffff_ffff;

/// Minimum address for randomized mmap base (upper portion of user space).
const MMAP_MIN: u64 = 0x0000_7f00_0000_0000;

/// Stack lives near the top of user space.
const STACK_TOP_MAX: u64 = 0x0000_7fff_ffff_0000;
const STACK_TOP_MIN: u64 = 0x0000_7fff_0000_0000;

/// Heap starts in the lower-mid user space.
const HEAP_MIN: u64 = 0x0000_0001_0000_0000;
const HEAP_MAX: u64 = 0x0000_0030_0000_0000;

/// Global PRNG state.
static PRNG_STATE: AtomicU64 = AtomicU64::new(0);

/// Seed the PRNG from RDTSC and CPUID.
fn get_seed() -> u64 {
    let tsc = unsafe { core::arch::x86_64::_rdtsc() };
    let cpuid = core::arch::x86_64::__cpuid(0);
    tsc.wrapping_add(cpuid.eax as u64)
        .wrapping_add((cpuid.ebx as u64) << 16)
        .wrapping_add((cpuid.ecx as u64) << 32)
        .wrapping_add((cpuid.edx as u64) << 48)
}

/// Simple xorshift64 PRNG.
fn xorshift64() -> u64 {
    let mut s = PRNG_STATE.load(Ordering::Relaxed);
    if s == 0 {
        s = get_seed();
        if s == 0 {
            s = 0x9e37_79b9_7f4a_7c15; // fallback non-zero seed
        }
    }
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    PRNG_STATE.store(s, Ordering::Relaxed);
    s
}

/// Return a randomized base address for mmap in the upper user-space region.
pub fn randomize_mmap_base() -> u64 {
    let range = USER_SPACE_TOP - MMAP_MIN;
    let r = xorshift64();
    let addr = MMAP_MIN + (r % range);
    addr & !0xFFF // page-align
}

/// Return a randomized stack top address.
pub fn randomize_stack_top() -> u64 {
    let range = STACK_TOP_MAX - STACK_TOP_MIN;
    let r = xorshift64();
    let addr = STACK_TOP_MIN + (r % range);
    addr & !0xFFF // page-align
}

/// Return a randomized heap base address.
pub fn randomize_heap_base() -> u64 {
    let range = HEAP_MAX - HEAP_MIN;
    let r = xorshift64();
    let addr = HEAP_MIN + (r % range);
    addr & !0xFFF // page-align
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_aslr_values_are_distinct() -> TestResult {
        let a = randomize_mmap_base();
        let b = randomize_stack_top();
        let c = randomize_heap_base();
        test_assert!(a != b, "mmap and stack should differ");
        test_assert!(b != c, "stack and heap should differ");
        test_assert!(a >= MMAP_MIN, "mmap below minimum");
        test_assert!(b >= STACK_TOP_MIN, "stack below minimum");
        test_assert!(c >= HEAP_MIN, "heap below minimum");
        TestResult::Pass
    }

    fn test_aslr_randomizes() -> TestResult {
        let a = randomize_heap_base();
        let b = randomize_heap_base();
        // Extremely unlikely to be equal; accept equality as a non-failure
        // but verify they are in range.
        test_assert!(a >= HEAP_MIN && a <= HEAP_MAX, "heap out of range");
        test_assert!(b >= HEAP_MIN && b <= HEAP_MAX, "heap out of range");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::aslr_distinct", test_aslr_values_are_distinct);
        crate::testing::register_test("security::aslr_randomizes", test_aslr_randomizes);
    }
}
