// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Hardware entropy driver using x86_64 RDRAND and RDSEED instructions.

use core::sync::atomic::{AtomicBool, Ordering};

static RDRAND_AVAILABLE: AtomicBool = AtomicBool::new(false);
static RDSEED_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Initialise entropy support by probing CPUID.
pub fn init() {
    let cpuid1 = unsafe { core::arch::x86_64::__cpuid(1) };
    if (cpuid1.ecx & (1 << 30)) != 0 {
        RDRAND_AVAILABLE.store(true, Ordering::Relaxed);
    }
    let cpuid7 = unsafe { core::arch::x86_64::__cpuid_count(7, 0) };
    if (cpuid7.ebx & (1 << 18)) != 0 {
        RDSEED_AVAILABLE.store(true, Ordering::Relaxed);
    }
}

/// Read a 64-bit value from RDRAND.
///
/// Retries up to 10 times to handle hardware carry-clear failures.
pub fn rdrand_u64() -> Option<u64> {
    if !RDRAND_AVAILABLE.load(Ordering::Relaxed) {
        return None;
    }
    for _ in 0..10 {
        let mut val: u64 = 0;
        let mut ok: u64 = 0;
        unsafe {
            core::arch::asm!(
                "rdrand {val}",
                "setc {ok}",
                val = out(reg) val,
                ok = inout(reg) ok,
                options(nomem, nostack),
            );
        }
        if ok != 0 {
            return Some(val);
        }
    }
    None
}

/// Read a 64-bit value from RDSEED.
///
/// Retries up to 10 times.
pub fn rdseed_u64() -> Option<u64> {
    if !RDSEED_AVAILABLE.load(Ordering::Relaxed) {
        return None;
    }
    for _ in 0..10 {
        let mut val: u64 = 0;
        let mut ok: u64 = 0;
        unsafe {
            core::arch::asm!(
                "rdseed {val}",
                "setc {ok}",
                val = out(reg) val,
                ok = inout(reg) ok,
                options(nomem, nostack),
            );
        }
        if ok != 0 {
            return Some(val);
        }
    }
    None
}

/// Fill `buf` with random bytes.
///
/// Prefers RDSEED, falls back to RDRAND, and finally to a ticks-based
/// deterministic LCG when neither instruction is available.
pub fn getrandom(buf: &mut [u8]) -> bool {
    let hw = RDRAND_AVAILABLE.load(Ordering::Relaxed)
        || RDSEED_AVAILABLE.load(Ordering::Relaxed);

    if hw {
        let mut ok = true;
        let mut filled = 0usize;
        while filled + 8 <= buf.len() {
            if let Some(val) = rdseed_u64().or_else(rdrand_u64) {
                buf[filled..filled + 8].copy_from_slice(&val.to_ne_bytes());
                filled += 8;
            } else {
                ok = false;
                break;
            }
        }
        if ok && filled < buf.len() {
            if let Some(val) = rdseed_u64().or_else(rdrand_u64) {
                let bytes = val.to_ne_bytes();
                let remaining = buf.len() - filled;
                buf[filled..].copy_from_slice(&bytes[..remaining]);
                filled = buf.len();
            } else {
                ok = false;
            }
        }
        if ok && filled == buf.len() {
            return true;
        }
    }

    // Honest fallback: no hardware RNG available or hardware failed.
    // Uses a simple LCG seeded from the timer tick counter.
    let ticks = crate::drivers::interrupts::ticks();
    let mut seed = ticks.wrapping_mul(0x9e3779b97f4a7c15);
    for i in 0..buf.len() {
        seed = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(i as u64);
        buf[i] = (seed >> (i % 8)) as u8;
    }
    true
}
