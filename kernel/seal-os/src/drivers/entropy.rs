// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Hardware entropy driver using x86_64 RDRAND and RDSEED instructions.

use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

static RDRAND_AVAILABLE: AtomicBool = AtomicBool::new(false);
static RDSEED_AVAILABLE: AtomicBool = AtomicBool::new(false);

/// Initialise entropy support by probing CPUID.
pub fn init() {
    let cpuid1 = core::arch::x86_64::__cpuid(1);
    if (cpuid1.ecx & (1 << 30)) != 0 {
        RDRAND_AVAILABLE.store(true, Ordering::Relaxed);
    }
    let cpuid7 = core::arch::x86_64::__cpuid_count(7, 0);
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
                "setc {ok:l}",
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
                "setc {ok:l}",
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
/// Prefers RDSEED and falls back to RDRAND. Returns `false` if hardware
/// entropy is unavailable or repeatedly fails.
pub fn getrandom(buf: &mut [u8]) -> bool {
    let hw = RDRAND_AVAILABLE.load(Ordering::Relaxed) || RDSEED_AVAILABLE.load(Ordering::Relaxed);

    if !hw {
        return false;
    }

    let mut filled = 0usize;
    while filled + 8 <= buf.len() {
        let Some(val) = rdseed_u64().or_else(rdrand_u64) else {
            return false;
        };
        buf[filled..filled + 8].copy_from_slice(&val.to_ne_bytes());
        filled += 8;
    }
    if filled < buf.len() {
        let Some(val) = rdseed_u64().or_else(rdrand_u64) else {
            return false;
        };
        let bytes = val.to_ne_bytes();
        let remaining = buf.len() - filled;
        buf[filled..].copy_from_slice(&bytes[..remaining]);
    }
    true
}

// ---------------------------------------------------------------------------
// Software fallback, for the rare CPU with neither RDRAND nor RDSEED.
//
// Single shared copy: two callers (net::dns's transaction IDs, net::udp's
// ephemeral ports) both need "hardware entropy, else something better than a
// bare counter" and had each grown their own xorshift64 construction before
// this was pulled out here -- exactly the drift risk a security primitive
// shouldn't have two copies of. `security::aslr` has a separate, unrelated
// xorshift64 seeded the same way for address-space randomization; it is not
// consolidated with this one -- different module, different purpose (memory
// layout, not network-protocol unpredictability), out of this item's scope.
// ---------------------------------------------------------------------------

static FALLBACK_RNG_STATE: Mutex<Option<u64>> = Mutex::new(None);

/// Draw a 64-bit value from the software fallback PRNG (xorshift64, seeded
/// once from RDTSC on first use).
///
/// This is *not* a security-grade fallback, stated plainly rather than left
/// for a future reader to assume: `xorshift64` is linear and fully
/// invertible, so an attacker who observes a few dozen of its (even
/// truncated) outputs can recover the complete 64-bit generator state and
/// then predict every subsequent value it will ever produce, with no
/// brute-forcing required. There is also no reseeding -- the RDTSC sample
/// taken on first use is the *only* entropy this generator ever has, so
/// under sustained observation the effective unpredictability converges
/// toward that one seed's entropy, and on a deterministic virtualized boot
/// (fixed vCPU cycle count before first use) that can be close to zero. It
/// is still strictly better than the bare incrementing counter it replaced
/// (unpredictable from a *single* induced observation, unlike a counter),
/// which is the only property callers here rely on.
///
/// This path only executes when the CPU has neither RDRAND nor RDSEED --
/// essentially no x86_64 hardware manufactured since 2012. Callers should
/// try `getrandom` (or `rdseed_u64().or_else(rdrand_u64)`) first and only
/// reach for this once those report no hardware source available.
pub fn fallback_random_u64() -> u64 {
    let mut state = FALLBACK_RNG_STATE.lock();
    let mut s = state.unwrap_or_else(|| {
        let seed = unsafe { core::arch::x86_64::_rdtsc() };
        if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        }
    });
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = Some(s);
    s
}
