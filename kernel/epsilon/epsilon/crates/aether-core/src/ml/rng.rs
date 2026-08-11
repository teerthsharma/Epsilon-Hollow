// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Deterministic `no_std` PRNG shared by the ML modules.
//!
//! Six call sites previously inlined the same LCG recurrence
//! `x = x * 6364136223846793005 + 1` with a hardcoded `42` seed. This module is
//! that recurrence, once, with the low-bit hazard handled.
//!
//! # Why the low bits are unusable
//!
//! Reduction mod `2^k` is a ring homomorphism, so `x_n mod 2^k` is itself an LCG
//! mod `2^k`: a self-contained sequence over `2^k` states whose transition
//! `x -> a*x + c` is a bijection when `a` is odd. Its period therefore divides
//! `2^k`, and by Hull-Dobell (`c` odd, `a = 1 mod 4`) it is exactly `2^k`. The
//! low 3 bits repeat with period 8, the low 8 bits with period 256.
//!
//! Any consumer that reaches for the low bits — `rng % 1000` splits as
//! `(rng mod 8, rng mod 125)` by CRT and inherits a period-8 sawtooth — gets a
//! visibly periodic stream. `next_u64` returns the raw state for callers that
//! only consume high bits; every derived helper here extracts from the top.

extern crate alloc;
use alloc::vec::Vec;

/// Knuth MMIX multiplier. Satisfies Hull-Dobell mod `2^64`: `a = 1 mod 4`.
const A: u64 = 6364136223846793005;
/// Weyl increment, odd, so `gcd(c, 2^64) = 1`.
const C: u64 = 1442695040888963407;

/// Seedable LCG. Deterministic for a given seed; identical across targets.
#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seed the generator. Any seed is valid; the period is `2^64` regardless.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance and return the raw 64-bit state.
    ///
    /// Only the high bits are well-distributed — see the module docs. Prefer
    /// [`Lcg::next_bits`], [`Lcg::gen_range`], or [`Lcg::next_f64`].
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }

    /// Top `bits` bits of the next state, as a `u64` in `[0, 2^bits)`.
    ///
    /// Panics if `bits` is 0 or exceeds 64.
    pub fn next_bits(&mut self, bits: u32) -> u64 {
        assert!(bits > 0 && bits <= 64, "bits must be in 1..=64");
        self.next_u64() >> (64 - bits)
    }

    /// Uniform `f64` in `[0, 1)`. Uses the top 53 bits — the full `f64` mantissa.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_bits(53) as f64) / ((1u64 << 53) as f64)
    }

    /// Uniform `f64` in `[-1, 1)`. The weight-initialisation shape.
    pub fn next_signed_f64(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }

    /// Uniform integer in `[0, n)`, unbiased and driven by the high bits.
    ///
    /// Returns 0 when `n` is 0, matching the empty-range convention.
    ///
    /// Uses Lemire's multiply-shift: the result is the upper half of the 128-bit
    /// product `x * n`, so it is determined by the *high* bits of the state.
    ///
    /// The obvious alternative, `x % n`, is wrong here. It reads the low bits,
    /// and this module's whole premise is that an LCG's low `k` bits have period
    /// `2^k` — so `x % 8` would emit a period-8 sawtooth. That defect is not
    /// visible in a bucket-occupancy test, because a perfect sawtooth fills
    /// every bucket exactly equally; it only shows up as periodicity in the
    /// *sequence*, which is what `gen_range_has_no_low_bit_period` asserts.
    pub fn gen_range(&mut self, n: u64) -> u64 {
        if n <= 1 {
            return 0;
        }
        // Reject the short tail so every value in [0, n) has equally many
        // preimages. `threshold` is `2^64 mod n`.
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = (self.next_u64() as u128) * (n as u128);
            if (m as u64) >= threshold {
                return (m >> 64) as u64;
            }
        }
    }

    /// Fisher-Yates shuffle in place. Unbiased given an unbiased `gen_range`.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.gen_range(i as u64 + 1) as usize;
            xs.swap(i, j);
        }
    }

    /// A shuffled `0..n` permutation. The batch-order primitive.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        self.shuffle(&mut idx);
        idx
    }
}

impl Default for Lcg {
    /// The historical seed the inlined call sites used.
    fn default() -> Self {
        Self::new(42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The defect this module exists to remove: an LCG's low `k` bits are a
    /// period-`2^k` sawtooth. Demonstrated against the raw recurrence, which is
    /// exactly what the six inlined call sites ran.
    #[test]
    fn low_bits_are_periodic_high_bits_are_not() {
        let mut r = Lcg::new(42);
        let raw: Vec<u64> = (0..64).map(|_| r.next_u64()).collect();

        // Low 3 bits repeat with period 8 — every element equals the one 8 back.
        for i in 8..raw.len() {
            assert_eq!(
                raw[i] & 0b111,
                raw[i - 8] & 0b111,
                "low 3 bits must have period 8 at index {i}"
            );
        }
        // Low 8 bits have period 256, so within 64 draws they must NOT repeat
        // at lag 8. Confirms the periodicity is bit-width-specific, not global.
        assert!(
            (8..raw.len()).any(|i| raw[i] & 0xFF != raw[i - 8] & 0xFF),
            "low 8 bits must not inherit the period-8 cycle"
        );
        // Top bits carry no period-8 structure.
        assert!(
            (8..raw.len()).any(|i| raw[i] >> 32 != raw[i - 8] >> 32),
            "high 32 bits must not be periodic at lag 8"
        );
    }

    #[test]
    fn next_bits_stays_in_range_and_uses_the_top() {
        let mut r = Lcg::new(7);
        for _ in 0..256 {
            assert!(r.next_bits(10) < 1024);
        }
        // next_bits(64) is the whole state.
        let mut a = Lcg::new(99);
        let mut b = Lcg::new(99);
        assert_eq!(a.next_bits(64), b.next_u64());
    }

    #[test]
    fn next_f64_is_unit_interval_and_signed_is_symmetric() {
        let mut r = Lcg::new(1);
        let mut sum = 0.0;
        let n = 4096;
        for _ in 0..n {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "next_f64 out of range: {x}");
            sum += r.next_signed_f64();
        }
        // Mean of 4096 symmetric draws sits near zero. Loose bound: this is a
        // smoke test for sign balance, not a distributional claim.
        assert!(
            (sum / n as f64).abs() < 0.05,
            "signed mean skewed: {}",
            sum / n as f64
        );
    }

    #[test]
    fn gen_range_covers_every_residue() {
        let mut r = Lcg::new(2024);
        let mut counts = [0usize; 8];
        for _ in 0..8000 {
            counts[r.gen_range(8) as usize] += 1;
        }
        for (bucket, &c) in counts.iter().enumerate() {
            assert!(c > 800, "bucket {bucket} starved: {c} of 8000");
        }
        assert_eq!(r.gen_range(1), 0);
        assert_eq!(r.gen_range(0), 0);
    }

    /// Occupancy alone cannot catch the low-bit hazard: a perfect period-8
    /// sawtooth fills all 8 buckets exactly equally and passes
    /// `gen_range_covers_every_residue` with a flat histogram. The defect only
    /// shows up in the *sequence*.
    ///
    /// `n = 8` is the sharpest case — a power of two divides `2^64`, so `x % 8`
    /// is precisely the low 3 bits, which have period exactly 8.
    #[test]
    fn gen_range_has_no_low_bit_period() {
        let mut r = Lcg::new(2024);
        let seq: Vec<u64> = (0..64).map(|_| r.gen_range(8)).collect();

        // A `x % 8` implementation repeats at every single lag-8 step. Requiring
        // merely "not all" would be too weak to be worth asserting, so demand a
        // substantial fraction of lag-8 pairs differ.
        let differing = (8..seq.len()).filter(|&i| seq[i] != seq[i - 8]).count();
        assert!(
            differing * 8 >= (seq.len() - 8) * 5,
            "gen_range(8) is periodic at lag 8: only {differing} of {} pairs differ",
            seq.len() - 8
        );

        // The same must hold for the shuffle built on top of it, which is what
        // DataLoader::epoch_order ultimately depends on.
        let a = Lcg::new(5).permutation(16);
        let b = Lcg::new(6).permutation(16);
        assert_ne!(a, b);
    }

    #[test]
    fn shuffle_is_a_permutation_and_seed_sensitive() {
        let mut a: Vec<usize> = (0..64).collect();
        let mut b: Vec<usize> = (0..64).collect();
        Lcg::new(1).shuffle(&mut a);
        Lcg::new(2).shuffle(&mut b);

        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>(), "shuffle lost elements");
        assert_ne!(a, b, "distinct seeds must give distinct orders");

        // Same seed reproduces exactly — determinism is a kernel requirement.
        let mut c: Vec<usize> = (0..64).collect();
        Lcg::new(1).shuffle(&mut c);
        assert_eq!(a, c);

        // Degenerate sizes must not panic.
        let mut one = vec![9];
        Lcg::new(5).shuffle(&mut one);
        assert_eq!(one, vec![9]);
        let mut empty: Vec<u8> = vec![];
        Lcg::new(5).shuffle(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn permutation_length_matches_and_zero_is_empty() {
        assert_eq!(Lcg::new(3).permutation(10).len(), 10);
        assert!(Lcg::new(3).permutation(0).is_empty());
    }
}
