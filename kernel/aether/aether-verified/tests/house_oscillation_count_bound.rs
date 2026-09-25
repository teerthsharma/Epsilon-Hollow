// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! The window-of-4 statistic in `aether_betti` is an oscillation count, not a
//! Betti number, and the gate that compared it against one could not fail.
//!
//! `betti_error_bound_check(data, tol, b)` returned `h <= b + (n - 3)` where
//! `h` is this count. A window at index `i` needs `i + 3 < n`, so there are at
//! most `n - 3` of them and `h <= n - 3 <= b + (n - 3)` for every `b >= 0`,
//! including `b = 0`. No input could make it return `false`, so it was removed
//! rather than repaired: there is no exact β₁ in the crate to compare against.
//!
//! What survives is the counting fact itself, `h <= n - 3`, proved in Lean as
//! `AetherVerified.Betti.oscillationCount_le_windows`. The tests below check it
//! exhaustively on small inputs and show that it is attained, so no bound that
//! depends only on `n` can be tighter.

use aether_verified::aether_betti::oscillation_count;

/// Every byte string of length 0..=8 over an alphabet whose values are pairwise
/// further apart than `tol = 3`, plus two tolerances that merge some of them.
/// 9,841 strings per tolerance.
#[test]
fn oscillation_count_never_exceeds_the_number_of_window_starts() {
    const ALPHABET: [u8; 3] = [0, 10, 20];
    for tol in [0usize, 3, 15] {
        for len in 0usize..=8 {
            let total = 3usize.pow(len as u32);
            for code in 0..total {
                let mut c = code;
                let data: Vec<u8> = (0..len)
                    .map(|_| {
                        let v = ALPHABET[c % 3];
                        c /= 3;
                        v
                    })
                    .collect();
                let h = oscillation_count(&data, tol);
                assert!(
                    h <= len.saturating_sub(3),
                    "count {h} exceeds {} window starts for {data:?} at tol {tol}",
                    len.saturating_sub(3)
                );
            }
        }
    }
}

/// The bound is attained. For a period-3 repetition of values pairwise further
/// apart than the tolerance, every window closes (`w[0] == w[3]`) and has a far
/// middle (`w[1] != w[0]`), so the count is exactly `len - 3`.
///
/// This is also why the statistic is not a homology rank: the same three
/// points sampled more times give a larger value, and β₁ of points on a line
/// is 0 at every scale.
#[test]
fn oscillation_count_attains_len_minus_three_on_a_period_three_pattern() {
    for len in 3usize..=64 {
        let data: Vec<u8> = [0u8, 60, 120].iter().cycle().take(len).copied().collect();
        assert_eq!(oscillation_count(&data, 3), len - 3, "len {len}");
    }
}
