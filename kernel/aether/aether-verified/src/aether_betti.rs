// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_betti.rs
//!
//! Provenance: Lean 4 (`AetherVerified.Betti`, `lean/AetherVerified/Betti.lean`)
//!
//! # Window-of-4 oscillation count
//!
//! [`oscillation_count`] slides a width-4 window along a byte stream and counts
//! windows whose last value returns within `tol` of the first while a middle
//! value moves further than `tol` away. It is a statistic about the *sequence*.
//!
//! This module used to call the count `betti1_heuristic`, "β̃₁", and to gate it
//! with `betti_error_bound_check`, documented as the Lean-backed bound
//! `β̃₁ ≤ β₁ + (n − 3)`. Neither name held:
//!
//! * **The count is not a homology rank.** It depends on arrival order and grows
//!   with sample count: the period-3 stream `[0, 60, 120, ...]` of length `n`
//!   gives exactly `n − 3`, while β₁ of points on a line is 0 at every scale.
//!   This is the same statistic `aether_core::topology` renamed
//!   `oscillation_count` for the same reason.
//! * **The gate could not fail.** A window at `i` needs `i + 3 < n`, so the
//!   count is at most `n − 3` and `count ≤ b + (n − 3)` held for every input and
//!   every `b`, including `b = 0`. It was removed, together with the identity
//!   function `betti1_bridge_value` that only it called.
//!
//! What remains proven is the counting fact `oscillation_count(data, tol) ≤ n − 3`
//! (`n = data.len()`, saturating), Lean theorem
//! `AetherVerified.Betti.oscillationCount_le_windows`. The bound is attained, so
//! it is the tightest bound that depends only on `n`.

/// Distance between two u8 values (absolute difference).
#[inline]
fn nat_dist_u8(a: u8, b: u8) -> usize {
    a.abs_diff(b) as usize
}

/// (Lean: `oscillationAt`)
///
/// Whether the width-4 window starting at `i` oscillates:
///   1. `|data[i] − data[i+3]| ≤ tol`  (the window returns near its start), and
///   2. `|data[i] − data[i+1]| > tol` or `|data[i] − data[i+2]| > tol`
///      (a middle value moves away).
///
/// Returns `false` when the window does not fit, i.e. `i + 3 ≥ data.len()`.
pub fn oscillation_at(data: &[u8], i: usize, tol: usize) -> bool {
    if i + 3 >= data.len() {
        return false;
    }
    let a = data[i];
    let b = data[i + 1];
    let c = data[i + 2];
    let d = data[i + 3];
    let close = nat_dist_u8(a, d) <= tol;
    let middle_far = nat_dist_u8(a, b) > tol || nat_dist_u8(a, c) > tol;
    close && middle_far
}

/// (Lean: `oscillationCount`)
///
/// Number of window starts `i` at which [`oscillation_at`] holds. O(n), single
/// pass. Never exceeds `data.len().saturating_sub(3)` (Lean:
/// `oscillationCount_le_windows`).
pub fn oscillation_count(data: &[u8], tol: usize) -> usize {
    (0..data.len())
        .filter(|i| oscillation_at(data, *i, tol))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillation_at() {
        // data: [1, 9, 8, 2] — endpoints close (|1-2|=1 ≤ 3), middle far (|1-9|=8 > 3)
        let data = [1u8, 9, 8, 2];
        assert!(oscillation_at(&data, 0, 3));
    }

    #[test]
    fn test_no_oscillation_flat() {
        // All values the same — no middle deviation
        let data = [5u8, 5, 5, 5];
        assert!(!oscillation_at(&data, 0, 3));
    }

    #[test]
    fn test_no_oscillation_endpoints_far() {
        // Endpoints are far apart — the window does not return
        let data = [0u8, 5, 5, 255];
        assert!(!oscillation_at(&data, 0, 3));
    }

    #[test]
    fn test_oscillation_count_positive() {
        let data = [1u8, 9, 8, 2, 7, 6, 1, 2];
        let h = oscillation_count(&data, 3);
        assert!(h > 0, "Expected at least one oscillating window");
    }

    #[test]
    fn test_oscillation_count_bounded_by_window_starts() {
        let data = [1u8, 9, 8, 2, 7, 6, 1, 2];
        let h = oscillation_count(&data, 3);
        let windows = data.len().saturating_sub(3);
        assert!(h <= windows);
    }

    #[test]
    fn test_short_data() {
        let data = [1u8, 2, 3];
        assert_eq!(oscillation_count(&data, 1), 0);
    }
}
