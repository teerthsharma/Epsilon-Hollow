// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_betti.rs
//!
//! Provenance: Lean 4 → C → Rust
//! Source: HeytingLean.Bridge.Sharma.AetherBetti
//!
//! # Betti Number Approximation with Error Bounds
//!
//! Heuristic β̃₁ for Betti-1 (loop detection in data streams).
//! The Lean 4 proof guarantees: `β̃₁ ≤ β₁ + (n − 3)`
//!
//! Heuristic overcounts are bounded by the window overlap factor.
//! Used for online topological feature detection in I/O streams
//! where exact persistent homology is too expensive.

/// Distance between two u8 values (absolute difference).
#[inline]
fn nat_dist_u8(a: u8, b: u8) -> usize {
    a.abs_diff(b) as usize
}

/// (Lean: isDetectedLoop)
///
/// Detect a potential 1-cycle (loop) at position i in the data stream.
///
/// A loop is detected when:
///   1. `|data[i] − data[i+3]| ≤ tol`  (endpoints are close: "closes the loop")
///   2. `|data[i] − data[i+1]| > tol OR |data[i] − data[i+2]| > tol`
///      (middle points deviate: "the loop has area")
///
/// This captures the topological signature of a 1-cycle: a path that
/// returns to its origin after passing through distant regions.
pub fn detected_loop_at(data: &[u8], i: usize, tol: usize) -> bool {
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

/// (Lean: betti1_heuristic)
///
/// Heuristic Betti-1 count: scan data for loop signatures.
/// O(n) single-pass, suitable for streaming.
pub fn betti1_heuristic(data: &[u8], tol: usize) -> usize {
    (0..data.len())
        .filter(|i| detected_loop_at(data, *i, tol))
        .count()
}

/// Runtime bridge value from an exact/proxy Betti provider.
///
/// In production, this would come from a persistent homology library
/// (e.g., Ripser, GUDHI) or a pre-computed database.
pub fn betti1_bridge_value(betti1_exact_or_proxy: usize) -> usize {
    betti1_exact_or_proxy
}

/// (Lean: betti_error_bound)
///
/// Verify the one-sided overlap bound: `β̃₁ ≤ β₁ + (n − 3)`
///
/// The heuristic can overcount loops because overlapping windows may
/// detect the same topological feature multiple times. The maximum
/// overcount is bounded by the window overlap factor (n − 3).
///
/// # Proof sketch (from Lean 4)
/// Each window of width 4 can share at most 3 positions with the next.
/// Therefore, the maximum number of duplicate detections per true cycle
/// is bounded by (n − 3) over the entire stream.
pub fn betti_error_bound_check(
    data: &[u8],
    tol: usize,
    betti1_exact_or_proxy: usize,
) -> bool {
    let h = betti1_heuristic(data, tol);
    let b = betti1_bridge_value(betti1_exact_or_proxy);
    let overlap = data.len().saturating_sub(3);
    h <= b + overlap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detected_loop_at() {
        // data: [1, 9, 8, 2] — endpoints close (|1-2|=1 ≤ 3), middle far (|1-9|=8 > 3)
        let data = [1u8, 9, 8, 2];
        assert!(detected_loop_at(&data, 0, 3));
    }

    #[test]
    fn test_no_loop_flat() {
        // All values the same — no middle deviation
        let data = [5u8, 5, 5, 5];
        assert!(!detected_loop_at(&data, 0, 3));
    }

    #[test]
    fn test_no_loop_endpoints_far() {
        // Endpoints are far apart — not a closed loop
        let data = [0u8, 5, 5, 255];
        assert!(!detected_loop_at(&data, 0, 3));
    }

    #[test]
    fn test_betti1_heuristic_count() {
        let data = [1u8, 9, 8, 2, 7, 6, 1, 2];
        let h = betti1_heuristic(&data, 3);
        assert!(h > 0, "Expected at least one loop detection");
    }

    #[test]
    fn test_betti_error_bound() {
        let data = [1u8, 9, 8, 2, 7, 6, 1, 2];
        assert!(betti_error_bound_check(&data, 3, 5));
    }

    #[test]
    fn test_betti_error_bound_zero_exact() {
        // Even with 0 exact loops, the heuristic is bounded by overlap
        let data = [1u8, 9, 8, 2, 7, 6, 1, 2];
        let h = betti1_heuristic(&data, 3);
        let overlap = data.len().saturating_sub(3);
        assert!(h <= 0 + overlap);
    }

    #[test]
    fn test_short_data() {
        let data = [1u8, 2, 3];
        assert_eq!(betti1_heuristic(&data, 1), 0);
    }
}
