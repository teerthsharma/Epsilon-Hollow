// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_chebyshev.rs
//!
//! Provenance: Lean 4 → C → Rust
//! Source: HeytingLean.Bridge.Sharma.AetherChebyshev
//!
//! # Chebyshev GC Guard Safety
//!
//! Chebyshev's inequality: `|{i : f_i ≤ f̄ − kσ}| ≤ n/k²`
//!
//! At most 1/k² false collection rate (25% at k = 2).
//! Used to bound the number of memory entries that can be safely
//! reclaimed during garbage collection without losing critical data.

/// Compute arithmetic mean.
fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / (x.len() as f64)
    }
}

/// Compute population standard deviation.
fn stddev(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let m = mean(x);
    let v = x
        .iter()
        .map(|v| {
            let d = *v - m;
            d * d
        })
        .sum::<f64>()
        / (x.len() as f64);
    libm::sqrt(v)
}

/// (Lean: reclaimable)
///
/// Count entries with value ≤ μ − kσ (reclaimable under Chebyshev bound).
pub fn reclaimable_count(x: &[f64], k: f64) -> usize {
    let m = mean(x);
    let sd = stddev(x);
    let thr = m - k * sd;
    x.iter().filter(|v| **v <= thr).count()
}

/// (Lean: chebyshev_finite)
///
/// Verify that the Chebyshev bound holds for the given data.
///
/// # Invariant
/// `reclaimable_count(x, k) ≤ n/k² + ε`  (ε for floating-point tolerance)
///
/// This is a runtime check that the theoretical bound holds for the actual data.
/// If it fails, the data distribution has a pathological shape that violates
/// the assumptions (should never happen for well-formed inputs).
pub fn chebyshev_guard_check(x: &[f64], k: f64) -> bool {
    if x.is_empty() || k <= 0.0 {
        return false;
    }
    let sd = stddev(x);
    if sd <= 0.0 {
        return true; // All values identical, nothing to reclaim
    }
    let reclaim = reclaimable_count(x, k) as f64;
    reclaim <= (x.len() as f64) / (k * k) + 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_basic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&x) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stddev_basic() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = stddev(&x);
        assert!((sd - 2.0).abs() < 0.1, "stddev={}", sd);
    }

    #[test]
    fn test_guard_check_normal_data() {
        let x = [1.0, 2.0, 3.0, 4.0, 10.0, 11.0];
        assert!(chebyshev_guard_check(&x, 2.0));
    }

    #[test]
    fn test_guard_check_uniform() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(chebyshev_guard_check(&x, 2.0));
        assert!(chebyshev_guard_check(&x, 3.0));
    }

    #[test]
    fn test_reclaimable_includes_threshold_equality() {
        let x = [0.0, 2.0, 4.0];
        // mean = 2, stddev = sqrt(8/3) ≈ 1.633
        // At k=0, threshold = mean - 0*sd = 2.0, values ≤ 2.0: {0, 2} = 2
        assert_eq!(reclaimable_count(&x, 0.0), 2);
    }

    #[test]
    fn test_empty_data() {
        assert!(!chebyshev_guard_check(&[], 2.0));
    }

    #[test]
    fn test_zero_k() {
        let x = [1.0, 2.0, 3.0];
        assert!(!chebyshev_guard_check(&x, 0.0));
    }
}
