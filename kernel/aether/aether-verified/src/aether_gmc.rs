// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_gmc.rs
//!
//! Provenance: Lean 4 → Rust
//! Source: EpsilonHollow.GeodesicConsolidation
//!
//! # Geodesic Memory Consolidation (GMC)
//!
//! Theorem 3: Entropy-reducing cluster merge.
//! ΔH ≤ 0 for every merge operation.

/// Shannon entropy of a discrete distribution (nats).
pub fn shannon_entropy(sizes: &[usize]) -> f64 {
    let n: usize = sizes.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let mut h = 0.0;
    for &s in sizes {
        if s > 0 {
            let p = s as f64 / n_f;
            h -= p * libm::log2(p);
        }
    }
    h
}

/// Compute entropy change when merging clusters of size a and b.
/// Returns a value ≤ 0 (entropy decreases or stays same).
pub fn entropy_change_on_merge(a: usize, b: usize, n: usize) -> f64 {
    if n == 0 || a == 0 || b == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let merged = (a + b) as f64;
    let a_f = a as f64;
    let b_f = b as f64;

    let neg_xlnx = |x: f64| -> f64 {
        if x <= 0.0 { 0.0 } else { -(x / n_f) * libm::log2(x / n_f) }
    };

    neg_xlnx(merged) - neg_xlnx(a_f) - neg_xlnx(b_f)
}

/// Verify: entropy never increases after merge.
pub fn verify_entropy_nonincreasing(a: usize, b: usize, n: usize) -> bool {
    entropy_change_on_merge(a, b, n) <= 1e-10
}

/// Maximum merges before termination: P₀ − 1.
pub fn max_merges(p_initial: usize) -> usize {
    if p_initial > 0 { p_initial - 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_decreases() {
        // Merging two clusters should not increase entropy
        assert!(verify_entropy_nonincreasing(100, 50, 1000));
        assert!(verify_entropy_nonincreasing(1, 1, 100));
        assert!(verify_entropy_nonincreasing(500, 500, 1000));
    }

    #[test]
    fn test_bounded_termination() {
        assert_eq!(max_merges(10), 9);
        assert_eq!(max_merges(1), 0);
        assert_eq!(max_merges(1000), 999);
    }

    #[test]
    fn test_entropy_values() {
        // Uniform distribution: max entropy
        let uniform = vec![100; 10]; // 10 clusters, 100 each
        let h = shannon_entropy(&uniform);
        assert!((h - libm::log2(10.0)).abs() < 0.01);

        // Single cluster: zero entropy
        let single = vec![1000];
        assert!(shannon_entropy(&single).abs() < 1e-10);
    }
}
