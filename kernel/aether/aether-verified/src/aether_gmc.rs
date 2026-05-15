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

/// Shannon entropy of a discrete distribution (bits).
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

/// Renyi entropy of order α for a discrete distribution (bits).
///
/// H_α(X) = 1/(1−α) · log₂(Σ pᵢ^α)
///
/// For α = 1, falls back to Shannon entropy (the limit).
/// The paper (T3) specifies α > 1 for the strict decrease property.
pub fn renyi_entropy(sizes: &[usize], alpha: f64) -> f64 {
    let n: usize = sizes.iter().sum();
    if n == 0 {
        return 0.0;
    }
    if (alpha - 1.0).abs() < 1e-12 {
        return shannon_entropy(sizes);
    }
    let n_f = n as f64;
    let sum_p_alpha: f64 = sizes
        .iter()
        .filter(|&&s| s > 0)
        .map(|&s| libm::pow(s as f64 / n_f, alpha))
        .sum();
    if sum_p_alpha <= 0.0 {
        return 0.0;
    }
    libm::log2(sum_p_alpha) / (1.0 - alpha)
}

/// Verify Renyi entropy non-increase when merging two clusters.
///
/// Constructs a concrete distribution with clusters of sizes `a`, `b`, and
/// `n - a - b` (the remainder), computes H_α before and after merging
/// clusters `a` and `b`, and checks ΔH_α ≤ 0.
///
/// For α > 1 this always holds by convexity of x^α:
///   (p_a + p_b)^α ≥ p_a^α + p_b^α, so Σ pᵢ^α increases,
///   and since 1/(1−α) < 0, H_α = log₂(Σ pᵢ^α)/(1−α) decreases.
pub fn verify_entropy_nonincreasing(a: usize, b: usize, n: usize) -> bool {
    verify_renyi_nonincreasing(a, b, n, 2.0)
}

/// Verify Renyi entropy decrease for a specific α.
pub fn verify_renyi_nonincreasing(a: usize, b: usize, n: usize, alpha: f64) -> bool {
    if n == 0 || a == 0 || b == 0 || a + b > n {
        return true;
    }
    let remainder = n - a - b;
    let before_3 = [a, b, remainder];
    let before_2 = [a, b];
    let after_2 = [a + b, remainder];
    let after_1 = [a + b];
    let (before, after): (&[usize], &[usize]) = if remainder > 0 {
        (&before_3, &after_2)
    } else {
        (&before_2, &after_1)
    };
    renyi_entropy(after, alpha) <= renyi_entropy(before, alpha) + 1e-10
}

/// Maximum merges before termination: P₀ − 1.
pub fn max_merges(p_initial: usize) -> usize {
    if p_initial > 0 {
        p_initial - 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[test]
    fn test_entropy_decreases() {
        assert!(verify_entropy_nonincreasing(100, 50, 1000));
        assert!(verify_entropy_nonincreasing(1, 1, 100));
        assert!(verify_entropy_nonincreasing(500, 500, 1000));
    }

    #[test]
    fn test_renyi_decreases_multiple_alpha() {
        for &alpha in &[1.5, 2.0, 3.0, 10.0] {
            assert!(
                verify_renyi_nonincreasing(100, 50, 1000, alpha),
                "failed for α={alpha}"
            );
            assert!(
                verify_renyi_nonincreasing(1, 1, 100, alpha),
                "failed for α={alpha}"
            );
            assert!(
                verify_renyi_nonincreasing(500, 500, 1000, alpha),
                "failed for α={alpha}"
            );
        }
    }

    #[test]
    fn test_renyi_alpha_1_equals_shannon() {
        let sizes = vec![100, 200, 300, 400];
        let h_s = shannon_entropy(&sizes);
        let h_r = renyi_entropy(&sizes, 1.0);
        assert!((h_s - h_r).abs() < 1e-10, "Shannon={h_s} Renyi(1)={h_r}");
    }

    #[test]
    fn test_renyi_uniform_distribution() {
        // For uniform distribution, all Renyi entropies equal log₂(k)
        let uniform = vec![100; 10];
        let expected = libm::log2(10.0);
        for &alpha in &[0.5, 1.0, 2.0, 5.0] {
            let h = renyi_entropy(&uniform, alpha);
            assert!(
                (h - expected).abs() < 0.01,
                "α={alpha}: H={h} expected={expected}"
            );
        }
    }

    #[test]
    fn test_renyi_monotone_in_alpha() {
        // H_α is non-increasing in α for fixed distribution
        let sizes = vec![100, 50, 200, 150, 500];
        let h1 = renyi_entropy(&sizes, 1.0);
        let h2 = renyi_entropy(&sizes, 2.0);
        let h5 = renyi_entropy(&sizes, 5.0);
        assert!(h1 >= h2 - 1e-10, "H_1={h1} < H_2={h2}");
        assert!(h2 >= h5 - 1e-10, "H_2={h2} < H_5={h5}");
    }

    #[test]
    fn test_bounded_termination() {
        assert_eq!(max_merges(10), 9);
        assert_eq!(max_merges(1), 0);
        assert_eq!(max_merges(1000), 999);
    }

    #[test]
    fn test_entropy_values() {
        let uniform = vec![100; 10];
        let h = shannon_entropy(&uniform);
        assert!((h - libm::log2(10.0)).abs() < 0.01);

        let single = vec![1000];
        assert!(shannon_entropy(&single).abs() < 1e-10);
    }
}
