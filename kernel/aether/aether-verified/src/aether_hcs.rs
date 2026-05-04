// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_hcs.rs
//!
//! Provenance: Lean 4 → Rust
//! Source: EpsilonHollow.HyperbolicCapacity
//!
//! # Hyperbolic Capacity Separation (HCS)
//!
//! Theorem 5: δ_hyp ≤ 2/(c·d·D), δ_euc ≥ ln(b)/d·(D−1)

/// Hyperbolic distortion bound: δ_hyp ≤ 2/(c·d·D)
pub fn hyp_distortion_bound(curvature: f64, dim: usize, depth: usize) -> f64 {
    if curvature <= 0.0 || dim == 0 || depth == 0 {
        return f64::INFINITY;
    }
    2.0 / (curvature * dim as f64 * depth as f64)
}

/// Euclidean distortion lower bound: δ_euc ≥ ln(b)/d·(D−1)
pub fn euc_distortion_bound(branching: usize, dim: usize, depth: usize) -> f64 {
    if dim == 0 || depth <= 1 {
        return 0.0;
    }
    libm::log(branching as f64) / dim as f64 * (depth - 1) as f64
}

/// Separation ratio: how many times better hyperbolic is.
/// δ_euc/δ_hyp ≥ c·ln(b)·D·(D−1)/2
pub fn separation_ratio(curvature: f64, branching: usize, depth: usize) -> f64 {
    if depth <= 1 {
        return 1.0;
    }
    curvature * libm::log(branching as f64) * depth as f64 * (depth - 1) as f64 / 2.0
}

/// Verify hyperbolic beats Euclidean for given parameters.
pub fn verify_hcs(curvature: f64, branching: usize, dim: usize, depth: usize) -> bool {
    let d_hyp = hyp_distortion_bound(curvature, dim, depth);
    let d_euc = euc_distortion_bound(branching, dim, depth);
    d_hyp < d_euc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        // b=4, D=10, d=128, c=1.0
        let d_hyp = hyp_distortion_bound(1.0, 128, 10);
        let d_euc = euc_distortion_bound(4, 128, 10);
        assert!(d_hyp < 0.001, "δ_hyp = {}", d_hyp);
        assert!(d_euc > 0.05, "δ_euc = {}", d_euc);
        assert!(verify_hcs(1.0, 4, 128, 10));
    }

    #[test]
    fn test_h100_scale() {
        // d=4096, c=0.1, b=64, D=50
        let d_hyp = hyp_distortion_bound(0.1, 4096, 50);
        let d_euc = euc_distortion_bound(64, 4096, 50);
        let ratio = separation_ratio(0.1, 64, 50);
        assert!(d_hyp < 0.0001, "δ_hyp = {}", d_hyp);
        assert!(ratio > 400.0, "ratio = {}", ratio);
    }

    #[test]
    fn test_separation_ratio() {
        let r = separation_ratio(1.0, 4, 10);
        // c·ln(4)·10·9/2 ≈ 62.4
        assert!(r > 60.0 && r < 65.0, "ratio = {}", r);
    }
}
