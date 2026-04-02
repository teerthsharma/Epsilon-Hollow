//! aether_tss.rs
//!
//! Provenance: Lean 4 → Rust
//! Source: EpsilonHollow.TopologicalStateSync
//!
//! # Topological State Synchronization (TSS)
//!
//! Theorem 1: The O(1) data transfer bound.
//! P_max = 4 / sin²(θ_min/2)
//! θ_min = 2 arcsin(ε_adaptive / 2)

/// Compute maximum cluster count from spherical cap packing.
///
/// P_max = 4 / sin²(θ_min / 2)
pub fn p_max(theta_min: f64) -> f64 {
    let sin_half = libm::sin(theta_min / 2.0);
    if sin_half.abs() < 1e-12 {
        return f64::INFINITY;
    }
    4.0 / (sin_half * sin_half)
}

/// Compute minimum angular separation from adaptive ε.
///
/// θ_min = 2 arcsin(ε / 2)
pub fn theta_min_from_epsilon(epsilon_adaptive: f64) -> f64 {
    let arg = (epsilon_adaptive / 2.0).max(-1.0).min(1.0);
    2.0 * libm::asin(arg)
}

/// Chebyshev-bounded adaptive ε.
///
/// ε = μ + k·σ,  P(|X−μ| ≥ kσ) ≤ 1/k²
pub fn epsilon_from_chebyshev(mu: f64, sigma: f64, k: f64) -> f64 {
    mu + k * sigma
}

/// Great-circle distance on S² between two points (θ₁,φ₁) and (θ₂,φ₂).
pub fn great_circle_distance(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let cos_d = libm::sin(t1) * libm::sin(t2)
        + libm::cos(t1) * libm::cos(t2) * libm::cos(p1 - p2);
    libm::acos(cos_d.max(-1.0).min(1.0))
}

/// Verify the TSS packing bound: P ≤ P_max.
pub fn verify_packing_bound(p: usize, theta_min: f64) -> bool {
    (p as f64) <= p_max(theta_min)
}

/// Verify all centroid pairs satisfy separation ≥ θ_min.
pub fn verify_separation(centroids: &[(f64, f64)], theta_min: f64) -> bool {
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let (t1, p1) = centroids[i];
            let (t2, p2) = centroids[j];
            if great_circle_distance(t1, p1, t2, p2) < theta_min - 1e-6 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p_max_basic() {
        // θ_min = π/6 (30°), sin(π/12) ≈ 0.2588
        // P_max = 4 / 0.2588² ≈ 59.7
        let pm = p_max(core::f64::consts::FRAC_PI_6);
        assert!(pm > 50.0 && pm < 70.0, "P_max = {}", pm);
    }

    #[test]
    fn test_theta_from_epsilon() {
        let eps = 0.5;
        let theta = theta_min_from_epsilon(eps);
        assert!(theta > 0.0, "θ_min must be positive");
        assert!(theta < core::f64::consts::PI, "θ_min must be < π");
    }

    #[test]
    fn test_separation() {
        // Two centroids far apart
        let centroids = vec![(0.5, 0.0), (2.5, 3.0)];
        assert!(verify_separation(&centroids, 0.1));
    }

    #[test]
    fn test_packing_bound_holds() {
        let theta = theta_min_from_epsilon(0.3);
        let pm = p_max(theta);
        assert!(verify_packing_bound(10, theta));
        assert!(10.0 <= pm);
    }
}
