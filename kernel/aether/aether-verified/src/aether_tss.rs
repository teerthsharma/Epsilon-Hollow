// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

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
#[inline]
pub fn p_max(theta_min: f64) -> f64 {
    let sin_half = libm::sin(theta_min / 2.0);
    if libm::fabs(sin_half) < 1e-12 {
        return f64::INFINITY;
    }
    4.0 / (sin_half * sin_half)
}

/// Compute minimum angular separation from adaptive ε.
///
/// θ_min = 2 arcsin(ε / 2)
pub fn theta_min_from_epsilon(epsilon_adaptive: f64) -> f64 {
    let arg = (epsilon_adaptive / 2.0).clamp(-1.0, 1.0);
    2.0 * libm::asin(arg)
}

/// Chebyshev-bounded adaptive ε.
///
/// ε = μ + k·σ,  P(|X−μ| ≥ kσ) ≤ 1/k²
pub fn epsilon_from_chebyshev(mu: f64, sigma: f64, k: f64) -> f64 {
    mu + k * sigma
}

/// Great-circle distance on S2 between two points given in the **colatitude**
/// convention: `theta` is measured from the north pole and lies in `[0, pi]`,
/// `phi` is the azimuth.
///
/// For unit vectors `[sin t cos p, sin t sin p, cos t]` — which is what
/// `aether-core`'s spherical index builds — the cosine of the central angle is
///
/// ```text
/// cos d = cos(t1) cos(t2) + sin(t1) sin(t2) cos(p1 - p2)
/// ```
///
///
/// # Numerical form
///
/// Computed by the haversine identity rather than by `acos` of the dot product.
/// `acos` has infinite derivative at 1, so the dot-product form loses roughly
/// half its significant digits as the separation approaches zero. Measured
/// against exact meridian separations:
///
/// | true separation | acos form | relative error | haversine | relative error |
/// | ---: | ---: | ---: | ---: | ---: |
/// | 1e-2 | 1.000000e-2 | 1.4e-13 | 1.000000e-2 | 8.7e-16 |
/// | 1e-4 | 1.000000e-4 | 2.6e-09 | 1.000000e-4 | 1.1e-13 |
/// | 1e-6 | 9.998224e-7 | **1.8e-04** | 1.000000e-6 | 8.2e-11 |
/// | 1e-8 | **0.0** | **100%** | 1.000000e-8 | 6.1e-09 |
///
/// The dot-product form returned exactly zero for points 1e-8 apart, reporting
/// distinct points as coincident, and gave `d(p, p)` up to 2.1e-8 over 200,000
/// random points instead of 0. Separation checks operate precisely in that
/// regime — `verify_separation` refuses a pair within its rounding radius of
/// `theta_min` — so the stable form is the one that belongs here.
///
/// The `cos(p1 - p2)` factor belongs on the `sin * sin` term. It was previously
/// on the `cos * cos` term, which is the **latitude** formula (theta measured
/// from the equator). Fed colatitude inputs it is wrong by up to 3.05 radians;
/// two points on the equator a quarter turn apart returned a distance of 0.
/// The two conventions coincide when either point is at the pole, which is why
/// the error survived: `tests/proptest_tss.rs` checks only `d(p,p) = 0` and
/// symmetry, and both hold under either convention.
pub fn great_circle_distance(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let half_dt = libm::sin((t1 - t2) * 0.5);
    let half_dp = libm::sin((p1 - p2) * 0.5);
    let h = half_dt * half_dt + libm::sin(t1) * libm::sin(t2) * half_dp * half_dp;
    2.0 * libm::asin(libm::sqrt(h.clamp(0.0, 1.0)))
}

/// Verify the TSS packing bound: P ≤ P_max.
pub fn verify_packing_bound(p: usize, theta_min: f64) -> bool {
    (p as f64) <= p_max(theta_min)
}

/// Bound on `|great_circle_distance(t1, p1, t2, p2) - exact distance|`.
///
/// The two half-angle differences round once each, moving their `sin` by at
/// most `u·|x|`; libm's `sin` is within an ulp; the products, sum and `sqrt`
/// add a few `u` more. So the `h` whose root is taken is within
/// `Δ = 8u·(8 + |t1| + |t2| + |p1| + |p2|)` of the exact one, with room to
/// spare. `2·asin(√h)` moves by at most `2·asin(√Δ) ≤ π√Δ` under that, the
/// worst case being the antipode, where its slope is unbounded; the slack
/// between the two covers `asin`'s own ulp and this line's rounding.
fn great_circle_radius(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let angles = libm::fabs(t1) + libm::fabs(t2) + libm::fabs(p1) + libm::fabs(p2);
    core::f64::consts::PI * libm::sqrt(4.0 * f64::EPSILON * (8.0 + angles))
}

/// Verify every centroid pair is certified more than θ_min apart.
///
/// A pair counts only when its distance minus its rounding radius still
/// exceeds θ_min. A pair within that radius of θ_min, or at a NaN distance, is
/// refused, not certified.
pub fn verify_separation(centroids: &[(f64, f64)], theta_min: f64) -> bool {
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let (t1, p1) = centroids[i];
            let (t2, p2) = centroids[j];
            let d = great_circle_distance(t1, p1, t2, p2);
            // from teerthsharma/separatrix separatrix/api.py:365: threshold trit
            let certified = d - great_circle_radius(t1, p1, t2, p2) > theta_min;
            if !certified {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

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
    fn separation_is_certified_only_beyond_the_rounding_radius() {
        use core::f64::consts::FRAC_PI_2;
        // 0.4999995 apart is short of 0.5, and was certified inside the old
        // `theta_min - 1e-6` slack.
        assert!(!verify_separation(&[(0.5, 0.0), (0.999_999_5, 0.0)], 0.5));

        // Two equator points pi - 1e-9 apart (to 1e-32). The haversine rounds
        // sin^2 of the half-angle to 1 and returns fl(pi), so a theta_min
        // between the two is short of the true separation yet below the
        // computed one.
        let phi = core::f64::consts::PI - 1e-9;
        let theta_min = phi + 5e-10;
        assert!(great_circle_distance(FRAC_PI_2, 0.0, FRAC_PI_2, phi) > theta_min);
        assert!(!verify_separation(
            &[(FRAC_PI_2, 0.0), (FRAC_PI_2, phi)],
            theta_min
        ));

        // A NaN distance decides nothing.
        assert!(!verify_separation(&[(f64::NAN, 0.0), (1.0, 0.0)], 0.1));

        // Control: 0.6 apart clears 0.5 by far more than any rounding.
        assert!(verify_separation(&[(0.5, 0.0), (1.1, 0.0)], 0.5));
    }

    #[test]
    fn test_packing_bound_holds() {
        let theta = theta_min_from_epsilon(0.3);
        let pm = p_max(theta);
        assert!(verify_packing_bound(10, theta));
        assert!(10.0 <= pm);
    }
}
