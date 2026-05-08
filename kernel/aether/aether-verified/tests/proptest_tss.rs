// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow
//
// Property-based tests for the O(1) primitives in `aether_tss`.
//
// Each property below corresponds to a closed-form algebraic invariant
// that the Lean source enforces. The tests use `proptest` with the default
// configuration so that shrinking is enabled and seeds are deterministic.

use aether_verified::aether_tss::{
    great_circle_distance, p_max, theta_min_from_epsilon, verify_packing_bound,
};
use proptest::prelude::*;

// θ ∈ (0, π) with a small inner margin so we stay strictly inside the open
// interval — this avoids both the sin(0) singularity (handled in p_max via
// the 1e-12 guard) and the trivial endpoint π.
fn theta_strategy() -> impl Strategy<Value = f64> {
    1e-6f64..(core::f64::consts::PI - 1e-6)
}

// ε ∈ [0, 2] — the valid input range for theta_min_from_epsilon, since the
// arcsin argument is clamped to [-1, 1].
fn epsilon_strategy() -> impl Strategy<Value = f64> {
    0.0f64..2.0f64
}

// Spherical coordinates: θ ∈ [0, π], φ ∈ [0, 2π).
fn point_strategy() -> impl Strategy<Value = (f64, f64)> {
    (
        0.0f64..core::f64::consts::PI,
        0.0f64..(2.0 * core::f64::consts::PI),
    )
}

proptest! {
    /// Property 1: p_max(θ) ≥ 0 for all θ ∈ (0, π).
    /// (4 / sin²(θ/2) is always positive when sin(θ/2) ≠ 0.)
    #[test]
    fn prop_p_max_non_negative(theta in theta_strategy()) {
        let pm = p_max(theta);
        prop_assert!(pm >= 0.0, "p_max({}) = {} should be non-negative", theta, pm);
    }

    /// Property 2: theta_min_from_epsilon(ε) ∈ [0, π] for ε ∈ [0, 2].
    #[test]
    fn prop_theta_min_in_range(eps in epsilon_strategy()) {
        let theta = theta_min_from_epsilon(eps);
        prop_assert!(theta >= 0.0, "θ_min({}) = {} must be ≥ 0", eps, theta);
        prop_assert!(
            theta <= core::f64::consts::PI + 1e-12,
            "θ_min({}) = {} must be ≤ π",
            eps,
            theta
        );
    }

    /// Property 3: great_circle_distance(p, p) ≈ 0 for all points p.
    /// Tolerance is loose because `acos` near ±1 amplifies fp error: a cos
    /// component slightly below 1.0 by O(ε) becomes a distance of O(√ε).
    #[test]
    fn prop_gcd_self_is_zero((t, p) in point_strategy()) {
        let d = great_circle_distance(t, p, t, p);
        prop_assert!(d.abs() < 1e-6, "gcd(p,p) = {} should be ≈ 0", d);
    }

    /// Property 4: great_circle_distance is symmetric.
    #[test]
    fn prop_gcd_symmetric(
        (t1, p1) in point_strategy(),
        (t2, p2) in point_strategy(),
    ) {
        let d_ab = great_circle_distance(t1, p1, t2, p2);
        let d_ba = great_circle_distance(t2, p2, t1, p1);
        prop_assert!(
            (d_ab - d_ba).abs() < 1e-9,
            "gcd asymmetric: {} vs {}",
            d_ab,
            d_ba
        );
    }

    /// Property 5: verify_packing_bound(p, θ) implies p ≤ ⌊p_max(θ)⌋ + 1.
    /// (`verify_packing_bound` returns `(p as f64) <= p_max(θ)`, so any `p`
    /// passing the check must be at most `p_max` rounded up.)
    #[test]
    fn prop_packing_bound_implies_floor_plus_one(
        p in 0usize..10_000,
        theta in theta_strategy(),
    ) {
        if verify_packing_bound(p, theta) {
            let pm = p_max(theta);
            // p ≤ p_max  ⇒  p ≤ ⌊p_max⌋ + 1 (the +1 absorbs floor + fp slack).
            let bound = if pm.is_finite() {
                (libm::floor(pm) as usize).saturating_add(1)
            } else {
                usize::MAX
            };
            prop_assert!(
                p <= bound,
                "p={} passed bound check but exceeds ⌊p_max({})⌋+1 = {}",
                p,
                pm,
                bound
            );
        }
    }
}
