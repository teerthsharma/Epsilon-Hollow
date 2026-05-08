// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow
//
// Property-based tests for the Spectral Contraction Mapping operator
// (Theorem 2 of the Epsilon-Hollow spec).

use aether_core::SpectralContractionOperator;
use proptest::prelude::*;

fn l2_diff_3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..3 {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

proptest! {
    /// For α ∈ (0, 1) and arbitrary state and pred, `apply(state, pred)` lies
    /// componentwise on the line segment between state and pred.
    #[test]
    fn prop_apply_is_in_segment(
        alpha in 0.0f64..1.0f64,
        s0 in -1e3f64..1e3f64,
        s1 in -1e3f64..1e3f64,
        s2 in -1e3f64..1e3f64,
        p0 in -1e3f64..1e3f64,
        p1 in -1e3f64..1e3f64,
        p2 in -1e3f64..1e3f64,
    ) {
        let scm = SpectralContractionOperator::<3>::new(alpha);
        let s = [s0, s1, s2];
        let p = [p0, p1, p2];
        let out = scm.apply(&s, &p);
        for i in 0..3 {
            let lo = s[i].min(p[i]);
            let hi = s[i].max(p[i]);
            prop_assert!(
                out[i] >= lo - 1e-9 && out[i] <= hi + 1e-9,
                "component {} not in [{}, {}]: got {}",
                i, lo, hi, out[i]
            );
        }
    }

    /// Iterating with fixed pred 50 times yields strictly smaller error than the
    /// initial error (contraction property), provided α ∈ (0, 1) and the initial
    /// error is non-trivial.
    #[test]
    fn prop_iteration_strictly_contracts(
        alpha in 0.01f64..0.99f64,
        s0 in -100.0f64..100.0f64,
        s1 in -100.0f64..100.0f64,
        s2 in -100.0f64..100.0f64,
        p0 in -100.0f64..100.0f64,
        p1 in -100.0f64..100.0f64,
        p2 in -100.0f64..100.0f64,
    ) {
        let s = [s0, s1, s2];
        let p = [p0, p1, p2];
        let err_0 = l2_diff_3(&s, &p);
        prop_assume!(err_0 > 1e-6);

        let scm = SpectralContractionOperator::<3>::new(alpha);
        let after = scm.iterate(s, p, 50);
        let err_t = l2_diff_3(&after, &p);
        prop_assert!(
            err_t < err_0,
            "err_0 = {}, err_50 = {}",
            err_0,
            err_t
        );
    }
}
