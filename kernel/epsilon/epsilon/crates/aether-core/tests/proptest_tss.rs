// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow
//
// Property-based tests for the TSS spherical Voronoi index.

use aether_core::SphericalVoronoiIndex;
use proptest::prelude::*;

#[inline]
fn great_circle(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let cos_d = t1.sin() * t2.sin() + t1.cos() * t2.cos() * (p1 - p2).cos();
    cos_d.clamp(-1.0, 1.0).acos()
}

fn fixed_centroids() -> [(f64, f64); 8] {
    let mut c = [(0.0, 0.0); 8];
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut i = 0;
    while i < 8 {
        c[i] = (core::f64::consts::FRAC_PI_2, (i as f64) * (two_pi / 8.0));
        i += 1;
    }
    c
}

proptest! {
    /// For any random query on S², `locate` returns the index whose centroid
    /// has minimum great-circle distance — proving the O(1) walk actually
    /// finds the global minimum.
    #[test]
    fn prop_locate_returns_argmin(
        theta in 0.0f64..core::f64::consts::PI,
        phi in 0.0f64..(2.0 * core::f64::consts::PI),
    ) {
        let centroids = fixed_centroids();
        let idx = SphericalVoronoiIndex::<8>::new(centroids);
        let chosen = idx.locate((theta, phi));
        let chosen_d = great_circle(theta, phi, centroids[chosen].0, centroids[chosen].1);
        for (i, c) in centroids.iter().enumerate() {
            let d = great_circle(theta, phi, c.0, c.1);
            prop_assert!(
                chosen_d <= d + 1e-12,
                "chosen={} d={} other={} d={}",
                chosen,
                chosen_d,
                i,
                d
            );
        }
    }
}
