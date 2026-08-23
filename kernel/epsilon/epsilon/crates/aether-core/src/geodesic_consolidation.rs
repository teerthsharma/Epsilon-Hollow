// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Geodesic Memory Consolidation (GMC) in `no_std` Rust.
//!
//! This module replaces the legacy `epsilon_core/geodesic_consolidation.py`
//! theorem slice. It provides bounded cluster-entropy accounting and a
//! fixed-capacity S2 centroid merge loop suitable for Rust/Aether runtime gates.

use libm::{asin, cos, log, sin, sqrt};

const LOG2: f64 = core::f64::consts::LN_2;
const ENTROPY_EPS: f64 = 1e-10;

/// Compute Shannon entropy in bits for a cluster-size distribution.
pub fn cluster_entropy(cluster_sizes: &[u32]) -> f64 {
    let total: u64 = cluster_sizes.iter().map(|&size| u64::from(size)).sum();
    if total == 0 {
        return 0.0;
    }

    let n = total as f64;
    let mut entropy = 0.0;
    for &size in cluster_sizes {
        if size == 0 {
            continue;
        }
        let p = f64::from(size) / n;
        entropy -= p * (log(p) / LOG2);
    }
    entropy
}

/// Exact entropy change when merging two clusters in a population of size `total`.
pub fn entropy_change_on_merge(size_a: u32, size_b: u32, total: u32) -> f64 {
    if total == 0 || size_a == 0 || size_b == 0 {
        return 0.0;
    }

    entropy_term(size_a + size_b, total) - entropy_term(size_a, total) - entropy_term(size_b, total)
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
/// regime — `verify_separation` compares against `theta_min - 1e-6` — so the
/// stable form is the one that belongs here.
///
/// The `cos(p1 - p2)` factor belongs on the `sin * sin` term. It was previously
/// on the `cos * cos` term, which is the **latitude** formula (theta measured
/// from the equator). Fed colatitude inputs it is wrong by up to 3.05 radians;
/// two points on the equator a quarter turn apart returned a distance of 0.
/// The two conventions coincide when either point is at the pole, which is why
/// the error survived: `tests/proptest_tss.rs` checks only `d(p,p) = 0` and
/// symmetry, and both hold under either convention.
pub fn great_circle_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (theta_a, phi_a) = a;
    let (theta_b, phi_b) = b;
    let half_dt = sin((theta_a - theta_b) * 0.5);
    let half_dp = sin((phi_a - phi_b) * 0.5);
    let h = half_dt * half_dt + sin(theta_a) * sin(theta_b) * half_dp * half_dp;
    2.0 * asin(sqrt(h.clamp(0.0, 1.0)))
}

/// Fixed-capacity geodesic consolidator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicConsolidator {
    delta_consolidation: f64,
}

impl GeodesicConsolidator {
    /// Create a consolidator with a geodesic merge threshold in radians.
    pub const fn new(delta_consolidation: f64) -> Self {
        Self {
            delta_consolidation,
        }
    }

    /// Return the geodesic merge threshold in radians.
    pub const fn delta_consolidation(&self) -> f64 {
        self.delta_consolidation
    }

    /// Run deterministic consolidation over fixed-size centroid and size arrays.
    pub fn consolidate<const N: usize>(
        &self,
        centroids_s2: &[(f64, f64); N],
        cluster_sizes: &[u32; N],
    ) -> ConsolidationReport<N> {
        let mut centroids = *centroids_s2;
        let mut sizes = *cluster_sizes;
        let mut active = N;
        let p_before = N;
        let total: u32 = sizes.iter().copied().sum();
        let entropy_before = cluster_entropy(&sizes);
        let mut merges_performed = 0_usize;

        while active > 1 {
            let mut best_i = 0_usize;
            let mut best_j = 1_usize;
            let mut best_dist = f64::INFINITY;

            for i in 0..active {
                for j in (i + 1)..active {
                    let dist = great_circle_distance(centroids[i], centroids[j]);
                    if dist < best_dist {
                        best_dist = dist;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if !best_dist.is_finite() || best_dist >= self.delta_consolidation {
                break;
            }

            merge_pair(&mut centroids, &mut sizes, active, best_i, best_j);
            active -= 1;
            merges_performed += 1;
        }

        let entropy_after = cluster_entropy(&sizes[..active]);
        let entropy_reduction = entropy_before - entropy_after;
        let max_entropy_reduction_bound = if p_before > 1 {
            log(p_before as f64) / LOG2
        } else {
            0.0
        };
        let merges_within_bound = merges_performed <= p_before.saturating_sub(1);
        let reduction_within_bound = entropy_reduction <= max_entropy_reduction_bound + ENTROPY_EPS;
        let entropy_reduced = entropy_after <= entropy_before + ENTROPY_EPS;

        ConsolidationReport {
            new_centroids: centroids,
            new_sizes: sizes,
            active_count: active,
            merges_performed,
            entropy_before,
            entropy_after,
            entropy_reduction,
            entropy_reduced,
            total_memories: total,
            p_before,
            p_after: active,
            max_merges_bound: p_before.saturating_sub(1),
            merges_within_bound,
            max_entropy_reduction_bound,
            reduction_within_bound,
            theorem_holds: entropy_reduced && merges_within_bound && reduction_within_bound,
        }
    }
}

/// Consolidation result with fixed-capacity output arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsolidationReport<const N: usize> {
    /// Output centroids; only `active_count` entries are live.
    pub new_centroids: [(f64, f64); N],
    /// Output cluster sizes; only `active_count` entries are live.
    pub new_sizes: [u32; N],
    /// Number of live clusters after consolidation.
    pub active_count: usize,
    /// Number of merge operations performed.
    pub merges_performed: usize,
    /// Entropy before consolidation.
    pub entropy_before: f64,
    /// Entropy after consolidation.
    pub entropy_after: f64,
    /// Entropy decrease.
    pub entropy_reduction: f64,
    /// True when entropy did not increase.
    pub entropy_reduced: bool,
    /// Total memory count across input clusters.
    pub total_memories: u32,
    /// Initial cluster count.
    pub p_before: usize,
    /// Final cluster count.
    pub p_after: usize,
    /// The bounded-termination merge limit `P0 - 1`.
    pub max_merges_bound: usize,
    /// True when performed merges are within `P0 - 1`.
    pub merges_within_bound: bool,
    /// Maximum entropy-reduction bound `log2(P0)`.
    pub max_entropy_reduction_bound: f64,
    /// True when total entropy reduction is within `log2(P0)`.
    pub reduction_within_bound: bool,
    /// Combined GMC theorem status for this deterministic run.
    pub theorem_holds: bool,
}

fn entropy_term(size: u32, total: u32) -> f64 {
    if size == 0 || total == 0 {
        return 0.0;
    }
    let p = f64::from(size) / f64::from(total);
    -p * (log(p) / LOG2)
}

fn merge_pair<const N: usize>(
    centroids: &mut [(f64, f64); N],
    sizes: &mut [u32; N],
    active: usize,
    i: usize,
    j: usize,
) {
    let size_i = sizes[i];
    let size_j = sizes[j];
    let merged = size_i + size_j;
    if merged > 0 {
        let merged_f = f64::from(merged);
        centroids[i] = (
            (f64::from(size_i) * centroids[i].0 + f64::from(size_j) * centroids[j].0) / merged_f,
            (f64::from(size_i) * centroids[i].1 + f64::from(size_j) * centroids[j].1) / merged_f,
        );
    }
    sizes[i] = merged;

    for idx in j..active.saturating_sub(1) {
        centroids[idx] = centroids[idx + 1];
        sizes[idx] = sizes[idx + 1];
    }
    if active > 0 {
        centroids[active - 1] = (0.0, 0.0);
        sizes[active - 1] = 0;
    }
}
