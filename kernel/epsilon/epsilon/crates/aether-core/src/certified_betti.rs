// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Certified β₀: a component count at a scale, or a refusal naming the pair
//! that makes the count ambiguous.
//!
//! Ported from planimeter's gap rule. The single-linkage merge heights of a
//! point set are exactly the n-1 edge weights of its Euclidean minimum
//! spanning tree (Gower & Ross 1969), so β₀ at threshold `t` is `n` minus the
//! number of tree edges shorter than `t`, and it can only change at those
//! n-1 heights.
//!
//! A fixed-threshold count is fragile: two points at distance `scale ± 1e-9`
//! give different integers, and the input does not say which is meant. This
//! module certifies a count only when no merge height lies in the band
//! `[scale / sqrt(ratio), scale * sqrt(ratio)]` — a band `ratio` wide with
//! the scale at its geometric centre. Every threshold in that band, under
//! either `<` or `<=`, gives the same integer, so no perturbation of the
//! threshold by less than a factor of `sqrt(ratio)` can flip it. Otherwise
//! the result is a refusal naming the tree edge whose height is nearest the
//! scale.
//!
//! This is stricter than planimeter's `--grid` rule, which accepts any scale
//! inside a `ratio`-wide gap; that rule would certify a scale sitting 1e-9
//! below a merge height.

use alloc::vec;

/// A β₀ that is either certified stable around the scale, or refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Beta0 {
    /// No merge height lies in the band around the scale.
    Certified {
        /// Number of connected components at every threshold in the band.
        value: u32,
        /// Largest merge height below the band, or `0.0` if there is none.
        gap_lo: f64,
        /// Smallest merge height above the band, or `f64::INFINITY` if there is none.
        gap_hi: f64,
    },
    /// A merge height lies in the band, so the count depends on a choice the
    /// input does not make. `(i, j)` with `i < j` are indices into the input
    /// slice of the spanning-tree edge whose height is nearest the scale.
    Refused {
        /// Smaller index of the ambiguous pair.
        i: usize,
        /// Larger index of the ambiguous pair.
        j: usize,
        /// Euclidean distance between `points[i]` and `points[j]`.
        height: f64,
    },
}

/// β₀ of `points` at Euclidean threshold `scale`, certified to be constant
/// over `[scale / sqrt(ratio), scale * sqrt(ratio)]`, or refused.
///
/// `ratio` below 1 (or NaN) is treated as 1, which certifies whenever no
/// height equals the scale exactly. A non-finite coordinate or a NaN scale
/// yields a refusal whenever there is more than one point.
// ponytail: O(n^2) all-pairs Prim, exact by construction; fine for the
// encoder's 64-point payloads. Upgrade path is a verified EMST (e.g. Delaunay
// in 2-D/3-D) with this Prim kept as the test control.
pub fn certified_beta0<const D: usize>(points: &[[f64; D]], scale: f64, ratio: f64) -> Beta0 {
    let n = points.len();
    let r = libm::sqrt(if ratio > 1.0 { ratio } else { 1.0 });
    let (lo, hi) = (scale / r, scale * r);

    let dist = |a: &[f64; D], b: &[f64; D]| {
        libm::sqrt(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum())
    };

    let mut inside = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut src = vec![0usize; n];
    let mut value = n as u32;
    let (mut gap_lo, mut gap_hi) = (0.0f64, f64::INFINITY);
    let mut nearest: Option<(usize, usize, f64)> = None;

    for k in 0..n {
        // Lowest-index minimum among points not yet in the tree. Points only
        // reachable through a NaN distance keep `best = INFINITY`.
        let mut u = usize::MAX;
        for v in 0..n {
            if !inside[v] && (u == usize::MAX || best[v] < best[u]) {
                u = v;
            }
        }
        inside[u] = true;

        if k > 0 {
            let h = best[u];
            if h < lo {
                value -= 1;
                gap_lo = gap_lo.max(h);
            } else if h > hi && h.is_finite() {
                gap_hi = gap_hi.min(h);
            } else {
                // In the band, or not a measurable distance: ambiguous.
                let closer = match nearest {
                    None => true,
                    Some((_, _, g)) => libm::fabs(h - scale) < libm::fabs(g - scale),
                };
                if closer {
                    let (i, j) = (src[u].min(u), src[u].max(u));
                    nearest = Some((i, j, dist(&points[i], &points[j])));
                }
            }
        }

        for v in 0..n {
            if !inside[v] {
                let d = dist(&points[u], &points[v]);
                if d < best[v] {
                    best[v] = d;
                    src[v] = u;
                }
            }
        }
    }

    match nearest {
        Some((i, j, height)) => Beta0::Refused { i, j, height },
        None => Beta0::Certified {
            value,
            gap_lo,
            gap_hi,
        },
    }
}
