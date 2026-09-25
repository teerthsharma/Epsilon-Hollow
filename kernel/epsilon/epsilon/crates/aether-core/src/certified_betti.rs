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
//! Heights are computed with a scale-safe norm, `m * sqrt(sum((d / m)^2))`
//! with `m` the largest coordinate difference, so separations near `1e-170`
//! or `1e170` are neither squared to zero nor to infinity. The band test runs
//! on those computed heights: an edge within a few ulp of a band end is
//! classified by its rounded value, which may sit on the other side of the
//! end than the exact real distance does.
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
    /// The count cannot be certified, for one of two reasons.
    ///
    /// - A merge height lies in the band (ends included), so the count
    ///   depends on a choice the input does not make. `(i, j)` with `i < j`
    ///   are indices into the input slice of the in-band spanning-tree edge
    ///   whose height is nearest the scale by absolute difference
    ///   `|height - scale|`, the earliest found on a tie. A spanning-tree
    ///   height that overflows to infinity is also refused this way.
    /// - A point has a non-finite coordinate, so no distance to it is
    ///   measurable. Then `i == j` is the index of the first such point and
    ///   `height` is NaN.
    Refused {
        /// Smaller index of the ambiguous pair, or the non-finite point.
        i: usize,
        /// Larger index of the ambiguous pair, or the non-finite point.
        j: usize,
        /// Euclidean distance between `points[i]` and `points[j]`, or NaN
        /// when `i == j`.
        height: f64,
    },
}

/// β₀ of `points` at Euclidean threshold `scale`, certified to be constant
/// over `[scale / sqrt(ratio), scale * sqrt(ratio)]`, or refused.
///
/// `ratio` below 1 (or NaN) is treated as 1, which certifies whenever no
/// height equals the scale exactly. With more than one point, a non-finite
/// coordinate refuses by naming that point, and a NaN scale refuses by
/// naming the first spanning-tree edge (no height is nearer a NaN scale).
// ponytail: O(n^2) all-pairs Prim, exact by construction; fine for the
// encoder's 64-point payloads. Upgrade path is a verified EMST (e.g. Delaunay
// in 2-D/3-D) with this Prim kept as the test control.
pub fn certified_beta0<const D: usize>(points: &[[f64; D]], scale: f64, ratio: f64) -> Beta0 {
    let n = points.len();
    let r = libm::sqrt(if ratio > 1.0 { ratio } else { 1.0 });
    let (lo, hi) = (scale / r, scale * r);

    if n > 1 {
        if let Some(p) = points.iter().position(|q| q.iter().any(|c| !c.is_finite())) {
            return Beta0::Refused {
                i: p,
                j: p,
                height: f64::NAN,
            };
        }
    }

    let mut inside = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut src = vec![0usize; n];
    let mut value = n as u32;
    let (mut gap_lo, mut gap_hi) = (0.0f64, f64::INFINITY);
    let mut nearest: Option<(usize, usize, f64)> = None;

    for k in 0..n {
        // Lowest-index minimum among points not yet in the tree. Points only
        // reachable through an overflowed distance keep `best = INFINITY`.
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
                // In the band, or overflowed to infinity: ambiguous.
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

/// Euclidean distance as `m * sqrt(sum((d / m)^2))` with `m = max |d|`, so a
/// difference of `1e-170` does not square to zero and one of `1e170` does not
/// square to infinity. Coordinates must be finite; a difference that still
/// overflows makes `m`, and so the result, infinite.
fn dist<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let m = a
        .iter()
        .zip(b)
        .map(|(x, y)| libm::fabs(x - y))
        .fold(0.0, f64::max);
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let q = libm::fabs(x - y) / m;
            q * q
        })
        .sum();
    m * libm::sqrt(s)
}
