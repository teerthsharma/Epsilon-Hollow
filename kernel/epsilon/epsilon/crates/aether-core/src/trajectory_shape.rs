// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Shape statistics of a delay-embedded loss trajectory.
//!
//! These are the pure `f64` kernels behind Seal OS's `stratum` fit detector
//! (`kernel/seal-os/src/ml_engine/stratum.rs`). They live here rather than in
//! the kernel crate because the kernel crate is outside the Cargo workspace and
//! its unit tests never run; here `cargo test --workspace` exercises them.
//!
//! Every function works on fixed stack buffers of [`MAX_POINTS`] entries and
//! allocates nothing, so the kernel can call them per stream without a heap.
//! A slice longer than [`MAX_POINTS`] is a caller bug and panics.

use crate::manifold::ManifoldPoint;

/// Largest point cloud (or series) any function here accepts.
pub const MAX_POINTS: usize = 64;

/// Embedding dimension the delay cloud is built in.
pub const EMBED_DIM: usize = 3;

/// A point of the delay cloud `p_t = (v_t, v_{t-1}, v_{t-2})`.
pub type DelayPoint = ManifoldPoint<EMBED_DIM>;

/// Scale at which cycle rank is evaluated, as a multiple of the H₀ death scale
/// `ε*` (the largest edge of the minimum spanning tree of the arc-length
/// resampled cloud).
///
/// Let `Δ` be the resampling step. Consecutive resampled points are at most `Δ`
/// apart, so `ε* ≤ Δ`. The admissible band for the margin is `(1.291, 1.732)`:
///
/// * **Floor `√5/√3 ≈ 1.291`.** The two arms of a fold at matched value sit
///   that many along-arm steps apart (see the `stratum` module thesis). Below
///   it a fold never connects.
/// * **Ceiling `√3 ≈ 1.732`.** On a stretch where the series only falls (or
///   only rises), every segment of the delay polyline lies in one closed
///   orthant of ℝ³, so a chord spanning `k` resampled steps is at least
///   `kΔ/√3 ≥ k·ε*/√3` long. Chords across two steps are quotiented out by
///   [`cycle_rank`], so the first chord that can close a cycle on such a
///   stretch spans three steps and needs a margin of at least `√3`.
///
/// 1.5 sits in the middle of that band. The ceiling used to be stated as `2.0`
/// on the assumption that the next-nearest point along an arc is `2·ε*` away;
/// that holds only for a straight arc. A monotone delay polyline turns by up to
/// 90° at a slope change, the two-step chord across such a corner is `√2·Δ`,
/// and at 1.5 every corner closed a triangle: the staircase counterexample
/// (`v` falling 0.001 per step with a 0.05 drop every third step) scored 0.969.
///
/// Measured on that staircase with the two-step chords quotiented out: 0.0 for
/// every margin from 1.3 to 1.8, 0.953 at 1.9. Monotone windows do not depend
/// on this constant at all — [`fold_score`] certifies them 0 before the complex
/// is built — so the ceiling only protects windows that are monotone apart from
/// noise.
pub const LOOP_SCALE_MARGIN: f64 = 1.5;

/// Below this a length is treated as zero.
const EPS_FLOOR: f64 = 1e-12;

fn check_len(n: usize) {
    assert!(
        n <= MAX_POINTS,
        "trajectory_shape: slice exceeds MAX_POINTS"
    );
}

/// Resample the polyline through `pts` at uniform arc length. Returns the number
/// of points written to `out`. Degenerate input is copied through unchanged.
pub fn arc_resample(pts: &[DelayPoint], out: &mut [DelayPoint; MAX_POINTS]) -> usize {
    let n = pts.len();
    check_len(n);
    if n < 3 {
        out[..n].copy_from_slice(pts);
        return n;
    }
    let mut cum = [0.0f64; MAX_POINTS];
    for i in 1..n {
        cum[i] = cum[i - 1] + pts[i - 1].distance(&pts[i]);
    }
    let total = cum[n - 1];
    if total < EPS_FLOOR {
        out[..n].copy_from_slice(pts);
        return n;
    }
    let mut seg = 0usize;
    for (k, slot) in out.iter_mut().enumerate().take(n) {
        let target = total * k as f64 / (n - 1) as f64;
        while seg + 2 < n && cum[seg + 1] < target {
            seg += 1;
        }
        let (a, b) = (cum[seg], cum[seg + 1]);
        let f = if b - a < EPS_FLOOR {
            0.0
        } else {
            ((target - a) / (b - a)).clamp(0.0, 1.0)
        };
        let mut c = [0.0f64; EMBED_DIM];
        for (d, cd) in c.iter_mut().enumerate() {
            *cd = pts[seg].coords[d] + f * (pts[seg + 1].coords[d] - pts[seg].coords[d]);
        }
        *slot = ManifoldPoint::new(c);
    }
    n
}

/// Minimum spanning tree edge statistics by Prim's algorithm, O(n²).
///
/// Returns `(max_edge, median_edge)`. The maximum is the H₀ death scale `ε*`:
/// the single-linkage merge height at which the last connected component dies,
/// i.e. the exact endpoint of the longest finite H₀ persistence bar.
pub fn mst_edge_stats(pts: &[DelayPoint]) -> (f64, f64) {
    let n = pts.len();
    check_len(n);
    if n < 2 {
        return (0.0, 0.0);
    }
    let mut included = [false; MAX_POINTS];
    let mut key = [f64::INFINITY; MAX_POINTS];
    let mut edges = [0.0f64; MAX_POINTS];
    let mut count = 0usize;
    key[0] = 0.0;
    for _ in 0..n {
        let mut best = usize::MAX;
        let mut best_key = f64::INFINITY;
        for i in 0..n {
            if !included[i] && key[i] < best_key {
                best_key = key[i];
                best = i;
            }
        }
        if best == usize::MAX {
            break;
        }
        included[best] = true;
        if best_key.is_finite() && best_key > 0.0 {
            edges[count] = best_key;
            count += 1;
        }
        for i in 0..n {
            if !included[i] {
                let d = pts[best].distance(&pts[i]);
                if d < key[i] {
                    key[i] = d;
                }
            }
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    edges[..count].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    (edges[count - 1], edges[count / 2])
}

fn uf_find(parent: &mut [usize; MAX_POINTS], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Cycle rank `E − V + β₀` of the Vietoris–Rips 1-skeleton at scale `eps`, with
/// β₀ counted exactly by union-find, after quotienting out the two-step chords
/// that a Rips triangle fills.
///
/// A chord `(i, i+2)` whose path edges `(i, i+1)` and `(i+1, i+2)` are both
/// present bounds the 2-simplex `{i, i+1, i+2}`, so in H₁ it is homologous to
/// that two-edge path and adds no class. Such chords are not counted. Path
/// edges are never dropped, so every cycle through a dropped chord has a
/// homologous cycle in the remaining graph, β₀ is unchanged, and the count
/// still upper-bounds Rips β₁. It is tighter than the raw 1-skeleton count by
/// exactly the corner triangles that made a monotone staircase read as a fold.
pub fn cycle_rank(pts: &[DelayPoint], eps: f64) -> u32 {
    let n = pts.len();
    check_len(n);
    if n == 0 {
        return 0;
    }
    let mut parent = [0usize; MAX_POINTS];
    for (i, slot) in parent.iter_mut().enumerate().take(n) {
        *slot = i;
    }
    let near = |i: usize, j: usize| pts[i].is_neighbor(&pts[j], eps);
    let mut edges: u64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if j == i + 2 && near(i, i + 1) && near(i + 1, j) {
                // Filled by the triangle {i, i+1, i+2}; see above.
                continue;
            }
            if near(i, j) {
                edges += 1;
                let (a, b) = (uf_find(&mut parent, i), uf_find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut b0 = 0u32;
    for i in 0..n {
        if uf_find(&mut parent, i) == i {
            b0 += 1;
        }
    }
    let cyc = edges as i64 - n as i64 + b0 as i64;
    if cyc > 0 {
        cyc as u32
    } else {
        0
    }
}

/// True when every coordinate of the cloud is non-increasing along the index,
/// or every coordinate is non-decreasing. For a τ = 1 delay cloud this is
/// exactly "the series in the window never turns back". O(n), no rounding: it
/// compares the stored values and nothing derived from them.
pub fn is_monotone(pts: &[DelayPoint]) -> bool {
    let along = |sign: f64| {
        pts.windows(2)
            .all(|w| (0..EMBED_DIM).all(|d| sign * (w[1].coords[d] - w[0].coords[d]) >= 0.0))
    };
    along(1.0) || along(-1.0)
}

/// The fold measurement of a delay cloud: `(ε*, loop_score)`.
///
/// `ε*` is the H₀ death scale of the arc-length resampled cloud. `loop_score`
/// is certified 0 when the window is monotone ([`is_monotone`]): a fold is a
/// revisit, and a series that never turns back revisits nothing. Otherwise it
/// is [`cycle_rank`] at `LOOP_SCALE_MARGIN · ε*` on the resampled cloud,
/// normalised by point count and capped at 1.
///
/// The certificate is the claim; the Rips count is not asked to prove it. The
/// count upper-bounds β₁ of a complex built on sampled, resampled points, and a
/// monotone polyline can still close spurious cycles in it (see
/// [`LOOP_SCALE_MARGIN`]). The caller handles the degenerate and non-finite
/// cases before calling this.
pub fn fold_score(raw: &[DelayPoint]) -> (f64, f64) {
    let mut resampled = [DelayPoint::zero(); MAX_POINTS];
    let m = arc_resample(raw, &mut resampled);
    if m == 0 {
        return (0.0, 0.0);
    }
    let cloud = &resampled[..m];
    let (eps_star, _) = mst_edge_stats(cloud);
    if is_monotone(raw) {
        return (eps_star, 0.0);
    }
    let cyc = cycle_rank(cloud, eps_star * LOOP_SCALE_MARGIN);
    (eps_star, (cyc as f64 / m as f64).min(1.0))
}

/// Participation ratio of the delay-embedding covariance of `x`.
///
/// `C` is symmetric Toeplitz in the autocovariances `c₀, c₁, c₂`, so
/// `tr(C) = 3c₀`, `‖C‖_F² = 3c₀² + 4c₁² + 2c₂²`, and
/// `PR = tr(C)²/(3‖C‖_F²) = 3c₀²/(3c₀² + 4c₁² + 2c₂²) ∈ [1/3, 1]`.
/// An exactly constant signal is reported as 1.0 — a flat loss is converged,
/// not a trend. Equality is checked on the stored values, so this is exact.
///
/// Otherwise the ratio is certified or refused, never floored. The ratio is
/// invariant under `x → s·x`, so any decision about whether `c₀` is "too small"
/// has to be relative too; an absolute floor on the `c⁴`-scale denominator
/// (formerly `1e-12`) read the underfit fixture scaled by `1e-3` as 1.0.
///
/// The bound is a priori. With `M = max|xᵢ|`, naive summation puts the mean
/// within `(n−1)·ε·M` of exact and the subtraction adds at most `2·ε·M`, so each
/// deviation `xᵢ − x̄` carries an error `e ≤ (n+1)·ε·M`. By Cauchy–Schwarz each
/// autocovariance then carries an error of at most `2e·√c₀ + e²`, which is
/// within `PR_REL_TOL · c₀` once `√c₀ ≥ 3e / PR_REL_TOL`. Below that the signal's
/// variation is not resolved from its own rounding, and the function returns
/// NaN — which `stratum`'s `FitSignals::measurable` fails closed on — rather
/// than a ratio it did not measure. A non-finite `M` or `c₀` also refuses.
pub fn participation_ratio(x: &[f64]) -> f64 {
    /// Relative accuracy each autocovariance is certified to before the ratio
    /// is reported. The ratio then moves by at most about `8 · PR_REL_TOL`.
    const PR_REL_TOL: f64 = 1e-6;

    let n = x.len();
    if n < EMBED_DIM || x.iter().all(|&v| v == x[0]) {
        return 1.0;
    }
    let big = x.iter().fold(0.0f64, |a, &v| a.max(libm::fabs(v)));
    let mean = x.iter().sum::<f64>() / n as f64;
    let cov = |lag: usize| -> f64 {
        let m = n - lag;
        let mut acc = 0.0;
        for i in 0..m {
            acc += (x[i] - mean) * (x[i + lag] - mean);
        }
        acc / m as f64
    };
    let c0 = cov(0);
    let resolved = 3.0 * (n as f64 + 1.0) * f64::EPSILON * big / PR_REL_TOL;
    // `>=` is false against NaN, so a NaN `c₀` refuses as well.
    let certified = resolved.is_finite() && libm::sqrt(c0) >= resolved;
    if !certified {
        return f64::NAN;
    }
    let (c1, c2) = (cov(1), cov(2));
    let denom = 3.0 * c0 * c0 + 4.0 * c1 * c1 + 2.0 * c2 * c2;
    (3.0 * c0 * c0 / denom).clamp(1.0 / 3.0, 1.0)
}

/// Late-quartile mean minus early-quartile mean, normalised to (−1, 1) by the
/// sum of their magnitudes. Scale equivariant, bounded even under geometric
/// blow-up, and robust to single-step spikes in a way an endpoint difference is
/// not.
pub fn quartile_drift(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return 0.0;
    }
    let q = n / 4;
    let early: f64 = x[..q].iter().sum::<f64>() / q as f64;
    let late: f64 = x[n - q..].iter().sum::<f64>() / q as f64;
    let denom = libm::fabs(early) + libm::fabs(late);
    if denom < EPS_FLOOR {
        0.0
    } else {
        ((late - early) / denom).clamp(-1.0, 1.0)
    }
}
