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
/// apart, so `ε* ≤ Δ`. The band is `(√8/√3, √3) ≈ (1.633, 1.732)`:
///
/// * **Floor `√8/√3 ≈ 1.633`, from a clean symmetric V.** Take `v` falling by
///   `s` per step to a vertex at `c` and rising by `s` per step after it. Every
///   delay step is `(±s, ±s, ±s)`, so the polyline has uniform step `√3·s`, the
///   resampled cloud is the raw cloud, and `ε* = √3·s`. The descending point
///   `p_{c−k} = (k, k+1, k+2)·s` and the ascending point
///   `p_{c+k+2} = (k+2, k+1, k)·s` differ by `(2, 0, −2)·s` for every `k`, so
///   the arms zip together at `√8·s = √(8/3)·ε*`. For `k = 0` that pair is a
///   two-step chord, quotiented out by [`cycle_rank`]; for `k ≥ 1` it is a
///   cross-arm edge and closes a cycle. Below the floor the V has no cross-arm
///   edge and scores 0. (Pairing points at *matched value*, `(v, v±s, v±2s)`,
///   gives `2√5·s ≈ 2.58·ε*` and is not the closest approach; the floor was
///   once stated from that pairing as `√5/√3 ≈ 1.291`, which is wrong on both
///   counts.)
/// * **Ceiling `√3 ≈ 1.732`, from a monotone stretch.** Where the series only
///   falls (or only rises), every segment of the delay polyline lies in one
///   closed orthant of ℝ³, so a chord spanning `k` resampled steps is at least
///   `kΔ/√3 ≥ k·ε*/√3` long. Two-step chords are quotiented out, so the first
///   chord that can close a cycle spans three steps and needs `√3`.
///
/// 1.68 is the midpoint (1.6825) of that band, rounded. Measured: the clean
/// symmetric V (`0.3 + 0.01·|t − 100|`) scores 0 up to 1.62 and 0.391 from
/// 1.64; the monotone staircase (0.001 per step, 0.05 drop every third step)
/// scores 0 through 1.8 even without the monotonicity certificate; the same
/// staircase with ±0.002 jitter (not monotone) scores 0 through 1.70 and 0.406
/// at 1.72.
///
/// The band is proved for the symmetric V only. An asymmetric fold closes
/// later: a V descending at 0.005 and climbing at 0.01 per step scores 0.016
/// at 1.72 and 0.406 only at 1.8, above the ceiling, so no margin in the band
/// detects it. The 1.5 used before detected no clean V at all; the staircase
/// counterexample (0.969 at 1.5 before two-step chords were quotiented) is why
/// the old `2.0` ceiling, argued from a straight arc, was also wrong.
pub const LOOP_SCALE_MARGIN: f64 = 1.68;

/// Below this a length is treated as zero.
const EPS_FLOOR: f64 = 1e-12;

fn check_len(n: usize) {
    assert!(
        n <= MAX_POINTS,
        "trajectory_shape: slice exceeds MAX_POINTS"
    );
}

/// Resample the polyline through `pts` at uniform arc length. Returns the number
/// of points written to `out`.
///
/// Degenerate input is copied through unchanged: an arc no longer than the
/// rounding of the coordinates themselves (`n · ε · max|coord|`), which is
/// not a measured length. The floor scales with the cloud, so rescaling the
/// loss does not change which windows are resampled. An absolute `1e-12`
/// floor here made a V scaled by `1e-11` a different shape from the same V
/// at scale 1.
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
    let big = pts
        .iter()
        .flat_map(|p| p.coords)
        .fold(0.0f64, |a, c| a.max(libm::fabs(c)));
    if total <= n as f64 * f64::EPSILON * big {
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
        // A zero-length segment has no interior to interpolate into.
        let f = if b > a {
            ((target - a) / (b - a)).clamp(0.0, 1.0)
        } else {
            0.0
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
/// [`LOOP_SCALE_MARGIN`]). On a resampled monotone cloud at a margin below
/// `√3` the count is in fact 0 as well — the MST of such a cloud is its path
/// (every cross-cut chord is at least `2Δ/√3 > Δ`), so every path edge is
/// present, every two-step chord is quotiented, and every longer chord exceeds
/// `√3·ε*`. The certificate does not rely on that argument, and it is the only
/// thing that zeroes a monotone window whose total arc length is under the
/// resampler's rounding floor, where the raw, unevenly spaced cloud is used.
/// The caller handles the degenerate and non-finite cases before calling this.
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
/// `PR = tr(C)²/(3‖C‖_F²) = 3/(3 + 4(c₁/c₀)² + 2(c₂/c₀)²) ∈ [1/3, 1]`.
/// An exactly constant signal is reported as 1.0 — a flat loss is converged,
/// not a trend. Equality is checked on the stored values, so this is exact.
///
/// Otherwise the ratio is certified or refused, never floored. The ratio is
/// invariant under `x → s·x`, so everything is computed on `y = x / M` with
/// `M = max|xᵢ|`: `y ∈ [−1, 1]`, so no square can overflow or underflow on
/// finite input at any scale, and the ratio is taken in the normalised form
/// above, which squares nothing larger than 1. The absolute floor this
/// replaces (`denominator < 1e-12`, on a quantity scaling as `s⁴`) read the
/// underfit fixture scaled by `1e-3` as 1.0; `3c₀²` itself underflows at
/// `s ≈ 1e-150` and overflows at `s ≈ 1e150`.
///
/// The certificate is a priori. Each `yᵢ` carries one rounding (`≤ ε`), naive
/// summation puts the mean within `n·ε` of exact, and the subtraction adds
/// `2ε`, so each deviation `yᵢ − ȳ` carries an error `e ≤ (n+3)·ε`. By
/// Cauchy–Schwarz each autocovariance then carries an error of at most
/// `2e·√c₀ + e²`, which is within `PR_REL_TOL · c₀` once
/// `√c₀ ≥ 3e / PR_REL_TOL`. Below that the variation is not resolved from its
/// own rounding and the function returns NaN rather than a ratio it did not
/// measure. The NaN is a refusal: [`classify`] skips the `Underfit` gate on it
/// and nothing else.
pub fn participation_ratio(x: &[f64]) -> f64 {
    /// Relative accuracy each autocovariance is certified to before the ratio
    /// is reported. The ratio then moves by at most about `8 · PR_REL_TOL`.
    const PR_REL_TOL: f64 = 1e-6;

    let n = x.len();
    if n < EMBED_DIM || x.iter().all(|&v| v == x[0]) {
        return 1.0;
    }
    let big = x.iter().fold(0.0f64, |a, &v| a.max(libm::fabs(v)));
    let y = |i: usize| x[i] / big;
    let mean = (0..n).map(y).sum::<f64>() / n as f64;
    let cov = |lag: usize| -> f64 {
        let m = n - lag;
        let mut acc = 0.0;
        for i in 0..m {
            acc += (y(i) - mean) * (y(i + lag) - mean);
        }
        acc / m as f64
    };
    let c0 = cov(0);
    let resolved = 3.0 * (n as f64 + 3.0) * f64::EPSILON / PR_REL_TOL;
    // `>=` is false against NaN, so a NaN `c₀` refuses as well.
    let certified = libm::sqrt(c0) >= resolved;
    if !certified {
        return f64::NAN;
    }
    let (r1, r2) = (cov(1) / c0, cov(2) / c0);
    (3.0 / (3.0 + 4.0 * r1 * r1 + 2.0 * r2 * r2)).clamp(1.0 / 3.0, 1.0)
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

// ── Regimes ─────────────────────────────────────────────────────────────────

/// The stratum a training run currently occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Delay embedding is still rank-1: the trend dominates the noise floor, so
    /// the run has not converged.
    Underfit,
    /// Coherent trajectory, no fold with upward residual drift, trend below the
    /// noise floor.
    WellFit,
    /// Validation trajectory has folded back through visited values *and* the
    /// generalisation residual is drifting upward.
    Overfit,
    /// Non-finite input, runaway loss, or a discontinuous trajectory.
    Collapsing,
}

impl Regime {
    /// Stable lowercase tag used in proof lines and the ABI.
    pub fn tag(self) -> &'static str {
        match self {
            Regime::Underfit => "underfit",
            Regime::WellFit => "wellfit",
            Regime::Overfit => "overfit",
            Regime::Collapsing => "collapsing",
        }
    }

    /// Numeric code returned across the Seal ABI.
    pub fn code(self) -> i64 {
        match self {
            Regime::Underfit => 0,
            Regime::WellFit => 1,
            Regime::Overfit => 2,
            Regime::Collapsing => 3,
        }
    }
}

// ── Signals ─────────────────────────────────────────────────────────────────

/// Topological signals measured from the current window. All dimensionless and
/// invariant under uniform rescaling of the loss axis.
///
/// The documented range of each measured field holds *when the field is
/// finite*. A stream whose losses overflow the squares these ratios are built
/// from reports NaN rather than a plausible number it did not measure; see
/// [`FitSignals::measurable`], which is what [`classify`] gates on.
#[derive(Debug, Clone, Copy)]
pub struct FitSignals {
    /// Finite observations accepted so far.
    pub samples: u64,
    /// Points currently in the window.
    pub points: usize,
    /// Cycle rank of the Rips 1-skeleton at `LOOP_SCALE_MARGIN · ε*` on the
    /// arc-length-reparameterised cloud, normalised by point count. Exactly 0
    /// for a monotone window, by a direct monotonicity certificate rather than
    /// by the complex. The fold signal.
    pub loop_score: f64,
    /// H₀ death scale `ε*` divided by the cloud's RMS radius, measured on the
    /// reparameterised cloud. Small = an extended path; near or above 1 = a
    /// diffuse ball (a converged run sitting in its noise floor). Reported as
    /// evidence; not gated on.
    pub h0_death: f64,
    /// Largest MST edge divided by the median MST edge, measured on the **raw**
    /// cloud. A discontinuity in sampling density: ≈1 for a smooth trajectory,
    /// enormous when successive steps grow geometrically. The collapse signal.
    pub shatter: f64,
    /// Participation ratio of the training-loss delay-embedding covariance, in
    /// [1/3, 1]. 1/3 = rank-1 (trend dominates), 1 = isotropic (converged).
    /// NaN when the loss varies by less than its own rounding can resolve (see
    /// [`participation_ratio`]); an exactly constant loss is 1.0. A NaN here is
    /// a refusal, not a failed measurement: it withholds the `Underfit` gate
    /// and nothing else (see [`classify`]).
    pub spread: f64,
    /// Bounded relative quartile drift of the residual `val − train`, in (−1, 1).
    pub resid_drift: f64,
    /// Bounded relative quartile drift of the training loss, in (−1, 1).
    pub train_drift: f64,
    /// Non-finite observations rejected. Any non-zero value latches `Collapsing`.
    pub nonfinite: u64,
}

impl FitSignals {
    /// True when every measured signal is a real number.
    ///
    /// The measurements are ratios of quantities derived from the losses
    /// themselves, so a loss large enough to overflow a square (`|v| > 1e154`)
    /// takes the numerator and the denominator to `inf` together and the ratio
    /// to NaN. Every comparison against NaN is false, so a NaN that reaches
    /// [`classify`] cannot fire a single gate — the verdict would be `WellFit`
    /// by default. A signal that could not be computed is not evidence of
    /// health, so `classify` fails closed on this instead.
    ///
    /// `spread` is not part of this. [`participation_ratio`] normalises by the
    /// series' own magnitude, so it cannot overflow on finite input; its only
    /// NaN is a refusal to read a trend out of rounding-level variation. That
    /// is an absent piece of evidence for one gate, not a geometry that failed
    /// to compute, and failing closed on it would turn a converged run whose
    /// loss sits still to the last bit into `Collapsing` — an intervention
    /// (`lr_scale` 0.1, heap clamp) triggered by a refusal.
    pub fn measurable(&self) -> bool {
        self.loop_score.is_finite()
            && self.h0_death.is_finite()
            && self.shatter.is_finite()
            && self.resid_drift.is_finite()
            && self.train_drift.is_finite()
    }

    /// The signal set of a window with no geometry yet: no loop, unit shatter,
    /// unit spread, no drift.
    pub const fn empty() -> Self {
        Self {
            samples: 0,
            points: 0,
            loop_score: 0.0,
            h0_death: 0.0,
            shatter: 1.0,
            spread: 1.0,
            resid_drift: 0.0,
            train_drift: 0.0,
            nonfinite: 0,
        }
    }
}

// ── Calibration ─────────────────────────────────────────────────────────────

/// Tunable decision boundaries.
///
/// Every field is a workload-dependent quantity, not a magic number, and every
/// one is settable at runtime through `SYS_FIT_CALIBRATE`. A fixed constant that
/// cannot be tuned is a bug: real trainers differ in step size, validation
/// cadence and noise floor, and those move where these boundaries belong.
#[derive(Debug, Clone, Copy)]
pub struct FitCalibration {
    /// Minimum normalised cycle rank to accept a fold as real.
    ///
    /// Basis: one noise-induced recurrence contributes `1/n = 0.0156` at n = 64.
    /// 0.125 requires 8 recurrence edges, i.e. the two arms of the fold overlap
    /// over roughly 8 steps. Measured: the embedded fold fixture scores 1.0 (the
    /// cap; 1.06 uncapped) and both monotone controls score exactly 0.0.
    pub loop_min: f64,
    /// Minimum residual drift for a fold to read as *divergence* rather than
    /// recovery.
    ///
    /// Basis: H₁ is orientation-blind, so drift supplies the sign. The drift
    /// statistic is bounded in (−1, 1); 0.05 means the late quartile mean
    /// exceeds the early quartile mean by ~10%, below which the quartile
    /// estimator is inside its own sampling noise.
    pub resid_rise_min: f64,
    /// Participation-ratio ceiling below which the trend still dominates.
    ///
    /// Basis: the floor of `PR` is exactly 1/3 ≈ 0.333 for a perfectly smooth
    /// trend. 0.45 allows ~35% above the floor before the run counts as
    /// converged. Measured: the underfit fixture scores 0.353, the converged
    /// fixture 0.814.
    pub spread_trend_max: f64,
    /// `shatter` at or above which the trajectory is judged discontinuous.
    ///
    /// Basis: 100 means the largest single step is two orders of magnitude
    /// larger than the typical one — a jump, not a trajectory. Measured: smooth
    /// fixtures score 1.0–2.1, the diverging fixture 1.1e4. Only fires when the
    /// loss is also rising, so a converged noise ball cannot trip it.
    pub collapse_shatter_min: f64,
    /// Training-loss drift that counts as divergence.
    ///
    /// Basis: the drift statistic is `(late − early)/(|late| + |early|)`, so 0.50
    /// means the late quartile mean is at least 3× the early quartile mean.
    /// Ordinary training noise does not move a quartile mean that far.
    pub collapse_rise: f64,
    /// Observations required before any verdict other than `WellFit` is issued.
    ///
    /// Basis: the drift estimator needs ≥4 points per quartile, and the cloud
    /// radius must not be dominated by the `EMBED_DIM` warm-up points.
    pub min_samples: u64,
}

/// Defaults. The basis for each value is documented on the field.
pub const DEFAULT_CALIBRATION: FitCalibration = FitCalibration {
    loop_min: 0.125,
    resid_rise_min: 0.05,
    spread_trend_max: 0.45,
    collapse_shatter_min: 100.0,
    collapse_rise: 0.50,
    min_samples: 16,
};

/// Largest accepted [`FitCalibration::min_samples`].
///
/// Every signal is computed from at most [`MAX_POINTS`] points, so this
/// field buys no extra evidence — it only delays the first verdict past an
/// early transient. 2²⁰ observations is 16384 full windows; past that the field
/// is not warming the detector up, it is switching it off. The ABI hands
/// `f64::from_bits` of a userspace word to [`FitCalibration::set_field`], and
/// Rust's float-to-integer cast *saturates*: without this ceiling `1e300`
/// becomes `u64::MAX`, a gate `samples` cannot reach in any run.
pub const MIN_SAMPLES_MAX: u64 = 1 << 20;

impl FitCalibration {
    /// Set one field by ABI field id. Returns false for an unknown id, a
    /// non-finite value, or a value outside the range its consumer can use.
    ///
    /// This is the ABI's trust boundary: `SYS_FIT_CALIBRATE` passes a userspace
    /// word here with no further checking. Each bound is the range of the signal
    /// the field is compared against, so a refused value is one that could not
    /// have moved the boundary anywhere the detector can reach:
    ///
    /// | id | field | accepted | why |
    /// |----|-------|----------|-----|
    /// | 0 | `loop_min` | `[0, 1]` | `loop_score` is a rank normalised by point count and capped at 1 |
    /// | 1 | `resid_rise_min` | `[-1, 1]` | `resid_drift` is a bounded quartile ratio |
    /// | 2 | `spread_trend_max` | `[0, 1]` | `spread` is a participation ratio in `[1/3, 1]` |
    /// | 3 | `collapse_shatter_min` | `[1, ∞)` | `shatter` is max/median of the same edge set, so never below 1; unbounded above |
    /// | 4 | `collapse_rise` | `[-1, 1]` | `train_drift` is a bounded quartile ratio |
    /// | 5 | `min_samples` | whole numbers in `[0, MIN_SAMPLES_MAX]` | see [`MIN_SAMPLES_MAX`]; the cast saturates in both directions |
    ///
    /// Out of range is refused, never clamped. A clamp would report success for
    /// a boundary the caller did not ask for, and the caller has no way to read
    /// back what it actually got except by inference from later verdicts.
    pub fn set_field(&mut self, field: u32, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match field {
            0 if (0.0..=1.0).contains(&value) => self.loop_min = value,
            1 if (-1.0..=1.0).contains(&value) => self.resid_rise_min = value,
            2 if (0.0..=1.0).contains(&value) => self.spread_trend_max = value,
            3 if value >= 1.0 => self.collapse_shatter_min = value,
            4 if (-1.0..=1.0).contains(&value) => self.collapse_rise = value,
            5 if (0.0..=MIN_SAMPLES_MAX as f64).contains(&value) && libm::trunc(value) == value => {
                self.min_samples = value as u64
            }
            _ => return false,
        }
        true
    }
}

// ── Measurement and verdict ─────────────────────────────────────────────────

/// Every signal of one window, from the chronologically ordered delay cloud and
/// the training loss and residual recorded at the same steps. This is the
/// whole of what the kernel's `FitStream` computes on read; the stream itself
/// only keeps the rings.
pub fn measure_window(
    raw: &[DelayPoint],
    train: &[f64],
    resid: &[f64],
    samples: u64,
    nonfinite: u64,
) -> FitSignals {
    let n = raw.len();
    let mut sig = FitSignals {
        samples,
        points: n,
        nonfinite,
        ..FitSignals::empty()
    };
    if n < EMBED_DIM {
        return sig;
    }

    // Cloud RMS radius: the intrinsic scale everything is measured against.
    let mut centre = [0.0f64; EMBED_DIM];
    for p in raw {
        for (c, x) in centre.iter_mut().zip(p.coords) {
            *c += x;
        }
    }
    for c in centre.iter_mut() {
        *c /= n as f64;
    }
    let centre = DelayPoint::new(centre);
    let mut sq = 0.0f64;
    for p in raw {
        let d = p.distance(&centre);
        sq += d * d;
    }
    let radius = libm::sqrt(sq / n as f64);

    // Sampling-density discontinuity, measured on the raw cloud.
    let (raw_max, raw_med) = mst_edge_stats(raw);

    if !radius.is_finite() {
        // The cloud's own scale overflowed: `d*d` in `distance` is `inf`, so
        // every pairwise distance is `inf`, so Prim's algorithm records no
        // finite edge and `raw_max` comes back 0 — indistinguishable, from
        // here, from a single coincident point. Reporting the degenerate
        // numbers would state that a trajectory oscillating between 1e200
        // and 2e200 sits still. Report the geometry as unmeasured instead
        // and let `classify` fail closed on it.
        sig.shatter = f64::NAN;
        sig.h0_death = f64::NAN;
        sig.loop_score = f64::NAN;
    } else if radius == 0.0 || raw_max == 0.0 {
        // Degenerate: every point coincides. Compared with zero rather than
        // an absolute floor, which read a loss scaled by 1e-11 as a point.
        sig.shatter = 1.0;
        sig.h0_death = 0.0;
        sig.loop_score = 0.0;
    } else {
        // `raw_med` is the median of positive tree edges, so it is positive
        // whenever `raw_max` is.
        sig.shatter = raw_max / raw_med;
        // Reparameterise by arc length so a varying step size cannot
        // masquerade as topology, then measure the fold.
        let (eps_star, loop_score) = fold_score(raw);
        sig.h0_death = eps_star / radius;
        sig.loop_score = loop_score;
    }

    sig.spread = participation_ratio(train);
    sig.train_drift = quartile_drift(train);
    sig.resid_drift = quartile_drift(resid);
    sig
}

/// Decision cascade. The order is load-bearing and is part of the rule:
///
/// 1. `Collapsing` first — a diverging run is also a smooth trend and would
///    otherwise read as `Underfit`.
/// 2. `Overfit` before `Underfit` — an overfitting run's *training* loss is
///    still descending smoothly, so its participation ratio sits near the rank-1
///    floor. The embedded fold fixture measures 0.424, below the 0.45 underfit
///    ceiling, so this ordering is not hypothetical.
/// 3. `Underfit` on the spread floor — only when `spread` was certified. A
///    refused spread (NaN) says the training loss moved by less than its own
///    rounding across the window, so it cannot show a dominating trend; the
///    gate is skipped and the run falls through, which is where an exactly
///    constant loss lands too.
/// 4. `WellFit` otherwise.
///
/// Both fail-closed checks — the latched non-finite *input* and the unmeasurable
/// *signal* — sit ahead of the warm-up gate deliberately. Waiting for more
/// samples cannot make either measurable, and behind the gate a `min_samples`
/// the run can never reach would suppress the only two verdicts that survive a
/// stream whose numbers have stopped meaning anything.
pub fn classify(sig: &FitSignals, cal: &FitCalibration) -> Regime {
    if sig.nonfinite > 0 {
        return Regime::Collapsing;
    }
    if !sig.measurable() {
        // Never let a NaN decide a branch: every comparison below is false
        // against one, which would elect `WellFit` by exhaustion.
        return Regime::Collapsing;
    }
    if sig.samples < cal.min_samples {
        return Regime::WellFit;
    }
    if sig.train_drift >= cal.collapse_rise
        || (sig.shatter >= cal.collapse_shatter_min && sig.train_drift > 0.0)
    {
        return Regime::Collapsing;
    }
    if sig.loop_score >= cal.loop_min && sig.resid_drift >= cal.resid_rise_min {
        return Regime::Overfit;
    }
    // Written out rather than left to NaN comparing false, so the refusal is a
    // stated branch and not an accident of IEEE 754.
    if sig.spread.is_finite() && sig.spread <= cal.spread_trend_max {
        return Regime::Underfit;
    }
    Regime::WellFit
}
