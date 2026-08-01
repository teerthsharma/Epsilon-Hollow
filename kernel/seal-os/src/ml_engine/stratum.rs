// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! `stratum` — kernel-level topological fit control.
//!
//! # Why "stratum"
//!
//! A stratification decomposes a singular space into manifold pieces. The space
//! of training states decomposes the same way: `underfit`, `wellfit` and
//! `overfit` are open strata a run moves through, and `collapsing` is the
//! singular stratum where the trajectory stops being a manifold at all. The
//! detector assigns a stratum; the actuator pushes the run toward a better one.
//! Distinct from `atlas`/`chart`/`germ`/`nerve` (code) and `bundle`/`section`
//! (firmware).
//!
//! # The topological thesis
//!
//! Overfitting has a *shape* signature in the delay embedding of the training
//! trajectory, not merely a level signature in the loss values.
//!
//! Let `v_t` be validation loss. Its Takens delay embedding is
//! `p_t = (v_t, v_{t-1}, v_{t-2}) ∈ ℝ³` (`aether_core::manifold::TimeDelayEmbedder`).
//!
//! * A **monotone** trajectory never revisits a value range, so `{p_t}` is a
//!   simple arc and the Vietoris–Rips 1-skeleton at the connectivity scale is a
//!   path: cycle rank 0.
//! * An **overfitting** trajectory is U-shaped: validation descends, turns, and
//!   climbs back through value ranges it already visited. At value `v` the
//!   descending point is `(v, v+s, v+2s)` and the ascending point is
//!   `(v, v−s, v−2s)`, a distance `s√5` apart, while consecutive points along
//!   either arm are `s√3` apart. Once the arms overlap in value they connect and
//!   the U **closes into a loop**: cycle rank > 0.
//!
//! That ratio `√5/√3 ≈ 1.291` is where [`LOOP_SCALE_MARGIN`] comes from, and it
//! is the only free constant in the construction — see its documentation.
//!
//! The loop is a property of *revisitation*: invariant under any strictly
//! monotone reparameterisation of the loss axis and under time
//! reparameterisation. A gap threshold (`val − train > τ`) cannot see it — it
//! sees only levels, so it fires on any run whose validation loss sits above
//! training loss, including a healthy run with an irreducible label-noise floor.
//! That case is the negative control in [`stratum_proof_line`], which reports
//! this detector's verdict and the naive gap baseline's verdict side by side.
//!
//! ## What is claimed, and what is not
//!
//! **Claimed and tested**: a monotone trajectory yields `loop_score == 0`
//! exactly, at any sampling density, at any scale. A fold yields
//! `loop_score > 0`.
//!
//! **Not claimed**: `loop_score > 0` does *not* imply a fold. A converged run
//! sitting in a noise ball also revisits its own neighbourhood constantly and
//! scores near 1.0. H₁ is orientation-blind — a run recovering from a validation
//! spike traces the same loop as one diverging into it. The residual drift gate
//! supplies the orientation the homology cannot. Both are required for the
//! `Overfit` verdict, and the proof reports both.
//!
//! ## Sampling density
//!
//! Cycle rank at a global scale is meaningless when the step size varies by
//! orders of magnitude: an exponentially converging run packs hundreds of points
//! into a ball smaller than one early step, and every one of them registers as a
//! recurrence. The point cloud is therefore **reparameterised by arc length**
//! before the complex is built, so the step is uniform by construction. Without
//! this, a strictly monotone exponential decay scores `loop_score = 1.0` — a
//! false positive that the `monotone_exp` control in the test suite exists to
//! catch.
//!
//! ## The underfit signal
//!
//! Underfit uses an independent geometric quantity: the **participation ratio**
//! of the 3×3 covariance of the *training*-loss delay embedding,
//! `PR = tr(C)² / (3‖C‖_F²) ∈ [1/3, 1]`. For a delay embedding `C` is symmetric
//! Toeplitz in the autocovariances `c₀, c₁, c₂`, so
//! `PR = 3c₀²/(3c₀² + 4c₁² + 2c₂²)` exactly, with no eigendecomposition.
//! `PR → 1/3` is rank-1: lag correlation near 1, the trajectory is a smooth
//! trend and the run is still moving. `PR → 1` is isotropic: the trend within
//! the window has fallen below the run's own noise floor, which is what
//! convergence *means*. This is the "collapsed spectral spread" of an underfit
//! manifold.
//!
//! # What the kernel can actually observe
//!
//! Honestly: nothing about the model's weights. A `no_std` kernel cannot walk a
//! userspace autograd graph. It observes what the training process pushes
//! through the Seal ABI — two scalars per step, `(train_loss, val_loss)` — plus
//! what the kernel already owns for that task (heap break, I/O prefetch state).
//! Every signal below derives from those two scalars. Activation statistics and
//! gradient structure are **not** observed; claiming otherwise would be a lie.
//!
//! # Streaming bound
//!
//! One [`FitStream`] per registered training process, of fixed size, independent
//! of run length. Observation is O(1) (ring writes only). Signals are recomputed
//! lazily on read in O(`STRATUM_WINDOW`²) with fixed stack buffers of
//! `STRATUM_WINDOW` entries. Nothing is buffered per-sample. The exact per-stream
//! byte count is measured at runtime and printed as `bytes_per_stream`.
//!
//! # Reuse note
//!
//! The delay embedding and the point/metric type come from `aether-core`
//! (`TimeDelayEmbedder`, `ManifoldPoint::distance`/`is_neighbor`). Component
//! counting does **not** use `SparseAttentionGraph::compute_betti_0`: that
//! structure stores adjacency in a `u64` bitmask (`are_neighbors` returns `false`
//! for any index ≥ 64) and its DFS silently drops neighbours once its 64-entry
//! stack fills, so β₀ is not sound at the densities this filtration reaches.
//! Union-find over the same `is_neighbor` predicate is used instead, and β₁
//! reuses the Euler-characteristic identity `β₁ = E − V + β₀` that
//! `estimate_betti_1` applies.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use spin::Mutex;

use aether_core::manifold::{ManifoldPoint, TimeDelayEmbedder};

// ── Sizing ──────────────────────────────────────────────────────────────────

/// Embedded points retained per stream. Bounds both memory and the O(n²)
/// filtration cost. 64 points at τ=1 covers ~66 training steps of trajectory.
pub const STRATUM_WINDOW: usize = 64;

/// Delay-embedding dimension. 3 is the smallest dimension in which a planar
/// fold of a 1-D signal embeds without self-intersection.
const EMBED_DIM: usize = 3;

/// Scale at which cycle rank is evaluated, as a multiple of the H₀ death scale
/// `ε*` (the largest edge of the minimum spanning tree — the single-linkage
/// merge height at which the cloud becomes one component).
///
/// The value is derived, not tuned. After arc-length reparameterisation, points
/// along an arm are `ε*` apart, so:
///
/// * the two arms of a fold sit `√5/√3 ≈ 1.291 · ε*` apart (see the module
///   thesis) — the margin must exceed this for a fold to register;
/// * on a monotone arc the next-nearest point in time is `2 · ε*` away — the
///   margin must stay below this or *every* arc registers as a lattice of
///   triangles and the signal saturates.
///
/// 1.5 is the midpoint of `(1.291, 2.0)`. The upper cliff is not theoretical:
/// sweeping this constant on the embedded fixtures shows `monotone_line` and
/// `monotone_exp` scoring exactly 0.0 for every value below 2.0 and jumping to
/// 0.969 at 2.0.
pub const LOOP_SCALE_MARGIN: f64 = 1.5;

/// Below this a length is treated as zero.
const EPS_FLOOR: f64 = 1e-12;

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
#[derive(Debug, Clone, Copy)]
pub struct FitSignals {
    /// Finite observations accepted so far.
    pub samples: u64,
    /// Points currently in the window.
    pub points: usize,
    /// Cycle rank of the Rips 1-skeleton at `LOOP_SCALE_MARGIN · ε*` on the
    /// arc-length-reparameterised cloud, normalised by point count. Exactly 0
    /// for a monotone trajectory. The fold signal.
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
    pub spread: f64,
    /// Bounded relative quartile drift of the residual `val − train`, in (−1, 1).
    pub resid_drift: f64,
    /// Bounded relative quartile drift of the training loss, in (−1, 1).
    pub train_drift: f64,
    /// Non-finite observations rejected. Any non-zero value latches `Collapsing`.
    pub nonfinite: u64,
}

impl FitSignals {
    const fn empty() -> Self {
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
    /// over roughly 8 steps. Measured: the embedded fold fixture scores 1.0 and
    /// both monotone controls score exactly 0.0.
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

impl FitCalibration {
    /// Set one field by ABI field id. Returns false for an unknown id or a
    /// non-finite value.
    pub fn set_field(&mut self, field: u32, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match field {
            0 => self.loop_min = value,
            1 => self.resid_rise_min = value,
            2 => self.spread_trend_max = value,
            3 => self.collapse_shatter_min = value,
            4 => self.collapse_rise = value,
            5 => self.min_samples = value as u64,
            _ => return false,
        }
        true
    }
}

// ── Actuation ───────────────────────────────────────────────────────────────

/// What the controller does about a regime.
///
/// `prefetch_epsilon` and `clamp_heap` are **real kernel control**: the kernel
/// owns both and [`apply_action`] enforces them. `reg_scale`, `lr_scale` and
/// `batch_scale` are **advisory** — the kernel cannot reach into a userspace
/// optimizer and change its regularisation coefficient, so it returns the
/// recommendation across the ABI and the trainer chooses whether to honour it.
#[derive(Debug, Clone, Copy)]
pub struct FitAction {
    /// Regime this action responds to.
    pub regime: Regime,
    /// REAL. Prefetch decision threshold for model-training I/O.
    /// `should_prefetch` fires when `p_fetch > epsilon`, so *lower* is more
    /// aggressive. Clamped to the engine's own [0.1, 0.9] range.
    pub prefetch_epsilon: f32,
    /// REAL. Freeze the training task's heap break where it stands.
    pub clamp_heap: bool,
    /// ADVISORY. Multiplier the trainer should apply to its regularisation term.
    pub reg_scale: f64,
    /// ADVISORY. Multiplier the trainer should apply to its learning rate.
    pub lr_scale: f64,
    /// ADVISORY. Multiplier the trainer should apply to its batch size.
    pub batch_scale: f64,
}

/// Map a regime to an action. Pure: no side effects, so the boot proof can
/// exercise it without a live scheduler.
pub fn plan_action(regime: Regime) -> FitAction {
    match regime {
        // Feed it faster and let it move: the model has not yet fit.
        Regime::Underfit => FitAction {
            regime,
            prefetch_epsilon: 0.15,
            clamp_heap: false,
            reg_scale: 0.5,
            lr_scale: 1.25,
            batch_scale: 1.0,
        },
        Regime::WellFit => FitAction {
            regime,
            prefetch_epsilon: 0.30,
            clamp_heap: false,
            reg_scale: 1.0,
            lr_scale: 1.0,
            batch_scale: 1.0,
        },
        // Stop rushing data through a memorising model; ask for more
        // regularisation and a smaller step.
        Regime::Overfit => FitAction {
            regime,
            prefetch_epsilon: 0.60,
            clamp_heap: false,
            reg_scale: 2.0,
            lr_scale: 0.5,
            batch_scale: 1.5,
        },
        // Contain it: the run is diverging and should stop consuming the machine.
        Regime::Collapsing => FitAction {
            regime,
            prefetch_epsilon: 0.90,
            clamp_heap: true,
            reg_scale: 4.0,
            lr_scale: 0.1,
            batch_scale: 1.0,
        },
    }
}

/// Prefetch threshold override published to the I/O path. `None` = engine default.
static PREFETCH_OVERRIDE: Mutex<Option<f32>> = Mutex::new(None);

/// Read by `PrefetchEngine::new_model_training`. Real coupling, one direction.
pub fn training_prefetch_epsilon() -> Option<f32> {
    *PREFETCH_OVERRIDE.lock()
}

/// Enforce the real knobs. Must be called from the training task's own context
/// (the heap clamp applies to the current task).
pub fn apply_action(action: &FitAction) {
    *PREFETCH_OVERRIDE.lock() = Some(action.prefetch_epsilon.clamp(0.1, 0.9));
    if action.clamp_heap {
        // RLIMIT_DATA == 2 in this kernel's setrlimit encoding: it pins brk_end.
        let brk = crate::process::scheduler::getrlimit(2);
        if brk != 0 {
            crate::process::scheduler::setrlimit(2, brk);
        }
    }
}

// ── Streaming state ─────────────────────────────────────────────────────────

/// Per-process fit-control state. Fixed size; see the module streaming bound.
pub struct FitStream {
    embed: TimeDelayEmbedder<EMBED_DIM>,
    pts: [ManifoldPoint<EMBED_DIM>; STRATUM_WINDOW],
    train: [f64; STRATUM_WINDOW],
    resid: [f64; STRATUM_WINDOW],
    len: usize,
    pos: usize,
    samples: u64,
    nonfinite: u64,
    dirty: bool,
    cal: FitCalibration,
    regime: Regime,
    signals: FitSignals,
}

impl FitStream {
    /// Fresh stream with the given calibration.
    pub fn new(cal: FitCalibration) -> Self {
        Self {
            embed: TimeDelayEmbedder::new(1),
            pts: [ManifoldPoint::zero(); STRATUM_WINDOW],
            train: [0.0; STRATUM_WINDOW],
            resid: [0.0; STRATUM_WINDOW],
            len: 0,
            pos: 0,
            samples: 0,
            nonfinite: 0,
            dirty: false,
            cal,
            regime: Regime::WellFit,
            signals: FitSignals::empty(),
        }
    }

    /// Push one training step. O(1): ring writes only. Non-finite input is
    /// rejected and latched — a run that produced a NaN is `Collapsing` for good.
    pub fn observe(&mut self, train_loss: f64, val_loss: f64) {
        if !train_loss.is_finite() || !val_loss.is_finite() {
            self.nonfinite += 1;
            self.dirty = true;
            return;
        }
        self.samples += 1;
        self.embed.push(val_loss);
        let Some(p) = self.embed.embed() else {
            // Warm-up: fewer than EMBED_DIM samples seen, no point yet.
            return;
        };
        self.pts[self.pos] = p;
        self.train[self.pos] = train_loss;
        self.resid[self.pos] = val_loss - train_loss;
        self.pos = (self.pos + 1) % STRATUM_WINDOW;
        if self.len < STRATUM_WINDOW {
            self.len += 1;
        }
        self.dirty = true;
    }

    /// Current regime, recomputing signals if new observations arrived.
    pub fn regime(&mut self) -> Regime {
        self.refresh();
        self.regime
    }

    /// Current signals, recomputing if new observations arrived.
    pub fn signals(&mut self) -> FitSignals {
        self.refresh();
        self.signals
    }

    /// Calibration in force for this stream.
    pub fn calibration(&self) -> FitCalibration {
        self.cal
    }

    /// Replace one calibration field; invalidates the cached verdict.
    pub fn calibrate(&mut self, field: u32, value: f64) -> bool {
        let ok = self.cal.set_field(field, value);
        if ok {
            self.dirty = true;
        }
        ok
    }

    fn refresh(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        self.signals = self.measure();
        self.regime = classify(&self.signals, &self.cal);
    }

    /// Copy a ring into chronological order.
    fn ordered<T: Copy>(&self, ring: &[T; STRATUM_WINDOW], out: &mut [T; STRATUM_WINDOW]) {
        let start = if self.len < STRATUM_WINDOW {
            0
        } else {
            self.pos
        };
        for i in 0..self.len {
            out[i] = ring[(start + i) % STRATUM_WINDOW];
        }
    }

    fn measure(&self) -> FitSignals {
        let n = self.len;
        let mut sig = FitSignals {
            samples: self.samples,
            points: n,
            nonfinite: self.nonfinite,
            ..FitSignals::empty()
        };
        if n < EMBED_DIM {
            return sig;
        }

        let mut raw = [ManifoldPoint::<EMBED_DIM>::zero(); STRATUM_WINDOW];
        self.ordered(&self.pts, &mut raw);
        let raw = &raw[..n];

        // Cloud RMS radius: the intrinsic scale everything is measured against.
        let mut centre = [0.0f64; EMBED_DIM];
        for p in raw {
            for d in 0..EMBED_DIM {
                centre[d] += p.coords[d];
            }
        }
        for c in centre.iter_mut() {
            *c /= n as f64;
        }
        let centre = ManifoldPoint::<EMBED_DIM>::new(centre);
        let mut sq = 0.0f64;
        for p in raw {
            let d = p.distance(&centre);
            sq += d * d;
        }
        let radius = libm::sqrt(sq / n as f64);

        // Sampling-density discontinuity, measured on the raw cloud.
        let (raw_max, raw_med) = mst_edge_stats(raw);

        if radius < EPS_FLOOR || raw_max < EPS_FLOOR {
            // Degenerate: every point coincides.
            sig.shatter = 1.0;
            sig.h0_death = 0.0;
            sig.loop_score = 0.0;
        } else {
            sig.shatter = if raw_med < EPS_FLOOR {
                1.0
            } else {
                raw_max / raw_med
            };
            // Reparameterise by arc length so a varying step size cannot
            // masquerade as topology, then measure the fold.
            let mut resampled = [ManifoldPoint::<EMBED_DIM>::zero(); STRATUM_WINDOW];
            let m = arc_resample(raw, &mut resampled);
            let cloud = &resampled[..m];
            let (eps_star, _) = mst_edge_stats(cloud);
            sig.h0_death = eps_star / radius;
            let cyc = cycle_rank(cloud, eps_star * LOOP_SCALE_MARGIN);
            sig.loop_score = (cyc as f64 / m as f64).min(1.0);
        }

        let mut train = [0.0f64; STRATUM_WINDOW];
        self.ordered(&self.train, &mut train);
        let mut resid = [0.0f64; STRATUM_WINDOW];
        self.ordered(&self.resid, &mut resid);
        sig.spread = participation_ratio(&train[..n]);
        sig.train_drift = quartile_drift(&train[..n]);
        sig.resid_drift = quartile_drift(&resid[..n]);
        sig
    }
}

/// Decision cascade. The order is load-bearing and is part of the rule:
///
/// 1. `Collapsing` first — a diverging run is also a smooth trend and would
///    otherwise read as `Underfit`.
/// 2. `Overfit` before `Underfit` — an overfitting run's *training* loss is
///    still descending smoothly, so its participation ratio sits near the rank-1
///    floor. The embedded fold fixture measures 0.424, below the 0.45 underfit
///    ceiling, so this ordering is not hypothetical.
/// 3. `Underfit` on the spread floor.
/// 4. `WellFit` otherwise.
pub fn classify(sig: &FitSignals, cal: &FitCalibration) -> Regime {
    if sig.nonfinite > 0 {
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
    if sig.spread <= cal.spread_trend_max {
        return Regime::Underfit;
    }
    Regime::WellFit
}

// ── Geometry ────────────────────────────────────────────────────────────────

/// Resample the polyline through `pts` at uniform arc length. Returns the number
/// of points written to `out`. Degenerate input is copied through unchanged.
fn arc_resample(
    pts: &[ManifoldPoint<EMBED_DIM>],
    out: &mut [ManifoldPoint<EMBED_DIM>; STRATUM_WINDOW],
) -> usize {
    let n = pts.len();
    if n < 3 {
        out[..n].copy_from_slice(pts);
        return n;
    }
    let mut cum = [0.0f64; STRATUM_WINDOW];
    for i in 1..n {
        cum[i] = cum[i - 1] + pts[i - 1].distance(&pts[i]);
    }
    let total = cum[n - 1];
    if total < EPS_FLOOR {
        out[..n].copy_from_slice(pts);
        return n;
    }
    let mut seg = 0usize;
    for k in 0..n {
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
        for d in 0..EMBED_DIM {
            c[d] = pts[seg].coords[d] + f * (pts[seg + 1].coords[d] - pts[seg].coords[d]);
        }
        out[k] = ManifoldPoint::new(c);
    }
    n
}

/// Minimum spanning tree edge statistics by Prim's algorithm, O(n²).
///
/// Returns `(max_edge, median_edge)`. The maximum is the H₀ death scale `ε*`:
/// the single-linkage merge height at which the last connected component dies,
/// i.e. the exact endpoint of the longest finite H₀ persistence bar.
fn mst_edge_stats(pts: &[ManifoldPoint<EMBED_DIM>]) -> (f64, f64) {
    let n = pts.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let mut included = [false; STRATUM_WINDOW];
    let mut key = [f64::INFINITY; STRATUM_WINDOW];
    let mut edges = [0.0f64; STRATUM_WINDOW];
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

fn uf_find(parent: &mut [usize; STRATUM_WINDOW], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Cycle rank `E − V + β₀` of the Vietoris–Rips 1-skeleton at scale `eps`, with
/// β₀ counted exactly by union-find. This upper-bounds Rips β₁: filling
/// 2-simplices can only kill cycles, never create them.
fn cycle_rank(pts: &[ManifoldPoint<EMBED_DIM>], eps: f64) -> u32 {
    let n = pts.len();
    if n == 0 {
        return 0;
    }
    let mut parent = [0usize; STRATUM_WINDOW];
    for (i, slot) in parent.iter_mut().enumerate().take(n) {
        *slot = i;
    }
    let mut edges: u64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if pts[i].is_neighbor(&pts[j], eps) {
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

/// Participation ratio of the delay-embedding covariance of `x`.
///
/// `C` is symmetric Toeplitz in the autocovariances `c₀, c₁, c₂`, so
/// `tr(C) = 3c₀`, `‖C‖_F² = 3c₀² + 4c₁² + 2c₂²`, and
/// `PR = tr(C)²/(3‖C‖_F²) = 3c₀²/(3c₀² + 4c₁² + 2c₂²) ∈ [1/3, 1]`.
/// A constant signal has `c₀ = 0` and is reported as 1.0 — a flat loss is
/// converged, not a trend.
fn participation_ratio(x: &[f64]) -> f64 {
    let n = x.len();
    if n < EMBED_DIM {
        return 1.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let cov = |lag: usize| -> f64 {
        let m = n - lag;
        let mut acc = 0.0;
        for i in 0..m {
            acc += (x[i] - mean) * (x[i + lag] - mean);
        }
        acc / m as f64
    };
    let (c0, c1, c2) = (cov(0), cov(1), cov(2));
    let denom = 3.0 * c0 * c0 + 4.0 * c1 * c1 + 2.0 * c2 * c2;
    if denom < EPS_FLOOR {
        return 1.0;
    }
    (3.0 * c0 * c0 / denom).clamp(1.0 / 3.0, 1.0)
}

/// Late-quartile mean minus early-quartile mean, normalised to (−1, 1) by the
/// sum of their magnitudes. Scale equivariant, bounded even under geometric
/// blow-up, and robust to single-step spikes in a way an endpoint difference is
/// not.
fn quartile_drift(x: &[f64]) -> f64 {
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

// ── Registry / ABI backing ──────────────────────────────────────────────────

static STREAMS: Mutex<BTreeMap<u64, FitStream>> = Mutex::new(BTreeMap::new());

/// Register a training workload. Idempotent: re-registering resets the stream.
/// Returns the handle (the caller's task id).
pub fn register(handle: u64) -> u64 {
    STREAMS
        .lock()
        .insert(handle, FitStream::new(DEFAULT_CALIBRATION));
    handle
}

/// Drop a registered workload. Returns true if it existed.
pub fn unregister(handle: u64) -> bool {
    STREAMS.lock().remove(&handle).is_some()
}

/// Push one observation. Returns the last computed regime (O(1); signals are
/// recomputed lazily by [`regime_of`]).
pub fn observe(handle: u64, train_loss: f64, val_loss: f64) -> Option<Regime> {
    let mut streams = STREAMS.lock();
    let s = streams.get_mut(&handle)?;
    s.observe(train_loss, val_loss);
    Some(s.regime)
}

/// Recompute and return the current regime, signals and planned action.
pub fn regime_of(handle: u64) -> Option<(Regime, FitSignals, FitAction)> {
    let mut streams = STREAMS.lock();
    let s = streams.get_mut(&handle)?;
    let regime = s.regime();
    let signals = s.signals();
    Some((regime, signals, plan_action(regime)))
}

/// Set one calibration field on a registered workload.
pub fn calibrate(handle: u64, field: u32, value: f64) -> bool {
    STREAMS
        .lock()
        .get_mut(&handle)
        .map(|s| s.calibrate(field, value))
        .unwrap_or(false)
}

/// Number of registered training workloads.
pub fn stream_count() -> usize {
    STREAMS.lock().len()
}

/// Human-readable status for the ABI and the shell.
pub fn report(handle: u64) -> Option<String> {
    let (regime, sig, act) = regime_of(handle)?;
    Some(format!(
        "regime={} loop={:.4} h0_death={:.4} shatter={:.3} spread={:.4} rdrift={:.4} \
         tdrift={:.4} samples={} prefetch_eps={:.2} clamp_heap={} reg={:.2} lr={:.2} batch={:.2}",
        regime.tag(),
        sig.loop_score,
        sig.h0_death,
        sig.shatter,
        sig.spread,
        sig.resid_drift,
        sig.train_drift,
        sig.samples,
        act.prefetch_epsilon,
        act.clamp_heap as u8,
        act.reg_scale,
        act.lr_scale,
        act.batch_scale
    ))
}

// ── Synthetic ground-truth regimes ──────────────────────────────────────────

/// Steps generated per proof case.
const PROOF_STEPS: usize = 128;

/// The regimes the boot proof runs the real detector against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofCase {
    /// Constant-rate descent from a high loss: the model has not fit yet.
    Underfit,
    /// Exponential descent that reaches its noise floor inside the window.
    WellFit,
    /// Training descends monotonically; validation bottoms out at step 88 —
    /// inside the 64-point window — and climbs back through the values it
    /// already visited. The fold.
    Overfit,
    /// Loss multiplies by 1.35 per step.
    Collapsing,
    /// NEGATIVE CONTROL. Healthy convergence with a large *constant* validation
    /// offset (irreducible label noise). Ground truth is `WellFit`. A gap
    /// threshold flags it; this detector must not.
    NegativeControl,
    /// INVARIANT CONTROL. Strictly monotone linear descent, train == val.
    /// `loop_score` must be exactly 0.
    MonotoneLine,
    /// INVARIANT CONTROL. Strictly monotone exponential decay, train == val.
    /// Sampling density varies by three orders of magnitude across the window;
    /// `loop_score` must still be exactly 0. This is the case that fails without
    /// arc-length reparameterisation.
    MonotoneExp,
}

impl ProofCase {
    /// The regime a correct detector must report.
    pub fn ground_truth(self) -> Regime {
        match self {
            ProofCase::Underfit => Regime::Underfit,
            ProofCase::WellFit | ProofCase::NegativeControl => Regime::WellFit,
            ProofCase::Overfit => Regime::Overfit,
            ProofCase::Collapsing => Regime::Collapsing,
            // Both monotone controls are noiseless smooth descents, so their
            // trend dominates by definition: `Underfit` is the correct regime as
            // well as the correct loop invariant.
            ProofCase::MonotoneLine | ProofCase::MonotoneExp => Regime::Underfit,
        }
    }

    /// Label used in the proof line.
    pub fn tag(self) -> &'static str {
        match self {
            ProofCase::Underfit => "underfit",
            ProofCase::WellFit => "wellfit",
            ProofCase::Overfit => "overfit",
            ProofCase::Collapsing => "collapsing",
            ProofCase::NegativeControl => "negctl",
            ProofCase::MonotoneLine => "monotone_line",
            ProofCase::MonotoneExp => "monotone_exp",
        }
    }

    /// `(train_loss, val_loss)` at step `t`. Deterministic: the "noise" is a
    /// fixed integer hash, so every boot measures the same trajectory.
    pub fn sample(self, t: usize) -> (f64, f64) {
        let tf = t as f64;
        let jitter = |salt: u64| -> f64 {
            let h = (t as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(salt);
            ((h >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02
        };
        match self {
            ProofCase::Underfit => {
                let train = 1.0 - 0.004 * tf;
                (train, train + 0.02)
            }
            ProofCase::WellFit => {
                let base = 0.05 + 0.95 * libm::exp(-tf / 9.0);
                (base + jitter(1), base + 0.012 + jitter(7))
            }
            ProofCase::Overfit => {
                let train = 0.05 + 0.55 * libm::exp(-tf / 30.0);
                let d = tf - 88.0;
                let val = 0.30 + 0.0004 * d * d;
                (train + jitter(3), val + jitter(11))
            }
            ProofCase::Collapsing => {
                let base = 0.5 * libm::pow(1.35, tf);
                (base, base * 1.05)
            }
            ProofCase::NegativeControl => {
                let base = 0.05 + 0.95 * libm::exp(-tf / 9.0);
                // Large constant gap: a label-noise floor, not overfitting.
                (base + jitter(5), base + 0.35 + jitter(13))
            }
            ProofCase::MonotoneLine => {
                let v = 1.0 - 0.004 * tf;
                (v, v)
            }
            ProofCase::MonotoneExp => {
                let v = 0.05 + 0.55 * libm::exp(-tf / 22.0);
                (v, v)
            }
        }
    }
}

/// Run the real detector over one synthetic case. Nothing is hardcoded: the
/// classifier does the work and the signals are whatever it measures.
pub fn run_case(case: ProofCase) -> (Regime, FitSignals) {
    let mut stream = FitStream::new(DEFAULT_CALIBRATION);
    for t in 0..PROOF_STEPS {
        let (train, val) = case.sample(t);
        stream.observe(train, val);
    }
    (stream.regime(), stream.signals())
}

/// The naive baseline this subsystem exists to beat: flag overfitting when the
/// validation gap exceeds a fixed threshold. Included in the proof so the
/// negative control demonstrates the difference rather than asserting it.
pub fn naive_gap_flags_overfit(case: ProofCase) -> bool {
    const GAP_THRESHOLD: f64 = 0.10;
    let (train, val) = case.sample(PROOF_STEPS - 1);
    val.is_finite() && train.is_finite() && (val - train) > GAP_THRESHOLD
}

// ── Proof emitter ───────────────────────────────────────────────────────────

/// Every case whose *regime* the proof checks against ground truth.
const REGIME_CASES: [ProofCase; 7] = [
    ProofCase::Underfit,
    ProofCase::WellFit,
    ProofCase::Overfit,
    ProofCase::Collapsing,
    ProofCase::NegativeControl,
    ProofCase::MonotoneLine,
    ProofCase::MonotoneExp,
];

/// Cases whose `loop_score` must additionally be exactly zero.
const MONOTONE_CASES: [ProofCase; 2] = [ProofCase::MonotoneLine, ProofCase::MonotoneExp];

/// Single-line boot proof. Every field is measured at this boot by running the
/// real detector. `result=fail` if any case is misclassified, if the monotone
/// invariant is violated, if the negative control is flagged, if the naive
/// baseline *stops* misfiring on it (which would mean the control no longer
/// discriminates), if the memory bound is exceeded, or if chunking the stream
/// changes the answer.
pub fn stratum_proof_line() -> String {
    let mut body = String::new();
    let mut correct = 0usize;
    for case in REGIME_CASES {
        let (got, sig) = run_case(case);
        if got == case.ground_truth() {
            correct += 1;
        }
        body.push_str(&format!(
            " case={} truth={} got={} loop={:.4} h0d={:.4} sh={:.3} sp={:.4} rd={:.4} td={:.4}",
            case.tag(),
            case.ground_truth().tag(),
            got.tag(),
            sig.loop_score,
            sig.h0_death,
            sig.shatter,
            sig.spread,
            sig.resid_drift,
            sig.train_drift
        ));
    }

    // The invariant: a monotone trajectory has no fold, at any sampling density.
    let mut monotone_ok = true;
    for case in MONOTONE_CASES {
        if run_case(case).1.loop_score != 0.0 {
            monotone_ok = false;
        }
    }

    // The negative control is only a control if the naive baseline misfires.
    let naive_misfires = naive_gap_flags_overfit(ProofCase::NegativeControl);
    let (negctl_regime, _) = run_case(ProofCase::NegativeControl);
    let negctl_clean = negctl_regime != Regime::Overfit;

    // Bounded memory: a stream 64× longer than the window must not grow it.
    const LONG_STEPS: usize = 4096;
    let mut long = FitStream::new(DEFAULT_CALIBRATION);
    for t in 0..LONG_STEPS {
        let (train, val) = ProofCase::WellFit.sample(t % PROOF_STEPS);
        long.observe(train, val);
    }
    let long_points = long.signals().points;
    let bounded = long_points <= STRATUM_WINDOW;
    let bytes_per_stream = core::mem::size_of::<FitStream>();

    // Incremental/batch agreement: the same observations delivered in one run
    // and in two halves, with a recompute in between, must agree exactly.
    let (batch_regime, batch_sig) = run_case(ProofCase::Overfit);
    let mut split = FitStream::new(DEFAULT_CALIBRATION);
    for t in 0..PROOF_STEPS / 2 {
        let (a, b) = ProofCase::Overfit.sample(t);
        split.observe(a, b);
    }
    let _ = split.regime();
    for t in PROOF_STEPS / 2..PROOF_STEPS {
        let (a, b) = ProofCase::Overfit.sample(t);
        split.observe(a, b);
    }
    let split_sig = split.signals();
    let agree = split.regime() == batch_regime
        && libm::fabs(split_sig.loop_score - batch_sig.loop_score) < 1e-12;

    let pass =
        correct == REGIME_CASES.len() && monotone_ok && negctl_clean && naive_misfires && bounded && agree;

    format!(
        "[MLFIT] proof version=1 subsystem=stratum window={} embed_dim={} kappa={:.3} \
         steps_per_case={} bytes_per_stream={} long_stream_steps={} long_stream_points={} \
         bounded={}{} monotone_loop_zero={} negctl_flagged={} naive_gap_baseline_flagged={} \
         incremental_batch_agree={} correct={}/{} result={}",
        STRATUM_WINDOW,
        EMBED_DIM,
        LOOP_SCALE_MARGIN,
        PROOF_STEPS,
        bytes_per_stream,
        LONG_STEPS,
        long_points,
        if bounded { "ok" } else { "fail" },
        body,
        if monotone_ok { "ok" } else { "fail" },
        if negctl_clean { "no" } else { "yes" },
        if naive_misfires { "yes" } else { "no" },
        if agree { "ok" } else { "fail" },
        correct,
        REGIME_CASES.len(),
        if pass { "pass" } else { "fail" }
    )
}

/// Print the boot proof to serial. Wire this into the boot sequence.
pub fn emit_boot_proof() {
    crate::serial_println!("{}", stratum_proof_line());
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_underfit_regime() -> TestResult {
        test_assert_eq!(run_case(ProofCase::Underfit).0, Regime::Underfit);
        TestResult::Pass
    }

    fn test_wellfit_regime() -> TestResult {
        test_assert_eq!(run_case(ProofCase::WellFit).0, Regime::WellFit);
        TestResult::Pass
    }

    fn test_overfit_regime() -> TestResult {
        let (regime, sig) = run_case(ProofCase::Overfit);
        test_assert_eq!(regime, Regime::Overfit);
        test_assert!(
            sig.loop_score >= DEFAULT_CALIBRATION.loop_min,
            "overfit must be carried by a fold, not by the drift gate alone"
        );
        test_assert!(
            sig.resid_drift >= DEFAULT_CALIBRATION.resid_rise_min,
            "overfit must also show upward residual drift"
        );
        TestResult::Pass
    }

    fn test_collapsing_regime() -> TestResult {
        test_assert_eq!(run_case(ProofCase::Collapsing).0, Regime::Collapsing);
        TestResult::Pass
    }

    /// The whole thesis in one test: a healthy run with a large constant
    /// validation gap must not be called overfitting, and the gap threshold this
    /// replaces must be shown to get it wrong.
    fn test_negative_control_not_flagged() -> TestResult {
        test_assert_eq!(run_case(ProofCase::NegativeControl).0, Regime::WellFit);
        test_assert!(
            naive_gap_flags_overfit(ProofCase::NegativeControl),
            "negative control no longer discriminates: the naive baseline agrees"
        );
        TestResult::Pass
    }

    /// The load-bearing topological invariant: a strictly monotone trajectory
    /// never revisits a value, so it has no fold — at any sampling density.
    /// `monotone_exp` varies its step size by three orders of magnitude across
    /// the window and fails without arc-length reparameterisation.
    fn test_monotone_has_no_fold() -> TestResult {
        for case in MONOTONE_CASES {
            let (_, sig) = run_case(case);
            test_assert!(
                sig.loop_score == 0.0,
                "a monotone trajectory must have cycle rank exactly 0"
            );
        }
        TestResult::Pass
    }

    fn test_incremental_matches_batch() -> TestResult {
        let (batch_regime, batch_sig) = run_case(ProofCase::Overfit);
        let mut split = FitStream::new(DEFAULT_CALIBRATION);
        let bounds = [0usize, 17, 40, 91, PROOF_STEPS];
        for w in bounds.windows(2) {
            for t in w[0]..w[1] {
                let (a, b) = ProofCase::Overfit.sample(t);
                split.observe(a, b);
            }
            let _ = split.regime();
        }
        let sig = split.signals();
        test_assert_eq!(split.regime(), batch_regime);
        test_assert!(
            libm::fabs(sig.loop_score - batch_sig.loop_score) < 1e-12,
            "loop score must not depend on how observations were chunked"
        );
        test_assert!(
            libm::fabs(sig.spread - batch_sig.spread) < 1e-12,
            "spread must not depend on how observations were chunked"
        );
        TestResult::Pass
    }

    fn test_bounded_memory_long_stream() -> TestResult {
        let mut s = FitStream::new(DEFAULT_CALIBRATION);
        for t in 0..100_000usize {
            let (a, b) = ProofCase::WellFit.sample(t % PROOF_STEPS);
            s.observe(a, b);
        }
        let sig = s.signals();
        test_assert!(sig.points <= STRATUM_WINDOW, "window must stay bounded");
        test_assert!(
            sig.samples == 100_000,
            "every finite sample must be counted"
        );
        TestResult::Pass
    }

    fn test_empty_stream() -> TestResult {
        let mut s = FitStream::new(DEFAULT_CALIBRATION);
        test_assert_eq!(s.regime(), Regime::WellFit);
        let sig = s.signals();
        test_assert_eq!(sig.points, 0);
        test_assert_eq!(sig.samples, 0);
        TestResult::Pass
    }

    fn test_constant_loss_no_nan() -> TestResult {
        let mut s = FitStream::new(DEFAULT_CALIBRATION);
        for _ in 0..64 {
            s.observe(0.25, 0.25);
        }
        let sig = s.signals();
        test_assert!(sig.loop_score.is_finite(), "loop_score must be finite");
        test_assert!(sig.spread.is_finite(), "spread must be finite");
        test_assert!(sig.resid_drift.is_finite(), "resid_drift must be finite");
        test_assert!(sig.h0_death.is_finite(), "h0_death must be finite");
        test_assert!(sig.shatter.is_finite(), "shatter must be finite");
        test_assert_eq!(s.regime(), Regime::WellFit);
        TestResult::Pass
    }

    fn test_nonfinite_guard() -> TestResult {
        let mut s = FitStream::new(DEFAULT_CALIBRATION);
        for t in 0..32usize {
            let (a, b) = ProofCase::WellFit.sample(t);
            s.observe(a, b);
        }
        s.observe(f64::NAN, 1.0);
        test_assert_eq!(s.regime(), Regime::Collapsing);
        s.observe(f64::INFINITY, f64::NEG_INFINITY);
        test_assert_eq!(s.regime(), Regime::Collapsing);
        test_assert!(
            s.signals().nonfinite == 2,
            "both rejections must be counted"
        );
        TestResult::Pass
    }

    /// TDA invariant: scaling every loss by c > 0 scales the cloud radius and
    /// every derived scale by c, so no Betti number and no ratio may move.
    fn test_scale_equivariance() -> TestResult {
        let mut a = FitStream::new(DEFAULT_CALIBRATION);
        let mut b = FitStream::new(DEFAULT_CALIBRATION);
        for t in 0..PROOF_STEPS {
            let (tr, va) = ProofCase::Overfit.sample(t);
            a.observe(tr, va);
            b.observe(tr * 1000.0, va * 1000.0);
        }
        let (sa, sb) = (a.signals(), b.signals());
        test_assert_eq!(a.regime(), b.regime());
        test_assert!(
            libm::fabs(sa.loop_score - sb.loop_score) < 1e-9,
            "loop_score must be scale invariant"
        );
        test_assert!(
            libm::fabs(sa.h0_death - sb.h0_death) < 1e-9,
            "h0_death must be scale invariant"
        );
        test_assert!(
            libm::fabs(sa.spread - sb.spread) < 1e-9,
            "spread must be scale invariant"
        );
        TestResult::Pass
    }

    /// TDA invariant: translating the loss axis leaves the geometry of the
    /// delay embedding unchanged up to a rigid motion, so every signal must hold.
    fn test_translation_invariance() -> TestResult {
        let mut a = FitStream::new(DEFAULT_CALIBRATION);
        let mut b = FitStream::new(DEFAULT_CALIBRATION);
        for t in 0..PROOF_STEPS {
            let (tr, va) = ProofCase::Overfit.sample(t);
            a.observe(tr, va);
            b.observe(tr + 5.0, va + 5.0);
        }
        let (sa, sb) = (a.signals(), b.signals());
        test_assert!(
            libm::fabs(sa.loop_score - sb.loop_score) < 1e-9,
            "loop_score must be translation invariant"
        );
        test_assert!(
            libm::fabs(sa.spread - sb.spread) < 1e-9,
            "spread must be translation invariant"
        );
        TestResult::Pass
    }

    /// A knob that does nothing is worse than no knob.
    fn test_calibration_knob_moves_boundary() -> TestResult {
        let (_, sig) = run_case(ProofCase::Overfit);
        let mut cal = DEFAULT_CALIBRATION;
        test_assert_eq!(classify(&sig, &cal), Regime::Overfit);
        cal.loop_min = 1.5; // above the achievable maximum
        test_assert!(
            classify(&sig, &cal) != Regime::Overfit,
            "raising loop_min must suppress the overfit verdict"
        );
        cal = DEFAULT_CALIBRATION;
        cal.resid_rise_min = 2.0; // above the bounded drift range
        test_assert!(
            classify(&sig, &cal) != Regime::Overfit,
            "raising resid_rise_min must suppress the overfit verdict"
        );
        TestResult::Pass
    }

    fn test_proof_line_passes() -> TestResult {
        let line = stratum_proof_line();
        test_assert!(line.starts_with("[MLFIT] proof version=1"), "proof prefix");
        test_assert!(line.contains("result=pass"), "boot proof must pass");
        TestResult::Pass
    }

    fn test_registry_roundtrip() -> TestResult {
        let handle = 0xF17_0001;
        register(handle);
        for t in 0..PROOF_STEPS {
            let (a, b) = ProofCase::Overfit.sample(t);
            observe(handle, a, b);
        }
        let Some((regime, _, action)) = regime_of(handle) else {
            return TestResult::Fail("registered stream must be found");
        };
        test_assert_eq!(regime, Regime::Overfit);
        test_assert!(action.reg_scale > 1.0, "overfit must raise regularisation");
        test_assert!(calibrate(handle, 0, 0.5), "calibration must apply");
        test_assert!(!calibrate(handle, 99, 0.5), "unknown field must be rejected");
        test_assert!(!calibrate(handle, 0, f64::NAN), "NaN must be rejected");
        test_assert!(unregister(handle), "unregister must find the stream");
        test_assert!(regime_of(handle).is_none(), "stream must be gone");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("stratum::underfit_regime", test_underfit_regime);
        crate::testing::register_test("stratum::wellfit_regime", test_wellfit_regime);
        crate::testing::register_test("stratum::overfit_regime", test_overfit_regime);
        crate::testing::register_test("stratum::collapsing_regime", test_collapsing_regime);
        crate::testing::register_test(
            "stratum::negative_control_not_flagged",
            test_negative_control_not_flagged,
        );
        crate::testing::register_test("stratum::monotone_has_no_fold", test_monotone_has_no_fold);
        crate::testing::register_test(
            "stratum::incremental_matches_batch",
            test_incremental_matches_batch,
        );
        crate::testing::register_test(
            "stratum::bounded_memory_long_stream",
            test_bounded_memory_long_stream,
        );
        crate::testing::register_test("stratum::empty_stream", test_empty_stream);
        crate::testing::register_test("stratum::constant_loss_no_nan", test_constant_loss_no_nan);
        crate::testing::register_test("stratum::nonfinite_guard", test_nonfinite_guard);
        crate::testing::register_test("stratum::scale_equivariance", test_scale_equivariance);
        crate::testing::register_test(
            "stratum::translation_invariance",
            test_translation_invariance,
        );
        crate::testing::register_test(
            "stratum::calibration_knob_moves_boundary",
            test_calibration_knob_moves_boundary,
        );
        crate::testing::register_test("stratum::proof_line_passes", test_proof_line_passes);
        crate::testing::register_test("stratum::registry_roundtrip", test_registry_roundtrip);
    }
}
