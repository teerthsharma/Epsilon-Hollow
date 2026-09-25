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
//! detector assigns a stratum; the controller names the way out of it (see
//! [`FitAction`] for how far "names" goes — it is short of enforcement today).
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
//! * A **monotone** trajectory never revisits a value range, so it has no fold.
//!   `{p_t}` is a simple arc — but *not* a straight one: a monotone delay
//!   polyline turns by up to 90° wherever the slope changes, so its Rips
//!   1-skeleton is not a path in general. The zero is certified by checking
//!   monotonicity directly, not read off the complex.
//! * An **overfitting** trajectory is U-shaped: validation descends, turns, and
//!   climbs back through value ranges it already visited. On a clean V with
//!   step `s`, consecutive points along either arm are `√3·s` apart and the
//!   descending point `(k, k+1, k+2)·s` sits `√8·s` from the ascending point
//!   `(k+2, k+1, k)·s` two steps further from the vertex. Once the scale
//!   passes that `√(8/3) ≈ 1.633` step ratio the arms zip together and the V
//!   **closes into a loop**: cycle rank > 0.
//!
//! That ratio is the floor of [`LOOP_SCALE_MARGIN`]'s band, and the margin is
//! the only free constant in the construction — see its documentation for the
//! band, the derivation, and the asymmetric folds it does not cover.
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
//! **Claimed and tested**: a monotone window yields `loop_score == 0` exactly,
//! at any sampling density, at any scale — by certificate:
//! `aether_core::trajectory_shape::fold_score` checks monotonicity in O(n) and
//! returns 0 before any complex is built. A fold yields `loop_score > 0`.
//!
//! **Refuted, and why the certificate exists**: this used to be claimed as a
//! property of the Rips cycle rank itself, on the argument that a monotone arc's
//! next-nearest point is `2·ε*` away. That holds only for a straight arc. The
//! staircase `v_t = v_{t−1} − (0.05 if t % 3 == 0 else 0.001)` from `v = 10`
//! over 128 steps is strictly decreasing and scored `loop_score = 0.969` — and,
//! with training loss on a widening gap below it, the verdict `Overfit`. The
//! host test `monotone_staircase_scores_no_fold` in
//! `aether-core/tests/trajectory_shape.rs` pins it.
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
//! userspace autograd graph. It observes exactly what the training process
//! pushes through the Seal ABI — two scalars per step, `(train_loss, val_loss)`
//! — and every signal below derives from those two. Nothing the kernel already
//! owns for the task is read: not the heap break, not the I/O prefetch state.
//! Activation statistics and gradient structure are **not** observed; claiming
//! otherwise would be a lie.
//!
//! # What the kernel can actually do about it
//!
//! Report. [`FitAction`] is returned across the ABI in full and enforced in no
//! part; each of its fields documents how far it reaches. A trainer that ignores
//! the verdict is not slowed, capped or throttled by this module.
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
//!
//! The measurement and the verdict — arc-length resampling, MST statistics,
//! cycle rank, participation ratio, quartile drift, [`FitSignals`],
//! [`FitCalibration`] and [`classify`] — live in
//! `aether_core::trajectory_shape` and are re-exported here. This crate is
//! outside the Cargo workspace and its unit tests never run; that module's host
//! tests do. What stays here is kernel state: the per-stream rings, the
//! registry, the actuation and the boot proof.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use spin::Mutex;

use aether_core::manifold::{ManifoldPoint, TimeDelayEmbedder};
use aether_core::trajectory_shape::{measure_window, MAX_POINTS};
pub use aether_core::trajectory_shape::{
    classify, FitCalibration, FitSignals, Regime, DEFAULT_CALIBRATION, LOOP_SCALE_MARGIN,
    MIN_SAMPLES_MAX,
};

// ── Sizing ──────────────────────────────────────────────────────────────────

/// Embedded points retained per stream. Bounds both memory and the O(n²)
/// filtration cost. 64 points at τ=1 covers ~66 training steps of trajectory.
pub const STRATUM_WINDOW: usize = 64;

// The geometry kernels size their stack buffers by `MAX_POINTS`.
const _: () = assert!(STRATUM_WINDOW <= MAX_POINTS);

/// Delay-embedding dimension. 3 is the smallest dimension in which a planar
/// fold of a 1-D signal embeds without self-intersection.
const EMBED_DIM: usize = aether_core::trajectory_shape::EMBED_DIM;

// ── Actuation ───────────────────────────────────────────────────────────────

/// What the controller recommends about a regime.
///
/// **Every field is advisory today, and nothing here is enforced against a task
/// that ignores it.** The kernel returns the whole action across the ABI and the
/// trainer chooses what to honour. Three of the fields could never be anything
/// else: a `no_std` kernel cannot reach into a userspace optimizer and change
/// its regularisation coefficient, learning rate or batch size. The other two
/// name kernel-owned quantities, and are documented per field with exactly how
/// far they currently reach — which is short of enforcement in both cases.
#[derive(Debug, Clone, Copy)]
pub struct FitAction {
    /// Regime this action responds to.
    pub regime: Regime,
    /// Prefetch decision threshold recommended for model-training I/O.
    /// `should_prefetch` fires when `p_fetch > epsilon`, so *lower* is more
    /// aggressive.
    ///
    /// [`apply_action`] publishes this kernel-side, clamped to the engine's own
    /// [0.1, 0.9] range, and `PrefetchEngine::new_model_training` adopts it at
    /// construction. **Nothing in this tree constructs that preset** — the AHCI
    /// read path and the shell both build `new_gaming` — so the published value
    /// currently reaches no I/O decision. Wiring a training read path to that
    /// constructor is what would make this real; until then it is a
    /// recommendation like the other four, one the kernel holds as well as
    /// returns.
    pub prefetch_epsilon: f32,
    /// Recommendation that the training task stop growing its heap.
    ///
    /// **Not enforced.** The kernel has no heap ceiling to set: `dispatch_brk`
    /// grows `brk_end` by request and consults no limit, and this kernel's
    /// `setrlimit(RLIMIT_DATA, ..)` assigns `brk_end` rather than bounding it,
    /// so there is no value of this flag that can refuse an allocation. Making
    /// it real means giving `dispatch_brk` a per-task ceiling to check.
    pub clamp_heap: bool,
    /// Multiplier the trainer should apply to its regularisation term.
    pub reg_scale: f64,
    /// Multiplier the trainer should apply to its learning rate.
    pub lr_scale: f64,
    /// Multiplier the trainer should apply to its batch size.
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

/// Prefetch threshold published to the I/O path. `None` = engine default.
///
/// ponytail: one global rather than one per stream. With two workloads
/// registered the later `SYS_FIT_REGIME` caller overwrites the earlier one's
/// threshold, and [`unregister`] clears the value for both. Move it into
/// [`FitStream`] and key [`training_prefetch_epsilon`] by task id when a second
/// training workload can actually exist.
static PREFETCH_OVERRIDE: Mutex<Option<f32>> = Mutex::new(None);

/// The threshold last published by [`apply_action`], for a model-training
/// prefetch engine to adopt at construction. `None` until a registered workload
/// asks for one, and again once it unregisters.
pub fn training_prefetch_epsilon() -> Option<f32> {
    *PREFETCH_OVERRIDE.lock()
}

/// Publish the kernel-side half of an action: the prefetch threshold, clamped to
/// the range `PrefetchEngine` itself holds `epsilon` in.
///
/// `clamp_heap` is not acted on, because there is nothing here to act on it
/// with. This kernel has no heap ceiling: `dispatch_brk` grows `brk_end` on
/// request and consults no limit, and `setrlimit(RLIMIT_DATA, ..)` assigns
/// `brk_end` directly rather than bounding it — so the only thing this function
/// could do with the flag is move the break, which is the opposite of freezing
/// it. See [`FitAction`] for what each field is worth.
pub fn apply_action(action: &FitAction) {
    *PREFETCH_OVERRIDE.lock() = Some(action.prefetch_epsilon.clamp(0.1, 0.9));
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
        let mut raw = [ManifoldPoint::<EMBED_DIM>::zero(); STRATUM_WINDOW];
        self.ordered(&self.pts, &mut raw);
        let mut train = [0.0f64; STRATUM_WINDOW];
        self.ordered(&self.train, &mut train);
        let mut resid = [0.0f64; STRATUM_WINDOW];
        self.ordered(&self.resid, &mut resid);
        measure_window(
            &raw[..n],
            &train[..n],
            &resid[..n],
            self.samples,
            self.nonfinite,
        )
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
///
/// Also drops the published prefetch threshold. A process that exits leaves no
/// claim on kernel I/O behind it, and the threshold is one global — left set, it
/// would steer every later model-training read for the rest of the boot.
pub fn unregister(handle: u64) -> bool {
    let existed = STREAMS.lock().remove(&handle).is_some();
    if existed {
        *PREFETCH_OVERRIDE.lock() = None;
    }
    existed
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

    let pass = correct == REGIME_CASES.len()
        && monotone_ok
        && negctl_clean
        && naive_misfires
        && bounded
        && agree;

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

    /// The counterexample to the old Rips-only argument: a strictly falling
    /// staircase whose delay polyline turns 90° at every drop. With training
    /// loss on a widening gap below it the drift gate is open, so before the
    /// monotonicity certificate this read `loop = 0.969` and `Overfit`.
    fn test_monotone_staircase_is_not_overfit() -> TestResult {
        let mut s = FitStream::new(DEFAULT_CALIBRATION);
        let mut v = 10.0;
        for t in 0..PROOF_STEPS {
            v -= if t % 3 == 0 { 0.05 } else { 0.001 };
            s.observe(v - 0.02 - 0.001 * t as f64, v);
        }
        test_assert!(
            s.signals().loop_score == 0.0,
            "a monotone staircase must have loop_score exactly 0"
        );
        test_assert!(
            s.regime() != Regime::Overfit,
            "a monotone run is not a fold"
        );
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

    /// A run that has diverged to 1e200 and plateaued there is not healthy, and
    /// every value it feeds in is finite, so the `nonfinite` latch never arms —
    /// what overflows is the cloud's own scale, not the input.
    ///
    /// Two variants. Alternating both losses once failed through
    /// `participation_ratio` (`inf/inf`); that ratio is now normalised by the
    /// series' own magnitude and stays finite, so this variant now fails closed
    /// through the overflowed validation cloud, as the second does. Diverging
    /// only the validation loss leaves every signal finite and *fabricated*:
    /// every pairwise distance in the delay embedding overflows, Prim's
    /// algorithm records no finite edge at all, and the degenerate branch then
    /// reports the same numbers a single coincident point would.
    fn test_diverged_stream_is_not_wellfit() -> TestResult {
        let mut both = FitStream::new(DEFAULT_CALIBRATION);
        for i in 0..20u32 {
            let v = if i % 2 == 0 { 1e200 } else { 2e200 };
            both.observe(v, v);
        }
        let sig = both.signals();
        test_assert!(
            sig.nonfinite == 0,
            "the trigger must not arm the non-finite latch, or it proves nothing"
        );
        test_assert!(
            sig.samples >= DEFAULT_CALIBRATION.min_samples,
            "the trigger must clear the warm-up gate, or it proves nothing"
        );
        test_assert_eq!(both.regime(), Regime::Collapsing);

        let mut val_only = FitStream::new(DEFAULT_CALIBRATION);
        for i in 0..40u32 {
            val_only.observe(0.5, if i % 2 == 0 { 1e200 } else { 2e200 });
        }
        test_assert!(
            !val_only.signals().measurable(),
            "a cloud whose own scale overflowed must not report measured signals"
        );
        test_assert_eq!(val_only.regime(), Regime::Collapsing);
        TestResult::Pass
    }

    /// Calibration is the ABI's trust boundary: `SYS_FIT_CALIBRATE` hands
    /// `f64::from_bits` of a userspace word straight to `set_field`. A float that
    /// no consumer can use has to be refused, because the alternative is a
    /// saturating cast — `1e300 as u64` is `u64::MAX`, a warm-up gate `samples`
    /// can never reach, which switches the detector off for the rest of the boot.
    fn test_calibrate_rejects_unusable_ranges() -> TestResult {
        let mut s = FitStream::new(DEFAULT_CALIBRATION);
        for t in 0..PROOF_STEPS {
            let (a, b) = ProofCase::Collapsing.sample(t);
            s.observe(a, b);
        }
        test_assert_eq!(s.regime(), Regime::Collapsing);

        test_assert!(
            !s.calibrate(5, 1e300),
            "a min_samples that saturates the cast must be refused"
        );
        test_assert!(
            s.calibration().min_samples == DEFAULT_CALIBRATION.min_samples,
            "a refused calibration must leave the field unchanged"
        );
        test_assert_eq!(s.regime(), Regime::Collapsing);
        test_assert!(
            !s.calibrate(5, -1.0),
            "a negative min_samples must be refused, not saturated to zero"
        );
        test_assert_eq!(s.regime(), Regime::Collapsing);

        // Every field, at a value outside the range its consumer can use.
        let unusable: [(u32, f64); 11] = [
            (0, 1.5),   // loop_score never exceeds 1
            (0, -0.1),  // nor drops below 0
            (1, 2.0),   // resid_drift is bounded in [-1, 1]
            (1, -2.0),
            (2, 1.5), // spread is bounded in [1/3, 1]
            (2, -0.5),
            (3, 0.5), // shatter is a max/median ratio, never below 1
            (4, 1.5), // train_drift is bounded in [-1, 1]
            (4, -1.5),
            (5, 1e19), // below u64::MAX, still unreachable by any real run
            (5, 0.5),  // not a whole number of observations
        ];
        for (field, bad) in unusable {
            test_assert!(
                !s.calibrate(field, bad),
                "a calibration outside its consumer's range must be refused"
            );
        }

        // The knob must still turn.
        let usable: [(u32, f64); 6] = [
            (0, 1.0),
            (1, 0.25),
            (2, 0.5),
            (3, 1.0),
            (4, 0.5),
            (5, 0.0),
        ];
        for (field, good) in usable {
            test_assert!(
                s.calibrate(field, good),
                "an in-range calibration must still apply"
            );
        }
        test_assert!(
            s.calibration().min_samples == 0,
            "an accepted calibration must be applied"
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

    /// The same invariant downward. An absolute floor on the participation
    /// ratio's denominator, which scales as c⁴, read the underfit fixture at
    /// 1e-3 as spread 1.0 and the verdict `WellFit`.
    fn test_scale_equivariance_downward() -> TestResult {
        let mut a = FitStream::new(DEFAULT_CALIBRATION);
        let mut b = FitStream::new(DEFAULT_CALIBRATION);
        for t in 0..PROOF_STEPS {
            let (tr, va) = ProofCase::Underfit.sample(t);
            a.observe(tr, va);
            b.observe(tr * 1e-3, va * 1e-3);
        }
        test_assert_eq!(b.regime(), Regime::Underfit);
        test_assert!(
            libm::fabs(a.signals().spread - b.signals().spread) < 1e-9,
            "spread must be scale invariant downward"
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

    /// The published prefetch threshold is a single global, so the workload that
    /// asked for it has to take it away when it leaves. Otherwise the last
    /// `SYS_FIT_REGIME` caller steers every later training read for the rest of
    /// the boot, including after that process has exited.
    fn test_unregister_clears_prefetch_override() -> TestResult {
        let handle = 0xF17_0002;
        register(handle);
        apply_action(&plan_action(Regime::Collapsing));
        test_assert_eq!(training_prefetch_epsilon(), Some(0.9));
        test_assert!(unregister(handle), "unregister must find the stream");
        test_assert_eq!(training_prefetch_epsilon(), None);
        TestResult::Pass
    }

    /// Publishing the threshold is the whole of what `apply_action` does, so the
    /// range it promises the I/O engine has to hold for any action it is handed.
    fn test_apply_action_clamps_published_epsilon() -> TestResult {
        let handle = 0xF17_0003;
        let mut action = plan_action(Regime::WellFit);
        action.prefetch_epsilon = 5.0;
        apply_action(&action);
        test_assert_eq!(training_prefetch_epsilon(), Some(0.9));
        action.prefetch_epsilon = -1.0;
        apply_action(&action);
        test_assert_eq!(training_prefetch_epsilon(), Some(0.1));
        // Leave the global as this test found it.
        register(handle);
        unregister(handle);
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
        test_assert!(
            !calibrate(handle, 99, 0.5),
            "unknown field must be rejected"
        );
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
            "stratum::monotone_staircase_is_not_overfit",
            test_monotone_staircase_is_not_overfit,
        );
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
        crate::testing::register_test(
            "stratum::diverged_stream_is_not_wellfit",
            test_diverged_stream_is_not_wellfit,
        );
        crate::testing::register_test(
            "stratum::calibrate_rejects_unusable_ranges",
            test_calibrate_rejects_unusable_ranges,
        );
        crate::testing::register_test("stratum::scale_equivariance", test_scale_equivariance);
        crate::testing::register_test(
            "stratum::scale_equivariance_downward",
            test_scale_equivariance_downward,
        );
        crate::testing::register_test(
            "stratum::translation_invariance",
            test_translation_invariance,
        );
        crate::testing::register_test(
            "stratum::calibration_knob_moves_boundary",
            test_calibration_knob_moves_boundary,
        );
        crate::testing::register_test(
            "stratum::unregister_clears_prefetch_override",
            test_unregister_clears_prefetch_override,
        );
        crate::testing::register_test(
            "stratum::apply_action_clamps_published_epsilon",
            test_apply_action_clamps_published_epsilon,
        );
        crate::testing::register_test("stratum::proof_line_passes", test_proof_line_passes);
        crate::testing::register_test("stratum::registry_roundtrip", test_registry_roundtrip);
    }
}
