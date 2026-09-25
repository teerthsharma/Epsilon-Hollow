//! Host tests for `aether_core::trajectory_shape`, the measurement and verdict
//! behind Seal OS's `stratum` fit detector. The kernel crate's own tests only
//! run under QEMU, so these are the checks `cargo test --workspace` actually
//! executes.
//!
//! The window construction below mirrors `FitStream::observe`: τ = 1 delay
//! embedding in ℝ³, the last `MAX_POINTS` points kept, and the training loss and
//! residual recorded at the same steps as the points. `measure_window` and
//! `classify` are the functions the kernel calls on read.

use aether_core::manifold::TimeDelayEmbedder;
use aether_core::trajectory_shape::{
    classify, cycle_rank, fold_score, is_monotone, measure_window, participation_ratio, DelayPoint,
    FitSignals, Regime, DEFAULT_CALIBRATION, EMBED_DIM, MAX_POINTS,
};

const LOOP_MIN: f64 = DEFAULT_CALIBRATION.loop_min;
const RESID_RISE_MIN: f64 = DEFAULT_CALIBRATION.resid_rise_min;
const SPREAD_TREND_MAX: f64 = DEFAULT_CALIBRATION.spread_trend_max;
/// `stratum::PROOF_STEPS`.
const STEPS: usize = 128;

struct Window {
    pts: Vec<DelayPoint>,
    train: Vec<f64>,
    resid: Vec<f64>,
    samples: u64,
}

fn window(train: &[f64], val: &[f64]) -> Window {
    let mut embed = TimeDelayEmbedder::<EMBED_DIM>::new(1);
    let mut w = Window {
        pts: Vec::new(),
        train: Vec::new(),
        resid: Vec::new(),
        samples: train.len() as u64,
    };
    for (&tr, &va) in train.iter().zip(val) {
        embed.push(va);
        if let Some(p) = embed.embed() {
            w.pts.push(p);
            w.train.push(tr);
            w.resid.push(va - tr);
        }
    }
    let skip = w.pts.len().saturating_sub(MAX_POINTS);
    w.pts.drain(..skip);
    w.train.drain(..skip);
    w.resid.drain(..skip);
    w
}

/// The kernel's verdict on one window at the default calibration.
fn verdict(w: &Window) -> (Regime, FitSignals) {
    let sig = measure_window(&w.pts, &w.train, &w.resid, w.samples, 0);
    (classify(&sig, &DEFAULT_CALIBRATION), sig)
}

/// `(loop_score, resid_drift, verdict is Overfit)`.
fn overfit_gate(w: &Window) -> (f64, f64, bool) {
    let (regime, sig) = verdict(w);
    (sig.loop_score, sig.resid_drift, regime == Regime::Overfit)
}

/// `stratum::ProofCase::sample`'s deterministic jitter.
fn jitter(t: usize, salt: u64) -> f64 {
    let h = (t as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt);
    ((h >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02
}

/// Validation loss that only ever falls: a slow 0.001 decline with a 0.05 drop
/// at every step in `drop_at`. Training loss sits below it on a widening gap,
/// so the residual drifts upward and the drift gate is open — only the loop
/// gate stands between this run and an `Overfit` verdict.
fn staircase(steps: usize, drop_at: impl Fn(usize) -> bool) -> (Vec<f64>, Vec<f64>) {
    let mut val = Vec::with_capacity(steps);
    let mut x = 10.0;
    for t in 0..steps {
        x -= if drop_at(t) { 0.05 } else { 0.001 };
        val.push(x);
    }
    let train = val
        .iter()
        .enumerate()
        .map(|(t, v)| v - 0.02 - 0.001 * t as f64)
        .collect();
    (train, val)
}

fn staircases() -> [(&'static str, Vec<f64>, Vec<f64>); 3] {
    let (t3, v3) = staircase(STEPS, |t| t % 3 == 0);
    let (t8, v8) = staircase(STEPS, |t| t % 8 == 0);
    let (t1, v1) = staircase(66, |t| t == 30);
    [
        ("period 3", t3, v3),
        ("period 8", t8, v8),
        ("one drop at t=30", t1, v1),
    ]
}

#[test]
fn monotone_staircase_scores_no_fold() {
    for (name, train, val) in staircases() {
        assert!(
            val.windows(2).all(|p| p[1] < p[0]),
            "{name}: fixture must be strictly monotone"
        );
        let w = window(&train, &val);
        let (loop_score, drift, overfit) = overfit_gate(&w);
        assert!(
            drift >= RESID_RISE_MIN,
            "{name}: the drift gate must be open, or this proves nothing (drift {drift})"
        );
        assert_eq!(
            loop_score, 0.0,
            "{name}: a monotone trajectory never revisits a value, so loop_score must be 0"
        );
        assert!(
            !overfit,
            "{name}: a monotone run must not be called Overfit"
        );
    }
}

/// The staircase plus a jitter larger than its slow step, so the window is no
/// longer monotone and the certificate does not apply. The treads wobble; the
/// trajectory still never comes back through a drop. The Rips count alone has
/// to keep it below the fold threshold.
#[test]
fn jittered_staircase_is_not_a_fold() {
    for period in [3usize, 8] {
        let (train, mut val) = staircase(STEPS, |t| t % period == 0);
        for (t, v) in val.iter_mut().enumerate() {
            *v += 0.2 * jitter(t, 11);
        }
        let w = window(&train, &val);
        let tail: Vec<f64> = val[STEPS - MAX_POINTS - 2..].to_vec();
        assert!(
            tail.windows(2).any(|p| p[1] > p[0]),
            "period {period}: the jitter must break monotonicity, or this repeats the certificate test"
        );
        let (loop_score, _, overfit) = overfit_gate(&w);
        assert!(
            loop_score < LOOP_MIN,
            "period {period}: loop_score {loop_score} reads a descending staircase as a fold"
        );
        assert!(!overfit, "period {period}: must not be called Overfit");
    }
}

/// `stratum::ProofCase::Overfit`: validation bottoms out at step 88 and climbs
/// back through values it already visited. The fix must not blind the detector.
#[test]
fn genuine_fold_still_scores_overfit() {
    let mut train = Vec::with_capacity(STEPS);
    let mut val = Vec::with_capacity(STEPS);
    for t in 0..STEPS {
        let tf = t as f64;
        let d = tf - 88.0;
        train.push(0.05 + 0.55 * (-tf / 30.0).exp() + jitter(t, 3));
        val.push(0.30 + 0.0004 * d * d + jitter(t, 11));
    }
    let w = window(&train, &val);
    let (loop_score, drift, overfit) = overfit_gate(&w);
    assert!(
        overfit,
        "the fold fixture must still read as Overfit (loop {loop_score}, drift {drift})"
    );
}

/// `stratum::ProofCase::MonotoneLine` and `MonotoneExp`.
#[test]
fn monotone_controls_score_zero() {
    for (name, f) in [
        ("line", (|t: f64| 1.0 - 0.004 * t) as fn(f64) -> f64),
        ("exp", |t: f64| 0.05 + 0.55 * (-t / 22.0).exp()),
    ] {
        let v: Vec<f64> = (0..STEPS).map(|t| f(t as f64)).collect();
        let w = window(&v, &v);
        assert_eq!(fold_score(&w.pts).1, 0.0, "monotone {name} must score 0");
    }
}

/// `stratum::ProofCase::Underfit`, times `scale`.
fn underfit(scale: f64) -> Window {
    let train: Vec<f64> = (0..STEPS)
        .map(|t| scale * (1.0 - 0.004 * t as f64))
        .collect();
    let val: Vec<f64> = train.iter().map(|v| v + scale * 0.02).collect();
    window(&train, &val)
}

/// Scaling every loss by `c > 0` scales every autocovariance by `c²`, so the
/// participation ratio cannot move. It did twice: an absolute `1e-12` floor on
/// the 4th-power denominator read the underfit fixture at `1e-3` as spread 1.0
/// (converged) instead of 0.353 (trend), and the unnormalised `3c₀²` then
/// underflowed at `1e-150` and overflowed at `1e150`.
///
/// Tolerance 1e-9: the scaled series differs from the exact scaling by one
/// rounding per element, which moves the ratio by O(1e-15).
#[test]
fn participation_ratio_is_scale_invariant() {
    let base = participation_ratio(&underfit(1.0).train);
    assert!(
        base <= SPREAD_TREND_MAX,
        "the fixture must read as a trend at scale 1, or this proves nothing (spread {base})"
    );
    for scale in [1e-2, 1e-3, 1e-9, 1e-150, 1e-300, 1e3, 1e150, 1e300] {
        let w = underfit(scale);
        let pr = participation_ratio(&w.train);
        assert!(
            (pr - base).abs() < 1e-9,
            "scale {scale:e}: spread {pr} differs from {base} at scale 1"
        );
        if scale <= 1e150 {
            // Above ~1e154 the validation cloud's own distances overflow and
            // the verdict fails closed, which is a separate, documented path.
            assert_eq!(verdict(&w).0, Regime::Underfit, "scale {scale:e}");
        }
    }
}

/// A variation at the level of rounding cannot be told apart from rounding, so
/// the ratio is refused (NaN) rather than reported. An exactly constant series
/// is not refused: equality is exact, and a flat loss is converged.
///
/// The sine inputs sit either side of the stated bound
/// `√c₀ ≥ 3(n+3)·ε / 1e-6 ≈ 4.5e-8` (n = 64, relative to `max|x|`), far enough
/// that each flips if the tolerance or the `3(n+3)` factor changes: a 1e-9
/// sine (σ ≈ 7e-10) flips without the factor, a 1e-10 sine (σ ≈ 7e-11) flips
/// at a tolerance of 1e-2, and a 1e-6 sine (σ ≈ 7e-7) flips at 1e-12.
#[test]
fn participation_ratio_certifies_at_the_stated_bound() {
    let sine = |amp: f64| -> Vec<f64> {
        (0..MAX_POINTS)
            .map(|t| 1.0 + amp * (t as f64).sin())
            .collect()
    };
    for amp in [1e-9, 1e-10] {
        let pr = participation_ratio(&sine(amp));
        assert!(pr.is_nan(), "a {amp:e} sine must be refused, got {pr}");
    }
    let pr = participation_ratio(&sine(1e-6));
    assert!(
        pr.is_finite(),
        "a 1e-6 sine is resolved and must be certified"
    );

    let wiggle: Vec<f64> = (0..MAX_POINTS)
        .map(|i| if i % 2 == 0 { 1.0 } else { 1.0 + f64::EPSILON })
        .collect();
    let pr = participation_ratio(&wiggle);
    assert!(pr.is_nan(), "a one-ulp wiggle must be refused, got {pr}");

    for c in [0.25, 0.1, 1e-300, 0.0] {
        assert_eq!(
            participation_ratio(&[c; MAX_POINTS]),
            1.0,
            "an exactly constant series at {c} is converged"
        );
    }
}

/// A refused spread withholds the `Underfit` gate and nothing else. It must not
/// become an intervention: before, NaN failed `measurable()` and the verdict was
/// `Collapsing` (`lr_scale` 0.1, heap clamp) for a run whose loss had simply
/// stopped moving. Both inputs are converged and must read `WellFit`.
#[test]
fn refused_spread_is_not_an_intervention() {
    let ln2 = core::f32::consts::LN_2; // 0.6931472, a converged cross-entropy
    let ulp_up = f64::from(f32::from_bits(ln2.to_bits() + 1));
    let mut f32_flat = vec![f64::from(ln2); MAX_POINTS + 2];
    f32_flat[40] = ulp_up;
    let tiny_sine: Vec<f64> = (0..MAX_POINTS + 2)
        .map(|t| 1.0 + 1e-10 * (t as f64).sin())
        .collect();
    for (name, train) in [("f32 flat + 1 ulp", f32_flat), ("1 + 1e-10 sin", tiny_sine)] {
        let val: Vec<f64> = train.iter().map(|v| v + 0.02).collect();
        let (regime, sig) = verdict(&window(&train, &val));
        assert!(
            sig.spread.is_nan(),
            "{name}: the refusal path must be the one exercised (spread {})",
            sig.spread
        );
        assert_eq!(regime, Regime::WellFit, "{name}: a refusal must not act");
    }
}

/// The noiseless symmetric V `0.3 + 0.01·|t − 100|`. Its arms close at
/// `√(8/3)·ε* ≈ 1.633·ε*` (see `LOOP_SCALE_MARGIN`); at the old margin of 1.5
/// it scored 0, so a clean fold was invisible.
#[test]
fn noiseless_v_is_a_fold() {
    let v: Vec<f64> = (0..STEPS)
        .map(|t| 0.3 + 0.01 * (t as f64 - 100.0).abs())
        .collect();
    let (_, loop_score) = fold_score(&window(&v, &v).pts);
    assert!(
        loop_score >= LOOP_MIN,
        "the noiseless V must read as a fold, loop_score {loop_score}"
    );
}

/// A monotone window whose whole arc is under `1e-12`. The resampler once
/// floored arc length there and measured the raw, unevenly spaced cloud,
/// where the Rips count does see the dense converged tail as recurrence; its
/// floor is now relative to the cloud's magnitude. The monotonicity
/// certificate zeroes it either way, falling or rising: loop_score must be
/// scale invariant too.
#[test]
fn monotone_certificate_holds_below_the_resampler_floor() {
    let fall: Vec<f64> = (0..STEPS)
        .map(|t| 1e-12 * (0.05 + 0.55 * (-(t as f64) / 22.0).exp()))
        .collect();
    let rise: Vec<f64> = fall.iter().map(|v| 1e-12 - v).collect();
    for (name, v) in [("falling", fall), ("rising", rise)] {
        let w = window(&v, &v);
        assert!(is_monotone(&w.pts), "{name}: the fixture must be monotone");
        assert_eq!(fold_score(&w.pts).1, 0.0, "{name}: monotone must score 0");
    }
}

fn pts(coords: &[[f64; 3]]) -> Vec<DelayPoint> {
    coords.iter().map(|&c| DelayPoint::new(c)).collect()
}

/// A two-step chord is quotiented only when the Rips triangle that fills it
/// exists, i.e. both path edges are present. Here `(0, 2)` and `(1, 3)` each
/// miss the path edge `(1, 2)` (length √2 > 1.2), so both stay, and together
/// with `(0, 1)` and `(2, 3)` they bound a real square hole.
#[test]
fn cycle_rank_keeps_a_chord_whose_path_edge_is_missing() {
    let square = pts(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    assert_eq!(cycle_rank(&square, 1.2), 1);
}

/// Only two-step chords are quotiented. The square visited in order has its
/// closing edge `(0, 3)` as a three-step chord and a real hole; the unit
/// equilateral triangle's two-step chord is filled and counts nothing.
#[test]
fn cycle_rank_counts_longer_chords_and_drops_filled_ones() {
    let square = pts(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    assert_eq!(cycle_rank(&square, 1.2), 1);
    let h = 3f64.sqrt() / 2.0;
    let triangle = pts(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]]);
    assert_eq!(cycle_rank(&triangle, 1.2), 0);
}

#[test]
fn is_monotone_reads_both_directions() {
    let up = pts(&[[1.0, 0.0, -1.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]]);
    let down = pts(&[[3.0, 4.0, 5.0], [2.0, 3.0, 4.0], [1.0, 2.0, 3.0]]);
    let turn = pts(&[[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]]);
    assert!(is_monotone(&up));
    assert!(is_monotone(&down));
    assert!(!is_monotone(&turn));
}

/// Loop score of the jittered parabolic V `0.30 + 0.0004·(t − 88)²` over
/// `t ∈ 56..122`, every loss scaled by `c`.
/// Checked through `fold_score` and through `measure_window`, which the kernel
/// calls; the two must agree.
fn scaled_v_loop(c: f64) -> f64 {
    let v: Vec<f64> = (56..122)
        .map(|t| c * (0.30 + 0.0004 * (t as f64 - 88.0).powi(2) + 0.02 * jitter(t, 7)))
        .collect();
    let w = window(&v, &v);
    let direct = fold_score(&w.pts).1;
    assert_eq!(verdict(&w).1.loop_score, direct, "measure_window at {c:e}");
    direct
}

/// `fold_score` compares lengths of the cloud with lengths of the same cloud,
/// so rescaling the loss must not move it. An absolute `1e-12` floor on arc
/// length made the V a different shape below `c ≈ 1e-11`.
#[test]
fn fold_score_is_scale_invariant() {
    let at_one = scaled_v_loop(1.0);
    for c in [1e-10, 1e-11, 1e-12, 1e-14, 1e11] {
        assert_eq!(scaled_v_loop(c), at_one, "loop_score at scale {c:e}");
    }
}
