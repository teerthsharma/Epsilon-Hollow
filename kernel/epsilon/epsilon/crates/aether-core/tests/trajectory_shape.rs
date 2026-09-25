//! Host tests for `aether_core::trajectory_shape`, the geometry behind Seal OS's
//! `stratum` fit detector. The kernel crate's own tests only run under QEMU, so
//! these are the checks `cargo test --workspace` actually executes.
//!
//! The window construction below mirrors `FitStream::observe`: τ = 1 delay
//! embedding in ℝ³, the last `MAX_POINTS` points kept, and the training loss and
//! residual recorded at the same steps as the points.

use aether_core::manifold::TimeDelayEmbedder;
use aether_core::trajectory_shape::{
    fold_score, participation_ratio, quartile_drift, DelayPoint, EMBED_DIM, MAX_POINTS,
};

/// `stratum::DEFAULT_CALIBRATION.loop_min`.
const LOOP_MIN: f64 = 0.125;
/// `stratum::DEFAULT_CALIBRATION.resid_rise_min`.
const RESID_RISE_MIN: f64 = 0.05;
/// `stratum::DEFAULT_CALIBRATION.spread_trend_max`.
const SPREAD_TREND_MAX: f64 = 0.45;
/// `stratum::PROOF_STEPS`.
const STEPS: usize = 128;

struct Window {
    pts: Vec<DelayPoint>,
    train: Vec<f64>,
    resid: Vec<f64>,
}

fn window(train: &[f64], val: &[f64]) -> Window {
    let mut embed = TimeDelayEmbedder::<EMBED_DIM>::new(1);
    let mut w = Window {
        pts: Vec::new(),
        train: Vec::new(),
        resid: Vec::new(),
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

/// The `Overfit` verdict's two gates, as `stratum::classify` applies them.
fn overfit_gate(w: &Window) -> (f64, f64, bool) {
    let (_, loop_score) = fold_score(&w.pts);
    let drift = quartile_drift(&w.resid);
    (
        loop_score,
        drift,
        loop_score >= LOOP_MIN && drift >= RESID_RISE_MIN,
    )
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

/// `stratum::ProofCase::Underfit`'s training loss, in the window, times `scale`.
fn underfit_train(scale: f64) -> Vec<f64> {
    let train: Vec<f64> = (0..STEPS)
        .map(|t| scale * (1.0 - 0.004 * t as f64))
        .collect();
    let val: Vec<f64> = train.iter().map(|v| v + scale * 0.02).collect();
    window(&train, &val).train
}

/// Scaling every loss by `c > 0` scales every autocovariance by `c²`, so the
/// participation ratio cannot move. It did below `c ≈ 1e-3`: an absolute
/// `1e-12` floor on the 4th-power denominator read the underfit fixture at
/// `1e-3` as spread 1.0 (converged) instead of 0.353 (trend).
///
/// Tolerance 1e-9: the scaled series differs from the exact scaling by one
/// rounding per element, which moves the ratio by O(1e-15).
#[test]
fn participation_ratio_is_scale_invariant_downward() {
    let base = participation_ratio(&underfit_train(1.0));
    assert!(
        base <= SPREAD_TREND_MAX,
        "the fixture must read as a trend at scale 1, or this proves nothing (spread {base})"
    );
    for scale in [1e-2, 1e-3, 1e-6, 1e-9, 1e3] {
        let pr = participation_ratio(&underfit_train(scale));
        assert!(
            (pr - base).abs() < 1e-9,
            "scale {scale}: spread {pr} differs from {base} at scale 1"
        );
    }
}

/// A variation at the level of rounding cannot be told apart from rounding, so
/// the ratio is refused (NaN, which `stratum`'s `measurable()` fails closed on)
/// rather than reported. An exactly constant series is not refused: equality
/// is exact, and a flat loss is converged.
#[test]
fn participation_ratio_refuses_rounding_level_variation() {
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
