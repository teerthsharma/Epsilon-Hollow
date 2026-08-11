// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! `tuner` — reclaim a converged training run's unused share, without the run
//! asking.
//!
//! # The premise
//!
//! A training run's demand for compute falls as it converges. The late epochs
//! of a run sitting in its own noise floor extract far less per unit of work
//! than the early ones did. Nothing in this kernel notices, so a converged job
//! holds everything it was granted until it exits, and the difference is never
//! available to anything else.
//!
//! The model's own code contains no limit and is not asked to. It cannot know
//! it has converged — that is a property of the trajectory, not of any step —
//! and a limit compiled into the trainer would be a guess made before the run
//! started. The kernel already measures the trajectory
//! ([`crate::ml_engine::stratum`]), so the kernel is where the decision belongs.
//!
//! # What the share is a share *of*
//!
//! **A dimensionless fraction of the run's own initial grant, in `[floor, 1.0]`.
//! It is not a share of any device.** Nothing in this tree apportions a
//! fraction of any hardware between tasks:
//!
//! * `ManifoldScheduler` dispatches by strict priority bucket
//!   (`process/scheduler.rs`); `Task` carries a `priority` and a
//!   `priority_boost` and no CPU budget, so there is no quantity a fraction
//!   could multiply.
//! * `dispatch_brk` grows `brk_end` on request and consults no limit — the same
//!   gap [`crate::ml_engine::stratum::FitAction::clamp_heap`] documents.
//! * The GPU tree probes and benchmarks devices; `drivers/gpu/gpu_bench.rs` is a
//!   benchmark, not a scheduler, and no path in it divides a device between two
//!   workloads.
//!
//! So this module maps a signal to a number and proves the number obeys its
//! five rules. To tune real hardware, exactly one thing has to exist first: a
//! per-task budget the dispatcher honours — a CPU-time quantum in [`Task`] that
//! `schedule()` decrements and refuses to re-arm past, or a heap ceiling
//! `dispatch_brk` checks. Multiply that budget by [`Tuner::share`] and the
//! reclaim is real. Claiming to tune anything before that exists would be a
//! confident wrong number.
//!
//! [`Task`]: crate::process::task::Task
//!
//! # The mapping
//!
//! The target is binary — [`Regime::WellFit`] aims at the floor, anything else
//! aims at [`CEILING`] — and the *approach* is what carries the confidence. A
//! run that stays converged accumulates more decisions and therefore glides
//! further down; a run that only brushes `WellFit` takes one 25% step and stops.
//! Depth of convergence is expressed in observations, not in a graded number
//! this module would have to invent. `stratum` reports `h0_death` (the
//! diffuseness of the converged noise ball) as evidence and explicitly does not
//! gate on it; grading the share by it would gate on it here, one layer up,
//! with no ground truth to calibrate against.
//!
//! From the first `WellFit` verdict to the floor is
//! `HYSTERESIS - 1 + ceil(ln(floor)/ln(1 - MAX_RECLAIM_FRAC))` observations:
//! 2 + 5 = 7 at the defaults.
//!
//! # The five rules
//!
//! 1. **Never reclaim on an unmeasurable signal** — [`read_signal`] returns
//!    [`Signal::Unmeasurable`] and [`step`] leaves `share` bit-identical. Not
//!    reduced, not raised. See that function for why each state qualifies.
//! 2. **Reclaim reverses** — any measured regime other than `WellFit` restores
//!    to [`CEILING`] in one decision. The model cannot ask for its capacity
//!    back, so the OS must give it back unprompted.
//! 3. **A floor** — set at construction by [`with_floor`], never derived from
//!    the signal, never crossed.
//! 4. **Hysteresis of [`HYSTERESIS`]** — see that constant for the number and
//!    the reason.
//! 5. **Bounded per step** — no decision reduces the share by more than
//!    [`MAX_RECLAIM_FRAC`] of its current value.
//!
//! # Why the unmeasurable rule is the load-bearing one
//!
//! `WellFit` is the verdict this module reclaims on, and `stratum` has already
//! shipped a bug that reported it wrongly. When every pairwise distance in the
//! delay embedding overflowed, Prim's algorithm recorded no finite edge and
//! returned `(0.0, 0.0)`; a `raw_max` below the epsilon floor is also what a
//! genuinely coincident cloud produces, so control fell into the
//! every-point-coincides branch and *fabricated* `shatter=1.000 h0_death=0
//! loop=0`. A run whose validation loss had diverged to 1e200 read `WellFit`
//! with every signal finite. That was fixed by marking the measurement
//! unmeasurable rather than inventing one
//! ([`FitSignals::measurable`]).
//!
//! A tuner that treated an uncomputable signal as evidence would have turned
//! that class of bug into a resource attack on the kernel's own workloads:
//! starve the diverging run precisely because its numbers stopped meaning
//! anything. [`read_signal`] therefore fails to *no action* rather than to a
//! verdict, which is a stricter posture than
//! [`classify`]'s — that function must return some
//! regime, so it fails closed to `Collapsing`.

use crate::ml_engine::stratum::{classify, FitCalibration, FitSignals, Regime};

// ── Policy constants ────────────────────────────────────────────────────────

/// The full grant. A share of 1.0 is everything the run was given, which is what
/// it holds before any reclaim and what it is restored to after one reverses.
pub const CEILING: f64 = 1.0;

/// Largest fraction of the *current* share a single decision may reclaim.
///
/// The bound exists because a wrong signal is wrong by less when it can only act
/// gradually: one spurious `WellFit` streak costs the run a quarter of its share
/// and no more, and the next non-`WellFit` verdict returns all of it. 0.25 is
/// the largest step for which the full descent from [`CEILING`] to the default
/// floor still takes more than one window of observations to complete (5
/// decisions), so a transient cannot empty the allocation before the detector
/// has re-measured.
///
/// The bound is asserted on reductions only. Restoring is bounded by nothing,
/// because the direction this rule protects against is starving a run, and a
/// restore can only hand back capacity this module itself took.
pub const MAX_RECLAIM_FRAC: f64 = 0.25;

/// Consecutive observations of the same class required before the share moves.
///
/// **N = 3.** Two failure modes have to be survived, and 3 is the smallest N
/// that survives both:
///
/// * an alternating flap (`WellFit`, `Overfit`, `WellFit`, …) never accumulates
///   two consecutive of either class, so N ≥ 2 already holds it;
/// * a *two-observation transient* — the regime that flips while a spike sits
///   inside the quartile the drift statistic is computed over, and flips back
///   once it leaves — needs N ≥ 3. `quartile_drift` averages `n/4 = 16` points
///   of a full window, so a single outlier moves the estimate for as long as it
///   remains in the tail quartile, which is more than one observation.
///
/// The cost is exactly two observations of reclaim latency on top of the
/// 5-decision glide. Raising it further buys nothing against a transient that
/// already persists longer than the detector's own quartile.
pub const HYSTERESIS: u32 = 3;

/// Floor used when a caller has no workload-specific deadline to set one from.
///
/// This is a policy input, not a measurement. The right floor is the share below
/// which the run misses a deadline, and the kernel cannot see a deadline — so
/// [`with_floor`] takes it as an argument and this value is only the default.
/// 0.25 keeps a converged run moving at a quarter rate, which bounds the damage
/// of a misclassified still-training run to a 4× slowdown rather than a stall.
pub const DEFAULT_FLOOR: f64 = 0.25;

/// Points below which the window holds no geometry at all.
///
/// `stratum` embeds in 3 dimensions and `FitStream::measure` returns the empty
/// signal set below 3 points — `spread = 1.0`, which [`classify`] reads as
/// converged. That default is a placeholder, not a measurement, and the
/// `min_samples` warm-up gate does not always cover it: `SYS_FIT_CALIBRATE`
/// accepts `min_samples = 0` (field 5, value 0.0 is in range), after which an
/// empty stream classifies `WellFit`.
const MIN_POINTS: usize = 3;

// ── Signal ──────────────────────────────────────────────────────────────────

/// What the tuner was able to learn from one reading of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// The regime was computed from evidence, and is safe to act on.
    Measured(Regime),
    /// The regime could not be computed from this reading. Not evidence of
    /// convergence, not evidence of collapse, not evidence of anything.
    Unmeasurable,
}

/// Classify one reading of a stream into something the tuner may act on.
///
/// The cascade mirrors [`classify`]'s, and diverges from it in exactly one
/// place: where `classify` must return some regime and so fails closed to
/// `Collapsing`, this fails closed to [`Signal::Unmeasurable`] and no action.
///
/// 1. A latched non-finite *input* is a real, positive observation of collapse —
///    the run produced a NaN. Ahead of the warm-up gate, as in `classify`,
///    because more samples cannot unmake it.
/// 2. A signal set that is not [`FitSignals::measurable`] is the historical
///    fabrication path: the cloud's own scale overflowed and the geometry is
///    unknown. Freeze.
/// 3. Fewer than [`MIN_POINTS`] points, or fewer samples than the calibration's
///    warm-up gate. `classify` answers `WellFit` in both cases *by default*, not
///    by measurement, and reclaiming on a default is how a run gets starved in
///    its first sixteen steps.
///
/// Past those, `classify` cannot reach either of its own fail-closed branches,
/// so whatever it returns was measured.
pub fn read_signal(sig: &FitSignals, cal: &FitCalibration) -> Signal {
    if sig.nonfinite > 0 {
        return Signal::Measured(Regime::Collapsing);
    }
    if !sig.measurable() {
        return Signal::Unmeasurable;
    }
    if sig.points < MIN_POINTS || sig.samples < cal.min_samples {
        return Signal::Unmeasurable;
    }
    Signal::Measured(classify(sig, cal))
}

// ── Tuner ───────────────────────────────────────────────────────────────────

/// The whole tuner state: plain `Copy` data, so every rule below is reachable
/// from a free function with no allocator, no lock and no live scheduler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tuner {
    /// Lower bound on [`Tuner::share`]. Fixed by [`with_floor`]; nothing in this
    /// module writes it again, and no signal contributes to it.
    pub floor: f64,
    /// Current target share of the run's initial grant, in `[floor, CEILING]`.
    pub share: f64,
    /// Consecutive [`Signal::Measured`] `WellFit` readings.
    pub steady: u32,
    /// Consecutive [`Signal::Measured`] readings of any other regime.
    pub unstable: u32,
}

/// Construct a tuner holding the full grant, with a floor it can never go below.
///
/// Returns `None` for a floor no allocation could use: non-finite, at or below
/// zero (convergence never means zero — a run at the floor must still make
/// progress), or above [`CEILING`]. Refused, never clamped, for the reason
/// [`FitCalibration::set_field`] gives: a clamp reports success for a bound the
/// caller did not ask for.
pub fn with_floor(floor: f64) -> Option<Tuner> {
    if !floor.is_finite() || floor <= 0.0 || floor > CEILING {
        return None;
    }
    Some(Tuner {
        floor,
        share: CEILING,
        steady: 0,
        unstable: 0,
    })
}

/// Advance the tuner by one observation.
///
/// Pure: state in, state out. The three arms are the three rules that decide
/// whether anything happens at all; the arithmetic inside the first is rules 3
/// and 5.
pub fn step(t: Tuner, signal: Signal) -> Tuner {
    match signal {
        // RULE 1. Nothing was measured, so nothing is evidence. The share is
        // returned untouched, and both streaks reset: "N consecutive
        // observations" means N consecutive *measurements*, and a gap in the
        // evidence breaks the chain. Resetting can only delay a reclaim.
        Signal::Unmeasurable => Tuner {
            steady: 0,
            unstable: 0,
            ..t
        },

        Signal::Measured(Regime::WellFit) => {
            let steady = t.steady.saturating_add(1);
            if steady < HYSTERESIS {
                // RULE 4.
                return Tuner {
                    steady,
                    unstable: 0,
                    ..t
                };
            }
            // RULE 5 sets how far this decision may go; RULE 3 stops it there.
            // `max` applies the floor to the bounded step rather than the other
            // way round, so the floor is a hard stop and not a target the step
            // could overshoot.
            let bounded = t.share * (1.0 - MAX_RECLAIM_FRAC);
            Tuner {
                share: bounded.max(t.floor).clamp(t.floor, CEILING),
                steady,
                unstable: 0,
                ..t
            }
        }

        // RULE 2. The regime left `WellFit`: the run destabilised, and the model
        // has no way to ask for its capacity back. Restore the whole grant in
        // one decision — `MAX_RECLAIM_FRAC` bounds reclaims, because the
        // direction that damages a run is the one that starves it, and this can
        // only return capacity this module itself took.
        Signal::Measured(_) => {
            let unstable = t.unstable.saturating_add(1);
            if unstable < HYSTERESIS {
                // RULE 4.
                return Tuner {
                    unstable,
                    steady: 0,
                    ..t
                };
            }
            Tuner {
                share: CEILING,
                steady: 0,
                unstable,
                ..t
            }
        }
    }
}

/// Capacity this tuner has released for other work, as a fraction of the grant.
pub fn reclaimed(t: Tuner) -> f64 {
    CEILING - t.share
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::ml_engine::stratum::{
        run_case, FitStream, ProofCase, DEFAULT_CALIBRATION, STRATUM_WINDOW,
    };
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Signals from a real converged run, straight out of the detector.
    fn converged() -> FitSignals {
        run_case(ProofCase::WellFit).1
    }

    /// Signals from a real overfitting run: the regime has left `WellFit`.
    fn destabilised() -> FitSignals {
        run_case(ProofCase::Overfit).1
    }

    /// The historical fabrication trigger: validation loss diverged to 1e200, so
    /// every pairwise distance in the delay embedding overflows and the cloud's
    /// own scale is unknown. Every input is finite, so the non-finite latch never
    /// arms. Before the `stratum` fix this reported `WellFit` with `shatter=1.000
    /// h0_death=0 loop=0`.
    fn scale_overflowed() -> FitSignals {
        let mut s = FitStream::new(DEFAULT_CALIBRATION);
        for i in 0..40u32 {
            s.observe(0.5, if i % 2 == 0 { 1e200 } else { 2e200 });
        }
        s.signals()
    }

    fn drive(mut t: Tuner, sig: &FitSignals, n: usize) -> Tuner {
        for _ in 0..n {
            t = step(t, read_signal(sig, &DEFAULT_CALIBRATION));
        }
        t
    }

    /// RULE 1. An unmeasurable signal leaves the allocation exactly as it is.
    ///
    /// Three ways to be unmeasurable, all built from a real stream rather than a
    /// hand-written struct: the overflowed cloud, the empty stream, and the
    /// warm-up window. All three are checked from the full grant *and* from an
    /// already-reclaimed share, because "not reduced" and "not raised" are two
    /// different failures.
    fn test_unmeasurable_never_moves_share() -> TestResult {
        let empty = FitStream::new(DEFAULT_CALIBRATION).signals();
        let mut warm = FitStream::new(DEFAULT_CALIBRATION);
        for t in 0..8usize {
            let (a, b) = ProofCase::WellFit.sample(t);
            warm.observe(a, b);
        }
        let warm = warm.signals();
        let overflowed = scale_overflowed();

        test_assert!(
            overflowed.nonfinite == 0,
            "the overflow trigger must not arm the non-finite latch, or it proves nothing"
        );
        test_assert!(
            warm.samples < DEFAULT_CALIBRATION.min_samples,
            "the warm-up fixture must actually be inside the warm-up gate"
        );
        for sig in [&overflowed, &empty, &warm] {
            test_assert_eq!(
                read_signal(sig, &DEFAULT_CALIBRATION),
                Signal::Unmeasurable
            );
        }

        // A `min_samples` of 0 is an accepted ABI calibration, so the warm-up
        // gate cannot be the only thing standing between an empty window and a
        // reclaim.
        let mut open = DEFAULT_CALIBRATION;
        test_assert!(open.set_field(5, 0.0), "min_samples=0 is an accepted value");
        test_assert_eq!(read_signal(&empty, &open), Signal::Unmeasurable);

        let full = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let reclaimed_state = drive(full, &converged(), 64);
        test_assert!(
            reclaimed_state.share < CEILING,
            "the reclaimed fixture must have actually reclaimed, or the raise check is vacuous"
        );

        for start in [full, reclaimed_state] {
            for sig in [&overflowed, &empty, &warm] {
                let after = drive(start, sig, 128);
                test_assert!(
                    after.share == start.share,
                    "an unmeasurable signal must leave the share bit-identical"
                );
            }
        }
        TestResult::Pass
    }

    /// RULE 2, first half. A converged run is reclaimed on.
    fn test_converged_run_is_reclaimed() -> TestResult {
        let sig = converged();
        test_assert_eq!(
            read_signal(&sig, &DEFAULT_CALIBRATION),
            Signal::Measured(Regime::WellFit)
        );
        let t = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let after = drive(t, &sig, HYSTERESIS as usize);
        test_assert!(
            after.share < CEILING,
            "a run measured WellFit for HYSTERESIS observations must be reclaimed on"
        );
        test_assert!(
            reclaimed(after) > 0.0,
            "the reclaimed capacity must be reported as available"
        );
        TestResult::Pass
    }

    /// RULE 2. The full round trip: converge, reclaim, destabilise, restore.
    ///
    /// This is the property that makes "no limit in the model code" safe. The
    /// model cannot ask for its capacity back, so the OS has to notice and hand
    /// it back unprompted.
    fn test_reclaim_round_trip() -> TestResult {
        let t = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let converged_state = drive(t, &converged(), 64);
        test_assert!(
            libm::fabs(converged_state.share - DEFAULT_FLOOR) < 1e-12,
            "a persistently converged run must settle at the floor"
        );

        let bad = destabilised();
        test_assert!(
            read_signal(&bad, &DEFAULT_CALIBRATION) != Signal::Measured(Regime::WellFit),
            "the destabilise fixture must have left WellFit, or the round trip proves nothing"
        );
        let restored = drive(converged_state, &bad, HYSTERESIS as usize);
        test_assert!(
            libm::fabs(restored.share - CEILING) < 1e-12,
            "leaving WellFit must give the whole reclaimed capacity back"
        );
        test_assert!(
            reclaimed(restored) < 1e-12,
            "nothing may remain reclaimed once the run has destabilised"
        );

        // And the trip is repeatable, not a one-shot latch.
        let again = drive(restored, &converged(), 64);
        test_assert!(
            libm::fabs(again.share - DEFAULT_FLOOR) < 1e-12,
            "the tuner must be able to reclaim a second time"
        );
        TestResult::Pass
    }

    /// A collapsing run — the regime this module must never starve — restores
    /// just as a destabilising one does.
    fn test_collapsing_run_is_restored() -> TestResult {
        let t = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let low = drive(t, &converged(), 64);
        test_assert!(
            low.share < CEILING,
            "the reclaimed fixture must have actually reclaimed, or the restore is vacuous"
        );
        let sig = run_case(ProofCase::Collapsing).1;
        test_assert_eq!(
            read_signal(&sig, &DEFAULT_CALIBRATION),
            Signal::Measured(Regime::Collapsing)
        );
        let after = drive(low, &sig, HYSTERESIS as usize);
        test_assert!(
            libm::fabs(after.share - CEILING) < 1e-12,
            "a collapsing run must get its capacity back, not lose more"
        );
        TestResult::Pass
    }

    /// RULE 3. The floor is never crossed, is hit exactly, and is not derived
    /// from the signal.
    fn test_floor_is_never_crossed() -> TestResult {
        for floor in [0.05, 0.25, 0.5, 0.999, CEILING] {
            let t = match with_floor(floor) {
                Some(t) => t,
                None => return TestResult::Fail("a floor in (0, CEILING] must construct"),
            };
            let mut cur = t;
            for _ in 0..10_000 {
                cur = step(cur, read_signal(&converged(), &DEFAULT_CALIBRATION));
                test_assert!(
                    cur.share >= floor,
                    "the share must never drop below the constructed floor"
                );
            }
            test_assert!(
                libm::fabs(cur.share - floor) < 1e-12,
                "a persistently converged run must settle exactly at its own floor"
            );
            test_assert!(
                cur.floor == floor,
                "the floor must not be rewritten by any observation"
            );
        }

        // A floor no allocation could use is refused, not clamped.
        for bad in [0.0, -0.25, 1.5, f64::NAN, f64::INFINITY] {
            test_assert!(
                with_floor(bad).is_none(),
                "an unusable floor must be refused"
            );
        }
        TestResult::Pass
    }

    /// RULE 4. HYSTERESIS - 1 observations change nothing; the HYSTERESIS'th
    /// acts. Both directions.
    fn test_hysteresis_is_exactly_n() -> TestResult {
        let t = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let short = drive(t, &converged(), HYSTERESIS as usize - 1);
        test_assert!(
            short.share == CEILING,
            "HYSTERESIS - 1 converged observations must not move the share"
        );
        let acted = step(short, read_signal(&converged(), &DEFAULT_CALIBRATION));
        test_assert!(
            acted.share < CEILING,
            "the HYSTERESIS'th converged observation must act"
        );

        let low = drive(t, &converged(), 64);
        let bad = destabilised();
        let short_up = drive(low, &bad, HYSTERESIS as usize - 1);
        test_assert!(
            short_up.share == low.share,
            "HYSTERESIS - 1 destabilised observations must not move the share"
        );
        let acted_up = step(short_up, read_signal(&bad, &DEFAULT_CALIBRATION));
        test_assert!(
            acted_up.share > low.share,
            "the HYSTERESIS'th destabilised observation must restore"
        );
        TestResult::Pass
    }

    /// RULE 4. A flapping regime must not oscillate the allocation. Thrashing an
    /// allocator is worse than never reclaiming at all.
    ///
    /// Two shapes: the 1-on-1-off alternation, and the two-observation transient
    /// that is the reason N is 3 rather than 2.
    fn test_flapping_does_not_move_share() -> TestResult {
        let t = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let good = read_signal(&converged(), &DEFAULT_CALIBRATION);
        let bad = read_signal(&destabilised(), &DEFAULT_CALIBRATION);
        test_assert!(good != bad, "the flap fixtures must actually differ");

        for start in [t, drive(t, &converged(), 64)] {
            let mut cur = start;
            for i in 0..1_000 {
                cur = step(cur, if i % 2 == 0 { good } else { bad });
                test_assert!(
                    cur.share == start.share,
                    "an alternating regime must not move the allocation at all"
                );
            }
            // Two-observation transient, repeated: still under N = 3.
            let mut cur = start;
            for i in 0..1_000 {
                cur = step(cur, if (i / 2) % 2 == 0 { good } else { bad });
                test_assert!(
                    cur.share == start.share,
                    "a two-observation transient must not move the allocation"
                );
            }
        }
        TestResult::Pass
    }

    /// RULE 5. No single decision reclaims more than MAX_RECLAIM_FRAC of the
    /// current share, on any path the tuner can take.
    fn test_per_step_reclaim_is_bounded() -> TestResult {
        let good = read_signal(&converged(), &DEFAULT_CALIBRATION);
        let bad = read_signal(&destabilised(), &DEFAULT_CALIBRATION);
        let unknown = read_signal(&scale_overflowed(), &DEFAULT_CALIBRATION);
        // A deterministic walk over every signal class, seeded by the index so
        // the same path is taken on every boot.
        for floor in [0.05, DEFAULT_FLOOR, 0.5] {
            let mut cur = match with_floor(floor) {
                Some(t) => t,
                None => return TestResult::Fail("a floor in (0, CEILING] must construct"),
            };
            for i in 0..2_000u64 {
                let sig = match (i.wrapping_mul(2_654_435_761) >> 8) % 8 {
                    0 => bad,
                    1 => unknown,
                    _ => good,
                };
                let next = step(cur, sig);
                test_assert!(
                    next.share >= cur.share * (1.0 - MAX_RECLAIM_FRAC) - 1e-12,
                    "no decision may reclaim more than MAX_RECLAIM_FRAC of the current share"
                );
                test_assert!(
                    next.share >= floor && next.share <= CEILING,
                    "the share must stay inside [floor, CEILING] on every path"
                );
                cur = next;
            }
        }
        TestResult::Pass
    }

    /// The documented descent length is a claim about this code, so it is
    /// asserted rather than left in prose: HYSTERESIS - 1 idle observations then
    /// 5 decisions to the default floor.
    fn test_descent_length_matches_documentation() -> TestResult {
        let t = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let good = read_signal(&converged(), &DEFAULT_CALIBRATION);
        let mut cur = t;
        let mut n = 0usize;
        while cur.share > DEFAULT_FLOOR + 1e-12 && n < 1_000 {
            cur = step(cur, good);
            n += 1;
        }
        test_assert!(
            n == HYSTERESIS as usize - 1 + 5,
            "the documented 7-observation descent must hold"
        );
        TestResult::Pass
    }

    /// The window is bounded, so the tuner must be too: driving it far past a
    /// full window must not overflow a streak counter into a wrap.
    fn test_streak_counters_saturate() -> TestResult {
        let mut t = match with_floor(DEFAULT_FLOOR) {
            Some(t) => t,
            None => return TestResult::Fail("with_floor(DEFAULT_FLOOR) must construct"),
        };
        let good = read_signal(&converged(), &DEFAULT_CALIBRATION);
        t.steady = u32::MAX;
        let after = step(t, good);
        test_assert!(
            after.steady >= HYSTERESIS,
            "a saturated streak must not wrap back below the hysteresis threshold"
        );
        test_assert!(
            STRATUM_WINDOW > 0,
            "the detector window must be non-empty for any of this to mean anything"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "tuner::unmeasurable_never_moves_share",
            test_unmeasurable_never_moves_share,
        );
        crate::testing::register_test(
            "tuner::converged_run_is_reclaimed",
            test_converged_run_is_reclaimed,
        );
        crate::testing::register_test("tuner::reclaim_round_trip", test_reclaim_round_trip);
        crate::testing::register_test(
            "tuner::collapsing_run_is_restored",
            test_collapsing_run_is_restored,
        );
        crate::testing::register_test("tuner::floor_is_never_crossed", test_floor_is_never_crossed);
        crate::testing::register_test("tuner::hysteresis_is_exactly_n", test_hysteresis_is_exactly_n);
        crate::testing::register_test(
            "tuner::flapping_does_not_move_share",
            test_flapping_does_not_move_share,
        );
        crate::testing::register_test(
            "tuner::per_step_reclaim_is_bounded",
            test_per_step_reclaim_is_bounded,
        );
        crate::testing::register_test(
            "tuner::descent_length_matches_documentation",
            test_descent_length_matches_documentation,
        );
        crate::testing::register_test(
            "tuner::streak_counters_saturate",
            test_streak_counters_saturate,
        );
    }
}
