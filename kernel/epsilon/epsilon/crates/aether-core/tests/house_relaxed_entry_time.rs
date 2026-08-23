//! The bridge from Sheehy's alpha-dependent relaxed distance to a fixed
//! filtration value that `persistence.rs` can consume.
//!
//! `d_alpha(p,q) = d + w_p(alpha) + w_q(alpha)` moves with alpha, so it is not
//! a distance matrix. But Lemma 4.1 says once `d_alpha(p,q) <= alpha` holds it
//! holds for every beta >= alpha, so the set of admitting alphas is an upward
//! ray and its infimum is a well-defined entry time.
use aether_core::nettree::{relaxed_distance, relaxed_entry_time};

const EPSS: [f64; 4] = [0.05, 0.1, 0.2, 1.0 / 3.0];

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[test]
fn the_entry_time_admits_and_the_instant_before_does_not() {
    // The defining property. At the entry time the pair is admitted; strictly
    // below it, it is not.
    let mut u = rng(4242);
    for &e in &EPSS {
        for _ in 0..3000 {
            let (d, tp, tq) = (3.0 * u() + 1e-6, 4.0 * u(), 4.0 * u());
            let a = relaxed_entry_time(d, tp, tq, e);
            assert!(
                a.is_finite(),
                "entry time not finite for d={d} tp={tp} tq={tq}"
            );
            assert!(
                relaxed_distance(d, a, tp, tq, e) <= a + 1e-9,
                "not admitted at its own entry time: d={d} tp={tp} tq={tq} alpha={a}"
            );
            let below = a * (1.0 - 1e-6) - 1e-9;
            if below > 0.0 {
                assert!(
                    relaxed_distance(d, below, tp, tq, e) > below,
                    "admitted strictly below the entry time: d={d} tp={tp} tq={tq} alpha={below}"
                );
            }
        }
    }
}

#[test]
fn the_entry_time_is_never_earlier_than_the_true_distance() {
    // d_alpha >= d always, so a pair cannot enter before its true distance.
    let mut u = rng(77);
    for &e in &EPSS {
        for _ in 0..3000 {
            let (d, tp, tq) = (3.0 * u() + 1e-6, 4.0 * u(), 4.0 * u());
            let a = relaxed_entry_time(d, tp, tq, e);
            assert!(
                a >= d - 1e-9,
                "entered at {a}, earlier than the true distance {d}"
            );
        }
    }
}

#[test]
fn zero_weight_recovers_the_exact_rips_value() {
    // With deletion times far beyond the scale, both weights are 0 at the
    // relevant alpha, so the entry time IS the true distance — the relaxed
    // filtration degenerates to the exact one, which is the reduction Sheehy's
    // construction must satisfy.
    for &e in &EPSS {
        for &d in &[0.1f64, 0.5, 1.0, 2.5] {
            let far = 1e6;
            let a = relaxed_entry_time(d, far, far, e);
            assert!(
                (a - d).abs() < 1e-9,
                "eps={e} d={d}: entry time {a} should equal the true distance"
            );
        }
    }
}

#[test]
fn the_entry_time_is_monotone_in_the_true_distance() {
    // Farther pairs enter no earlier. Without this the filtration is not a
    // filtration.
    let mut u = rng(31337);
    for &e in &EPSS {
        for _ in 0..2000 {
            let (tp, tq) = (4.0 * u(), 4.0 * u());
            let d1 = 3.0 * u() + 1e-6;
            let d2 = d1 + 2.0 * u();
            let (a1, a2) = (
                relaxed_entry_time(d1, tp, tq, e),
                relaxed_entry_time(d2, tp, tq, e),
            );
            assert!(
                a2 >= a1 - 1e-9,
                "monotonicity broken: d {d1}->{d2} gave entry {a1}->{a2}"
            );
        }
    }
}

#[test]
fn a_shorter_deletion_time_never_makes_a_pair_enter_earlier() {
    // Weights only ever grow the distance, and deleting sooner grows them
    // sooner, so a pair can only enter later.
    let mut u = rng(8888);
    for &e in &EPSS {
        for _ in 0..2000 {
            let d = 2.0 * u() + 1e-6;
            let t_long = 4.0 * u() + 1.0;
            let t_short = t_long * u();
            let a_long = relaxed_entry_time(d, t_long, t_long, e);
            let a_short = relaxed_entry_time(d, t_short, t_short, e);
            assert!(
                a_short >= a_long - 1e-9,
                "deleting sooner made the pair enter earlier: {a_short} < {a_long}"
            );
        }
    }
}

#[test]
fn malformed_distances_are_rejected_rather_than_silently_accepted() {
    // No caller can produce these — `ManifoldPoint::distance` is a norm — but
    // the guard exists, so it is tested. An untested guard is indistinguishable
    // from an absent one, and the mutation gate demonstrated exactly that by
    // loosening the bound and surviving.
    for &bad in &[-1e-9f64, -0.5, -1.0, -1e9, f64::NAN, f64::NEG_INFINITY] {
        let a = relaxed_entry_time(bad, 1.0, 1.0, 0.1);
        assert!(
            a.is_infinite() && a > 0.0,
            "malformed distance {bad} produced entry time {a}, not +inf"
        );
    }
    // And a distance of exactly zero is legitimate: coincident points.
    let z = relaxed_entry_time(0.0, 1.0, 1.0, 0.1);
    assert_eq!(z, 0.0, "coincident points must enter at 0, got {z}");
}

#[test]
fn the_early_exit_cannot_report_a_time_below_the_true_distance() {
    // `relaxed_entry_time` short-circuits when `admits(d)` holds. If that probe
    // were taken at the wrong point the function could return a time below `d`,
    // which would silently admit edges that do not belong in the filtration.
    // d_alpha >= d always, so the entry time can never be less than d.
    let mut u = rng(0xC0FFEE);
    for &e in &EPSS {
        for _ in 0..5000 {
            let d = 4.0 * u() + 1e-9;
            let (tp, tq) = (5.0 * u(), 5.0 * u());
            let a = relaxed_entry_time(d, tp, tq, e);
            assert!(
                a >= d - 1e-12,
                "entry {a} below the true distance {d} (tp={tp} tq={tq} eps={e})"
            );
        }
    }
}
