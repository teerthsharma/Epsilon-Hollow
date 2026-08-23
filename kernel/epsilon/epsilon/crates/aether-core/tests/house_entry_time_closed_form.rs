//! The entry time in closed form, checked against the bisection it replaces.
//!
//! `relaxed_entry_time` finds the root of a monotone predicate by 200 halvings.
//! The predicate is piecewise **affine** with at most five pieces, so the root
//! is available exactly. This file is the control on that claim: the bisection
//! is retained as a reference implementation and the closed form must reproduce
//! it everywhere, which is the same shape of gate that `stratum` runs against
//! its naive baseline and `foliation` against its Belady oracle.

use aether_core::nettree::{relaxed_distance, relaxed_entry_time, relaxed_entry_time_exact};

/// A deterministic spread of arguments covering every ordering of the four
/// breakpoints, including the degenerate ones the clamps produce.
fn sweep() -> Vec<(f64, f64, f64, f64)> {
    let ds = [0.0, 1e-9, 0.01, 0.1, 0.5, 1.0, 2.0, 7.5, 100.0];
    let ts = [0.0, 1e-9, 0.05, 0.3, 1.0, 3.0, 50.0, f64::INFINITY];
    let es = [1.0 / 3.0, 0.25, 0.1, 0.02, 1e-6];
    let mut out = Vec::new();
    for &d in &ds {
        for &t_p in &ts {
            for &t_q in &ts {
                for &eps in &es {
                    out.push((d, t_p, t_q, eps));
                }
            }
        }
    }
    out
}

fn close(a: f64, b: f64) -> bool {
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn closed_form_reproduces_the_bisection_everywhere_on_the_sweep() {
    let cases = sweep();
    let mut worst = 0.0f64;
    let mut worst_at = (0.0, 0.0, 0.0, 0.0);
    let mut disagreements = 0usize;
    for &(d, t_p, t_q, eps) in &cases {
        let bisected = relaxed_entry_time(d, t_p, t_q, eps);
        let exact = relaxed_entry_time_exact(d, t_p, t_q, eps);
        if !close(bisected, exact) {
            disagreements += 1;
        }
        let gap = (bisected - exact).abs() / bisected.abs().max(1.0);
        if gap.is_finite() && gap > worst {
            worst = gap;
            worst_at = (d, t_p, t_q, eps);
        }
    }
    println!(
        "swept {} tuples; worst relative gap {:.3e} at d={} t_p={} t_q={} eps={}",
        cases.len(),
        worst,
        worst_at.0,
        worst_at.1,
        worst_at.2,
        worst_at.3
    );
    assert_eq!(
        disagreements,
        0,
        "closed form disagrees with the bisection on {} of {} tuples",
        disagreements,
        cases.len()
    );
}

#[test]
fn the_returned_time_admits_the_pair_and_nothing_strictly_below_it_does() {
    // The defining property, checked on the closed form directly rather than
    // inherited from the bisection: alpha admits exactly when d_alpha <= alpha.
    for &(d, t_p, t_q, eps) in &sweep() {
        let a = relaxed_entry_time_exact(d, t_p, t_q, eps);
        if !a.is_finite() {
            continue;
        }
        let slack = relaxed_distance(d, a, t_p, t_q, eps) - a;
        assert!(
            slack <= 1e-9 * a.abs().max(1.0),
            "entry time {a} does not admit (d={d} t_p={t_p} t_q={t_q} eps={eps}, slack {slack})"
        );
        if a > 0.0 {
            let below = a * (1.0 - 1e-6);
            let s2 = relaxed_distance(d, below, t_p, t_q, eps) - below;
            assert!(
                s2 > -1e-9 * a.abs().max(1.0),
                "a strictly smaller alpha={below} also admits, so {a} is not the entry time"
            );
        }
    }
}

/// **Lemma D.** `d_alpha(p,q) >= d(p,q)` for every alpha, because both weights
/// are non-negative. Hence the entry time is never below the true distance, and
/// a pair can only appear in the filtration truncated at `alpha_max` when
/// `d <= alpha_max`.
///
/// That is what makes edge discovery a **fixed-radius** query rather than a
/// per-pair one: the candidate set is exactly the pairs within `alpha_max`, and
/// the filter has no false negatives.
#[test]
fn lemma_d_entry_time_is_never_below_the_true_distance() {
    for &(d, t_p, t_q, eps) in &sweep() {
        let a = relaxed_entry_time_exact(d, t_p, t_q, eps);
        assert!(
            a >= d - 1e-12,
            "entry time {a} is below the true distance {d} (t_p={t_p} t_q={t_q} eps={eps})"
        );
        // Without this the test passes for `fn(..) -> f64 { f64::INFINITY }`,
        // which satisfies "never below the true distance" trivially. Every
        // finite non-negative distance HAS a finite entry time, because the
        // final interval has slope `1 - 2 eps >= 1/3`, so requiring finiteness
        // is not an extra assumption - it is the other half of the lemma.
        assert!(
            d.is_finite() == a.is_finite(),
            "a finite distance {d} must have a finite entry time, got {a}              (t_p={t_p} t_q={t_q} eps={eps})"
        );
    }
}

/// **Lemma E.** Once `alpha >= max(t_p, t_q)` both weights are `eps * alpha`, so
/// admission reads `d + 2 eps alpha <= alpha`, that is `alpha >= d / (1 - 2 eps)`.
/// Whenever the entry time lands in that regime it equals `d / (1 - 2 eps)`
/// exactly. A closed form, not a limit.
#[test]
fn lemma_e_above_both_deletion_times_the_entry_time_is_d_over_one_minus_two_eps() {
    let mut checked = 0usize;
    for &(d, t_p, t_q, eps) in &sweep() {
        let e = if eps.is_finite() && eps > 0.0 && eps <= 1.0 / 3.0 {
            eps
        } else {
            1.0 / 3.0
        };
        let tp = if t_p.is_finite() && t_p > 0.0 {
            t_p
        } else {
            0.0
        };
        let tq = if t_q.is_finite() && t_q > 0.0 {
            t_q
        } else {
            0.0
        };
        let a = relaxed_entry_time_exact(d, t_p, t_q, eps);
        if !a.is_finite() || a < tp.max(tq) {
            continue;
        }
        checked += 1;
        let predicted = d / (1.0 - 2.0 * e);
        assert!(
            close(a, predicted),
            "in the eps regime the entry time should be {predicted}, got {a} (d={d} t_p={t_p} t_q={t_q} eps={eps})"
        );
    }
    println!("Lemma E exercised on {checked} tuples that reached the eps regime");
    assert!(checked > 100, "only {checked} tuples reached the regime");
}

/// **Lemma F.** With both deletion times far beyond the scale both weights
/// vanish and the entry time **is** the true distance: the relaxed filtration
/// degenerates to the exact Rips filtration. That is the reduction the whole
/// construction has to satisfy, restated for the closed form.
#[test]
fn lemma_f_far_deletion_times_degenerate_to_the_exact_rips_value() {
    for &d in &[0.0, 1e-6, 0.25, 1.0, 4.0, 91.0] {
        for &eps in &[1.0 / 3.0, 0.1, 1e-4] {
            let far = 1e6 * (d + 1.0);
            let a = relaxed_entry_time_exact(d, far, far, eps);
            assert!(
                close(a, d),
                "with deletion times at {far} the entry time should be the distance {d}, got {a} (eps={eps})"
            );
        }
    }
}

#[test]
fn monotone_in_the_true_distance() {
    for &t_p in &[0.2, 1.0, 9.0] {
        for &t_q in &[0.05, 2.0, 40.0] {
            for &eps in &[1.0 / 3.0, 0.2, 0.01] {
                let mut prev = f64::NEG_INFINITY;
                for k in 0..200 {
                    let d = k as f64 * 0.05;
                    let a = relaxed_entry_time_exact(d, t_p, t_q, eps);
                    assert!(
                        a >= prev - 1e-9,
                        "entry time fell from {prev} to {a} as d rose to {d}"
                    );
                    prev = a;
                }
            }
        }
    }
}

#[test]
fn rejects_the_inputs_the_bisection_rejects() {
    for &bad in &[-1.0, -1e-9, f64::NAN, f64::NEG_INFINITY] {
        assert!(
            relaxed_entry_time_exact(bad, 1.0, 1.0, 0.1).is_infinite(),
            "a distance of {bad} should not produce a finite entry time"
        );
    }
    assert!(relaxed_entry_time_exact(f64::INFINITY, 1.0, 1.0, 0.1).is_infinite());
}

/// The closed form exists to remove 200 halvings from an O(n^2) pair loop.
/// The ratio is machine dependent and is reported rather than asserted; the
/// assertion is only that the closed form is not *slower*, which is the outcome
/// that would invalidate the change.
#[test]
fn closed_form_is_not_slower_than_the_bisection() {
    use std::time::Instant;
    let cases = sweep();
    let reps = 40;

    let t0 = Instant::now();
    let mut sink = 0.0f64;
    for _ in 0..reps {
        for &(d, t_p, t_q, eps) in &cases {
            sink += relaxed_entry_time(d, t_p, t_q, eps).min(1e12);
        }
    }
    let bisect = t0.elapsed();

    let t1 = Instant::now();
    let mut sink2 = 0.0f64;
    for _ in 0..reps {
        for &(d, t_p, t_q, eps) in &cases {
            sink2 += relaxed_entry_time_exact(d, t_p, t_q, eps).min(1e12);
        }
    }
    let exact = t1.elapsed();

    let n = reps * cases.len();
    println!(
        "{n} evaluations: bisection {:?}, closed form {:?}, ratio {:.1}x (sinks {sink:.3} / {sink2:.3})",
        bisect,
        exact,
        bisect.as_secs_f64() / exact.as_secs_f64().max(1e-12)
    );
    assert!(
        exact <= bisect,
        "closed form ({exact:?}) is slower than the bisection ({bisect:?}) it replaces"
    );
}

/// The sweep above caps deletion times at 50.0 and never probes the top of the
/// f64 range. An adversarial reading of the interval walk found that
/// `0.5 * (lo + hi)` forms the sum **before** halving, so it overflows to
/// infinity whenever `lo + hi > f64::MAX`. `weight_affine(inf, ..)` then falls
/// through both guards and reports the `eps` piece for both points, whatever
/// piece the interval is actually in, and the walk reads the wrong slope.
///
/// f64::MAX is 1.7976931348623157e308 and 1e308 + 1.11111e308 = 2.11111e308,
/// so the overflow is reachable with finite, in-range arguments. This checks the
/// returned value against a from-scratch scan rather than against the bisection,
/// which brackets by doubling and has the same range problem.
#[test]
fn very_large_deletion_times_do_not_overflow_the_interval_probe() {
    let cases = [
        (9.2e307f64, 1e308f64, 1.11111e308f64, 0.05f64),
        (1.0e307, 1.5e308, 1.6e308, 0.1),
        (5.0e306, 9.0e307, 1.7e308, 1.0 / 3.0),
    ];
    for &(d, t_p, t_q, eps) in &cases {
        let got = relaxed_entry_time_exact(d, t_p, t_q, eps);

        // Independent least-root scan: walk the same breakpoints, but bisect
        // inside each interval on the raw predicate instead of solving a piece.
        let mut cuts = [
            0.0,
            (1.0 - 2.0 * eps) * t_p,
            t_p,
            (1.0 - 2.0 * eps) * t_q,
            t_q,
        ];
        cuts.sort_by(f64::total_cmp);
        let admits = |a: f64| relaxed_distance(d, a, t_p, t_q, eps) <= a;
        let mut expected = f64::INFINITY;
        for i in 0..cuts.len() {
            let lo = cuts[i];
            // Halve the WIDTH, never form lo + hi.
            let hi = if i + 1 < cuts.len() {
                cuts[i + 1]
            } else {
                f64::MAX
            };
            if hi <= lo || !admits(hi) {
                continue;
            }
            let (mut a, mut b) = (lo, hi);
            for _ in 0..300 {
                let mid = a + 0.5 * (b - a);
                if admits(mid) {
                    b = mid;
                } else {
                    a = mid;
                }
            }
            expected = b;
            break;
        }
        let rel = (got - expected).abs() / expected.abs().max(1.0);
        assert!(
            rel <= 1e-9,
            "at d={d} t_p={t_p} t_q={t_q} eps={eps}: closed form returned {got}, \
             least root is {expected}, relative gap {rel:.3e}"
        );
    }
}
