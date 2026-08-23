//! Iteration 9 — the relaxed distance layer, tested against Sheehy's OWN stated
//! properties. Every assertion here is a claim the paper makes about its own
//! construction (arXiv:1203.6786 section 4), turned into an executable check.
use aether_core::nettree::{relaxed_distance, weight};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

const EPSS: [f64; 4] = [0.05, 0.1, 0.2, 1.0 / 3.0];

#[test]
fn weight_is_continuous_at_both_breakpoints() {
    for &e in &EPSS {
        for &t in &[0.0f64, 0.5, 1.0, 7.3] {
            let b1 = (1.0 - 2.0 * e) * t;
            for &brk in &[b1, t] {
                let h = 1e-9;
                let (lo, hi) = (weight(brk - h, t, e), weight(brk + h, t, e));
                let at = weight(brk, t, e);
                assert!(
                    (lo - at).abs() < 1e-6 && (hi - at).abs() < 1e-6,
                    "eps={e} t={t}: discontinuity at {brk}: {lo} / {at} / {hi}"
                );
            }
        }
    }
}

#[test]
fn weight_is_half_lipschitz_in_alpha() {
    // Paper: "The weight of a point is 1/2-Lipschitz in alpha".
    let mut u = rng(4242);
    for &e in &EPSS {
        for _ in 0..4000 {
            let t = 5.0 * u();
            let (a, b) = (5.0 * u(), 5.0 * u());
            let d = (weight(a, t, e) - weight(b, t, e)).abs();
            assert!(
                d <= 0.5 * (a - b).abs() + 1e-12,
                "eps={e} t={t}: |w({a})-w({b})|={d} > 0.5*|a-b|"
            );
        }
    }
}

#[test]
fn relaxed_distance_dominates_and_is_monotone() {
    // Paper: "d_alpha is monotonically non-decreasing in alpha. In particular,
    // d_alpha >= d_0 = d for all alpha >= 0."
    let mut u = rng(77);
    for &e in &EPSS {
        for _ in 0..2000 {
            let (d, tp, tq) = (3.0 * u(), 4.0 * u(), 4.0 * u());
            assert!(
                (relaxed_distance(d, 0.0, tp, tq, e) - d).abs() < 1e-12,
                "d_0 must equal d"
            );
            let mut prev = d;
            let mut a = 0.0;
            for _ in 0..60 {
                a += 0.1;
                let cur = relaxed_distance(d, a, tp, tq, e);
                assert!(
                    cur >= prev - 1e-12,
                    "not monotone at alpha={a}: {prev} -> {cur}"
                );
                assert!(cur >= d - 1e-12, "relaxed distance fell below d");
                prev = cur;
            }
        }
    }
}

#[test]
fn lemma_4_1_holds() {
    // Paper, Lemma 4.1: if d_alpha(p,q) <= alpha <= beta then d_beta(p,q) <= beta.
    let mut u = rng(31337);
    let mut exercised = 0u32;
    for &e in &EPSS {
        for _ in 0..6000 {
            let (d, tp, tq) = (3.0 * u(), 4.0 * u(), 4.0 * u());
            let a = 4.0 * u();
            if relaxed_distance(d, a, tp, tq, e) > a {
                continue;
            } // hypothesis fails
            exercised += 1;
            let b = a + 4.0 * u();
            let db = relaxed_distance(d, b, tp, tq, e);
            assert!(
                db <= b + 1e-12,
                "Lemma 4.1 violated: eps={e} d={d} tp={tp} tq={tq} alpha={a} beta={b} -> {db}"
            );
        }
    }
    assert!(
        exercised > 500,
        "hypothesis almost never held; test was near-vacuous ({exercised} cases)"
    );
    println!("Lemma 4.1 exercised on {exercised} cases where the hypothesis actually held");
}

#[test]
fn the_sandwich_both_directions() {
    // Paper: if d(p,q) <= alpha/c with 1/c = 1 - 2*eps then d_alpha(p,q) <= alpha.
    // And conversely d_alpha >= d, so d_alpha(p,q) <= alpha implies d(p,q) <= alpha.
    let mut u = rng(8888);
    for &e in &EPSS {
        for _ in 0..4000 {
            let a = 4.0 * u() + 0.01;
            let (tp, tq) = (4.0 * u(), 4.0 * u());
            let d = (1.0 - 2.0 * e) * a * u(); // d <= (1-2eps)*alpha
            let da = relaxed_distance(d, a, tp, tq, e);
            assert!(
                da <= a + 1e-9,
                "forward sandwich failed: eps={e} d={d} alpha={a} -> {da}"
            );
            if da <= a {
                assert!(d <= a + 1e-12, "reverse sandwich failed");
            }
        }
    }
}
