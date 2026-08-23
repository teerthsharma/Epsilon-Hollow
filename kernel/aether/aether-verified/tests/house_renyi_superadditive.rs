//! D11 — T3 Step 3 stated the power inequality backwards.
//!
//! For alpha > 1 the power function is SUPERadditive:
//!     (p_i + p_j)^alpha >= p_i^alpha + p_j^alpha
//! The document asserted `<=`, which is false for every alpha > 1.
//!
//! The conclusion (Renyi entropy falls per merge) is correct, but only because
//! H_alpha = log(sum p^alpha) / (1 - alpha) and 1 - alpha < 0, so a LARGER
//! power sum is a SMALLER entropy. The corrected inequality is the one the
//! argument needs; the reversed one would break it.
use aether_verified::aether_gmc::verify_renyi_nonincreasing;

/// The named mutant, written out: the claim as it appeared in the document.
fn old_claim_holds(pi: f64, pj: f64, alpha: f64) -> bool {
    (pi + pj).powf(alpha) <= pi.powf(alpha) + pj.powf(alpha)
}

fn superadditive(pi: f64, pj: f64, alpha: f64) -> bool {
    (pi + pj).powf(alpha) >= pi.powf(alpha) + pj.powf(alpha) - 1e-12
}

#[test]
fn the_document_inequality_is_false_at_every_tested_point() {
    // THE MISFIRE. If this ever passes, the correction was wrong.
    let mut counterexamples = 0;
    for &alpha in &[1.5f64, 2.0, 3.0] {
        for &(pi, pj) in &[(1.0f64, 1.0f64), (0.3, 0.2), (0.5, 0.1)] {
            if !old_claim_holds(pi, pj, alpha) { counterexamples += 1; }
            assert!(superadditive(pi, pj, alpha),
                "superadditivity failed at alpha={alpha} pi={pi} pj={pj}");
        }
    }
    println!("counterexamples to the document's `<=`: {counterexamples} of 9");
    assert_eq!(counterexamples, 9,
        "the reversed claim must fail at every tested point; got {counterexamples}/9");
}

#[test]
fn superadditivity_holds_across_alpha_and_magnitudes() {
    let mut s: u64 = 0xC0FFEE_1234_5678;
    let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64) / ((1u64 << 53) as f64) };
    for _ in 0..5000 {
        let alpha = 1.0 + 4.0 * u();          // alpha in (1, 5]
        let (pi, pj) = (u() + 1e-6, u() + 1e-6);
        assert!(superadditive(pi, pj, alpha),
            "superadditivity failed: alpha={alpha} pi={pi} pj={pj}");
    }
}

#[test]
fn renyi_entropy_falls_for_every_alpha_above_one_not_just_two() {
    // The shipped `verify_entropy_nonincreasing` hardcodes alpha = 2.0, while
    // the document claims the result for all alpha > 1. Exercise the general
    // form the document actually asserts.
    for &alpha in &[1.01f64, 1.5, 2.0, 3.0, 5.0, 10.0] {
        for &(a, b, n) in &[(100usize, 50usize, 1000usize), (500, 500, 1000), (1, 1, 3), (7, 11, 40)] {
            assert!(verify_renyi_nonincreasing(a, b, n, alpha),
                "Renyi entropy rose on merge: alpha={alpha} a={a} b={b} n={n}");
        }
    }
}
