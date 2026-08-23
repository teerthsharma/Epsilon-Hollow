//! D7 — Theorem T2's "theoretical error bound" is the closed form the iteration
//! realizes, so the convergence check compares a value against itself.
//! Gate misfire input: alpha = 0, where nothing moves and nothing converges.
use aether_core::scm::{SpectralContractionVerifier, TelemetryOperator};

#[test]
fn alpha_zero_must_not_report_convergence() {
    // THE REQUIRED MISFIRE. alpha = 0 makes step the identity: the state never
    // moves. Any honest convergence check must reject it.
    let v = SpectralContractionVerifier::<4>::new(0.0);
    let start = [10.0, 10.0, 10.0, 10.0];
    let pred = [0.0, 0.0, 0.0, 0.0];
    let r = v.verify_convergence(start, pred, 50, 1e-6);
    println!("alpha=0: initial={} final={} bound={} converged={}",
        r.initial_error, r.final_error, r.theoretical_error_bound, r.converged);
    assert_eq!(r.final_error, r.initial_error, "alpha=0 must not move the state");
    assert!(!r.converged, "alpha=0 does not converge; the check reported that it did");
}

#[test]
fn the_predicted_error_is_exact_and_converged_does_not_consult_it() {
    // For this operator the step is (1-a)*state + a*pred, so the error obeys
    // e_{n+1} = (1-a) e_n and the predicted value is EXACT, not an upper bound.
    // That is correct mathematics. The defect was using an exact value as a
    // check on itself. Both halves are pinned here.
    for &alpha in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
        let v = SpectralContractionVerifier::<3>::new(alpha);
        for &steps in &[1usize, 5, 20] {
            let r = v.verify_convergence([3.0, -4.0, 12.0], [0.0, 0.0, 0.0], steps, 1e-12);
            let ratio = r.final_error / r.theoretical_error_bound;
            assert!((ratio - 1.0).abs() < 1e-12,
                "alpha={alpha} steps={steps}: predicted value is not exact, ratio={ratio}");
        }
    }

    // And `converged` must be independent of it. Here the predicted value is
    // satisfied trivially (realized == predicted, as just shown) yet the run
    // has NOT reached tolerance, so an honest check reports false.
    let v = SpectralContractionVerifier::<3>::new(0.5);
    let r = v.verify_convergence([1000.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2, 1e-9);
    println!("alpha=0.5 steps=2 tol=1e-9: final={:.6} predicted={:.6} converged={}",
        r.final_error, r.theoretical_error_bound, r.converged);
    assert!((r.final_error / r.theoretical_error_bound - 1.0).abs() < 1e-12);
    assert!(!r.converged,
        "two steps from 1000 cannot reach 1e-9; converged must be false, so it          cannot be reading the predicted value");
}

#[test]
fn telemetry_lipschitz_must_be_the_true_worst_case() {
    // Constructor does not enforce alpha_min <= alpha_max. When swapped, the
    // reported constant is optimistic and the real worst case is larger.
    let op = TelemetryOperator::<2>::new(0.5, 0.1, 0.1, 0.9);
    let reported = op.lipschitz_constant();
    let mut worst: f64 = 0.0;
    for i in 0..=1000 {
        let eps = 0.9 * (i as f64) / 1000.0;
        let a = op.adaptive_gain(eps);
        worst = worst.max(1.0 - a);
    }
    println!("TelemetryOperator(0.5,0.1,0.1,0.9): reported={reported} measured_worst={worst}");
    assert!(reported >= worst - 1e-12,
        "reported Lipschitz {reported} is below the measured worst case {worst}: \
         the constant is optimistic and the contraction claim is unsound");
}
