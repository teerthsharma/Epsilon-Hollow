use aether_core::scm::{LatentPredictor, SpectralContractionVerifier, TelemetryOperator};

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Q1: can `ConvergenceVerification::converged` ever be false for alpha in [0,1]?
#[test]
fn probe_can_converged_ever_be_false() {
    let mut false_count = 0usize;
    let mut total = 0usize;
    for ai in 0..=100 {
        let alpha = ai as f64 / 100.0;
        for &steps in &[0usize, 1, 3, 10, 200, 5000] {
            for &mag in &[1e-8f64, 1.0, 1e3, 1e12] {
                let v = SpectralContractionVerifier::<2>::new(alpha);
                let r = v.verify_convergence([mag, -mag], [0.0, 0.0], steps, 1e-6);
                total += 1;
                if !r.converged {
                    false_count += 1;
                    if false_count <= 6 {
                        println!(
                            "PROBE FAIL alpha={} steps={} mag={:e} final={:e} bound={:e} rel_excess={:e}",
                            alpha,
                            steps,
                            mag,
                            r.final_error,
                            r.theoretical_error_bound,
                            r.final_error / r.theoretical_error_bound - 1.0
                        );
                    }
                }
            }
        }
    }
    println!("PROBE converged==false in {}/{} configurations", false_count, total);

    // The headline case: alpha = 0 is NOT a contraction, the state never moves.
    let v0 = SpectralContractionVerifier::<2>::new(0.0);
    let r0 = v0.verify_convergence([10.0, -10.0], [0.0, 0.0], 10_000, 1e-6);
    println!(
        "PROBE alpha=0: initial={} final={} bound={} converged={}",
        r0.initial_error, r0.final_error, r0.theoretical_error_bound, r0.converged
    );

    // Is the "bound" just the same closed form the iteration realizes?
    let v = SpectralContractionVerifier::<2>::new(0.25);
    let r = v.verify_convergence([10.0, -10.0], [0.0, 0.0], 20, 1e-30);
    println!(
        "PROBE alpha=.25 steps=20: final={:.20e} bound={:.20e} ratio={:.17}",
        r.final_error,
        r.theoretical_error_bound,
        r.final_error / r.theoretical_error_bound
    );

    // full_verification: which alphas make theorem_holds false?
    let mut fail = vec![];
    for ai in 0..=100 {
        let alpha = ai as f64 / 100.0;
        let rep = SpectralContractionVerifier::<2>::new(alpha)
            .full_verification([10.0, -10.0], [0.0, 0.0], 5, 1e-6);
        if !rep.theorem_holds {
            fail.push(alpha);
        }
    }
    println!("PROBE theorem_holds==false only for alpha in {:?}", fail);
}

/// Q2: is `TelemetryOperator::lipschitz_constant()` a valid bound on the operator?
#[test]
fn probe_telemetry_lipschitz_bound_is_asserted_not_measured() {
    // alpha_max < alpha_min. Nothing in the constructor rejects this.
    let op = TelemetryOperator::<2>::new(0.5, 0.1, 0.1, 0.9);
    let reported = op.lipschitz_constant();
    let s = [10.0, -10.0];
    let p = [0.0, 0.0];
    let mut worst: f64 = 0.0;
    for k in 0..=90 {
        let eps = k as f64 / 100.0;
        let out = op.apply(&s, &p, eps);
        let measured = l2(&out, &p) / l2(&s, &p);
        worst = worst.max(measured);
    }
    println!(
        "PROBE TelemetryOperator(0.5,0.1,0.1,0.9): reported_lipschitz={} measured_worst={} gain_at_eps0.9={}",
        reported,
        worst,
        op.adaptive_gain(0.9)
    );
    println!("PROBE bound violated = {}", worst > reported);

    // Also: does adaptive_gain ever reach the documented "minimum adaptive gain"?
    let op2 = TelemetryOperator::<2>::new(0.01, 0.1, 0.1, 0.9);
    println!(
        "PROBE alpha_min field = 0.01, smallest attainable gain = {}",
        op2.adaptive_gain(f64::NEG_INFINITY)
    );
}

/// Q3: is the LatentPredictor attractor the module doc's `S_pred`?
#[test]
fn probe_latent_predictor_attractor() {
    let w = [[1.0, 0.0], [0.0, 1.0]];
    let alpha = 0.1;
    let lp = LatentPredictor::<2, 2>::new(w, alpha);
    let pred = [0.0, 0.0];
    let action = [1.0, 1.0];
    let mut s = [0.0, 0.0];
    for _ in 0..5000 {
        s = lp.step(&s, &action, &pred);
    }
    println!(
        "PROBE LatentPredictor fixed point with action=[1,1], pred=[0,0], alpha={}: s={:?}",
        alpha, s
    );
    println!(
        "PROBE predicted analytic offset (1-a)/a * W*a = {}",
        (1.0 - alpha) / alpha
    );
    println!("PROBE distance from claimed attractor S_pred = {}", l2(&s, &pred));
}
