use aether_core::governor::*;

#[test]
fn probe_real_governor_saturates_clamps() {
    // Exactly what manifold_fs.rs:561 does: governor.adapt(1.0, 0.01)
    let mut g = GeometricGovernor::new();
    println!("initial eps = {}", g.epsilon());
    for i in 0..8 {
        let e = g.adapt(1.0, 0.01);
        println!("tick {i}: eps = {e:.12}  last_error = {}", g.last_error());
    }
    // the manifold_fs teleport sequence: tick(1.0), tick(0.0), tick(1.0)
    let mut h = GeometricGovernor::new();
    h.adapt(1.0, 0.01);
    println!("teleport pre  eps = {}", h.epsilon());
    h.adapt(0.0, 0.01);
    h.adapt(1.0, 0.01);
    println!("teleport post eps = {}", h.epsilon());

    // test_high_load_raises_epsilon reproduction: is it the PD law or the clamp?
    let mut k = GeometricGovernor::new();
    let one = k.adapt(10000.0, 0.001);
    println!("high load after ONE adapt: eps = {one}");
}

#[test]
fn probe_contraction_rate_vs_measured() {
    let alpha = 0.01;
    let beta = 0.05;
    let dt = 1.0;
    let rho = contraction_rate(alpha, beta, dt);
    println!("claimed rho = {rho}");
    println!("half_life   = {}", half_life(rho));
    println!("settle99    = {}", settling_time(rho, 0.01));

    // The analyzer's own simulation, per the in-file test.
    let a = GovernorConvergenceAnalyzer::new(alpha, beta, dt, 0.1, 0.9, 0.3);
    let s = a.simulate_constant(500, 0.5, 0.2);
    println!(
        "sim: init={} final={} empirical={} theoretical={}",
        s.initial_error, s.final_error, s.empirical_rate, s.theoretical_rate
    );

    // Does rho depend on the plant at all? Vary the operating point.
    for (r_target, delta) in [(0.3, 0.2), (1000.0, 0.2), (0.3, 50.0)] {
        let a2 = GovernorConvergenceAnalyzer::new(alpha, beta, dt, 1e-6, 1e6, r_target);
        let s2 = a2.simulate_constant(500, 0.5, delta);
        println!(
            "r_target={r_target} delta={delta}: empirical={} theoretical={} (identical rho)",
            s2.empirical_rate, s2.theoretical_rate
        );
    }

    let t = a.theoretical_analysis();
    println!(
        "lyapunov={} contraction={} (rho^2 < rho forced for rho in (0,1))",
        t.lyapunov_rate, t.contraction_rate
    );
}

#[test]
fn probe_theorem_holds_is_falsifiable() {
    // sweep gains and operating points; report any config where theorem_holds == false
    let mut total = 0;
    let mut failed = 0;
    for &alpha in &[0.001, 0.01, 0.1, 0.5, 0.9, 5.0, 50.0] {
        for &beta in &[0.0, 0.05, 0.5, 5.0, 100.0] {
            for &r_target in &[0.3, 0.5, 10.0, 1000.0] {
                for &delta in &[0.01, 0.2, 5.0] {
                    let a = GovernorConvergenceAnalyzer::new(alpha, beta, 1.0, 0.05, 1.0, r_target);
                    let rep = a.verify_theorem(200, 0.5, delta);
                    total += 1;
                    if !rep.theorem_holds {
                        failed += 1;
                        if failed <= 6 {
                            println!("FAIL a={alpha} b={beta} R={r_target} d={delta}: margin_ok={} rate_ok={} emp={} theo={} init={} fin={}",
                                rep.theory.gain_margin_stable, rep.rate_agreement,
                                rep.simulation.empirical_rate, rep.simulation.theoretical_rate,
                                rep.simulation.initial_error, rep.simulation.final_error);
                        }
                    }
                }
            }
        }
    }
    println!("theorem_holds false in {failed}/{total} configs");

    // How much divergence does the +-0.05 window actually admit at n=200?
    // empirical = (final/initial)^(1/200). Solve for the ratio at the window edge.
    let theo = contraction_rate(0.01, 0.05, 1.0);
    let edge = theo - 0.05;
    println!(
        "theoretical={theo}, window low edge={edge}, ratio needed to fail low = {}",
        libm::pow(edge, 200.0)
    );
}

#[test]
fn probe_gain_margin_semantics() {
    let a = GovernorConvergenceAnalyzer::new(0.01, 0.05, 1.0, 0.1, 0.9, 0.3);
    for row in a.gain_tuning_table() {
        println!(
            "{}: alpha={} beta={} margin={} stable={} rho={}",
            row.label, row.alpha, row.beta, row.gain_margin, row.stable, row.contraction_rate
        );
    }
    // Does the "stable" flag ever go false at the real governor's dt?
    let fast = GovernorConvergenceAnalyzer::new(0.01, 0.05, 0.01, 0.1, 0.9, 0.3);
    let tf = fast.theoretical_analysis();
    println!(
        "dt=0.01 (manifold_fs value): margin={} stable={} rho={}",
        tf.gain_margin, tf.gain_margin_stable, tf.contraction_rate
    );
}
