// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! `governor_step` was documented as guaranteeing `V(e') <= V(e)` for
//! `V(e) = e^2`, citing a Lean theorem `govStep_lyapunov` that does not exist.
//! Lean proves `V(rho * e) <= V(e)` for a scalar `|rho| <= 1`
//! (`Governor.lyapunov_descent`); the PD step is not multiplication of `e` by a
//! fixed scalar, because its derivative term reads `e_prev`.
//!
//! The first test is a step on stable default gains (`alpha + beta / dt = 0.06`)
//! where `|e|` rises. `lyapunov_descent_holds` is the runtime check that refuses
//! it; the second test shows the same check certifies an ordinary step.

use aether_verified::aether_governor::{
    gain_margin_refined, governor_error, governor_step, lyapunov_descent_holds,
};

#[test]
fn a_step_with_a_stale_derivative_raises_the_error_and_is_refused() {
    let (epsilon, e_prev, delta, dt) = (0.28, 0.5, 0.2, 1.0);
    let (alpha, beta, eps_min, eps_max, r_target) = (0.01, 0.05, 0.1, 0.9, 0.3);

    let next = governor_step(
        epsilon, e_prev, delta, dt, alpha, beta, eps_min, eps_max, r_target,
    );
    let before = governor_error(r_target, delta, epsilon).abs();
    let after = governor_error(r_target, delta, next).abs();

    assert!((next - 0.279_857_142_857).abs() < 1e-9, "epsilon' = {next}");
    assert!((before - 0.414_285_714_286).abs() < 1e-9, "|e| = {before}");
    assert!((after - 0.414_650_331_632).abs() < 1e-9, "|e'| = {after}");
    assert!(after > before);

    // The gain margin passes on these gains: it is not a descent certificate.
    assert!(gain_margin_refined(alpha, beta, dt));
    assert!(!lyapunov_descent_holds(r_target, delta, epsilon, next));
}

#[test]
fn a_step_that_shrinks_the_error_is_certified() {
    // e = 0.2 / 0.5 - 0.3 = 0.1, epsilon' = 0.5 + 0.1 * 0.1 + 0.05 * 0.1 = 0.515,
    // e' = 0.2 / 0.515 - 0.3 = 0.088350...
    let (epsilon, delta, r_target) = (0.5, 0.2, 0.3);
    let next = governor_step(epsilon, 0.0, delta, 1.0, 0.1, 0.05, 0.1, 0.9, r_target);
    assert!((next - 0.515).abs() < 1e-12);
    assert!(lyapunov_descent_holds(r_target, delta, epsilon, next));
}
