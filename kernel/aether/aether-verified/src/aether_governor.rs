// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_governor.rs
//!
//! Provenance: Lean 4 (`AetherVerified.Governor`, `lean/AetherVerified/Governor.lean`)
//!
//! # PD governor
//!
//! A proportional-derivative controller that adapts the sparsity threshold ε
//! at runtime.
//!
//! No theorem shows that [`governor_step`] decreases `|e|`, and it does not
//! always: `tests/house_governor_step_descent.rs` has a step on gains with
//! `α + β/dt = 0.06` where `|e|` rises from 0.414286 to 0.414650. What Lean
//! proves is scalar: `Governor.lyapunov_descent` gives `V(ρ·e) ≤ V(e)` for
//! `V(e) = e²` and `|ρ| ≤ 1`, and `Governor.geometric_bound` iterates it. The PD
//! step is not multiplication of `e` by a fixed `ρ`, because its derivative
//! term reads `e_prev`. [`lyapunov_descent_holds`] checks descent for a
//! computed step at runtime instead.

/// Clamp a value to [lo, hi].
#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}

/// (Lean: govError)
///
/// Compute the governor error signal:
///   `e = δ/ε − r_target`
///
/// where δ is the measured sparsity delta and ε is the current threshold.
pub fn governor_error(r_target: f64, delta: f64, epsilon: f64) -> f64 {
    debug_assert!(epsilon.abs() > f64::EPSILON, "epsilon must be non-zero");
    delta / epsilon - r_target
}

/// (Lean: govStep)
///
/// One PD governor step.
///
/// Update rule:
///   `ε_{t+1} = clamp(ε_t + α·e + β·ė, ε_min, ε_max)`
///
/// where:
///   - `e = δ/ε − r_target` (proportional error)
///   - `ė = (e − e_prev) / dt` (derivative error)
///   - `α` = proportional gain
///   - `β` = derivative gain
///
/// The step does not guarantee `|e_{t+1}| ≤ |e_t|` (see the module docs); check
/// a computed step with [`lyapunov_descent_holds`].
#[allow(clippy::too_many_arguments)] // PID controller naturally takes all gains/limits.
pub fn governor_step(
    epsilon: f64,
    e_prev: f64,
    delta: f64,
    dt: f64,
    alpha: f64,
    beta: f64,
    eps_min: f64,
    eps_max: f64,
    r_target: f64,
) -> f64 {
    let e = governor_error(r_target, delta, epsilon);
    let d_error = (e - e_prev) / dt;
    let adjustment = alpha * e + beta * d_error;
    clamp(epsilon + adjustment, eps_min, eps_max)
}

/// (Lean: `Governor.gainMarginRefined`)
///
/// `dt ≥ 1` and `0.01 + 0.05/dt < 1`: the gain margin of the default gains
/// α = 0.01, β = 0.05, not of the gains a caller passes to [`governor_step`].
///
/// This is a gain condition, not a descent certificate, and would stay one with
/// the real gains: the step in `tests/house_governor_step_descent.rs` runs on
/// exactly these gains and still raises `|e|`.
pub fn gain_margin_refined(dt: f64) -> bool {
    // ponytail: hard-codes alpha = 0.01, beta = 0.05. Upgrade to
    // `gain_margin_refined(alpha, beta, dt)` = `dt >= 1.0 &&
    // aether_agcr::gain_margin_stable(alpha, beta, dt)` together with its one
    // caller, aether-link/examples/world_model_demo.rs, which is outside this
    // change's scope.
    dt >= 1.0 && (0.01 + 0.05 / dt) < 1.0
}

/// Whether moving the threshold from `epsilon` to `epsilon_next` under the
/// same measurement `delta` does not increase `|e|`, i.e. `V(e') ≤ V(e)` for
/// `V(e) = e²`. A runtime check of one computed step, not a proof about
/// [`governor_step`].
pub fn lyapunov_descent_holds(r_target: f64, delta: f64, epsilon: f64, epsilon_next: f64) -> bool {
    governor_error(r_target, delta, epsilon_next).abs()
        <= governor_error(r_target, delta, epsilon).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governor_error_zero() {
        // When δ/ε == r_target, error is zero
        let e = governor_error(0.5, 1.0, 2.0);
        assert!((e).abs() < 1e-10);
    }

    #[test]
    fn test_governor_step_convergence() {
        // Run several steps and check the error stays bounded. Per-step descent
        // is not asserted because the step does not guarantee it (module docs).
        let mut epsilon = 0.5;
        let mut e_prev = 0.0;
        let r_target = 0.3;
        let alpha = 0.1;
        let beta = 0.05;
        let dt = 1.0;

        let mut prev_error_abs = f64::INFINITY;
        for _ in 0..20 {
            let delta = 0.2; // Simulated measurement
            let e = governor_error(r_target, delta, epsilon);
            let error_abs = e.abs();

            epsilon = governor_step(epsilon, e_prev, delta, dt, alpha, beta, 0.1, 0.9, r_target);
            e_prev = e;

            // After initial transient, error should be bounded
            if prev_error_abs < 1.0 {
                // Not strictly monotone due to clamp, but bounded
                assert!(error_abs < 10.0, "Error diverged: {}", error_abs);
            }
            prev_error_abs = error_abs;
        }
    }

    #[test]
    fn test_gain_margin_refined_stable() {
        assert!(gain_margin_refined(1.0));
        assert!(gain_margin_refined(10.0));
        assert!(gain_margin_refined(100.0));
    }

    #[test]
    fn test_gain_margin_refined_unstable() {
        assert!(!gain_margin_refined(0.001));
        assert!(!gain_margin_refined(0.5));
    }

    #[test]
    fn test_clamp_bounds() {
        let eps = governor_step(0.5, 0.0, 100.0, 1.0, 10.0, 0.0, 0.1, 0.9, 0.3);
        assert!((0.1..=0.9).contains(&eps), "Clamp violated: {}", eps);
    }
}
