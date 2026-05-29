// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! # Spectral Contraction Mapping (SCM) — runnable operator
//!
//! Realizes Theorem 2 of the Epsilon-Hollow spec: the telemetry operator
//! `T(S) = (1 − α) · S + α · S_pred` is a contraction mapping with
//! Lipschitz constant `1 − α < 1`. By Banach's fixed-point theorem,
//! iterating `T` from any starting state converges geometrically to the
//! unique fixed point (here: `S_pred`, the environment's predicted state).
//!
//! Convention matches `aether-verified/src/aether_scm.rs`:
//!   * `α ∈ [0, 1]` is the assimilation rate (how much we trust the prediction).
//!   * `α = 0` ⇒ pure inertia (state never updates).
//!   * `α = 1` ⇒ instantaneous snap-to-prediction.
//!
//! This module also provides [`LatentPredictor`] which implements the
//! transition dynamics `s_{t+1} = SCM(s_t + W·a_t)` from §3.4.

/// Componentwise spectral contraction operator over a `D`-dimensional state.
///
/// For each component `i`, computes
/// `T(S, S_pred)_i = (1 − α) · S_i + α · S_pred_i`.
///
/// The Lipschitz constant of `T(·, S_pred)` (with `S_pred` fixed) is
/// `1 − α`, which is strictly less than 1 for any `α ∈ (0, 1]`.
/// This is what makes `T` a Banach contraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralContractionOperator<const D: usize> {
    /// Assimilation rate. Must lie in `[0, 1]`.
    pub alpha: f64,
}

impl<const D: usize> SpectralContractionOperator<D> {
    /// Construct a new operator with the given assimilation rate `α`.
    ///
    /// In debug builds, asserts `α ∈ [0, 1]`. Release builds trust the caller.
    #[inline]
    pub fn new(alpha: f64) -> Self {
        debug_assert!((0.0..=1.0).contains(&alpha), "alpha must lie in [0, 1]");
        Self { alpha }
    }

    /// Apply the operator once: `(1 − α) · state + α · pred`, componentwise.
    #[inline]
    pub fn apply(&self, state: &[f64; D], pred: &[f64; D]) -> [f64; D] {
        let mut out = [0.0_f64; D];
        let one_minus_alpha = 1.0 - self.alpha;
        let alpha = self.alpha;
        let mut i = 0;
        while i < D {
            out[i] = one_minus_alpha * state[i] + alpha * pred[i];
            i += 1;
        }
        out
    }

    /// Lipschitz constant of `T(·, pred)`: `1 − α`.
    ///
    /// Strictly less than 1 for `α > 0`, which is the contraction certificate.
    #[inline]
    pub fn lipschitz_constant(&self) -> f64 {
        1.0 - self.alpha
    }

    /// Apply the operator `steps` times, holding `pred` fixed.
    ///
    /// By Banach's theorem, the result converges to `pred` geometrically:
    /// `‖S_t − pred‖ ≤ (1 − α)^t · ‖S_0 − pred‖`.
    #[inline]
    pub fn iterate(&self, state: [f64; D], pred: [f64; D], steps: usize) -> [f64; D] {
        let mut s = state;
        let mut k = 0;
        while k < steps {
            s = self.apply(&s, &pred);
            k += 1;
        }
        s
    }
}

/// Adaptive telemetry prediction-correction operator from the Python SCM slice.
///
/// For fixed prediction `S_pred`, this computes
/// `T(S) = (1 - alpha_n) * S + alpha_n * S_pred`, where `alpha_n` is derived
/// from the current governor epsilon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetryOperator<const D: usize> {
    /// Minimum adaptive gain.
    pub alpha_min: f64,
    /// Maximum adaptive gain.
    pub alpha_max: f64,
    /// Minimum epsilon used by the controlling governor.
    pub epsilon_min: f64,
    /// Maximum epsilon used to normalize adaptive gain.
    pub epsilon_max: f64,
}

impl<const D: usize> TelemetryOperator<D> {
    /// Construct a telemetry operator with explicit gain and epsilon bounds.
    #[inline]
    pub fn new(alpha_min: f64, alpha_max: f64, epsilon_min: f64, epsilon_max: f64) -> Self {
        Self {
            alpha_min,
            alpha_max,
            epsilon_min,
            epsilon_max,
        }
    }

    /// Compute `alpha_min + (epsilon_t / epsilon_max) * (alpha_max - alpha_min)`.
    #[inline]
    pub fn adaptive_gain(&self, epsilon_t: f64) -> f64 {
        if self.epsilon_max <= 0.0 {
            return self.alpha_min;
        }
        let epsilon_t = epsilon_t.clamp(self.epsilon_min, self.epsilon_max);
        self.alpha_min + (epsilon_t / self.epsilon_max) * (self.alpha_max - self.alpha_min)
    }

    /// Apply the adaptive telemetry operator once.
    #[inline]
    pub fn apply(&self, state: &[f64; D], pred: &[f64; D], epsilon_t: f64) -> [f64; D] {
        let alpha = self.adaptive_gain(epsilon_t);
        let mut out = [0.0; D];
        let mut i = 0;
        while i < D {
            out[i] = (1.0 - alpha) * state[i] + alpha * pred[i];
            i += 1;
        }
        out
    }

    /// Conservative Lipschitz constant `1 - alpha_min`.
    #[inline]
    pub fn lipschitz_constant(&self) -> f64 {
        1.0 - self.alpha_min
    }

    /// Iterations needed to halve the error under the conservative rate.
    #[inline]
    pub fn convergence_half_life(&self) -> f64 {
        let lip = self.lipschitz_constant();
        if lip <= 0.0 || lip >= 1.0 {
            f64::INFINITY
        } else {
            libm::log(2.0) / libm::fabs(libm::log(lip))
        }
    }
}

/// Result of checking whether an SCM operator is a Banach contraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContractionVerification {
    /// Lipschitz constant used by the theorem.
    pub lipschitz_constant: f64,
    /// Whether the constant is finite and strictly below one.
    pub contraction_holds: bool,
}

/// Result of checking convergence against the theorem error bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConvergenceVerification {
    /// Initial L2 error between state and prediction.
    pub initial_error: f64,
    /// Final L2 error after the requested iterations.
    pub final_error: f64,
    /// Theoretical upper bound `(1-alpha)^steps * initial_error`.
    pub theoretical_error_bound: f64,
    /// Number of iterations simulated.
    pub steps: usize,
    /// Whether the final error is within tolerance and theorem bound.
    pub converged: bool,
}

/// Aggregate SCM theorem verification report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralContractionReport {
    /// Contraction check output.
    pub contraction: ContractionVerification,
    /// Convergence check output.
    pub convergence: ConvergenceVerification,
    /// True only when the contraction and convergence checks both pass.
    pub theorem_holds: bool,
}

/// Public verifier for the legacy spectral-contraction theorem API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralContractionVerifier<const D: usize> {
    operator: SpectralContractionOperator<D>,
}

impl<const D: usize> SpectralContractionVerifier<D> {
    /// Construct a verifier with the supplied assimilation rate.
    #[inline]
    pub fn new(alpha: f64) -> Self {
        Self {
            operator: SpectralContractionOperator::new(alpha),
        }
    }

    /// Verify that the operator is a strict contraction.
    #[inline]
    pub fn verify_contraction(&self) -> ContractionVerification {
        let lipschitz_constant = self.operator.lipschitz_constant();
        ContractionVerification {
            lipschitz_constant,
            contraction_holds: lipschitz_constant.is_finite()
                && (0.0..1.0).contains(&lipschitz_constant),
        }
    }

    /// Verify convergence from `start` toward `pred` within `steps`.
    pub fn verify_convergence(
        &self,
        start: [f64; D],
        pred: [f64; D],
        steps: usize,
        tolerance: f64,
    ) -> ConvergenceVerification {
        let initial_error = l2_diff(&start, &pred);
        let final_state = self.operator.iterate(start, pred, steps);
        let final_error = l2_diff(&final_state, &pred);
        let rate = self.operator.lipschitz_constant().clamp(0.0, 1.0);
        let theoretical_error_bound = libm::pow(rate, steps as f64) * initial_error;
        let bound_slack = theoretical_error_bound.max(tolerance) + 1e-9;

        ConvergenceVerification {
            initial_error,
            final_error,
            theoretical_error_bound,
            steps,
            converged: final_error <= tolerance || final_error <= bound_slack,
        }
    }

    /// Run contraction and convergence checks as one theorem report.
    pub fn full_verification(
        &self,
        start: [f64; D],
        pred: [f64; D],
        steps: usize,
        tolerance: f64,
    ) -> SpectralContractionReport {
        let contraction = self.verify_contraction();
        let convergence = self.verify_convergence(start, pred, steps, tolerance);
        SpectralContractionReport {
            contraction,
            convergence,
            theorem_holds: contraction.contraction_holds && convergence.converged,
        }
    }
}

#[inline]
fn l2_diff<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut s = 0.0_f64;
    let mut i = 0;
    while i < D {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    libm::sqrt(s)
}

#[cfg(test)]
mod telemetry_port_tests {
    use super::*;

    #[test]
    fn telemetry_operator_uses_adaptive_gain_and_half_life() {
        let op = TelemetryOperator::<3>::new(0.01, 0.1, 0.1, 0.9);
        let alpha = op.adaptive_gain(0.45);
        assert!((alpha - 0.055).abs() < 1e-12);

        let state = [10.0, -10.0, 2.0];
        let pred = [0.0, 0.0, 2.0];
        let out = op.apply(&state, &pred, 0.45);
        assert!((out[0] - 9.45).abs() < 1e-12);
        assert!((out[1] + 9.45).abs() < 1e-12);
        assert_eq!(out[2], 2.0);

        assert!((op.lipschitz_constant() - 0.99).abs() < 1e-12);
        assert!(op.convergence_half_life() > 68.0);
        assert!(op.convergence_half_life() < 70.0);
    }
}

/// Latent transition predictor implementing `s_{t+1} = SCM(s_t + W·a_t)`.
///
/// `D` is the latent state dimension; `A` is the action dimension.
/// `W` is a learnable action-to-state projection matrix stored row-major
/// as `[[f64; D]; A]` (row `i` is the contribution of action component `i`).
#[derive(Debug, Clone, Copy)]
pub struct LatentPredictor<const D: usize, const A: usize> {
    /// Action projection matrix. `w[i]` is the contribution of `action[i]`
    /// to the latent state delta.
    pub w: [[f64; D]; A],
    /// The spectral contraction operator providing stable convergence.
    pub scm: SpectralContractionOperator<D>,
}

impl<const D: usize, const A: usize> LatentPredictor<D, A> {
    /// Construct a new latent predictor with projection matrix `w` and
    /// assimilation rate `α`.
    #[inline]
    pub fn new(w: [[f64; D]; A], alpha: f64) -> Self {
        Self {
            w,
            scm: SpectralContractionOperator::new(alpha),
        }
    }

    /// Project the action vector through `W` to a state-space delta.
    ///
    /// Computes `Σ_i action[i] · w[i]` componentwise.
    #[inline]
    pub fn project(&self, action: &[f64; A]) -> [f64; D] {
        let mut out = [0.0_f64; D];
        let mut i = 0;
        while i < A {
            let a_i = action[i];
            let mut j = 0;
            while j < D {
                out[j] += a_i * self.w[i][j];
                j += 1;
            }
            i += 1;
        }
        out
    }

    /// Compute one transition: `SCM(state + W·action, pred)`.
    #[inline]
    pub fn step(&self, state: &[f64; D], action: &[f64; A], pred: &[f64; D]) -> [f64; D] {
        let proj = self.project(action);
        let mut shifted = [0.0_f64; D];
        let mut j = 0;
        while j < D {
            shifted[j] = state[j] + proj[j];
            j += 1;
        }
        self.scm.apply(&shifted, pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_is_componentwise_convex_combination() {
        let scm = SpectralContractionOperator::<3>::new(0.4);
        let s = [1.0, 2.0, 3.0];
        // Fixed point: apply(s, s) == s.
        let out = scm.apply(&s, &s);
        for i in 0..3 {
            assert!((out[i] - s[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn apply_collapses_to_pred_at_alpha_1() {
        let scm = SpectralContractionOperator::<2>::new(1.0);
        let s = [0.0, 0.0];
        let p = [7.0, -3.0];
        let out = scm.apply(&s, &p);
        assert_eq!(out, p);
    }

    #[test]
    fn apply_preserves_state_at_alpha_0() {
        let scm = SpectralContractionOperator::<2>::new(0.0);
        let s = [5.0, -2.5];
        let p = [99.0, 99.0];
        let out = scm.apply(&s, &p);
        assert_eq!(out, s);
    }

    #[test]
    fn lipschitz_constant_is_one_minus_alpha() {
        let scm = SpectralContractionOperator::<1>::new(0.3);
        assert!((scm.lipschitz_constant() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn banach_convergence() {
        // Iterating T from any starting state converges to pred.
        let scm = SpectralContractionOperator::<4>::new(0.25);
        let pred = [1.0, -1.0, 0.5, 2.0];
        let start = [100.0, -100.0, 50.0, -50.0];
        let out = scm.iterate(start, pred, 200);
        let err = l2_diff(&out, &pred);
        assert!(err < 1e-6, "after 200 iterations err = {}", err);
    }

    #[test]
    fn latent_predictor_zero_action_is_pure_scm() {
        let w = [[1.0, 2.0], [3.0, 4.0]];
        let lp = LatentPredictor::<2, 2>::new(w, 0.5);
        let s = [10.0, -5.0];
        let p = [0.0, 0.0];
        let action = [0.0, 0.0];
        let out = lp.step(&s, &action, &p);
        let expected = lp.scm.apply(&s, &p);
        assert_eq!(out, expected);
    }

    #[test]
    fn latent_predictor_step_is_deterministic() {
        let w = [[0.1, 0.2, 0.3]];
        let lp = LatentPredictor::<3, 1>::new(w, 0.4);
        let s = [1.0, 2.0, 3.0];
        let a = [0.5];
        let p = [-1.0, 0.0, 1.0];
        let o1 = lp.step(&s, &a, &p);
        let o2 = lp.step(&s, &a, &p);
        assert_eq!(o1, o2);
    }

    #[test]
    fn convergence_rate_matches_theorem() {
        // Theorem: ‖S_t − pred‖ ≤ (1−α)^t · ‖S_0 − pred‖.
        // To reach error ≤ 1e-6 starting from err_0 = ‖S_0 − pred‖,
        // we need t ≥ ⌈(log(1e-6) − log(err_0)) / log(1−α)⌉.
        let alpha = 0.2_f64;
        let scm = SpectralContractionOperator::<2>::new(alpha);
        let pred = [0.0, 0.0];
        let start = [10.0, 0.0];
        let err_0 = l2_diff(&start, &pred);
        let target = 1e-6_f64;
        let theory_t =
            libm::ceil((libm::log(target) - libm::log(err_0)) / libm::log(1.0 - alpha)) as usize;
        let budget = theory_t + 5;

        let mut s = start;
        let mut hit_at: Option<usize> = None;
        for k in 1..=budget {
            s = scm.apply(&s, &pred);
            if l2_diff(&s, &pred) <= target {
                hit_at = Some(k);
                break;
            }
        }
        let k = hit_at.expect("did not converge within budget");
        assert!(
            k <= budget,
            "took {} iters, theoretical bound + 5 = {}",
            k,
            budget
        );
    }

    #[test]
    fn spectral_contraction_verifier_reports_full_theorem_status() {
        let verifier = SpectralContractionVerifier::<2>::new(0.25);
        let report = verifier.full_verification([10.0, -10.0], [0.0, 0.0], 200, 1e-6);
        assert!(report.contraction.contraction_holds);
        assert!(report.convergence.final_error < report.convergence.initial_error);
        assert!(report.theorem_holds);
    }
}
