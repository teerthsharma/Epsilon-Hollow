// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//! AEGIS Geometric Governor
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//!
//! Implements the Adaptive Threshold Controller using Nonlinear Control Theory
//! (PID-on-Manifold).
//!
//! Mathematical Foundation:
//!   Error Signal: e(t) = R_target - Î”(t)/Îµ(t)
//!   Update Law: Îµ(t+1) = Îµ(t) + Î±Â·e(t) + Î²Â·de/dt
//!
//! Intuition:
//!   - If kernel wakes too often (e < 0): raise Îµ (decrease sensitivity)
//!   - If kernel is sluggish (e > 0): lower Îµ (increase sensitivity)
//!
//! This ensures the kernel doesn't:
//!   - "Stutter" (thrash) during high load
//!   - "Sleep" during critical transients
//!
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Governor Constants
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

use alloc::vec::Vec;

/// Target "Frame Rate" - ideal kernel tick rate in Hz
/// The governor tries to maintain this balance between responsiveness and efficiency
const TARGET_TICK_RATE: f64 = 1000.0;

/// Proportional gain (Î±)
/// Controls response to instantaneous error
const ALPHA: f64 = 0.01;

/// Derivative gain (Î²)
/// Controls response to rate of change of error
/// Helps dampen oscillations
const BETA: f64 = 0.05;

/// Minimum allowed epsilon (prevents runaway sensitivity)
const EPSILON_MIN: f64 = 0.001;

/// Maximum allowed epsilon (prevents system from sleeping too long)
const EPSILON_MAX: f64 = 10.0;

/// Default initial epsilon
const EPSILON_INITIAL: f64 = 0.1;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Geometric Governor
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// The Geometric Governor: Adaptive Threshold Controller
///
/// This is the "How" of AEGIS - it dynamically adjusts the sensitivity
/// threshold Îµ(t) based on system behavior, using classical nonlinear
/// control theory (PID controller on the state manifold).
///
/// # Control Law
/// ```text
/// Îµ(t+1) = Îµ(t) + Î±Â·e(t) + Î²Â·de/dt
///
/// where:
///   e(t) = R_target - R_actual
///   R_actual = Î”/Îµ (effective "frame rate")
/// ```
///
/// # Stability Properties
/// - Bounded: Îµ âˆˆ [EPSILON_MIN, EPSILON_MAX]
/// - Asymptotically stable around R_target
/// - Damped oscillation via derivative term
#[derive(Debug, Clone)]
pub struct GeometricGovernor {
    /// Current adaptive threshold Îµ(t)
    epsilon: f64,

    /// Previous error (for derivative calculation)
    last_error: f64,

    /// Accumulated integral error (for potential PID extension)
    integral_error: f64,

    /// Number of adjustments made (for statistics)
    adjustment_count: u64,

    /// Custom gains (optional override)
    alpha: f64,
    beta: f64,
}

impl GeometricGovernor {
    /// Create a new governor with default parameters
    pub fn new() -> Self {
        Self {
            epsilon: EPSILON_INITIAL,
            last_error: 0.0,
            integral_error: 0.0,
            adjustment_count: 0,
            alpha: ALPHA,
            beta: BETA,
        }
    }

    /// Create a governor with custom initial epsilon
    pub fn with_epsilon(epsilon: f64) -> Self {
        let mut gov = Self::new();
        gov.epsilon = epsilon.clamp(EPSILON_MIN, EPSILON_MAX);
        gov
    }

    /// Create a governor with custom gains
    pub fn with_gains(alpha: f64, beta: f64) -> Self {
        Self {
            epsilon: EPSILON_INITIAL,
            last_error: 0.0,
            integral_error: 0.0,
            adjustment_count: 0,
            alpha,
            beta,
        }
    }

    /// Get current epsilon value
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get adjustment statistics
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count
    }

    /// Adapt epsilon based on observed deviation
    ///
    /// Implements the PID-on-Manifold control law:
    /// ```text
    /// Îµ(t+1) = Îµ(t) + Î±Â·e(t) + Î²Â·de/dt
    /// ```
    ///
    /// # Arguments
    /// * `deviation_delta` - The observed deviation Î”(t)
    /// * `dt` - Time delta since last adaptation (in seconds)
    ///
    /// # Returns
    /// The new epsilon value
    pub fn adapt(&mut self, deviation_delta: f64, dt: f64) -> f64 {
        // Prevent division by zero
        if dt <= 0.0 || self.epsilon <= 0.0 {
            return self.epsilon;
        }

        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        // Step 1: Calculate the "Effective Rate" we are seeing
        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        //
        // The effective rate is how often the kernel WOULD wake up
        // given the current deviation and threshold.
        //
        // Rate = Î” / Îµ
        //
        // If Î” is high relative to Îµ, we're waking up often.
        // If Î” is low relative to Îµ, we're barely waking up.

        let current_rate = deviation_delta / self.epsilon;

        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        // Step 2: Calculate Control Error
        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        //
        // Error = Target - Actual
        //
        // Positive error: We're too slow (need to lower Îµ, increase sensitivity)
        // Negative error: We're too fast (need to raise Îµ, decrease sensitivity)

        let error = TARGET_TICK_RATE - current_rate;

        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        // Step 3: Calculate Derivative of Error
        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        //
        // de/dt = (e(t) - e(t-1)) / dt
        //
        // This term helps dampen oscillations and provides predictive control.

        let d_error = (error - self.last_error) / dt;

        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        // Step 4: Apply Control Law (PD Controller)
        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        //
        // Î”Îµ = Î±Â·e + Î²Â·de/dt
        //
        // Note: We could add an integral term (Î³Â·âˆ«eÂ·dt) for PID,
        // but PD is sufficient for our stability requirements.

        let adjustment = (self.alpha * error) + (self.beta * d_error);

        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        // Step 5: Update State
        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        self.epsilon -= adjustment;
        self.last_error = error;
        self.adjustment_count += 1;

        // Update integral for potential future use
        self.integral_error += error * dt;

        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        // Step 6: Safety Clamps
        // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        //
        // Prevent epsilon from:
        // - Vanishing (â†’ system never sleeps, 100% CPU)
        // - Exploding (â†’ system never wakes, misses events)

        self.epsilon = self.epsilon.clamp(EPSILON_MIN, EPSILON_MAX);

        self.epsilon
    }

    /// Check if a deviation exceeds the current threshold
    ///
    /// This is the core decision function: Î”(t) â‰¥ Îµ(t)?
    pub fn should_trigger(&self, deviation: f64) -> bool {
        deviation >= self.epsilon
    }

    /// Reset the governor to initial state
    pub fn reset(&mut self) {
        self.epsilon = EPSILON_INITIAL;
        self.last_error = 0.0;
        self.integral_error = 0.0;
        self.adjustment_count = 0;
    }

    /// Get the current error (for diagnostics)
    pub fn last_error(&self) -> f64 {
        self.last_error
    }
}

#[cfg(test)]
mod convergence_port_tests {
    use super::*;

    #[test]
    fn convergence_rate_helpers_match_agcr_defaults() {
        let rho = contraction_rate(0.01, 0.05, 1.0);
        assert!(rho > 0.990 && rho < 0.991);
        assert!(half_life(rho) > 72.0 && half_life(rho) < 74.0);
        assert!(settling_time(rho, 0.01) > 480.0 && settling_time(rho, 0.01) < 485.0);

        let alpha = required_alpha(0.05, 1.0, 100.0, 0.01);
        assert!(alpha > 0.045 && alpha < 0.0475);
    }

    #[test]
    fn convergence_analyzer_reports_stable_gain_margin() {
        let analyzer = GovernorConvergenceAnalyzer::new(0.01, 0.05, 1.0, 0.1, 0.9, 0.3);
        let theory = analyzer.theoretical_analysis();
        assert!(theory.gain_margin_stable);
        assert!(theory.lyapunov_rate < theory.contraction_rate);

        let sim = analyzer.simulate_constant(500, 0.5, 0.2);
        assert!(sim.final_error < sim.initial_error);
        assert!(sim.theoretical_rate > 0.990 && sim.theoretical_rate < 0.991);

        let table = analyzer.gain_tuning_table();
        assert_eq!(table.len(), 5);
        assert!(table[0].stable);
        assert!(table[4].contraction_rate < table[0].contraction_rate);
    }
}

impl Default for GeometricGovernor {
    fn default() -> Self {
        Self::new()
    }
}

/// Geometric contraction rate for the adaptive governor theorem.
///
/// `rho = 1 - alpha / (1 + beta / dt)`.
#[inline]
pub fn contraction_rate(alpha: f64, beta: f64, dt: f64) -> f64 {
    if dt <= 0.0 {
        return f64::INFINITY;
    }
    1.0 - alpha / (1.0 + beta / dt)
}

/// Steps needed to halve the error under contraction rate `rho`.
#[inline]
pub fn half_life(rho: f64) -> f64 {
    if rho <= 0.0 || rho >= 1.0 {
        f64::INFINITY
    } else {
        libm::log(2.0) / libm::fabs(libm::log(rho))
    }
}

/// Steps needed to reduce error to `target_ratio` of its initial value.
#[inline]
pub fn settling_time(rho: f64, target_ratio: f64) -> f64 {
    if rho <= 0.0 || rho >= 1.0 || target_ratio <= 0.0 {
        f64::INFINITY
    } else {
        libm::log(1.0 / target_ratio) / libm::fabs(libm::log(rho))
    }
}

/// Required proportional gain for a target settling time.
#[inline]
pub fn required_alpha(beta: f64, dt: f64, t_target: f64, target_ratio: f64) -> f64 {
    if dt <= 0.0 || t_target <= 0.0 || target_ratio <= 0.0 {
        return f64::NAN;
    }
    let rho_required = libm::exp(-libm::log(1.0 / target_ratio) / t_target);
    (1.0 + beta / dt) * (1.0 - rho_required)
}

/// Closed-form AGCR theorem outputs for a governor configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernorTheory {
    /// Geometric contraction rate.
    pub contraction_rate: f64,
    /// PD gain margin `alpha + beta / dt`.
    pub gain_margin: f64,
    /// Whether the gain margin is below one.
    pub gain_margin_stable: bool,
    /// Error half-life in steps.
    pub half_life_steps: f64,
    /// Steps to 99 percent convergence.
    pub settling_99_steps: f64,
    /// Steps to 99.9 percent convergence.
    pub settling_999_steps: f64,
    /// Lyapunov contraction rate for `V(e) = e^2`.
    pub lyapunov_rate: f64,
}

/// Compact result of a constant-measurement governor simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernorSimulation {
    /// Initial absolute tracking error.
    pub initial_error: f64,
    /// Final absolute tracking error.
    pub final_error: f64,
    /// Empirical geometric rate inferred from endpoints.
    pub empirical_rate: f64,
    /// The theoretical AGCR rate for this configuration.
    pub theoretical_rate: f64,
    /// Whether the final error is below one percent of the initial error.
    pub converged_99: bool,
    /// Final epsilon after simulation.
    pub final_epsilon: f64,
}

/// Full governor simulation trace for arbitrary measurement slices.
#[derive(Debug, Clone, PartialEq)]
pub struct GovernorSimulationHistory {
    /// Absolute tracking error at each simulated step.
    pub errors: Vec<f64>,
    /// Epsilon values, including the initial value at index zero.
    pub epsilons: Vec<f64>,
    /// Empirical geometric rate from the legacy log-linear fit.
    pub empirical_rate: f64,
    /// The theoretical AGCR rate for this configuration.
    pub theoretical_rate: f64,
    /// Initial absolute tracking error.
    pub initial_error: f64,
    /// Final absolute tracking error.
    pub final_error: f64,
    /// Whether the final error is below one percent of the initial error.
    pub converged_99: bool,
}

/// Aggregate AGCR theorem verification report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernorTheoremVerification {
    /// Closed-form theorem outputs.
    pub theory: GovernorTheory,
    /// Simulation result used to compare against the closed-form rate.
    pub simulation: GovernorSimulation,
    /// Whether empirical and theoretical rates agree within a conservative bound.
    pub rate_agreement: bool,
    /// True when stability, simulation improvement, and rate agreement all hold.
    pub theorem_holds: bool,
}

/// Gain tuning table row for the AGCR theorem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainTuningRow {
    /// Human-readable preset label.
    pub label: &'static str,
    /// Proportional gain.
    pub alpha: f64,
    /// Derivative gain.
    pub beta: f64,
    /// Gain margin `alpha + beta / dt`.
    pub gain_margin: f64,
    /// Whether the margin is stable.
    pub stable: bool,
    /// Geometric contraction rate.
    pub contraction_rate: f64,
    /// Error half-life.
    pub half_life: f64,
    /// Steps to 99 percent convergence.
    pub settling_99: f64,
}

/// Analyzer for the Adaptive Governor Convergence Rate theorem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernorConvergenceAnalyzer {
    /// Proportional gain.
    pub alpha: f64,
    /// Derivative gain.
    pub beta: f64,
    /// Time step.
    pub dt: f64,
    /// Minimum epsilon clamp.
    pub eps_min: f64,
    /// Maximum epsilon clamp.
    pub eps_max: f64,
    /// Target ratio used in the Python theorem slice.
    pub r_target: f64,
}

impl GovernorConvergenceAnalyzer {
    /// Construct an analyzer from explicit governor parameters.
    #[inline]
    pub fn new(alpha: f64, beta: f64, dt: f64, eps_min: f64, eps_max: f64, r_target: f64) -> Self {
        Self {
            alpha,
            beta,
            dt,
            eps_min,
            eps_max,
            r_target,
        }
    }

    /// Compute the closed-form theorem bounds.
    #[inline]
    pub fn theoretical_analysis(&self) -> GovernorTheory {
        let rho = contraction_rate(self.alpha, self.beta, self.dt);
        let gain_margin = if self.dt > 0.0 {
            self.alpha + self.beta / self.dt
        } else {
            f64::INFINITY
        };

        GovernorTheory {
            contraction_rate: rho,
            gain_margin,
            gain_margin_stable: gain_margin < 1.0,
            half_life_steps: half_life(rho),
            settling_99_steps: settling_time(rho, 0.01),
            settling_999_steps: settling_time(rho, 0.001),
            lyapunov_rate: rho * rho,
        }
    }

    /// Simulate a constant sparsity measurement without storing history.
    pub fn simulate_constant(
        &self,
        n_steps: usize,
        initial_epsilon: f64,
        measurement_delta: f64,
    ) -> GovernorSimulation {
        if n_steps == 0 || self.dt <= 0.0 {
            return GovernorSimulation {
                initial_error: 0.0,
                final_error: 0.0,
                empirical_rate: f64::NAN,
                theoretical_rate: contraction_rate(self.alpha, self.beta, self.dt),
                converged_99: false,
                final_epsilon: initial_epsilon,
            };
        }

        let mut epsilon = initial_epsilon.clamp(self.eps_min, self.eps_max);
        let mut e_prev = 0.0;
        let mut initial_error = 0.0;
        let mut final_error = 0.0;

        let mut t = 0;
        while t < n_steps {
            let e = measurement_delta / epsilon - self.r_target;
            let abs_e = libm::fabs(e);
            if t == 0 {
                initial_error = abs_e;
            }
            final_error = abs_e;

            let d_error = (e - e_prev) / self.dt;
            let adjustment = self.alpha * e + self.beta * d_error;
            epsilon = (epsilon + adjustment).clamp(self.eps_min, self.eps_max);
            e_prev = e;
            t += 1;
        }

        let empirical_rate = if initial_error > 1e-12 && final_error > 1e-12 {
            libm::pow(final_error / initial_error, 1.0 / n_steps as f64)
        } else {
            0.0
        };

        GovernorSimulation {
            initial_error,
            final_error,
            empirical_rate,
            theoretical_rate: contraction_rate(self.alpha, self.beta, self.dt),
            converged_99: final_error < initial_error * 0.01,
            final_epsilon: epsilon,
        }
    }

    /// Simulate arbitrary sparsity measurements and return error/epsilon history.
    ///
    /// This restores the legacy `governor_convergence.py` history-producing
    /// simulation path while keeping the data structure bounded by the caller's
    /// input slice length. `epsilons[0]` is the clamped initial epsilon, and
    /// each later epsilon is the post-update value for that step.
    pub fn simulate_measurements(
        &self,
        initial_epsilon: f64,
        delta_measurements: &[f64],
    ) -> GovernorSimulationHistory {
        let mut epsilon = initial_epsilon.clamp(self.eps_min, self.eps_max);
        let mut e_prev = 0.0;
        let mut errors = Vec::with_capacity(delta_measurements.len());
        let mut epsilons = Vec::with_capacity(delta_measurements.len().saturating_add(1));
        epsilons.push(epsilon);

        if self.dt <= 0.0 {
            return GovernorSimulationHistory {
                errors,
                epsilons,
                empirical_rate: f64::NAN,
                theoretical_rate: contraction_rate(self.alpha, self.beta, self.dt),
                initial_error: 0.0,
                final_error: 0.0,
                converged_99: false,
            };
        }

        for &delta in delta_measurements {
            let e = delta / epsilon - self.r_target;
            let d_error = (e - e_prev) / self.dt;
            let adjustment = self.alpha * e + self.beta * d_error;

            errors.push(libm::fabs(e));
            epsilon = (epsilon + adjustment).clamp(self.eps_min, self.eps_max);
            epsilons.push(epsilon);
            e_prev = e;
        }

        let initial_error = errors.first().copied().unwrap_or(0.0);
        let final_error = errors.last().copied().unwrap_or(0.0);
        let empirical_rate = empirical_contraction_rate(&errors);

        GovernorSimulationHistory {
            errors,
            epsilons,
            empirical_rate,
            theoretical_rate: contraction_rate(self.alpha, self.beta, self.dt),
            initial_error,
            final_error,
            converged_99: final_error < initial_error * 0.01,
        }
    }

    /// Simulate the legacy constant 0.2 measurement path with full history.
    pub fn simulate_default_history(
        &self,
        n_steps: usize,
        initial_epsilon: f64,
    ) -> GovernorSimulationHistory {
        let mut epsilon = initial_epsilon.clamp(self.eps_min, self.eps_max);
        let mut e_prev = 0.0;
        let mut errors = Vec::with_capacity(n_steps);
        let mut epsilons = Vec::with_capacity(n_steps.saturating_add(1));
        epsilons.push(epsilon);

        if self.dt <= 0.0 {
            return GovernorSimulationHistory {
                errors,
                epsilons,
                empirical_rate: f64::NAN,
                theoretical_rate: contraction_rate(self.alpha, self.beta, self.dt),
                initial_error: 0.0,
                final_error: 0.0,
                converged_99: false,
            };
        }

        let mut t = 0;
        while t < n_steps {
            let e = 0.2 / epsilon - self.r_target;
            let d_error = (e - e_prev) / self.dt;
            let adjustment = self.alpha * e + self.beta * d_error;

            errors.push(libm::fabs(e));
            epsilon = (epsilon + adjustment).clamp(self.eps_min, self.eps_max);
            epsilons.push(epsilon);
            e_prev = e;
            t += 1;
        }

        let initial_error = errors.first().copied().unwrap_or(0.0);
        let final_error = errors.last().copied().unwrap_or(0.0);
        let empirical_rate = empirical_contraction_rate(&errors);

        GovernorSimulationHistory {
            errors,
            epsilons,
            empirical_rate,
            theoretical_rate: contraction_rate(self.alpha, self.beta, self.dt),
            initial_error,
            final_error,
            converged_99: final_error < initial_error * 0.01,
        }
    }

    /// Return the Python gain-tuning presets as a fixed-size no_std table.
    pub fn gain_tuning_table(&self) -> [GainTuningRow; 5] {
        [
            self.row("Conservative", 0.005, 0.02),
            self.row("Default", 0.01, 0.05),
            self.row("Moderate", 0.02, 0.08),
            self.row("Aggressive", 0.05, 0.10),
            self.row("Very Aggressive", 0.10, 0.15),
        ]
    }

    /// Verify the aggregate governor theorem against closed form and simulation.
    pub fn verify_theorem(
        &self,
        n_steps: usize,
        initial_epsilon: f64,
        measurement_delta: f64,
    ) -> GovernorTheoremVerification {
        let theory = self.theoretical_analysis();
        let simulation = self.simulate_constant(n_steps, initial_epsilon, measurement_delta);
        let rate_delta = libm::fabs(simulation.empirical_rate - simulation.theoretical_rate);
        let rate_agreement = simulation.empirical_rate.is_finite()
            && simulation.theoretical_rate.is_finite()
            && rate_delta <= 0.05;
        let improved = simulation.final_error <= simulation.initial_error;
        let stable_rate = theory.contraction_rate.is_finite()
            && theory.contraction_rate > 0.0
            && theory.contraction_rate < 1.0;

        GovernorTheoremVerification {
            theory,
            simulation,
            rate_agreement,
            theorem_holds: theory.gain_margin_stable && stable_rate && improved && rate_agreement,
        }
    }

    #[inline]
    fn row(&self, label: &'static str, alpha: f64, beta: f64) -> GainTuningRow {
        let gain_margin = if self.dt > 0.0 {
            alpha + beta / self.dt
        } else {
            f64::INFINITY
        };
        let rho = contraction_rate(alpha, beta, self.dt);
        GainTuningRow {
            label,
            alpha,
            beta,
            gain_margin,
            stable: gain_margin < 1.0,
            contraction_rate: rho,
            half_life: half_life(rho),
            settling_99: settling_time(rho, 0.01),
        }
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Unit Tests
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

fn empirical_contraction_rate(errors: &[f64]) -> f64 {
    if errors.len() <= 50 || errors[10] <= 1e-12 {
        return f64::NAN;
    }

    let mut count = 0.0;
    let mut sum_t = 0.0;
    let mut sum_y = 0.0;
    let mut sum_tt = 0.0;
    let mut sum_ty = 0.0;

    for &error in &errors[10..] {
        if error <= 1e-12 {
            continue;
        }

        let t = 10.0 + count;
        let y = libm::log(error);
        count += 1.0;
        sum_t += t;
        sum_y += y;
        sum_tt += t * t;
        sum_ty += t * y;
    }

    if count <= 10.0 {
        return f64::NAN;
    }

    let denominator = count * sum_tt - sum_t * sum_t;
    if libm::fabs(denominator) <= f64::EPSILON {
        return f64::NAN;
    }

    libm::exp((count * sum_ty - sum_t * sum_y) / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_epsilon() {
        let gov = GeometricGovernor::new();
        assert!((gov.epsilon() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_clamped_min() {
        let mut gov = GeometricGovernor::new();

        // Drive epsilon down with high deviation (system waking too often)
        for _ in 0..10000 {
            gov.adapt(1000.0, 0.001);
        }

        assert!(gov.epsilon() >= EPSILON_MIN);
    }

    #[test]
    fn test_epsilon_clamped_max() {
        let mut gov = GeometricGovernor::new();

        // Drive epsilon up with low deviation (system sleeping too much)
        for _ in 0..10000 {
            gov.adapt(0.0001, 0.001);
        }

        assert!(gov.epsilon() <= EPSILON_MAX);
    }

    #[test]
    fn test_high_load_raises_epsilon() {
        let mut gov = GeometricGovernor::new();
        let initial = gov.epsilon();

        // High deviation = waking too often = raise epsilon
        gov.adapt(10000.0, 0.001);

        // Epsilon should increase (after initial transient)
        // Note: May need multiple iterations due to derivative term
        for _ in 0..10 {
            gov.adapt(10000.0, 0.001);
        }

        assert!(gov.epsilon() > initial);
    }

    #[test]
    fn test_trigger_threshold() {
        let gov = GeometricGovernor::with_epsilon(0.5);

        assert!(!gov.should_trigger(0.4));
        assert!(gov.should_trigger(0.5));
        assert!(gov.should_trigger(0.6));
    }

    #[test]
    fn convergence_analyzer_verifies_aggregate_theorem() {
        let analyzer = GovernorConvergenceAnalyzer::new(0.01, 0.05, 1.0, 0.05, 1.0, 0.5);
        let report = analyzer.verify_theorem(200, 0.5, 0.2);
        assert!(report.theory.gain_margin_stable);
        assert!(report.simulation.final_error <= report.simulation.initial_error);
        assert!(report.theorem_holds);
    }

    #[test]
    fn simulate_measurements_returns_legacy_histories() {
        let analyzer = GovernorConvergenceAnalyzer::new(0.1, 0.05, 1.0, 0.1, 0.9, 0.3);
        let history = analyzer.simulate_measurements(0.5, &[0.2, 0.25, 0.15]);

        assert_eq!(history.errors.len(), 3);
        assert_eq!(history.epsilons.len(), 4);
        assert!((history.errors[0] - 0.1).abs() < 1e-12);
        assert!((history.epsilons[0] - 0.5).abs() < 1e-12);
        assert!((history.epsilons[1] - 0.515).abs() < 1e-12);
        assert!(history.empirical_rate.is_nan());
        assert_eq!(history.initial_error, history.errors[0]);
        assert_eq!(history.final_error, history.errors[2]);
    }

    #[test]
    fn default_history_matches_constant_simulation_summary() {
        let analyzer = GovernorConvergenceAnalyzer::new(0.01, 0.05, 1.0, 0.1, 0.9, 0.3);
        let history = analyzer.simulate_default_history(500, 0.5);
        let summary = analyzer.simulate_constant(500, 0.5, 0.2);

        assert_eq!(history.errors.len(), 500);
        assert_eq!(history.epsilons.len(), 501);
        assert!((history.initial_error - summary.initial_error).abs() < 1e-12);
        assert!((history.final_error - summary.final_error).abs() < 1e-12);
        assert!((history.epsilons[500] - summary.final_epsilon).abs() < 1e-12);
        assert!(history.empirical_rate.is_finite());
        assert!((history.empirical_rate - history.theoretical_rate).abs() < 0.05);
    }
}
