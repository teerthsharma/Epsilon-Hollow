// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_agcr.rs
//!
//! Provenance: Lean 4 → Rust
//! Source: EpsilonHollow.GovernorConvergence
//!
//! # Adaptive Governor Convergence Rate (AGCR)
//!
//! Theorem 4: ρ = 1 − α/(1 + β/dt), t_{1/2} = ln2/|lnρ|

/// Contraction rate of the PD governor.
///
/// ρ = 1 − α/(1 + β/dt) ∈ (0, 1)
pub fn contraction_rate(alpha: f64, beta: f64, dt: f64) -> f64 {
    1.0 - alpha / (1.0 + beta / dt)
}

/// Half-life in steps.
pub fn half_life(rho: f64) -> f64 {
    if rho <= 0.0 || rho >= 1.0 {
        return f64::INFINITY;
    }
    core::f64::consts::LN_2 / libm::fabs(libm::log(rho))
}

/// Settling time for target_ratio convergence (e.g., 0.01 = 99%).
pub fn settling_time(rho: f64, target_ratio: f64) -> f64 {
    if rho <= 0.0 || rho >= 1.0 {
        return f64::INFINITY;
    }
    libm::log(1.0 / target_ratio) / libm::fabs(libm::log(rho))
}

/// Required α for a desired settling time.
pub fn required_alpha(beta: f64, dt: f64, t_target: f64, target_ratio: f64) -> f64 {
    let rho_req = libm::exp(-libm::log(1.0 / target_ratio) / t_target);
    (1.0 + beta / dt) * (1.0 - rho_req)
}

/// Verify gain margin stability: α + β/dt < 1.
pub fn gain_margin_stable(alpha: f64, beta: f64, dt: f64) -> bool {
    alpha + beta / dt < 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rate() {
        let rho = contraction_rate(0.01, 0.05, 1.0);
        // ρ = 1 − 0.01/1.05 ≈ 0.9905
        assert!((rho - 0.9905).abs() < 0.001, "ρ = {}", rho);
    }

    #[test]
    fn test_half_life_default() {
        let rho = contraction_rate(0.01, 0.05, 1.0);
        let hl = half_life(rho);
        // ~73 steps
        assert!(hl > 65.0 && hl < 80.0, "t_1/2 = {}", hl);
    }

    #[test]
    fn test_gain_margin() {
        assert!(gain_margin_stable(0.01, 0.05, 1.0));
        assert!(gain_margin_stable(0.1, 0.3, 1.0));
        assert!(!gain_margin_stable(0.5, 0.6, 1.0));
    }

    #[test]
    fn test_settling_99() {
        let rho = contraction_rate(0.01, 0.05, 1.0);
        let t99 = settling_time(rho, 0.01);
        assert!(t99 > 400.0 && t99 < 550.0, "t_99 = {}", t99);
    }
}
