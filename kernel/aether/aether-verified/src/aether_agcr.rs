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

/// Eigenvalues of the 2×2 PD state matrix.
///
/// The paper defines: A = [[1 − β/dt, β/dt], [1, 0]]
/// Characteristic polynomial: r² − (1 − β/dt)r − β/dt = 0
/// which reduces to r² + (β/dt)r + (α/dt) = 0 after substitution.
///
/// Returns (real parts, imaginary parts) of the two eigenvalues.
/// Overdamped: β² > 4α·dt → two distinct real eigenvalues.
/// Underdamped: β² < 4α·dt → complex conjugate pair, |r| = √(α/dt).
pub fn eigenvalues(alpha: f64, beta: f64, dt: f64) -> ((f64, f64), (f64, f64)) {
    let discriminant = beta * beta - 4.0 * alpha * dt;
    let neg_b_over_2dt = -beta / (2.0 * dt);

    if discriminant >= 0.0 {
        let sqrt_d = libm::sqrt(discriminant) / (2.0 * dt);
        (
            (neg_b_over_2dt + sqrt_d, 0.0),
            (neg_b_over_2dt - sqrt_d, 0.0),
        )
    } else {
        let imag = libm::sqrt(-discriminant) / (2.0 * dt);
        ((neg_b_over_2dt, imag), (neg_b_over_2dt, -imag))
    }
}

/// Spectral radius of the PD state matrix.
///
/// ρ(A) = max(|r₁|, |r₂|) where r₁, r₂ are the eigenvalues.
/// Overdamped: max of the two real eigenvalue magnitudes.
/// Underdamped: |r| = √(α/dt) for both (complex conjugate pair).
pub fn spectral_radius(alpha: f64, beta: f64, dt: f64) -> f64 {
    let ((r1, i1), (r2, i2)) = eigenvalues(alpha, beta, dt);
    let mag1 = libm::sqrt(r1 * r1 + i1 * i1);
    let mag2 = libm::sqrt(r2 * r2 + i2 * i2);
    if mag1 > mag2 {
        mag1
    } else {
        mag2
    }
}

/// Check if the system is overdamped (real eigenvalues) or underdamped (complex).
pub fn is_overdamped(alpha: f64, beta: f64, dt: f64) -> bool {
    beta * beta > 4.0 * alpha * dt
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

    #[test]
    fn test_overdamped_default() {
        // Default params: β²=0.0025, 4αdt=0.04 → underdamped
        assert!(!is_overdamped(0.01, 0.05, 1.0));
    }

    #[test]
    fn test_overdamped_large_beta() {
        // β=1.0, α=0.01, dt=1.0 → β²=1.0 > 4αdt=0.04 → overdamped
        assert!(is_overdamped(0.01, 1.0, 1.0));
    }

    #[test]
    fn test_spectral_radius_less_than_one() {
        let rho = spectral_radius(0.01, 0.05, 1.0);
        assert!(rho < 1.0, "spectral_radius = {}", rho);
    }

    #[test]
    fn test_underdamped_magnitude() {
        // Underdamped: |r| = √(α/dt)
        let alpha = 0.01;
        let dt = 1.0;
        let rho = spectral_radius(alpha, 0.05, dt);
        let expected = libm::sqrt(alpha / dt);
        assert!(
            (rho - expected).abs() < 1e-10,
            "ρ={rho} expected=√(α/dt)={expected}"
        );
    }

    #[test]
    fn test_eigenvalues_overdamped_real() {
        let ((r1, i1), (r2, i2)) = eigenvalues(0.01, 1.0, 1.0);
        assert!(i1.abs() < 1e-12 && i2.abs() < 1e-12, "should be real");
        assert!(r1.abs() < 1.0 && r2.abs() < 1.0, "both < 1 for stability");
    }
}
