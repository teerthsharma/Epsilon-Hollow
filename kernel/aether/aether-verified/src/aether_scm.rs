//! aether_scm.rs
//!
//! Provenance: Lean 4 → Rust
//! Source: EpsilonHollow.SpectralContraction
//!
//! # Spectral Contraction Mapping (SCM)
//!
//! Theorem 2: Banach fixed-point for I/O prediction.
//! Lip(T) = 1 − α_min < 1

/// Lipschitz constant of the telemetry operator.
///
/// Lip(T) = 1 − α_min
pub fn lipschitz_constant(alpha_min: f64) -> f64 {
    1.0 - alpha_min
}

/// Convergence half-life: t_{1/2} = ln(2) / |ln(ρ)|
pub fn convergence_half_life(rho: f64) -> f64 {
    if rho <= 0.0 || rho >= 1.0 {
        return f64::INFINITY;
    }
    core::f64::consts::LN_2 / libm::fabs(libm::log(rho))
}

/// Verify contraction: ‖T(S₁) − T(S₂)‖ ≤ Lip · ‖S₁ − S₂‖
pub fn verify_contraction(
    dist_before: f64,
    dist_after: f64,
    alpha_min: f64,
) -> bool {
    let lip = lipschitz_constant(alpha_min);
    dist_after <= lip * dist_before + 1e-10
}

/// Apply telemetry operator: T(S) = (1−α)S + α·S_pred
pub fn apply_operator(s: f64, s_pred: f64, alpha: f64) -> f64 {
    (1.0 - alpha) * s + alpha * s_pred
}

/// Error after t iterations: err_t ≤ ρ^t · err_0
pub fn error_bound(rho: f64, t: u64, err_0: f64) -> f64 {
    libm::pow(rho, t as f64) * err_0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lipschitz_less_than_one() {
        assert!(lipschitz_constant(0.01) < 1.0);
        assert!(lipschitz_constant(0.5) < 1.0);
        assert!(lipschitz_constant(0.99) < 1.0);
    }

    #[test]
    fn test_contraction() {
        let s1 = 5.0;
        let s2 = 3.0;
        let pred = 4.0;
        let alpha = 0.1;
        let t_s1 = apply_operator(s1, pred, alpha);
        let t_s2 = apply_operator(s2, pred, alpha);
        assert!(verify_contraction(
            (s1 - s2).abs(),
            (t_s1 - t_s2).abs(),
            alpha,
        ));
    }

    #[test]
    fn test_half_life() {
        let hl = convergence_half_life(0.99);
        assert!(hl > 60.0 && hl < 80.0, "half_life = {}", hl);
    }

    #[test]
    fn test_error_decreases() {
        let rho = 0.95;
        let err_0 = 100.0;
        let err_10 = error_bound(rho, 10, err_0);
        let err_100 = error_bound(rho, 100, err_0);
        assert!(err_10 < err_0);
        assert!(err_100 < err_10);
    }
}
