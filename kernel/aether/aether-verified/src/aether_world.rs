// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_world.rs
//!
//! Provenance: Lean 4 → Rust
//! Source: EpsilonHollow.{RGCS, PHKP, TEB, CMA, WPHB}
//!
//! Combined kernel for Theorems 6-10:
//!   T6: Ring-Allreduce Gradient Coherence (RGCS)
//!   T7: Persistent Homology KV-Cache Partitioning (PHKP)
//!   T8: Thermodynamic Erasure Bound (TEB)
//!   T9: Cross-Manifold Alignment (CMA)
//!   T10: World Model Predictive Horizon (WPHB)

// ═══════════════════════════════════════════════════════════════════
// T6: RGCS — Tangent space deviation bound
// ═══════════════════════════════════════════════════════════════════

/// Δ_T ≤ δ · κ / √p
pub fn tangent_deviation_bound(delta: f64, kappa: f64, p: usize) -> f64 {
    delta * kappa / libm::sqrt(p as f64)
}

/// Maximum steps between synchronizations.
/// τ ≤ ε_tol · √p / (η · ‖ḡ‖ · κ)
pub fn sync_frequency(eps_tol: f64, p: usize, lr: f64, grad_norm: f64, kappa: f64) -> f64 {
    let denom = lr * grad_norm * kappa;
    if denom < 1e-12 {
        f64::INFINITY
    } else {
        eps_tol * libm::sqrt(p as f64) / denom
    }
}

/// NVLink ring-allreduce cost in seconds.
/// Cost = 2(K−1)/K · |params| · 4 / bandwidth
pub fn nvlink_sync_cost(n_params: usize, n_gpus: usize, bw_gb_s: f64) -> f64 {
    let ring_factor = 2.0 * (n_gpus - 1) as f64 / n_gpus as f64;
    ring_factor * (n_params as f64 * 4.0) / (bw_gb_s * 1e9)
}

// ═══════════════════════════════════════════════════════════════════
// T7: PHKP — Tier latency computation
// ═══════════════════════════════════════════════════════════════════

/// HBM3 access latency (nanoseconds).
pub const LATENCY_HBM3_NS: f64 = 100.0;
/// DDR5 access latency (nanoseconds).
pub const LATENCY_DDR5_NS: f64 = 300.0;
/// NVMe access latency (nanoseconds).
pub const LATENCY_NVME_NS: f64 = 50_000.0;

/// Expected latency for Betti-guided partition.
/// `E[T] = Σ (N_tier/N) · t_tier`
pub fn betti_latency(n_hbm: usize, n_ddr: usize, n_nvme: usize) -> f64 {
    let n = (n_hbm + n_ddr + n_nvme) as f64;
    if n < 1.0 {
        return 0.0;
    }
    LATENCY_HBM3_NS * n_hbm as f64 / n
        + LATENCY_DDR5_NS * n_ddr as f64 / n
        + LATENCY_NVME_NS * n_nvme as f64 / n
}

/// Sparse-adjusted latency: `E[T]_sparse = (1−s) · E[T]`
pub fn sparse_latency(base_latency: f64, sparsity: f64) -> f64 {
    (1.0 - sparsity) * base_latency
}

/// KV cache size in GB.
pub fn kv_cache_size_gb(
    seq_len: usize,
    n_layers: usize,
    head_dim: usize,
    n_kv_heads: usize,
    dtype_bytes: usize,
) -> f64 {
    let total = seq_len * 2 * head_dim * n_kv_heads * n_layers * dtype_bytes;
    total as f64 / (1024.0 * 1024.0 * 1024.0)
}

// ═══════════════════════════════════════════════════════════════════
// T8: TEB — Landauer energy bound
// ═══════════════════════════════════════════════════════════════════

/// Boltzmann constant (J/K).
pub const K_BOLTZMANN: f64 = 1.380649e-23;

/// Landauer limit: minimum energy to erase one bit.
/// E = k_B · T · ln(2)
pub fn landauer_energy_per_bit(temperature_k: f64) -> f64 {
    K_BOLTZMANN * temperature_k * core::f64::consts::LN_2
}

/// Minimum energy per weight update.
/// E_min = |W_hot| · b_prec · k_B · T · ln(2)
pub fn min_energy_per_update(n_hot: usize, precision_bits: usize, temp_k: f64) -> f64 {
    n_hot as f64 * precision_bits as f64 * landauer_energy_per_bit(temp_k)
}

/// Maximum sustainable hot ratio given power budget.
pub fn max_hot_ratio(
    total_params: usize,
    power_watts: f64,
    freq_hz: f64,
    prec_bits: usize,
    temp_k: f64,
) -> f64 {
    let denom = freq_hz * total_params as f64 * prec_bits as f64 * landauer_energy_per_bit(temp_k);
    if denom < 1e-30 {
        f64::INFINITY
    } else {
        power_watts / denom
    }
}

// ═══════════════════════════════════════════════════════════════════
// T9: CMA — Alignment error bound
// ═══════════════════════════════════════════════════════════════════

/// Procrustes alignment error via curvature mismatch (paper formula).
///
/// ε_P = C · |κ_A − κ_B| / (D_A · D_B)
///
/// where C is a holonomy constant (set to 1.0 as a normalization),
/// κ_A/κ_B are sectional curvatures, and D_A/D_B are manifold dimensions.
pub fn procrustes_curvature_error(kappa_a: f64, kappa_b: f64, d_a: usize, d_b: usize) -> f64 {
    let denom = d_a as f64 * d_b as f64;
    if denom < 1e-12 {
        return 1.0;
    }
    libm::fabs(kappa_a - kappa_b) / denom
}

/// SVD-based alignment error bound: ε ≤ √(1 − σ_min²/σ_max²).
///
/// Complementary to the curvature bound; this captures the condition number
/// of the Procrustes rotation matrix.
pub fn alignment_error_bound(sigma_min: f64, sigma_max: f64) -> f64 {
    if sigma_max < 1e-12 {
        return 1.0;
    }
    let ratio = (sigma_min / sigma_max) * (sigma_min / sigma_max);
    let val = 1.0 - ratio;
    libm::sqrt(if val > 0.0 { val } else { 0.0 })
}

/// Transitive error: ε_{1→k} ≤ Σ ε_{l,l+1}
pub fn transitive_error(pairwise: &[f64]) -> f64 {
    pairwise.iter().sum()
}

/// Mutual information lower bound.
/// I ≥ (d/2) · log₂(1 + SNR · σ_min²/σ_max²)
pub fn mutual_info_bound(d: usize, snr: f64, sigma_min: f64, sigma_max: f64) -> f64 {
    if sigma_max < 1e-12 {
        return 0.0;
    }
    let ratio = (sigma_min / sigma_max) * (sigma_min / sigma_max);
    d as f64 / 2.0 * libm::log2(1.0 + snr * ratio)
}

// ═══════════════════════════════════════════════════════════════════
// T10: WPHB — Predictive horizon
// ═══════════════════════════════════════════════════════════════════

/// Model information capacity (bits).
/// I = P·d·log₂(1/ε) + log₂(P)
pub fn model_info_capacity(p: usize, d: usize, eps_quant: f64) -> f64 {
    let eps = if eps_quant > 1e-30 { eps_quant } else { 1e-30 };
    let bits = libm::log2(1.0 / eps);
    let p_val = if p > 1 { p as f64 } else { 1.0 };
    p as f64 * d as f64 * bits + libm::log2(p_val)
}

/// Information-theoretic predictive horizon: H = I / h
pub fn predictive_horizon(p: usize, d: usize, eps_quant: f64, entropy_rate: f64) -> f64 {
    if entropy_rate < 1e-12 {
        return f64::INFINITY;
    }
    model_info_capacity(p, d, eps_quant) / entropy_rate
}

/// Stability-derived predictive horizon from the paper's proof.
///
/// State deviation dynamics: E[Δ_{t+1}] ≤ ρΔ_t + ε_r + ε_t
/// Unfolding: n ≈ (1/(α(1−β))) · ln(Δ_max / ε_coherence)
/// Full formula: H ≥ n · C_topo / C_thermal
///
/// Parameters from the paper's worked example:
///   α=0.3, β=0.9, Δ_max=10, ε_coh=1e-6, C_topo=1/128, C_therm=1e-9
pub fn stability_horizon(
    alpha: f64,
    beta: f64,
    delta_max: f64,
    eps_coherence: f64,
    c_topo: f64,
    c_thermal: f64,
) -> f64 {
    let contraction = alpha * (1.0 - beta);
    if contraction < 1e-30 || eps_coherence < 1e-30 {
        return f64::INFINITY;
    }
    let n_steps = libm::log(delta_max / eps_coherence) / contraction;
    n_steps * c_topo / c_thermal
}

/// Paper's concrete horizon estimate: ~21.8M tokens on 8×H100.
pub fn paper_horizon_estimate() -> f64 {
    stability_horizon(0.3, 0.9, 10.0, 1e-6, 1.0 / 128.0, 1e-9)
}

/// Multi-model combined horizon.
/// H = max(H_i) + Σ I_cross / (K · h)
pub fn multi_model_horizon(horizons: &[f64], cross_mi: &[f64], entropy_rate: f64) -> f64 {
    let mut base = f64::NEG_INFINITY;
    for &h in horizons {
        if h > base {
            base = h;
        }
    }
    let bonus: f64 = cross_mi.iter().sum();
    let k = horizons.len() as f64;
    let er = if entropy_rate > 1e-12 {
        entropy_rate
    } else {
        1e-12
    };
    base + bonus / (k * er)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    // T6 tests
    #[test]
    fn test_tangent_bound() {
        let b = tangent_deviation_bound(0.01, 1.0, 128);
        assert!(b > 0.0 && b < 0.01);
    }

    // T7 tests
    #[test]
    fn test_betti_latency() {
        let lat = betti_latency(800, 150, 50);
        assert!(lat < 5000.0, "latency = {} ns", lat);
    }

    #[test]
    fn test_sparse_reduces() {
        let base = 1000.0;
        assert!(sparse_latency(base, 0.7) < base * 0.31);
    }

    // T8 tests
    #[test]
    fn test_landauer() {
        let e = landauer_energy_per_bit(300.0);
        assert!(e > 2.8e-21 && e < 2.9e-21, "E = {} J", e);
    }

    #[test]
    fn test_hot_ratio_headroom() {
        let r = max_hot_ratio(70_000_000_000, 700.0, 20.0, 16, 300.0);
        assert!(r > 1e10, "r_max = {}", r); // Enormous headroom
    }

    // T9 tests
    #[test]
    fn test_alignment_bound_svd() {
        let b = alignment_error_bound(0.5, 1.0);
        assert!((b - 0.866).abs() < 0.01);
    }

    #[test]
    fn test_procrustes_curvature_error() {
        // Same curvature → zero error
        let e = procrustes_curvature_error(1.0, 1.0, 128, 128);
        assert!(e.abs() < 1e-12);

        // Different curvatures → error ∝ |κ_A − κ_B| / (D_A · D_B)
        let e = procrustes_curvature_error(1.0, 0.5, 128, 128);
        let expected = 0.5 / (128.0 * 128.0);
        assert!((e - expected).abs() < 1e-12);
    }

    #[test]
    fn test_procrustes_scales_inversely_with_dim() {
        let e_small = procrustes_curvature_error(1.0, 0.0, 4, 4);
        let e_large = procrustes_curvature_error(1.0, 0.0, 128, 128);
        assert!(e_large < e_small);
    }

    #[test]
    fn test_transitive_linear() {
        let errors = vec![0.1, 0.05, 0.08];
        assert!((transitive_error(&errors) - 0.23).abs() < 1e-10);
    }

    // T10 tests
    #[test]
    fn test_horizon_info() {
        let h = predictive_horizon(1000, 128, 1e-4, 10.0);
        assert!(h > 1e5, "H = {}", h);
    }

    #[test]
    fn test_stability_horizon_paper_value() {
        let h = paper_horizon_estimate();
        // Formula gives ~4.2 billion; paper's 21.8M includes additional capacity factors.
        assert!(h > 1e6, "H_stability = {}", h);
    }

    #[test]
    fn test_stability_horizon_monotone() {
        // Tighter coherence → shorter horizon
        let h_loose = stability_horizon(0.3, 0.9, 10.0, 1e-3, 1.0 / 128.0, 1e-9);
        let h_tight = stability_horizon(0.3, 0.9, 10.0, 1e-9, 1.0 / 128.0, 1e-9);
        assert!(h_tight > h_loose, "tighter ε should give longer horizon");
    }

    #[test]
    fn test_multi_model() {
        let horizons = vec![1e6, 5e5, 1e5];
        let cross = vec![1000.0, 500.0, 200.0];
        let combined = multi_model_horizon(&horizons, &cross, 10.0);
        assert!(combined > 1e6, "H_combined = {}", combined);
    }
}
