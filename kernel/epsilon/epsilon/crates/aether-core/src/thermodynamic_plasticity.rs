// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Thermodynamic Erasure Bound (TEB) formulas in `no_std` Rust.
//!
//! This module replaces the legacy `epsilon_core/thermodynamic_plasticity.py`
//! theorem slice. It keeps the Landauer energy floor, hot-partition power
//! bound, Gibbs entropy learning-rate equivalence, and Helmholtz free-energy
//! checks available to the Rust/Aether theorem runtime without NumPy or Python.

use libm::{fabs, log, log10};

/// Boltzmann constant in joules per kelvin.
pub const K_BOLTZMANN: f64 = 1.380_649e-23;
/// Natural logarithm of 2.
pub const LN2: f64 = core::f64::consts::LN_2;
/// Landauer limit in joules per bit at 1 kelvin.
pub const LANDAUER_LIMIT: f64 = K_BOLTZMANN * LN2;
/// Default total parameter count for the H100 theorem slice.
pub const H100_TOTAL_PARAMS: u64 = 70_000_000_000;
/// Default hot partition ratio for continuous learning.
pub const H100_HOT_RATIO: f64 = 0.005;
/// Default precision bits for fp16 hot weights.
pub const H100_PRECISION_BITS: u32 = 16;
/// Default operating temperature in kelvin.
pub const ROOM_TEMPERATURE_K: f64 = 300.0;
/// H100 SXM thermal design power used by the legacy theorem slice.
pub const H100_GPU_POWER_WATTS: f64 = 700.0;
const ENTROPY_EPS: f64 = 1e-30;

/// Minimum energy to erase one bit at `temperature_k`.
pub fn landauer_energy_per_bit(temperature_k: f64) -> f64 {
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return 0.0;
    }
    K_BOLTZMANN * temperature_k * LN2
}

/// Minimum energy per hot-partition update.
pub fn min_energy_per_update(n_hot_params: u64, precision_bits: u32, temperature_k: f64) -> f64 {
    n_hot_params as f64 * f64::from(precision_bits) * landauer_energy_per_bit(temperature_k)
}

/// Maximum hot ratio sustainable inside a power budget.
pub fn max_sustainable_hot_ratio(
    total_params: u64,
    power_budget_watts: f64,
    update_freq_hz: f64,
    precision_bits: u32,
    temperature_k: f64,
) -> f64 {
    let denom = update_freq_hz
        * total_params as f64
        * f64::from(precision_bits)
        * landauer_energy_per_bit(temperature_k);
    if !denom.is_finite() || denom < ENTROPY_EPS {
        return f64::INFINITY;
    }
    power_budget_watts / denom
}

/// Gibbs entropy `S = -k_B * sum(p_i ln p_i)` for a signed distribution slice.
pub fn gibbs_entropy(distribution: &[f64]) -> f64 {
    let total = abs_sum_with_floor(distribution);
    if total <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for value in distribution {
        let p = (fabs(*value) + ENTROPY_EPS) / total;
        entropy -= K_BOLTZMANN * p * log(p);
    }
    entropy
}

/// Entropy-adaptive learning rate in thermodynamic form.
pub fn thermodynamic_lr(alpha: f64, gibbs_s: f64, s_max: f64) -> f64 {
    if !s_max.is_finite() || s_max < ENTROPY_EPS {
        return alpha;
    }
    let ratio = (s_max - gibbs_s) / s_max;
    alpha * ratio.max(0.0)
}

/// Helmholtz free-energy change `delta_F = delta_U - T * delta_S`.
pub fn helmholtz_free_energy_change(delta_u: f64, temperature_k: f64, delta_s: f64) -> f64 {
    delta_u - temperature_k * delta_s
}

/// Analyzer configuration for TEB energy and learning-rate reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermodynamicAnalyzer {
    /// Total model parameter count.
    pub total_params: u64,
    /// Ratio of parameters in the hot mutable partition.
    pub hot_ratio: f64,
    /// Number of hot mutable parameters.
    pub hot_params: u64,
    /// Precision bits per hot parameter.
    pub precision_bits: u32,
    /// Operating temperature in kelvin.
    pub temperature_k: f64,
    /// Available GPU power in watts.
    pub gpu_power_watts: f64,
}

impl ThermodynamicAnalyzer {
    /// Create an analyzer from explicit model and hardware parameters.
    pub fn new(
        total_params: u64,
        hot_ratio: f64,
        precision_bits: u32,
        temperature_k: f64,
        gpu_power_watts: f64,
    ) -> Self {
        let bounded_ratio = if hot_ratio.is_finite() && hot_ratio > 0.0 {
            hot_ratio
        } else {
            0.0
        };
        Self {
            total_params,
            hot_ratio: bounded_ratio,
            hot_params: (total_params as f64 * bounded_ratio) as u64,
            precision_bits,
            temperature_k,
            gpu_power_watts,
        }
    }

    /// Create the default H100/70B analyzer used by the legacy theorem slice.
    pub fn h100_default() -> Self {
        Self::new(
            H100_TOTAL_PARAMS,
            H100_HOT_RATIO,
            H100_PRECISION_BITS,
            ROOM_TEMPERATURE_K,
            H100_GPU_POWER_WATTS,
        )
    }

    /// Compute the Landauer floor and headroom for a given update frequency.
    pub fn energy_analysis(&self, update_freq_hz: f64) -> EnergyAnalysis {
        let energy_per_update =
            min_energy_per_update(self.hot_params, self.precision_bits, self.temperature_k);
        let power_floor_watts = energy_per_update * update_freq_hz;
        let max_hot_ratio_thermodynamic = max_sustainable_hot_ratio(
            self.total_params,
            self.gpu_power_watts,
            update_freq_hz,
            self.precision_bits,
            self.temperature_k,
        );
        let orders_above_landauer = if power_floor_watts > 0.0 {
            log10(self.gpu_power_watts / power_floor_watts)
        } else {
            f64::INFINITY
        };
        let thermodynamic_headroom = if self.hot_ratio > 0.0 {
            max_hot_ratio_thermodynamic / self.hot_ratio
        } else {
            f64::INFINITY
        };

        EnergyAnalysis {
            hot_params: self.hot_params,
            precision_bits: self.precision_bits,
            temperature_k: self.temperature_k,
            landauer_per_bit_j: landauer_energy_per_bit(self.temperature_k),
            energy_per_update_j: energy_per_update,
            power_floor_watts,
            gpu_power_watts: self.gpu_power_watts,
            orders_above_landauer,
            max_hot_ratio_thermodynamic,
            current_hot_ratio: self.hot_ratio,
            thermodynamic_headroom,
        }
    }

    /// Compare Shannon and Gibbs formulations of entropy-adaptive learning.
    pub fn lr_analysis(&self, output_distribution: &[f64], alpha: f64) -> LearningRateAnalysis {
        let h_shannon = shannon_entropy(output_distribution);
        let h_max = if output_distribution.len() > 1 {
            log(output_distribution.len() as f64) / LN2
        } else {
            0.0
        };
        let s_gibbs = gibbs_entropy(output_distribution);
        let s_max = if output_distribution.len() > 1 {
            K_BOLTZMANN * log(output_distribution.len() as f64)
        } else {
            0.0
        };
        let eta_shannon = if h_max > ENTROPY_EPS {
            alpha * (1.0 - h_shannon / h_max).max(0.0)
        } else {
            alpha
        };
        let eta_gibbs = thermodynamic_lr(alpha, s_gibbs, s_max);
        let delta_s = s_max - s_gibbs;
        let delta_f = helmholtz_free_energy_change(0.001, self.temperature_k, delta_s);

        LearningRateAnalysis {
            h_shannon,
            h_max,
            s_gibbs,
            s_max,
            eta_shannon,
            eta_gibbs,
            lr_agreement: fabs(eta_shannon - eta_gibbs) < 1e-10,
            delta_f,
            learning_favorable: delta_f < 0.0,
        }
    }

    /// Analyze a multi-H100 cluster with `n_gpus` devices.
    pub fn h100_cluster_analysis(&self, n_gpus: u32) -> ClusterAnalysis {
        let energy = self.energy_analysis(20.0);
        let total_power = self.gpu_power_watts * f64::from(n_gpus);
        let max_hot_ratio_cluster = max_sustainable_hot_ratio(
            self.total_params,
            total_power,
            20.0,
            self.precision_bits,
            self.temperature_k,
        );

        ClusterAnalysis {
            n_gpus,
            total_gpu_power_kw: total_power / 1000.0,
            landauer_floor_watts: energy.power_floor_watts,
            orders_above_landauer: energy.orders_above_landauer,
            max_hot_ratio_cluster,
            practical_limit: "Compute-bound, not thermodynamic-bound",
        }
    }
}

/// Landauer floor report for a hot-partition update loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyAnalysis {
    /// Number of hot mutable parameters.
    pub hot_params: u64,
    /// Precision bits per hot parameter.
    pub precision_bits: u32,
    /// Operating temperature in kelvin.
    pub temperature_k: f64,
    /// Landauer floor per bit in joules.
    pub landauer_per_bit_j: f64,
    /// Minimum energy per update in joules.
    pub energy_per_update_j: f64,
    /// Minimum continuous update power in watts.
    pub power_floor_watts: f64,
    /// Available GPU power in watts.
    pub gpu_power_watts: f64,
    /// Base-10 orders of magnitude between GPU power and Landauer floor.
    pub orders_above_landauer: f64,
    /// Maximum hot ratio allowed by the power budget.
    pub max_hot_ratio_thermodynamic: f64,
    /// Current configured hot ratio.
    pub current_hot_ratio: f64,
    /// Ratio between thermodynamic maximum and configured hot ratio.
    pub thermodynamic_headroom: f64,
}

/// Learning-rate equivalence report for Shannon and Gibbs entropy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningRateAnalysis {
    /// Shannon entropy in bits.
    pub h_shannon: f64,
    /// Maximum Shannon entropy in bits.
    pub h_max: f64,
    /// Gibbs entropy in joules per kelvin.
    pub s_gibbs: f64,
    /// Maximum Gibbs entropy in joules per kelvin.
    pub s_max: f64,
    /// Entropy-adaptive learning rate via Shannon entropy.
    pub eta_shannon: f64,
    /// Entropy-adaptive learning rate via Gibbs entropy.
    pub eta_gibbs: f64,
    /// True when both learning-rate formulations agree numerically.
    pub lr_agreement: bool,
    /// Helmholtz free-energy change.
    pub delta_f: f64,
    /// True when the configured free-energy change is favorable.
    pub learning_favorable: bool,
}

/// H100 cluster-scale thermodynamic report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterAnalysis {
    /// Number of H100 GPUs in the cluster.
    pub n_gpus: u32,
    /// Total GPU power in kilowatts.
    pub total_gpu_power_kw: f64,
    /// Landauer continuous-update floor in watts.
    pub landauer_floor_watts: f64,
    /// Orders of magnitude between GPU power and Landauer floor.
    pub orders_above_landauer: f64,
    /// Maximum hot ratio allowed by total cluster power.
    pub max_hot_ratio_cluster: f64,
    /// Practical bottleneck classification.
    pub practical_limit: &'static str,
}

fn shannon_entropy(distribution: &[f64]) -> f64 {
    let total = abs_sum_with_floor(distribution);
    if total <= 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for value in distribution {
        let p = (fabs(*value) + ENTROPY_EPS) / total;
        entropy -= p * (log(p) / LN2);
    }
    entropy
}

fn abs_sum_with_floor(distribution: &[f64]) -> f64 {
    if distribution.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    for value in distribution {
        total += fabs(*value) + ENTROPY_EPS;
    }
    total
}
