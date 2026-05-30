// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! World Model Predictive Horizon Bound (WPHB) in `no_std` Rust.
//!
//! This replaces the legacy `epsilon_core/world_model_horizon.py` theorem
//! slice with deterministic Rust formulas for topological horizon, flat
//! horizon, compression advantage, and multi-model alignment bonus.

use libm::log;

const LOG2: f64 = core::f64::consts::LN_2;
const EPS: f64 = 1e-12;
const BYTES_PER_TIB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;

/// Compute the topological predictive horizon.
pub fn predictive_horizon(
    clusters: u64,
    dimension: u32,
    epsilon_quant: f64,
    entropy_rate: f64,
) -> f64 {
    if entropy_rate < EPS {
        return f64::INFINITY;
    }
    let p = clusters.max(1) as f64;
    let bits_quant = quantization_bits(epsilon_quant);
    let model_info = p * f64::from(dimension) * bits_quant + log2(p);
    model_info / entropy_rate
}

/// Compute the flat, non-topological predictive horizon.
pub fn flat_horizon(nodes: u64, dimension: u32, epsilon_quant: f64, entropy_rate: f64) -> f64 {
    if entropy_rate < EPS {
        return f64::INFINITY;
    }
    nodes as f64 * f64::from(dimension) * quantization_bits(epsilon_quant) / entropy_rate
}

/// Compression advantage of topological clustering over flat memory.
pub fn topological_advantage(nodes: u64, clusters: u64, dimension: u32, epsilon_quant: f64) -> f64 {
    if clusters == 0 {
        return f64::INFINITY;
    }
    let bits_quant = quantization_bits(epsilon_quant);
    let topo_info =
        clusters as f64 * f64::from(dimension) * bits_quant + log2(clusters.max(1) as f64);
    let flat_info = nodes as f64 * f64::from(dimension) * bits_quant;
    flat_info / topo_info.max(EPS)
}

/// Combined horizon with cross-manifold mutual-information bonus.
pub fn multi_model_horizon(
    individual_horizons: &[f64],
    cross_mutual_info: &[f64],
    entropy_rate: f64,
) -> f64 {
    if individual_horizons.is_empty() {
        return 0.0;
    }
    let mut base = f64::NEG_INFINITY;
    for horizon in individual_horizons {
        base = base.max(*horizon);
    }
    let cross_sum = cross_mutual_info.iter().copied().sum::<f64>();
    base + cross_sum / (individual_horizons.len() as f64 * entropy_rate.max(EPS))
}

/// Analyzer for WPHB model configurations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldModelAnalyzer {
    entropy_rate: f64,
}

impl WorldModelAnalyzer {
    /// Create an analyzer with environment entropy in bits per step.
    pub const fn new(entropy_rate: f64) -> Self {
        Self { entropy_rate }
    }

    /// Return the configured entropy rate.
    pub const fn entropy_rate(&self) -> f64 {
        self.entropy_rate
    }

    /// Analyze a single model.
    pub fn single_model_analysis(
        &self,
        dimension: u32,
        nodes: u64,
        clusters: u64,
        epsilon_quant: f64,
    ) -> SingleModelAnalysis {
        let horizon_topological =
            predictive_horizon(clusters, dimension, epsilon_quant, self.entropy_rate);
        let horizon_flat = flat_horizon(nodes, dimension, epsilon_quant, self.entropy_rate);
        let advantage = topological_advantage(nodes, clusters, dimension, epsilon_quant);

        SingleModelAnalysis {
            dimension,
            nodes,
            clusters,
            epsilon_quant,
            entropy_rate: self.entropy_rate,
            horizon_topological,
            horizon_flat,
            topological_advantage: advantage,
            horizon_tokens: horizon_topological,
        }
    }

    /// Analyze the Architect, Logic-Gate, Foreman stack.
    pub fn three_model_stack(&self) -> StackAnalysis {
        let architect = self.single_model_analysis(8192, 1_000_000, 2000, 1e-4);
        let logic_gate = self.single_model_analysis(4096, 500_000, 1000, 1e-3);
        let foreman = self.single_model_analysis(2048, 100_000, 500, 1e-2);
        let individual_horizons = [
            architect.horizon_topological,
            logic_gate.horizon_topological,
            foreman.horizon_topological,
        ];
        let cross_mutual_information = [4096.0 * 5.0, 2048.0 * 3.0, 1024.0 * 1.5];
        let combined_horizon = multi_model_horizon(
            &individual_horizons,
            &cross_mutual_information,
            self.entropy_rate,
        );
        let best = max_slice(&individual_horizons).max(EPS);

        StackAnalysis {
            individual_models: [architect, logic_gate, foreman],
            individual_horizons,
            cross_mutual_information,
            combined_horizon,
            combined_horizon_millions: combined_horizon / 1e6,
            advantage_over_best_single: combined_horizon / best,
        }
    }

    /// Analyze an H100 cluster running the three-model stack.
    pub fn h100_cluster_analysis(&self, n_gpus: u32, params_per_model: u64) -> H100ClusterAnalysis {
        let total_pflops = f64::from(n_gpus) * 1979.0e12 / 1.0e15;
        let total_hbm_gb = f64::from(n_gpus) * 80.0;
        let model_memory_tb = 3.0 * params_per_model as f64 * 2.0 / BYTES_PER_TIB;
        let stack = self.three_model_stack();
        let tokens_per_sec = 1000.0;
        let time_to_exhaust = stack.combined_horizon / tokens_per_sec;

        H100ClusterAnalysis {
            n_gpus,
            total_pflops,
            total_hbm_gb,
            model_memory_tb,
            requires_offloading: model_memory_tb * 1024.0 > total_hbm_gb,
            combined_horizon_millions: stack.combined_horizon_millions,
            time_to_exhaust_hours: time_to_exhaust / 3600.0,
            time_to_exhaust_days: time_to_exhaust / 86_400.0,
        }
    }

    /// Verify the core WPHB inequalities.
    pub fn verify_theorem(&self) -> HorizonVerification {
        let single = self.single_model_analysis(128, 100_000, 1000, 1e-4);
        let expected_ratio = 100_000.0 / 1000.0;
        let ratio_correct =
            (single.topological_advantage - expected_ratio).abs() / expected_ratio < 0.1;
        let stack = self.three_model_stack();
        let combined_exceeds_individuals =
            stack.combined_horizon >= max_slice(&stack.individual_horizons);
        let h100_analysis = self.h100_cluster_analysis(8, 1_000_000_000_000);

        HorizonVerification {
            single_model: single,
            topological_advantage_ratio: single.topological_advantage,
            expected_ratio,
            ratio_correct,
            combined_exceeds_individuals,
            h100_analysis,
            theorem_holds: ratio_correct && combined_exceeds_individuals,
        }
    }
}

/// Single-model WPHB report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleModelAnalysis {
    /// State dimension.
    pub dimension: u32,
    /// Flat memory node count.
    pub nodes: u64,
    /// Betti-0 cluster count.
    pub clusters: u64,
    /// Quantization precision.
    pub epsilon_quant: f64,
    /// Environment entropy rate in bits per step.
    pub entropy_rate: f64,
    /// Topological predictive horizon.
    pub horizon_topological: f64,
    /// Flat predictive horizon.
    pub horizon_flat: f64,
    /// Flat/topological information ratio.
    pub topological_advantage: f64,
    /// Horizon interpreted as tokens.
    pub horizon_tokens: f64,
}

/// Three-model WPHB stack report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackAnalysis {
    /// Architect, Logic-Gate, and Foreman reports.
    pub individual_models: [SingleModelAnalysis; 3],
    /// Individual topological horizons.
    pub individual_horizons: [f64; 3],
    /// Per-model cross mutual-information estimates.
    pub cross_mutual_information: [f64; 3],
    /// Combined horizon.
    pub combined_horizon: f64,
    /// Combined horizon in millions of steps.
    pub combined_horizon_millions: f64,
    /// Combined horizon divided by best single-model horizon.
    pub advantage_over_best_single: f64,
}

/// H100 cluster WPHB report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H100ClusterAnalysis {
    /// Number of H100 GPUs.
    pub n_gpus: u32,
    /// Total FP8 PFLOPS estimate.
    pub total_pflops: f64,
    /// Total HBM in GiB.
    pub total_hbm_gb: f64,
    /// Model memory in TiB.
    pub model_memory_tb: f64,
    /// True when model memory exceeds HBM capacity.
    pub requires_offloading: bool,
    /// Combined horizon in millions of steps.
    pub combined_horizon_millions: f64,
    /// Time to exhaust the horizon at 1000 tokens/s.
    pub time_to_exhaust_hours: f64,
    /// Time to exhaust the horizon at 1000 tokens/s.
    pub time_to_exhaust_days: f64,
}

/// WPHB theorem verification report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonVerification {
    /// Single-model verification fixture.
    pub single_model: SingleModelAnalysis,
    /// Measured topology advantage.
    pub topological_advantage_ratio: f64,
    /// Expected compression ratio.
    pub expected_ratio: f64,
    /// True when measured advantage is within 10 percent of expected.
    pub ratio_correct: bool,
    /// True when combined stack horizon exceeds all individual horizons.
    pub combined_exceeds_individuals: bool,
    /// H100 stack report.
    pub h100_analysis: H100ClusterAnalysis,
    /// Combined theorem status.
    pub theorem_holds: bool,
}

fn quantization_bits(epsilon_quant: f64) -> f64 {
    log2(1.0 / epsilon_quant.max(1e-30))
}

fn log2(value: f64) -> f64 {
    log(value) / LOG2
}

fn max_slice(values: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for value in values {
        max = max.max(*value);
    }
    max
}
