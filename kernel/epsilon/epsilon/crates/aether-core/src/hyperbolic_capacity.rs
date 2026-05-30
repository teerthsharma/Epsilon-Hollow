// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Hyperbolic Capacity Separation (HCS) formulas in `no_std` Rust.
//!
//! This module replaces the legacy `epsilon_core/hyperbolic_capacity.py`
//! theorem slice with deterministic, allocation-free Rust. It keeps the
//! original analytical contract: tree hierarchies embedded in a Poincare ball
//! have bounded distortion while Euclidean embeddings pay a depth-amplified
//! capacity penalty.

use libm::{log, pow};

/// Default embedding dimension used for accelerator-scale analysis.
pub const H100_DEFAULT_DIM: u32 = 4096;
/// Default positive curvature magnitude for the H100-scale Poincare ball.
pub const H100_DEFAULT_CURVATURE: f64 = 0.1;
/// Default branching factor for large hierarchical memory analysis.
pub const H100_DEFAULT_BRANCHING: u32 = 64;
/// Default hierarchy depth for large hierarchical memory analysis.
pub const H100_DEFAULT_DEPTH: u32 = 50;
/// H100 HBM3 memory budget used by the legacy theorem slice.
pub const H100_MEMORY_GB: f64 = 80.0;
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Upper bound on Poincare-ball embedding distortion.
///
/// The legacy theorem states:
///
/// ```text
/// delta_hyp <= 2 / (c * d * D)
/// ```
///
/// Invalid curvature, dimension, or depth returns infinity, matching the
/// legacy Python guard behavior for an impossible bound.
pub fn hyperbolic_distortion_bound(curvature: f64, dim: u32, depth: u32) -> f64 {
    if !curvature.is_finite() || curvature <= 0.0 || dim == 0 || depth == 0 {
        return f64::INFINITY;
    }

    2.0 / (curvature * f64::from(dim) * f64::from(depth))
}

/// Lower bound on Euclidean embedding distortion for a `b`-ary tree.
///
/// The legacy theorem states:
///
/// ```text
/// delta_euc >= ln(b) / d * (D - 1)
/// ```
///
/// A non-branching tree or depth 0/1 has no meaningful separation pressure and
/// therefore returns 0.
pub fn euclidean_distortion_bound(branching_factor: u32, dim: u32, depth: u32) -> f64 {
    if branching_factor <= 1 || dim == 0 || depth <= 1 {
        return 0.0;
    }

    log(f64::from(branching_factor)) / f64::from(dim) * f64::from(depth - 1)
}

/// Lower bound on the hyperbolic-vs-Euclidean separation ratio.
///
/// The legacy theorem states:
///
/// ```text
/// delta_euc / delta_hyp >= c * ln(b) * D * (D - 1) / 2
/// ```
pub fn separation_ratio(curvature: f64, branching_factor: u32, depth: u32) -> f64 {
    if depth <= 1 {
        return 1.0;
    }
    if !curvature.is_finite() || curvature <= 0.0 || branching_factor <= 1 {
        return 0.0;
    }

    curvature * log(f64::from(branching_factor)) * f64::from(depth) * f64::from(depth - 1) / 2.0
}

/// Distortion and capacity report for a hierarchical embedding shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcsAnalysis {
    /// Embedding dimension.
    pub dim: u32,
    /// Positive curvature magnitude `c`; the manifold curvature is `-c`.
    pub curvature: f64,
    /// Tree branching factor.
    pub branching_factor: u32,
    /// Tree depth.
    pub depth: u32,
    /// Analytical hyperbolic distortion upper bound.
    pub hyperbolic_distortion: f64,
    /// Analytical Euclidean distortion lower bound.
    pub euclidean_distortion: f64,
    /// Analytical separation ratio lower bound.
    pub separation_ratio: f64,
    /// Tree node count capped at depth 20 to keep the integer witness bounded.
    pub total_tree_nodes: u128,
    /// Memory required for one float64 embedding per capped tree node.
    pub memory_gb: f64,
    /// Whether the capped tree embedding fits within an 80 GB H100 memory budget.
    pub fits_h100_80gb: bool,
}

/// Run the default H100-scale HCS analysis.
pub fn h100_analysis() -> HcsAnalysis {
    hcs_analysis(
        H100_DEFAULT_DIM,
        H100_DEFAULT_CURVATURE,
        H100_DEFAULT_BRANCHING,
        H100_DEFAULT_DEPTH,
    )
}

/// Run an HCS analysis for a caller-provided shape.
pub fn hcs_analysis(dim: u32, curvature: f64, branching_factor: u32, depth: u32) -> HcsAnalysis {
    let capped_depth = depth.min(20);
    let total_tree_nodes = checked_pow_u128(u128::from(branching_factor), capped_depth);
    let bytes_per_embedding = f64::from(dim) * 8.0;
    let node_count_float = pow(f64::from(branching_factor), f64::from(capped_depth));
    let memory_gb = node_count_float * bytes_per_embedding / BYTES_PER_GIB;

    HcsAnalysis {
        dim,
        curvature,
        branching_factor,
        depth,
        hyperbolic_distortion: hyperbolic_distortion_bound(curvature, dim, depth),
        euclidean_distortion: euclidean_distortion_bound(branching_factor, dim, depth),
        separation_ratio: separation_ratio(curvature, branching_factor, depth),
        total_tree_nodes,
        memory_gb,
        fits_h100_80gb: memory_gb < H100_MEMORY_GB,
    }
}

/// Deterministic HCS verifier with no random sampling or heap allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcsVerifier {
    dim: u32,
    curvature: f64,
}

impl HcsVerifier {
    /// Create a verifier for `dim` dimensions and positive curvature magnitude `curvature`.
    pub const fn new(dim: u32, curvature: f64) -> Self {
        Self { dim, curvature }
    }

    /// Return the configured embedding dimension.
    pub const fn dim(&self) -> u32 {
        self.dim
    }

    /// Return the configured positive curvature magnitude.
    pub const fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Verify the theorem using analytical bounds and bounded tree accounting.
    pub fn verify_theorem(&self, branching_factor: u32, depth: u32) -> HcsVerification {
        let effective_depth = depth.min(6);
        let tree_nodes = full_tree_node_count(branching_factor, effective_depth);
        let hyperbolic_bound = hyperbolic_distortion_bound(self.curvature, self.dim, depth);
        let euclidean_bound = euclidean_distortion_bound(branching_factor, self.dim, depth);
        let ratio_theory = separation_ratio(self.curvature, branching_factor, depth);
        let hyperbolic_better = hyperbolic_bound.is_finite()
            && euclidean_bound.is_finite()
            && hyperbolic_bound < euclidean_bound;

        HcsVerification {
            tree_nodes,
            branching_factor,
            depth,
            effective_depth,
            hyperbolic_distortion_bound: hyperbolic_bound,
            euclidean_distortion_bound: euclidean_bound,
            separation_ratio_theory: ratio_theory,
            hyperbolic_better,
            h100_analysis: h100_analysis(),
        }
    }
}

/// Result of a deterministic HCS theorem verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcsVerification {
    /// Number of nodes in the bounded verification tree.
    pub tree_nodes: u128,
    /// Requested branching factor.
    pub branching_factor: u32,
    /// Requested depth.
    pub depth: u32,
    /// Depth actually used for bounded tree accounting.
    pub effective_depth: u32,
    /// Analytical hyperbolic distortion upper bound.
    pub hyperbolic_distortion_bound: f64,
    /// Analytical Euclidean distortion lower bound.
    pub euclidean_distortion_bound: f64,
    /// Analytical separation ratio lower bound.
    pub separation_ratio_theory: f64,
    /// True when the analytical hyperbolic bound is lower than the Euclidean bound.
    pub hyperbolic_better: bool,
    /// Default accelerator-scale analysis bundled with the verification.
    pub h100_analysis: HcsAnalysis,
}

fn full_tree_node_count(branching_factor: u32, depth: u32) -> u128 {
    if branching_factor == 0 {
        return 1;
    }
    if branching_factor == 1 {
        return u128::from(depth) + 1;
    }

    let mut total = 0_u128;
    let mut level_nodes = 1_u128;
    for _ in 0..=depth {
        total = total.saturating_add(level_nodes);
        level_nodes = level_nodes.saturating_mul(u128::from(branching_factor));
    }
    total
}

fn checked_pow_u128(base: u128, exp: u32) -> u128 {
    let mut acc = 1_u128;
    for _ in 0..exp {
        acc = acc.saturating_mul(base);
    }
    acc
}
