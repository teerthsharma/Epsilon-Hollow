// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Core Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Platform-agnostic mathematical foundation for AEGIS.
//! Works on both `no_std` (bare-metal kernel) and `std` (CLI/apps).
//!
//! Core Modules:
//!   - topology: TDA, Betti numbers, shape verification
//!   - manifold: Time-delay embedding, sparse attention graphs
//!   - aether: AETHER geometric primitives, hierarchical blocks
//!   - ml: Regression engine, convergence detection
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// Module Exports
// ═══════════════════════════════════════════════════════════════════════════════

pub mod aether;
pub mod cross_manifold_alignment;
pub mod geodesic_consolidation;
pub mod governor;
pub mod hyperbolic_capacity;
pub mod hyperbolic_geometry;
pub mod manifold;
/// Memory subsystem primitives (Chebyshev liveness, GC, etc.).
#[allow(missing_docs)] // Internal compute kernels; field semantics documented at module level.
pub mod memory;
pub mod meta_controller;
/// Internal ML primitives: linear algebra, autograd, classifiers, clustering.
#[allow(missing_docs)] // Heavy internal compute kernels; not part of stable public API surface.
#[allow(rustdoc::broken_intra_doc_links, rustdoc::invalid_html_tags)]
// Notation in field comments uses bracket pseudo-indexing.
pub mod ml;
/// OS-level integration primitives (page tables, syscalls).
#[allow(missing_docs)] // Internal OS scaffolding; not part of stable public API surface.
pub mod os;
pub mod persistent_kv_partition;
/// Spectral Contraction Mapping (SCM) runnable operator and latent predictor (Theorem 2).
pub mod scm;
/// Spectral coherence, entropy, wavelet entropy, and manifold decay helpers.
pub mod spectral_entropy;
pub mod state;
pub mod thermodynamic_plasticity;
pub mod topology;
/// Topological State Synchronization (TSS): O(1)-amortized spherical Voronoi index.
pub mod tss;
pub mod world_model_horizon;

// Re-export key types for convenience
pub use aether::{BlockMetadata, DriftDetector, HierarchicalBlockTree};
pub use manifold::{ManifoldPoint, SparseAttentionGraph, TimeDelayEmbedder, TopologicalPipeline};
pub use scm::{LatentPredictor, SpectralContractionOperator};
pub use topology::{
    compute_betti_0, compute_betti_1, compute_shape, verify_shape, TopologicalShape, VerifyResult,
};
pub use tss::SphericalVoronoiIndex;
