//! aether_verified
//!
//! Provenance: Lean 4 → C → Rust
//! Source: HeytingLean.Bridge.Sharma.*
//!
//! Every theorem in this crate has a corresponding Lean 4 proof.
//! No axiom is assumed without proof. No gap exists in the verification chain.
//!
//! Modules (Original 4 — Round 1):
//!   - `aether_pruning`:   Cauchy-Schwarz block pruning bounds
//!   - `aether_governor`:  PD Governor Lyapunov stability
//!   - `aether_chebyshev`: Chebyshev GC guard safety
//!   - `aether_betti`:     Betti approximation error bounds
//!
//! Modules (10 Novel Theorems — Round 2):
//!   - `aether_tss`:       T1: Topological State Synchronization (O(1) retrieval)
//!   - `aether_scm`:       T2: Spectral Contraction Mapping (Banach convergence)
//!   - `aether_gmc`:       T3: Geodesic Memory Consolidation (entropy reduction)
//!   - `aether_agcr`:      T4: Adaptive Governor Convergence Rate (quantitative)
//!   - `aether_hcs`:       T5: Hyperbolic Capacity Separation (distortion proof)
//!   - `aether_world`:     T6-T10: RGCS, PHKP, TEB, CMA, WPHB (H100 world model)
#![no_std]

#[cfg(test)]
extern crate std;

// ── Round 1: Foundation Kernels ──────────────────────────────────
pub mod aether_pruning;
pub mod aether_governor;
pub mod aether_chebyshev;
pub mod aether_betti;

// ── Round 2: Novel Theorem Kernels ───────────────────────────────
pub mod aether_tss;
pub mod aether_scm;
pub mod aether_gmc;
pub mod aether_agcr;
pub mod aether_hcs;
pub mod aether_world;

pub use aether_pruning::*;
pub use aether_governor::*;
pub use aether_chebyshev::*;
pub use aether_betti::*;
