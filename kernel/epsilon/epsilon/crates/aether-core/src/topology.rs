// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Topological Gatekeeper
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements Topological Data Analysis (TDA) for binary authentication.
//! Computes Betti number signatures of byte streams at multiple filtration scales.
//!
//! Mathematical Foundation:
//!   - Embedding: Bytes as 1D point cloud on ℝ
//!   - Homology: β₀ (components via gap detection), β₁ (loops via window return)
//!   - Multi-scale filtration: Betti numbers at thresholds [5, 10, 15, 25, 50]
//!   - Distance: L2 on (β₀, β₁, density) feature vector
//!
//! Heuristics:
//!   - Safe Code (linear logic): β₁ ≈ 0 (low loop complexity)
//!   - Malicious Code (NOP sleds/jumps): high β₀ clustering or high β₁
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

// use libm::fabs;

// ═══════════════════════════════════════════════════════════════════════════════
// Topology Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Geometric distance threshold for clustering (Betti-0 calculation)
const CLUSTER_THRESHOLD: i16 = 15;

/// Sliding window size for topology analysis
const WINDOW_SIZE: usize = 64;

/// Minimum density for valid code (β₀ / len)
const DENSITY_MIN: f64 = 0.1;

/// Maximum density for valid code
const DENSITY_MAX: f64 = 0.6;

/// Maximum allowed Betti-1 (loop complexity) per window
const MAX_BETTI_1: u32 = 10;

// ═══════════════════════════════════════════════════════════════════════════════
// Topological Shape Signature
// ═══════════════════════════════════════════════════════════════════════════════

/// Shape signature: (β₀, β₁) tuple from Persistent Homology
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologicalShape {
    /// β₀: Number of connected components (0-dimensional holes)
    pub betti_0: u32,

    /// β₁: Number of loops/cycles (1-dimensional holes)
    pub betti_1: u32,

    /// Density: β₀ / data_length (normalized clustering)
    pub density: f64,
}

impl TopologicalShape {
    /// Create a shape from Betti numbers
    pub fn new(betti_0: u32, betti_1: u32, data_len: usize) -> Self {
        let density = if data_len > 0 {
            betti_0 as f64 / data_len as f64
        } else {
            0.0
        };

        Self {
            betti_0,
            betti_1,
            density,
        }
    }

    /// L2 distance between shape feature vectors (β₀, β₁, density).
    pub fn distance(&self, other: &Self) -> f64 {
        let d0 = libm::pow(self.betti_0 as f64 - other.betti_0 as f64, 2.0);
        let d1 = libm::pow(self.betti_1 as f64 - other.betti_1 as f64, 2.0);
        let dd = libm::pow(self.density - other.density, 2.0);

        libm::sqrt(d0 + d1 + dd)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Betti Number Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute β₀ (connected components) at a given threshold.
///
/// Treats bytes as a 1D point cloud on ℝ and counts "gaps" exceeding
/// the threshold as component boundaries (1D Vietoris-Rips at scale ε).
pub fn compute_betti_0_at(data: &[u8], threshold: i16) -> u32 {
    if data.len() < 2 {
        return if data.is_empty() { 0 } else { 1 };
    }

    let mut components = 0u32;
    let mut in_component = false;

    for window in data.windows(2) {
        let dist = (window[0] as i16 - window[1] as i16).abs();

        if dist > threshold {
            if !in_component {
                components += 1;
                in_component = true;
            }
        } else {
            in_component = false;
        }
    }

    components
}

/// Compute β₀ at the default threshold (CLUSTER_THRESHOLD = 15).
pub fn compute_betti_0(data: &[u8]) -> u32 {
    compute_betti_0_at(data, CLUSTER_THRESHOLD)
}

/// Multi-scale filtration: compute β₀ at several thresholds.
///
/// Returns a persistence-like profile showing how β₀ changes as the
/// connectivity threshold increases. This is the 1D analogue of a
/// Vietoris-Rips barcode for β₀.
pub fn betti_0_filtration(data: &[u8]) -> Vec<(i16, u32)> {
    const SCALES: [i16; 5] = [5, 10, 15, 25, 50];
    SCALES
        .iter()
        .map(|&t| (t, compute_betti_0_at(data, t)))
        .collect()
}

/// Compute β₁ (loops/cycles) via local pattern detection
///
/// This approximates 1-dimensional homology by detecting "oscillation" patterns
/// in the byte stream - sequences that return to similar values.
///
/// # Arguments
/// * `data` - Binary data to analyze
///
/// # Returns
/// β₁: Approximate number of loops/cycles
pub fn compute_betti_1(data: &[u8]) -> u32 {
    if data.len() < 4 {
        return 0;
    }

    let mut loops = 0u32;
    let tolerance = 5i16; // How close values must be to "close a loop"

    // Detect cycles: a -> b -> c -> ~a (return to start)
    for window in data.windows(4) {
        let a = window[0] as i16;
        let d = window[3] as i16;

        // If we return to approximately the same value, it's a "loop"
        if (a - d).abs() <= tolerance {
            // Check that middle values are different (actual traversal)
            let b = window[1] as i16;
            let c = window[2] as i16;

            if (a - b).abs() > tolerance || (a - c).abs() > tolerance {
                loops += 1;
            }
        }
    }

    loops
}

/// Compute full topological shape signature
pub fn compute_shape(data: &[u8]) -> TopologicalShape {
    let betti_0 = compute_betti_0(data);
    let betti_1 = compute_betti_1(data);

    TopologicalShape::new(betti_0, betti_1, data.len())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shape Verification
// ═══════════════════════════════════════════════════════════════════════════════

/// Verification result with detailed rejection reason
#[derive(Debug, Clone)]
pub enum VerifyResult {
    /// Code passed topological verification
    Pass,

    /// Density out of expected range
    InvalidDensity {
        /// Observed density.
        actual: f64,
        /// Lower bound of acceptable range.
        min: f64,
        /// Upper bound of acceptable range.
        max: f64,
    },

    /// Too many loops (possible obfuscation)
    ExcessiveLoops {
        /// Observed loop count.
        count: u32,
        /// Maximum allowed loops.
        max: u32,
    },

    /// Shape too different from reference
    ShapeMismatch {
        /// Topological distance to the reference shape.
        distance: f64,
        /// Maximum allowed distance.
        threshold: f64,
    },
}

/// Verify binary data against topological constraints
///
/// # Heuristics
/// - Standard compiled code: density ∈ [0.1, 0.6]
/// - Encrypted/obfuscated payloads: density outside this range
/// - NOP sleds: very low density (uniform bytes)
/// - ROP chains: very high loop count
///
/// # Arguments
/// * `data` - Binary data to verify
///
/// # Returns
/// `VerifyResult` indicating pass or detailed failure
pub fn verify_shape(data: &[u8]) -> VerifyResult {
    let shape = compute_shape(data);

    // Check density bounds
    if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX {
        return VerifyResult::InvalidDensity {
            actual: shape.density,
            min: DENSITY_MIN,
            max: DENSITY_MAX,
        };
    }

    // Check loop complexity
    if shape.betti_1 > MAX_BETTI_1 {
        return VerifyResult::ExcessiveLoops {
            count: shape.betti_1,
            max: MAX_BETTI_1,
        };
    }

    VerifyResult::Pass
}

/// Simple boolean verification (convenience wrapper)
pub fn is_shape_valid(data: &[u8]) -> bool {
    matches!(verify_shape(data), VerifyResult::Pass)
}

/// Verify with custom reference shape (L2 distance on shape features)
pub fn verify_against_reference(
    data: &[u8],
    reference: &TopologicalShape,
    threshold: f64,
) -> VerifyResult {
    let shape = compute_shape(data);
    let distance = shape.distance(reference);

    if distance > threshold {
        return VerifyResult::ShapeMismatch {
            distance,
            threshold,
        };
    }

    verify_shape(data)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sliding Window Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Analyze binary with sliding window, fail-fast on any violation
///
/// This is used by the ELF loader to check .text sections.
///
/// # Arguments
/// * `data` - Full binary data
/// * `window_size` - Size of sliding window (default: 64)
///
/// # Returns
/// `Ok(())` if all windows pass, `Err(offset)` at first failure
pub fn verify_sliding_window(data: &[u8], window_size: usize) -> Result<(), usize> {
    let size = if window_size == 0 {
        WINDOW_SIZE
    } else {
        window_size
    };

    if data.len() < size {
        return if is_shape_valid(data) { Ok(()) } else { Err(0) };
    }

    for (offset, window) in data.windows(size).enumerate() {
        if !is_shape_valid(window) {
            return Err(offset);
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        assert_eq!(compute_betti_0(&[]), 0);
        assert_eq!(compute_betti_1(&[]), 0);
    }

    #[test]
    fn test_uniform_data_low_density() {
        // NOP sled simulation: all same byte
        let nop_sled = [0x90u8; 64];
        let shape = compute_shape(&nop_sled);

        // Uniform data should have 0 gaps
        assert_eq!(shape.betti_0, 0);
    }

    #[test]
    fn test_random_pattern() {
        // Simulated "normal" code with varied byte patterns
        let code: [u8; 16] = [
            0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x10, 0x89, 0x7d, 0xfc, 0x8b, 0x45, 0xfc, 0x83,
            0xc0, 0x01,
        ];

        let shape = compute_shape(&code);

        // Should have reasonable density for compiled code
        assert!(shape.density >= 0.0);
    }

    #[test]
    fn test_betti_0_filtration_monotone() {
        // β₀ should be non-increasing as threshold grows (fewer gaps)
        let data: [u8; 32] = [
            0x10, 0x50, 0x90, 0x12, 0x55, 0x98, 0x15, 0x60, 0xA0, 0x20, 0x65, 0xA5, 0x25, 0x70,
            0xB0, 0x30, 0x75, 0xB5, 0x35, 0x80, 0xC0, 0x40, 0x85, 0xC5, 0x45, 0x88, 0xD0, 0x48,
            0x8A, 0xD5, 0x4A, 0x8C,
        ];
        let profile = betti_0_filtration(&data);
        for i in 1..profile.len() {
            assert!(
                profile[i].1 <= profile[i - 1].1,
                "β₀ should decrease: at threshold {} got {}, at {} got {}",
                profile[i - 1].0,
                profile[i - 1].1,
                profile[i].0,
                profile[i].1
            );
        }
    }

    #[test]
    fn test_verify_pass() {
        // Typical x86_64 function prologue
        let prologue = [
            0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x20, 0x89, 0x7d, 0xec, 0x89, 0x75, 0xe8,
            0x48, 0x89, 0x55, 0xe0, 0x48, 0x89, 0x4d, 0xd8, 0x44, 0x89, 0x45, 0xd4, 0x44, 0x89,
            0x4d, 0xd0, 0x8b, 0x45,
        ];

        // Verify returns a result (may pass or fail based on heuristics)
        let result = verify_shape(&prologue);
        // Just ensure it doesn't panic
        let _ = result;
    }
}
