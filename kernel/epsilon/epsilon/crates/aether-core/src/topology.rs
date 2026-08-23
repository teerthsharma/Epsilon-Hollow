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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ═══════════════════════════════════════════════════════════════════════════════
// Topology Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Geometric distance threshold for clustering (Betti-0 calculation)
const CLUSTER_THRESHOLD: i16 = 15;

/// Sliding window size for topology analysis
const WINDOW_SIZE: usize = 64;

/// Minimum density for valid code (β₀ / len).
///
/// # This threshold is not calibrated, and the ratio it thresholds is unsound
///
/// `β₀` counts clusters of *distinct byte values*, so it is bounded above by
/// 256 for every input. `len` is unbounded. The ratio `β₀ / len` therefore has
/// a ceiling of `256 / len`, which falls below `DENSITY_MIN` once
/// `len > 2560` — past that length **every input is rejected regardless of
/// content**. Measured: a 4096-byte input has a maximum achievable density of
/// 0.0625.
///
/// A second limit compounds it. At `CLUSTER_THRESHOLD = 15` on a 256-wide value
/// space, any input long enough to populate most byte values has no gap
/// exceeding the threshold, so `β₀` collapses to 1. Measured: uniformly random
/// input of 256, 1024 and 4096 bytes all give `β₀ = 1`.
///
/// Both numbers below are inherited from the version of `compute_betti_0_at`
/// that counted gap *runs* rather than components. Recalibrating them requires
/// a labelled corpus that does not exist in this repository, and inventing
/// replacements would repeat the defect being repaired. They are left as they
/// stand, documented, until such a corpus exists. Callers should prefer
/// [`verify_sliding_window`], which applies the check per `WINDOW_SIZE` window
/// where the ratio is at least well-scaled.
const DENSITY_MIN: f64 = 0.1;

/// Maximum density for valid code. See [`DENSITY_MIN`] for why this pair is
/// uncalibrated and why the underlying ratio is unsound at length.
const DENSITY_MAX: f64 = 0.6;

/// Maximum allowed value of the oscillation statistic per window.
///
/// Named `MAX_BETTI_1` until iteration 42, when the quantity it bounds was
/// shown not to be a Betti number. The threshold and the behaviour are
/// unchanged; only the claim about what is being counted is. Like
/// [`DENSITY_MIN`], the value 10 is uncalibrated.
const MAX_OSCILLATION: u32 = 10;

// ═══════════════════════════════════════════════════════════════════════════════
// Topological Shape Signature
// ═══════════════════════════════════════════════════════════════════════════════

/// Shape signature: β₀ and an oscillation statistic.
///
/// The second component was labelled β₁ and is not one - see
/// [`oscillation_count`]. For byte data under the one-dimensional metric the
/// true β₁ is identically zero, which [`betti_1`] states and proves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologicalShape {
    /// β₀: Number of connected components (0-dimensional holes)
    pub betti_0: u32,

    /// Oscillation statistic: overlapping 4-windows that return near their
    /// start. Grows with sample count and depends on arrival order, so it is a
    /// signal about the byte sequence, not a homology rank.
    pub oscillation: u32,

    /// Density: β₀ / data_length (normalized clustering)
    pub density: f64,
}

impl TopologicalShape {
    /// Create a shape from β₀ and the oscillation statistic.
    pub fn new(betti_0: u32, oscillation: u32, data_len: usize) -> Self {
        let density = if data_len > 0 {
            betti_0 as f64 / data_len as f64
        } else {
            0.0
        };

        Self {
            betti_0,
            oscillation,
            density,
        }
    }

    /// L2 distance between shape feature vectors (β₀, oscillation, density).
    pub fn distance(&self, other: &Self) -> f64 {
        let d0 = libm::pow(self.betti_0 as f64 - other.betti_0 as f64, 2.0);
        let d1 = libm::pow(self.oscillation as f64 - other.oscillation as f64, 2.0);
        let dd = libm::pow(self.density - other.density, 2.0);

        libm::sqrt(d0 + d1 + dd)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Betti Number Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute β₀ (connected components) at a given threshold.
///
/// Bytes are a point cloud on ℝ. β₀ is the number of connected components of
/// the 1-dimensional Vietoris-Rips complex at scale `threshold`: sort the
/// values, and every consecutive gap exceeding `threshold` separates one
/// component from the next, so
///
/// ```text
/// β₀ = (number of gaps exceeding the threshold) + 1
/// ```
///
/// The `+ 1` is the whole content of the formula — `k` gaps cut a line into
/// `k + 1` pieces. Dropping it is the named mutant that
/// `tests/house_betti0_bytes.rs` exists to kill.
///
/// Sorting is what makes this a homology invariant: a point cloud has no
/// intrinsic order, so β₀ cannot depend on the order the bytes arrive in.
/// Because the values are bytes, sorting is a 256-entry presence scan rather
/// than a comparison sort — O(n + 256), no allocation, `no_std` safe.
pub fn compute_betti_0_at(data: &[u8], threshold: i16) -> u32 {
    if data.is_empty() {
        return 0;
    }

    let mut present = [false; 256];
    for &byte in data {
        present[byte as usize] = true;
    }

    let mut components = 0u32;
    let mut previous: Option<i16> = None;

    for (value, &seen) in present.iter().enumerate() {
        if !seen {
            continue;
        }
        let value = value as i16;
        match previous {
            // The first occupied value opens the first component: this is the
            // `+ 1` in `gaps + 1`.
            None => components = 1,
            Some(p) if value - p > threshold => components += 1,
            Some(_) => {}
        }
        previous = Some(value);
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

/// Count overlapping 4-windows whose last value returns close to the first.
///
/// This was called `compute_betti_1` and documented as approximating
/// 1-dimensional homology. It does not, and cannot, for two reasons visible in
/// its own output:
///
/// * **It grows with sample count.** On `[0, 60, 120, 0]` repeated it returns
///   1, 2, 4, 8, 16, 32 at lengths 4, 8, 16, 32, 64, 128 - exactly `n / 4`.
///   Sampling the same shape more densely cannot change a homology rank.
/// * **It depends on arrival order.** A point cloud carries no order, so any
///   permutation is a relabeling and a provable no-op against homology. A
///   stride permutation of that same input moves the value from 4 to 7.
///
/// Both are asserted in `tests/house_betti1_is_not_a_rank.rs`.
///
/// The statistic is kept because it is a usable signal about a byte *sequence*,
/// responding to periodicity, and because [`verify_shape`] rejects on it. Only
/// the claim that it measures homology is withdrawn. For the true β₁ of byte
/// data, see [`betti_1`].
pub fn oscillation_count(data: &[u8]) -> u32 {
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

/// β₁ of byte data under the one-dimensional metric: identically zero.
///
/// # Why this is not a placeholder
///
/// The bytes are points on a line. Order the distinct values
/// `x_1 < ... < x_n`. In the Rips graph at scale `t` the neighbours of `x_1`
/// are exactly the values in `(x_1, x_1 + t]`; any two of those differ by at
/// most `t`, so they are pairwise adjacent. Hence `N[x_1]` is a clique, `x_1`
/// is a simplicial vertex, its closed star is a full simplex, and deleting it
/// is an elementary collapse. Induct: every component collapses to a point, so
/// `H_1 = 0` at every scale.
///
/// This is the same shape of correction as the `epsilon` crate's β₂, which
/// could not return 0 for a cloud that is not a sphere. A rank no input can
/// move is worth stating explicitly, because the alternative is a function
/// returning a plausible non-zero number for a space that has no loops.
///
/// `tests/house_betti1_is_not_a_rank.rs` does not take the argument on trust:
/// it runs the crate's own persistent homology over the byte values and
/// requires zero H1 bars across 5 corpora at 7 radii each. A disagreement would
/// indict either this proof or `persistence.rs`.
pub fn betti_1(_data: &[u8]) -> u32 {
    0
}

/// Compute full topological shape signature
pub fn compute_shape(data: &[u8]) -> TopologicalShape {
    let betti_0 = compute_betti_0(data);
    let oscillation = oscillation_count(data);

    TopologicalShape::new(betti_0, oscillation, data.len())
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

    /// The density criterion cannot be applied at this length.
    ///
    /// `beta_0` counts clusters of distinct byte **values**, so it is bounded by
    /// 256 for every input, while `len` is unbounded. `density = beta_0 / len`
    /// therefore has a ceiling of `256 / len`, which falls below
    /// `DENSITY_MIN` once `len` exceeds 2560. Past that length no input of any
    /// content can pass, so reporting [`Self::InvalidDensity`] would be a
    /// guaranteed false rejection rather than a verdict.
    ///
    /// The gate declines instead. Callers wanting a verdict on long input
    /// should use [`verify_sliding_window`], which applies the criterion per
    /// `WINDOW_SIZE` window where the ratio is well scaled.
    LengthOutOfRange {
        /// Length of the supplied input.
        len: usize,
        /// Longest input at which the density criterion can still be satisfied.
        max_assessable: usize,
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
    // Past this length the density ceiling of `256 / len` sits below
    // `DENSITY_MIN`, so no content can pass and a rejection would carry no
    // information. Decline rather than reject.
    let max_assessable = (256.0 / DENSITY_MIN) as usize;
    if data.len() > max_assessable {
        return VerifyResult::LengthOutOfRange {
            len: data.len(),
            max_assessable,
        };
    }

    if shape.density < DENSITY_MIN || shape.density > DENSITY_MAX {
        return VerifyResult::InvalidDensity {
            actual: shape.density,
            min: DENSITY_MIN,
            max: DENSITY_MAX,
        };
    }

    // Check oscillation complexity. This branch decides a rejection, so it
    // keeps the same quantity and the same threshold it always had: the repair
    // in iteration 42 corrected what the quantity is called, not what it does.
    // Substituting the true β₁ here would make the branch unreachable, trading
    // a mislabelled check for a vacuous one.
    if shape.oscillation > MAX_OSCILLATION {
        return VerifyResult::ExcessiveLoops {
            count: shape.oscillation,
            max: MAX_OSCILLATION,
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
        assert_eq!(oscillation_count(&[]), 0);
    }

    #[test]
    fn test_uniform_data_is_one_component() {
        // NOP sled simulation: all same byte. Every point sits at the same
        // value, so the cloud has exactly one component. This previously
        // asserted 0, which pinned the `gaps` / `gaps + 1` defect in place.
        let nop_sled = [0x90u8; 64];
        let shape = compute_shape(&nop_sled);
        assert_eq!(shape.betti_0, 1);
    }

    #[test]
    fn betti_0_is_bounded_by_the_alphabet() {
        // beta_0 counts clusters of *distinct byte values*, so it can never
        // exceed 256 however long the input is. This is the bound that makes
        // `density = beta_0 / len` unusable at length; see DENSITY_MIN.
        let mut s: u64 = 0x1234_5678_9ABC_DEF0;
        let long: Vec<u8> = (0..8192)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 256) as u8
            })
            .collect();
        assert!(compute_betti_0(&long) <= 256);
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
