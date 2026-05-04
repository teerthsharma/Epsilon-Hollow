// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! aether_pruning.rs
//!
//! Provenance: Lean 4 → C → Rust
//! Source: HeytingLean.Bridge.Sharma.AetherPruning
//!
//! # Cauchy-Schwarz Block Pruning
//!
//! Core theorem: `⟨q, k⟩ ≤ ‖q‖ · ‖k‖ ≤ ‖q‖ · (‖μ_B‖ + r_B)`
//!
//! This gives a sound upper bound on the dot product between a query
//! and any key within a block, using only the block centroid and radius.
//! Zero false negatives: every relevant token is preserved.

/// Compute L2 norm of a vector.
#[inline]
fn l2_norm(x: &[f64]) -> f64 {
    libm::sqrt(x.iter().map(|v| v * v).sum::<f64>())
}

/// (Lean: upperBoundScore)
///
/// Upper bound on `⟨q, k⟩` for any k in a block with given centroid and radius.
///
/// # Mathematical Guarantee
/// `⟨q, k⟩ ≤ ‖q‖ · (‖centroid‖ + radius)`
///
/// By Cauchy-Schwarz: `⟨q, k⟩ ≤ ‖q‖ · ‖k‖`
/// By triangle inequality: `‖k‖ ≤ ‖centroid‖ + ‖k - centroid‖ ≤ ‖centroid‖ + radius`
pub fn upper_bound_score(q: &[f64], centroid: &[f64], radius: f64) -> f64 {
    let qn = l2_norm(q);
    let cn = l2_norm(centroid);
    qn * (cn + radius)
}

/// (Lean: can_prune_sound)
///
/// Returns `true` if the block can be safely pruned.
/// Sound: if this returns `true`, then `⟨q, k⟩ < threshold` for ALL k in the block.
pub fn can_prune(q: &[f64], centroid: &[f64], radius: f64, threshold: f64) -> bool {
    upper_bound_score(q, centroid, radius) < threshold
}

/// (Lean: prune_safe)
///
/// Double-check: verify that a specific key k also satisfies the pruning condition.
/// This is the "belt and suspenders" check used during debug/validation.
pub fn prune_safe_check(
    q: &[f64], k: &[f64], centroid: &[f64], radius: f64, threshold: f64,
) -> bool {
    if !can_prune(q, centroid, radius, threshold) {
        return false;
    }
    let dot: f64 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
    dot < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upper_bound_score() {
        let q = [1.0, 0.0, 0.0];
        let centroid = [0.5, 0.0, 0.0];
        let radius = 0.1;
        let bound = upper_bound_score(&q, &centroid, radius);
        // ‖q‖ = 1, ‖c‖ = 0.5, bound = 1 * (0.5 + 0.1) = 0.6
        assert!((bound - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_can_prune_true() {
        let q = [1.0, 0.0, 0.0];
        let centroid = [0.1, 0.0, 0.0];
        assert!(can_prune(&q, &centroid, 0.1, 0.5));
    }

    #[test]
    fn test_can_prune_false() {
        let q = [1.0, 0.0, 0.0];
        let centroid = [0.9, 0.0, 0.0];
        assert!(!can_prune(&q, &centroid, 0.2, 0.5));
    }

    #[test]
    fn test_prune_safe_check() {
        let q = [1.0, 0.0, 0.0];
        let centroid = [0.1, 0.0, 0.0];
        let k = [0.15, 0.0, 0.0];
        assert!(can_prune(&q, &centroid, 0.1, 0.5));
        assert!(prune_safe_check(&q, &k, &centroid, 0.1, 0.5));
    }

    #[test]
    fn test_soundness_no_false_negatives() {
        // If can_prune says safe, then the actual dot product must be below threshold
        let q = [0.7, 0.3, 0.1];
        let centroid = [0.05, 0.02, 0.01];
        let radius = 0.08;
        let threshold = 0.2;
        if can_prune(&q, &centroid, radius, threshold) {
            // Generate any k within radius of centroid
            let k = [centroid[0] + 0.05, centroid[1] + 0.03, centroid[2] + 0.01];
            let dot: f64 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            assert!(dot < threshold, "Soundness violation: dot={} >= threshold={}", dot, threshold);
        }
    }
}
