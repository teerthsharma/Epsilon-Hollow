// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Regression: `EmbeddingBridge::build_graph` must accept token counts above 64.
//!
//! # What this locks
//!
//! `build_graph` ends in a fail-closed topological gate: it computes β₀ over the
//! projected embedding graph and returns `BridgeError::DisconnectedGraph` unless
//! β₀ == 1. That gate was reading a β₀ produced by a `[u64; MAX_POINTS]`
//! adjacency bitmask with `MAX_POINTS = 256`, so every point at index ≥ 64 was
//! invisible to the graph and counted as its own component. β₀ came back as
//! `1 + (n - 64)` for any connected cloud of `n > 64` points.
//!
//! The consequence was not a slightly wrong statistic. It was a validation gate
//! that **rejected every input above 64 tokens**, unconditionally, reporting
//! them as disconnected when they were not.
//!
//! The in-module test `test_sphere_betti_0_is_one` did not catch this because it
//! builds `MIN_TOKENS + 10` = 30 embeddings, and 30 < 64. This file exercises
//! the same gate on both sides of the boundary. It is deliberately an
//! integration test against the public API rather than a unit test, so it
//! checks what a caller actually experiences.

use epsilon::bridge::{EmbeddingBridge, MIN_TOKENS};

/// Deterministic embeddings on a smooth curve — no RNG, so a failure is
/// reproducible and is never a seed artefact.
fn embeddings<const E: usize>(n: usize) -> Vec<[f64; E]> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let mut v = [0.0; E];
            for (k, c) in v.iter_mut().enumerate() {
                *c = ((t * 6.0 + k as f64 * 0.37).sin() + 1.5) * 0.5;
            }
            v
        })
        .collect()
}

#[test]
fn build_graph_accepts_token_counts_on_both_sides_of_64() {
    let bridge = EmbeddingBridge::<32, 3>::with_seed(42);

    // 65, 100 and 200 are the sizes that the bitmask defect made unreachable.
    // 30 is the size the pre-existing in-module test used, kept as the control:
    // if it ever fails, the cause is something other than the 64 boundary.
    for n in [MIN_TOKENS + 10, 63, 64, 65, 100, 200] {
        let emb = embeddings::<32>(n);
        let graph = bridge
            .build_graph(&emb)
            .unwrap_or_else(|e| panic!("build_graph rejected {n} tokens: {e:?}"));

        assert_eq!(
            graph.compute_betti_0(),
            1,
            "projected graph of {n} tokens must be one component"
        );
    }
}

/// The component count must not depend on how many points happen to sit above
/// index 64 — the defining symptom of the bitmask defect was β₀ growing by
/// exactly one per point past the boundary.
#[test]
fn component_count_does_not_grow_with_token_count() {
    let bridge = EmbeddingBridge::<32, 3>::with_seed(7);

    let mut counts = Vec::new();
    for n in [40usize, 70, 120, 200] {
        if let Ok(g) = bridge.build_graph(&embeddings::<32>(n)) {
            counts.push((n, g.compute_betti_0()));
        }
    }

    for (n, b0) in &counts {
        assert_eq!(
            *b0, 1,
            "β₀ = {b0} at n = {n}; a connected cloud must stay one component \
             regardless of size (observed: {counts:?})"
        );
    }
}
