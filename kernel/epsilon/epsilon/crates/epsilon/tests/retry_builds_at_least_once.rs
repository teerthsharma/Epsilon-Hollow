// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Regression: `build_graph_with_retry` must build at least once, and any
//! error it returns must carry a measured beta_0.
//!
//! The loop checked `epsilon > MAX_RETRY_EPSILON` before its first build and
//! fell through to an error seeded as `DisconnectedGraph { beta0: 0 }`. A
//! bridge constructed with epsilon above the cap therefore rejected a cloud
//! that `build_graph` accepts, and reported an impossible beta_0 of 0 for a
//! non-empty cloud.

use epsilon::{BridgeError, EmbeddingBridge, MIN_TOKENS};

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
fn epsilon_above_retry_cap_still_builds() {
    let bridge = EmbeddingBridge::<32, 3>::new(42, 3.0);
    let e = embeddings::<32>(MIN_TOKENS + 10);
    let direct = bridge
        .build_graph(&e)
        .expect("build_graph accepts this cloud");
    let retried = bridge
        .build_graph_with_retry(&e, 5)
        .expect("retry must build at least once");
    assert_eq!(retried.point_count, direct.point_count);
    assert_eq!(retried.compute_betti_0(), 1);
}

#[test]
fn exhausted_retry_reports_the_measured_beta0() {
    let bridge = EmbeddingBridge::<32, 3>::new(42, 1e-6);
    let e = embeddings::<32>(MIN_TOKENS + 10);
    let direct = bridge.build_graph(&e).unwrap_err();
    assert!(matches!(direct, BridgeError::DisconnectedGraph { beta0 } if beta0 > 1));
    assert_eq!(bridge.build_graph_with_retry(&e, 0).unwrap_err(), direct);
}
