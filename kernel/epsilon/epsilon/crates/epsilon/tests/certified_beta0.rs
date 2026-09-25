// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Every beta_0 this crate decides is certified or refused.
//!
//! `SparseGraph::compute_betti_0` is union-find at one strict threshold, so a
//! pair of points one ulp either side of epsilon gives 1 or 2 with nothing in
//! the input to choose between them. The certified count answers only when no
//! minimum-spanning-tree edge lies within a factor `sqrt(BETA0_RATIO)` of
//! epsilon, and otherwise names the ambiguous pair.

use epsilon::manifold::Beta0;
use epsilon::{EpsilonPoint, SparseGraph};

fn pair(x: f64) -> SparseGraph<3> {
    let mut g = SparseGraph::new(1.0);
    g.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
    g.add_point(EpsilonPoint::new([x, 0.0, 0.0]));
    g
}

#[test]
fn graph_beta0_one_ulp_from_epsilon_is_refused() {
    let below = pair(0.999_999_999_999_999_9);
    let at = pair(1.0);
    // The uncertified count flips across one ulp.
    assert_eq!(below.compute_betti_0(), 1);
    assert_eq!(at.compute_betti_0(), 2);

    assert_eq!(
        below.certified_betti_0(),
        Beta0::Refused {
            i: 0,
            j: 1,
            height: 0.999_999_999_999_999_9
        }
    );
    assert_eq!(
        at.certified_betti_0(),
        Beta0::Refused {
            i: 0,
            j: 1,
            height: 1.0
        }
    );
}

#[test]
fn graph_beta0_far_from_epsilon_is_certified() {
    assert!(matches!(
        pair(0.01).certified_betti_0(),
        Beta0::Certified { value: 1, .. }
    ));
    assert!(matches!(
        pair(10.0).certified_betti_0(),
        Beta0::Certified { value: 2, .. }
    ));
}

// ─── EmbeddingBridge ─────────────────────────────────────────────────────────

use epsilon::manifold::BETA0_RATIO;
use epsilon::{BridgeError, EmbeddingBridge};

/// Deterministic embeddings on a smooth curve. Projected with seed 42, 30 of
/// them have a longest spanning-tree edge of about 0.089.
fn curve<const E: usize>(n: usize) -> Vec<[f64; E]> {
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

fn assert_ambiguous_pair(
    bridge: &EmbeddingBridge<32, 3>,
    e: &[[f64; 32]],
    eps: f64,
    err: BridgeError,
) {
    match err {
        BridgeError::AmbiguousBeta0 { i, j, height } => {
            assert!(i < j && j < e.len(), "pair ({i},{j})");
            let (pi, pj) = (
                bridge.project_single(&e[i]).unwrap(),
                bridge.project_single(&e[j]).unwrap(),
            );
            assert_eq!(pi.distance(&pj), height, "height is the named pair's distance");
            let r = BETA0_RATIO.sqrt();
            assert!(eps / r <= height && height <= eps * r, "height {height} outside the band");
        }
        other => panic!("expected AmbiguousBeta0, got {other:?}"),
    }
}

#[test]
fn build_graph_refuses_a_beta0_with_no_certified_gap() {
    // Every spanning-tree edge is below 0.1, so the graph at epsilon 0.1 is
    // connected, but the longest sits within a factor 1.13 of epsilon.
    let bridge = EmbeddingBridge::<32, 3>::new(42, 0.1);
    let e = curve::<32>(30);
    let err = bridge.build_graph(&e).unwrap_err();
    assert_ambiguous_pair(&bridge, &e, 0.1, err);
}

#[test]
fn retry_returns_a_graph_whose_beta0_is_certified() {
    // At 0.05 the graph is disconnected. Widening by 1.2 first connects it at
    // 0.1037, within 1.17x of the longest edge: connected, but not certified.
    let bridge = EmbeddingBridge::<32, 3>::new(42, 0.05);
    let e = curve::<32>(30);
    let graph = bridge
        .build_graph_with_retry(&e, 30)
        .expect("a certified epsilon exists below the cap");
    assert!(
        matches!(graph.certified_betti_0(), Beta0::Certified { value: 1, .. }),
        "got {:?}",
        graph.certified_betti_0()
    );
}

#[test]
fn exhausted_retry_names_the_ambiguous_pair() {
    let bridge = EmbeddingBridge::<32, 3>::new(42, 0.1);
    let e = curve::<32>(30);
    let err = bridge.build_graph_with_retry(&e, 0).unwrap_err();
    assert_ambiguous_pair(&bridge, &e, 0.1, err);
}
