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
            assert_eq!(
                pi.distance(&pj),
                height,
                "height is the named pair's distance"
            );
            let r = BETA0_RATIO.sqrt();
            assert!(
                eps / r <= height && height <= eps * r,
                "height {height} outside the band"
            );
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

// ─── HollowCubeManifold::assimilate ──────────────────────────────────────────

use epsilon::{HollowCubeManifold, ManifoldPayload, SurgeryError};

fn shell() -> HollowCubeManifold<3> {
    let mut m = HollowCubeManifold::<3>::new(1.5);
    m.add_shell_point(EpsilonPoint::new([1.0, 0.0, 0.0]));
    m.add_shell_point(EpsilonPoint::new([0.9, 0.1, 0.0]));
    m.add_shell_point(EpsilonPoint::new([0.9, 0.0, 0.1]));
    m
}

fn payload(points: &[[f64; 3]]) -> ManifoldPayload<3> {
    let mut src = SparseGraph::<3>::new(2.0);
    for &c in points {
        src.add_point(EpsilonPoint::new(c));
    }
    ManifoldPayload::from_graph(&src, 1.0)
}

#[test]
fn assimilate_refuses_a_merge_with_no_certified_gap() {
    let mut m = shell();
    let before = m.shell_shape();
    // The payload reaches the shell by an edge of sqrt(1.05) = 1.025, so the
    // merged graph at epsilon 1.5 is connected, with a margin of 1.46x.
    m.inject_into_void(payload(&[[0.0, 0.5, 0.5], [0.1, 0.5, 0.5]]))
        .unwrap();
    match m.assimilate() {
        Err(SurgeryError::AmbiguousAssimilation { i, j, height }) => {
            // Merged indices: shell 0..3, then payload 3..5.
            assert!(
                i < 3 && j == 4,
                "pair ({i},{j}) is not the shell-payload edge"
            );
            assert!((height - 1.05f64.sqrt()).abs() < 1e-12, "height = {height}");
        }
        other => panic!("expected AmbiguousAssimilation, got {other:?}"),
    }
    assert_eq!(
        m.shell_shape(),
        before,
        "refusal must leave the shell untouched"
    );
    assert!(m.void_is_empty());
}

#[test]
fn assimilate_commits_a_certified_merge() {
    let mut m = shell();
    // Every spanning-tree edge of the merged cloud is at most 0.141, below
    // 1.5 / sqrt(10) = 0.474.
    m.inject_into_void(payload(&[[0.8, 0.1, 0.0], [0.8, 0.2, 0.0]]))
        .unwrap();
    assert_eq!(m.assimilate(), Ok(2));
    assert_eq!(m.shell_shape().0, 1);
}
