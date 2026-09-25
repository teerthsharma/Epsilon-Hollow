// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Regression: a payload's Betti signature must describe the points it carries.
//!
//! `ManifoldPayload::from_graph` carries at most 64 points but took its
//! signature from the whole source graph (up to 256 points). A bridge point at
//! index 64 connected two clusters in the graph, so the payload claimed
//! `signature_b0 = 1` while the 64 points it actually carried had `beta_0 = 2`,
//! and `inject_into_void` accepted a disconnected payload.

use epsilon::{EpsilonPoint, HollowCubeManifold, ManifoldPayload, SparseGraph, SurgeryError};

#[test]
fn signature_is_computed_over_carried_points_only() {
    let mut g = SparseGraph::<3>::new(1.0);
    for i in 0..32 {
        g.add_point(EpsilonPoint::new([i as f64 * 0.01, 0.0, 0.0]));
    }
    for i in 0..32 {
        g.add_point(EpsilonPoint::new([4.0 + i as f64 * 0.01, 0.0, 0.0]));
    }
    // Indices 64..68: the only link between the clusters, and not carried.
    // The carried clusters sit 3.69 apart, above the certificate band
    // [1/sqrt(10), sqrt(10)] at epsilon 1.0, so their beta_0 = 2 is certified.
    for x in [0.8, 1.6, 2.4, 3.2] {
        g.add_point(EpsilonPoint::new([x, 0.0, 0.0]));
    }
    assert_eq!(g.compute_betti_0(), 1, "the full graph is connected");

    let payload = ManifoldPayload::from_graph(&g, 1.0);
    assert_eq!(payload.point_count, 64);

    let mut carried = SparseGraph::<3>::new(1.0);
    for p in &payload.points[..payload.point_count] {
        carried.add_point(*p);
    }
    assert_eq!(
        (
            payload.signature_b0,
            payload.signature_b1,
            payload.signature_b2
        ),
        carried.full_shape(),
        "signature must be the shape of the carried points"
    );
    assert_eq!(payload.signature_b0, 2);

    let mut m = HollowCubeManifold::<3>::new(1.0);
    m.add_shell_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
    assert_eq!(
        m.inject_into_void(payload),
        Err(SurgeryError::TopologyMismatch {
            expected_b0: 1,
            actual_b0: 2
        })
    );
}
