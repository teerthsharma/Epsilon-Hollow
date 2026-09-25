// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Regression: the wake-up rescan must not break the shell's `beta_0 = 1`
//! invariant, and must not report success when capacity drops points.
//!
//! `HollowCubeManifold::assimilate` was documented as verifying Betti
//! boundaries and merging only "if topologies align", but it merged every
//! payload point unconditionally. A connected payload placed out of reach of a
//! connected shell produced `Success { points_assimilated: 2 }` and left the
//! shell with `beta_0 = 2`, after which every later teleport was rejected as
//! `DegenerateShell`. At the 256-point cap `add_point` returned `None` for each
//! payload point and the teleport still reported `Success { points_assimilated: 0 }`.

use epsilon::{
    sys_teleport_context, EpsilonPoint, HollowCubeManifold, ManifoldPayload, SparseGraph,
    SurgeryError, SurgeryGovernor, TeleportResult, TeleportTarget,
};

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

fn teleport(m: &mut HollowCubeManifold<3>, p: ManifoldPayload<3>) -> TeleportResult {
    let mut gov = SurgeryGovernor::new();
    sys_teleport_context(m, p, &mut gov, TeleportTarget::LocalVoid)
}

#[test]
fn out_of_reach_payload_is_refused_and_shell_stays_connected() {
    let mut m = shell();
    let before = m.shell_shape();
    assert_eq!(before.0, 1);

    // 1.8 from the shell at eps 1.5: disconnected at 1.5, but within 1.2x of
    // it, so the merged beta_0 is not certified and the merge is refused as
    // ambiguous rather than disconnected.
    let near_miss = payload(&[[-1.0, 0.0, 0.0], [-0.9, 0.1, 0.0]]);
    assert_eq!(near_miss.signature_b0, 1);
    match teleport(&mut m, near_miss) {
        TeleportResult::TopologyRejected(SurgeryError::AmbiguousAssimilation { i, j, height }) => {
            // Merged indices: shell [0.9, 0.1, 0] is 1, payload [-0.9, 0.1, 0] is 4.
            assert_eq!((i, j), (1, 4));
            assert!((height - 1.8).abs() < 1e-12, "height = {height}");
        }
        other => panic!("expected AmbiguousAssimilation, got {other:?}"),
    }
    assert_eq!(
        m.shell_shape(),
        before,
        "refusal must leave the shell untouched"
    );
    assert!(m.void_is_empty());

    // Connected on its own (distance 0.1 < 2.0), and 5.0 from the shell, above
    // 1.5 * sqrt(10) = 4.74: certified disconnected.
    let far = payload(&[[-4.2, 0.1, 0.0], [-4.1, 0.1, 0.0]]);
    assert_eq!(far.signature_b0, 1);

    let r = teleport(&mut m, far);
    match r {
        TeleportResult::TopologyRejected(SurgeryError::DisconnectedAssimilation {
            merged_b0,
            shell_index,
            payload_index,
            distance,
        }) => {
            assert_eq!(merged_b0, 2);
            // Closest pair: shell [0.9, 0.1, 0] and payload [-4.1, 0.1, 0].
            assert_eq!((shell_index, payload_index), (1, 1));
            assert!((distance - 5.0).abs() < 1e-12, "distance = {distance}");
        }
        other => panic!("expected DisconnectedAssimilation, got {other:?}"),
    }

    assert_eq!(
        m.shell_shape(),
        before,
        "refusal must leave the shell untouched"
    );
    assert!(m.void_is_empty());

    // The shell is still usable: a payload within 0.141 of it now succeeds.
    let near = payload(&[[0.8, 0.1, 0.0], [0.8, 0.2, 0.0]]);
    assert_eq!(
        teleport(&mut m, near),
        TeleportResult::Success {
            points_assimilated: 2
        }
    );
    assert_eq!(m.shell_shape().0, 1);
}

#[test]
fn full_shell_refuses_instead_of_reporting_success() {
    let mut m = HollowCubeManifold::<3>::new(1.5);
    for i in 0..256 {
        assert!(m
            .add_shell_point(EpsilonPoint::new([i as f64 * 0.01, 0.0, 0.0]))
            .is_some());
    }
    let before = m.shell_shape();
    assert_eq!(before.0, 1);

    let r = teleport(&mut m, payload(&[[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]]));
    assert_eq!(
        r,
        TeleportResult::TopologyRejected(SurgeryError::ShellCapacityExceeded {
            shell_points: 256,
            payload_points: 2,
            capacity: 256,
        })
    );
    assert_eq!(
        m.shell_shape(),
        before,
        "refusal must leave the shell untouched"
    );
    assert!(m.void_is_empty());
}
