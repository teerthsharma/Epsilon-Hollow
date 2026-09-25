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

    // Connected on its own (distance 0.141 < 2.0), but 1.8 from the shell at eps 1.5.
    let far = payload(&[[-1.0, 0.0, 0.0], [-0.9, 0.1, 0.0]]);
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
            // Closest pair: shell [0.9, 0.1, 0] and payload [-0.9, 0.1, 0].
            assert_eq!((shell_index, payload_index), (1, 1));
            assert!((distance - 1.8).abs() < 1e-12, "distance = {distance}");
        }
        other => panic!("expected DisconnectedAssimilation, got {other:?}"),
    }

    assert_eq!(m.shell_shape(), before, "refusal must leave the shell untouched");
    assert!(m.void_is_empty());

    // The shell is still usable: a reachable payload now succeeds.
    let near = payload(&[[0.0, 0.5, 0.5], [0.1, 0.5, 0.5]]);
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
    assert_eq!(m.shell_shape(), before, "refusal must leave the shell untouched");
    assert!(m.void_is_empty());
}
