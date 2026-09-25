// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Contract: `sys_teleport_context` runs no governor tick.
//!
//! The permit brackets the surgery (prepare_for_surgery zeroes beta and
//! last_error, complete_surgery restores them), but no `adapt` runs inside
//! the window, so the governor's (epsilon, last_error, beta, tick_count) is
//! bit-identical before and after a successful teleport. An earlier claim that
//! the clutch zeroes beta "for exactly one tick" had no tick behind it; this
//! test pins what the code actually does.

use epsilon::{
    sys_teleport_context, EpsilonPoint, HollowCubeManifold, ManifoldPayload, SparseGraph,
    SurgeryGovernor, TeleportResult, TeleportTarget,
};

fn snapshot(g: &SurgeryGovernor) -> (u64, u64, u64, u64) {
    (
        g.epsilon().to_bits(),
        g.last_error().to_bits(),
        g.beta().to_bits(),
        g.tick_count(),
    )
}

#[test]
fn successful_teleport_leaves_governor_bit_identical() {
    let mut gov = SurgeryGovernor::with_gains(1e-7, 1e-7);
    gov.adapt(0.05, 1.0);
    gov.adapt(0.07, 1.0);
    let before = snapshot(&gov);
    assert_ne!(
        before.1, 0,
        "non-zero error history, so a lost restore shows"
    );

    let mut m = HollowCubeManifold::<3>::new(1.5);
    m.add_shell_point(EpsilonPoint::new([1.0, 0.0, 0.0]));
    m.add_shell_point(EpsilonPoint::new([0.9, 0.1, 0.0]));
    let mut src = SparseGraph::<3>::new(2.0);
    // Within 0.1 of the shell, so the merged beta_0 = 1 is certified at
    // epsilon 1.5; a payload ~1.02 away sits inside the certificate band.
    src.add_point(EpsilonPoint::new([0.8, 0.1, 0.0]));
    src.add_point(EpsilonPoint::new([0.8, 0.2, 0.0]));

    let r = sys_teleport_context(
        &mut m,
        ManifoldPayload::from_graph(&src, 1.0),
        &mut gov,
        TeleportTarget::LocalVoid,
    );
    assert_eq!(
        r,
        TeleportResult::Success {
            points_assimilated: 2
        }
    );
    assert_eq!(snapshot(&gov), before);
}
