// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-kernel check of the boot theorem gate: T4/AGCR is certified exactly
//! when `alpha + beta/dt < 1` holds at the step the runtime drives the
//! governor with (`crate::GOVERNOR_DT`), and the other nine gates hold.

use aether_verified::aether_agcr::gain_margin_stable;

use crate::testing::TestResult;
use crate::{GOVERNOR_ALPHA, GOVERNOR_BETA, GOVERNOR_DT};

const T4: usize = 3;

fn test_t4_gate_matches_runtime_governor_dt() -> TestResult {
    let gates = crate::verify_topology_theorems();
    test_assert!(
        gates[T4] == gain_margin_stable(GOVERNOR_ALPHA, GOVERNOR_BETA, GOVERNOR_DT),
        "T4 gate disagrees with alpha+beta/dt < 1 at the runtime GOVERNOR_DT"
    );
    test_assert!(
        gates.iter().enumerate().all(|(idx, ok)| idx == T4 || *ok),
        "a theorem gate other than T4 failed"
    );
    TestResult::Pass
}

pub fn register_all() {
    crate::testing::register_test(
        "kernel_foundation::t4_gate_matches_runtime_governor_dt",
        test_t4_gate_matches_runtime_governor_dt,
    );
}
