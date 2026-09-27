// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-kernel checks of the boot theorem verdicts (`crate::theorems`): each
//! verdict follows the live state it is computed from, is read from the
//! running instances, and a theorem with no running instance is never
//! certified.

use aether_verified::aether_agcr::gain_margin_stable;

use crate::testing::TestResult;
use crate::theorems::{evaluate, LiveState, Status};
use crate::{GOVERNOR_ALPHA, GOVERNOR_BETA, GOVERNOR_DT};

const T1: usize = 0;
const T2: usize = 1;
const T4: usize = 3;

fn test_t4_gate_matches_runtime_governor_dt() -> TestResult {
    let verdicts = evaluate(&LiveState::read());
    test_assert!(
        (verdicts[T4].status == Status::Certified)
            == gain_margin_stable(GOVERNOR_ALPHA, GOVERNOR_BETA, GOVERNOR_DT),
        "T4 verdict disagrees with alpha+beta/dt < 1 at the runtime GOVERNOR_DT"
    );
    test_assert!(
        verdicts[T1].status == Status::Certified && verdicts[T2].status == Status::Certified,
        "T1 or T2 is not certified at the running kernel's parameters"
    );
    TestResult::Pass
}

/// T3, T5 and T6-T10 have no running instance carrying their parameters, so
/// there is no live state to certify them against. A line computed only from
/// literals must not read as certified.
fn test_theorems_without_live_state_are_not_certified() -> TestResult {
    let verdicts = evaluate(&LiveState::read());
    test_assert!(
        [2usize, 4, 5, 6, 7, 8, 9]
            .iter()
            .all(|&idx| verdicts[idx].status == Status::NotChecked),
        "T3, T5 or T6-T10 reported certified with no kernel subsystem running them"
    );
    TestResult::Pass
}

/// T1 refuses a table whose cells coincide, and a live epsilon whose
/// `theta_min = 2 asin(eps/2)` exceeds the tables' separation.
fn test_t1_follows_the_table_and_the_live_epsilon() -> TestResult {
    let mut state = LiveState::read();
    state.tables[1].centroids[1] = state.tables[1].centroids[0];
    test_assert!(
        evaluate(&state)[T1].status == Status::NotCertified,
        "a table with two coincident cells was certified"
    );
    let mut state = LiveState::read();
    state.epsilon = 1.9;
    test_assert!(
        evaluate(&state)[T1].status == Status::NotCertified,
        "eps=1.9 (theta_min 2.50 rad) was certified"
    );
    TestResult::Pass
}

/// T2 refuses gains outside (0, 1]: alpha = 0 is the identity (Lipschitz 1),
/// and alpha = 2.5 stretches distances by 1.5 while `1 - alpha` claims -1.5.
fn test_t2_refuses_non_contracting_gains() -> TestResult {
    for alpha in [0.0, 2.5] {
        let mut state = LiveState::read();
        state.operators[0].1 = alpha;
        test_assert!(
            evaluate(&state)[T2].status == Status::NotCertified,
            "a non-contracting gain was certified"
        );
    }
    TestResult::Pass
}

/// T2 reads the running firewall predictor, not a copy of its gain.
fn test_t2_reads_the_running_operator() -> TestResult {
    let saved = crate::net::firewall::scm_alpha();
    crate::net::firewall::set_scm_alpha(0.0);
    let refused = evaluate(&LiveState::read())[T2].status == Status::NotCertified;
    crate::net::firewall::set_scm_alpha(saved);
    test_assert!(
        refused,
        "T2 stayed certified after the running firewall operator's alpha became 0"
    );
    TestResult::Pass
}

pub fn register_all() {
    crate::testing::register_test(
        "kernel_foundation::theorems_without_live_state_are_not_certified",
        test_theorems_without_live_state_are_not_certified,
    );
    crate::testing::register_test(
        "kernel_foundation::t4_gate_matches_runtime_governor_dt",
        test_t4_gate_matches_runtime_governor_dt,
    );
    crate::testing::register_test(
        "kernel_foundation::t1_follows_the_table_and_the_live_epsilon",
        test_t1_follows_the_table_and_the_live_epsilon,
    );
    crate::testing::register_test(
        "kernel_foundation::t2_refuses_non_contracting_gains",
        test_t2_refuses_non_contracting_gains,
    );
    crate::testing::register_test(
        "kernel_foundation::t2_reads_the_running_operator",
        test_t2_reads_the_running_operator,
    );
}
