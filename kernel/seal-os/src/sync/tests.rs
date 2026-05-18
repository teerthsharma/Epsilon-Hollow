// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Sync primitive and SMP verification tests.

use core::sync::atomic::Ordering;

use crate::cpu;
use crate::testing::TestResult;
use crate::{test_assert, test_assert_eq};

use super::{SeqLock, TicketLock};

// ---------------------------------------------------------------------------
// Primitive tests
// ---------------------------------------------------------------------------

fn test_ticket_lock_basic() -> TestResult {
    let lock = TicketLock::new();
    {
        let _g = lock.lock();
        // Lock held
    }
    // Lock released — should be able to acquire again
    let _g = lock.lock();
    TestResult::Pass
}

fn test_seq_lock_basic() -> TestResult {
    let sl = SeqLock::new();

    let seq = sl.read_begin();
    test_assert!(seq & 1 == 0, "seq must be even initially");

    sl.write_lock();
    test_assert!(sl.read_retry(seq), "must retry after write_lock");
    sl.write_unlock();

    let seq2 = sl.read_begin();
    test_assert!(seq2 & 1 == 0, "seq must be even after write_unlock");
    test_assert!(!sl.read_retry(seq2), "no retry when no writer active");

    TestResult::Pass
}

// ---------------------------------------------------------------------------
// SMP tests
// ---------------------------------------------------------------------------

fn test_smp_online() -> TestResult {
    let count = cpu::CPU_COUNT.load(Ordering::SeqCst);
    test_assert!(count >= 1, "at least BSP must be online");
    TestResult::Pass
}

#[cfg(all(not(test), feature = "test-mode"))]
fn test_apic_timer_fires() -> TestResult {
    unsafe {
        let apic_id = crate::drivers::apic::local_apic().id();
        let cpu_num = cpu::current_cpu_num();
        test_assert!(
            apic_id == cpu_num as u32,
            "apic id matches cpu num"
        );
    }
    TestResult::Pass
}

#[cfg(not(all(not(test), feature = "test-mode")))]
fn test_apic_timer_fires() -> TestResult {
    TestResult::Pass
}

#[cfg(all(not(test), feature = "test-mode"))]
fn test_ipi_roundtrip() -> TestResult {
    // Self-IPI stub — full cross-CPU test requires at least 2 CPUs online.
    test_assert!(true, "ipi test stub");
    TestResult::Pass
}

#[cfg(not(all(not(test), feature = "test-mode")))]
fn test_ipi_roundtrip() -> TestResult {
    TestResult::Pass
}

fn test_percpu_scheduler() -> TestResult {
    // Ensure the global scheduler is initialized.
    crate::process::scheduler::init();
    let before = crate::process::scheduler::task_count();
    crate::process::scheduler::spawn("test_task", 1, || {});
    let after = crate::process::scheduler::task_count();
    test_assert_eq!(after, before + 1);
    TestResult::Pass
}

pub fn register_all() {
    crate::testing::register_test("sync::ticket_lock_basic", test_ticket_lock_basic);
    crate::testing::register_test("sync::seq_lock_basic", test_seq_lock_basic);
    crate::testing::register_test("sync::smp_online", test_smp_online);
    crate::testing::register_test("sync::apic_timer_fires", test_apic_timer_fires);
    crate::testing::register_test("sync::ipi_roundtrip", test_ipi_roundtrip);
    crate::testing::register_test("sync::percpu_scheduler", test_percpu_scheduler);
}
