// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Test harness for different execution environments.

use alloc::vec::Vec;
use super::{TestResult, TestSuite, SuiteResult, TEST_REGISTRY};
use crate::serial_println;

/// Run every test that was registered via `seal_test!` or `register_test`.
pub fn run_all_tests() -> bool {
    let registry = TEST_REGISTRY.lock();
    let suite = TestSuite {
        name: "All Registered Tests",
        cases: registry.clone(),
    };
    drop(registry);
    run_suite(&suite)
}

/// Run only the kernel-foundation group (names prefixed with `kernel_foundation::`).
pub fn run_group_a_tests() -> bool {
    run_group("kernel_foundation")
}

/// Run only the filesystem group (names prefixed with `filesystem::`).
pub fn run_group_b_tests() -> bool {
    run_group("filesystem")
}

/// Run only the network group (names prefixed with `network::`).
pub fn run_group_c_tests() -> bool {
    run_group("network")
}

fn run_group(prefix: &str) -> bool {
    let registry = TEST_REGISTRY.lock();
    let cases: Vec<_> = registry
        .iter()
        .filter(|tc| tc.name.starts_with(prefix))
        .cloned()
        .collect();
    drop(registry);
    let suite = TestSuite {
        name: "group",
        cases,
    };
    run_suite(&suite)
}

/// Run a single suite, printing results to serial and returning `true` if all passed.
pub fn run_suite(suite: &TestSuite) -> bool {
    serial_println!("\n[TEST] === Suite: {} ===", suite.name);
    let mut passed = 0;
    let mut failed = 0;
    let mut panicked = 0;

    for case in &suite.cases {
        let result = (case.func)();
        match result {
            TestResult::Pass => {
                serial_println!("TEST_PASS: {}", case.name);
                passed += 1;
            }
            TestResult::Fail(msg) => {
                serial_println!("TEST_FAIL: {} - {}", case.name, msg);
                failed += 1;
            }
            TestResult::Panic => {
                serial_println!("TEST_FAIL: {} - PANIC", case.name);
                panicked += 1;
            }
        }
    }

    let total = suite.cases.len();
    serial_println!(
        "[TEST] Suite {}: {}/{} passed, {} failed, {} panicked",
        suite.name, passed, total, failed, panicked
    );

    failed == 0 && panicked == 0
}

/// Run all registered tests, print summary, exit QEMU, and halt.
pub fn run_all_tests_and_exit() -> ! {
    let success = run_all_tests();
    if success {
        serial_println!("[TEST] ALL TESTS PASSED");
    } else {
        serial_println!("[TEST] TEST FAILED");
    }

    // Attempt QEMU isa-debug-exit
    #[cfg(not(test))]
    exit_qemu(success);

    loop {
        x86_64::instructions::hlt();
    }
}

/// Write to the QEMU isa-debug-exit device at port 0x501.
#[cfg(not(test))]
fn exit_qemu(success: bool) {
    use x86_64::instructions::port::Port;
    let code: u32 = if success { 0 } else { 1 };
    unsafe {
        let mut port = Port::<u32>::new(0x501);
        port.write(code);
    }
}
