// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! QEMU integration test runner.
//!
//! When the `test-mode` feature is enabled, `kernel_main` calls `test_main()`
//! which runs all registered tests and then exits QEMU via the isa-debug-exit
//! device (port 0x501).

use super::harness::run_all_tests_and_exit;

/// Entry point for the in-kernel test runner.
pub fn test_main() -> ! {
    // Register all in-kernel tests from each module
    crate::memory::tests::register_all();
    crate::fs::block_store::tests::register_all();
    crate::fs::manifold_fs::tests::register_all();
    crate::process::scheduler::tests::register_all();
    crate::security::tests::register_all();
    crate::syscall::table::tests::register_all();
    crate::drivers::apic::tests::register_all();
    crate::cpu::tests::register_all();
    crate::sync::tests::register_all();

    run_all_tests_and_exit()
}
