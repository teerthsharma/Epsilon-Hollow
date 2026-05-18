// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-kernel testing framework for Seal OS.
//!
//! Works in `no_std` environments. Tests register themselves into a global
//! registry and are executed by the harness.  Macros return `TestResult::Fail`
//! instead of panicking so that a suite can continue after a failure.

use alloc::vec::Vec;
use spin::Mutex;

/// Result of a single test function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestResult {
    Pass,
    Fail(&'static str),
    Panic,
}

/// A single test case.
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: &'static str,
    pub func: fn() -> TestResult,
}

/// A named collection of test cases.
#[derive(Debug)]
pub struct TestSuite {
    pub name: &'static str,
    pub cases: Vec<TestCase>,
}

/// Aggregated result after running a suite.
#[derive(Debug)]
pub struct SuiteResult {
    pub suite_name: &'static str,
    pub passed: usize,
    pub failed: usize,
    pub panicked: usize,
    pub details: Vec<(&'static str, TestResult)>,
}

/// Global registry of all tests compiled into the kernel.
static TEST_REGISTRY: Mutex<Vec<TestCase>> = Mutex::new(Vec::new());

/// Register a test function so that the harness can discover it.
pub fn register_test(name: &'static str, func: fn() -> TestResult) {
    TEST_REGISTRY.lock().push(TestCase { name, func });
}

/// Declarative macro to register a test function by name.
#[macro_export]
macro_rules! seal_test {
    ($name:ident) => {
        $crate::testing::register_test(stringify!($name), $name);
    };
}

/// Assert that a condition is true, returning `TestResult::Fail` on failure.
#[macro_export]
macro_rules! test_assert {
    ($cond:expr) => {
        if !$cond {
            return $crate::testing::TestResult::Fail(concat!(
                "assertion failed: ",
                stringify!($cond)
            ));
        }
    };
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return $crate::testing::TestResult::Fail($msg);
        }
    };
}

/// Assert that two values are equal, returning `TestResult::Fail` on mismatch.
#[macro_export]
macro_rules! test_assert_eq {
    ($left:expr, $right:expr) => {
        if $left != $right {
            return $crate::testing::TestResult::Fail(concat!(
                stringify!($left),
                " != ",
                stringify!($right)
            ));
        }
    };
}

pub mod harness;
pub mod mock;
pub mod runner;
