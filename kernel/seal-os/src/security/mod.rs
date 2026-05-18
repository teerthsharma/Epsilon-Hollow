// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Security subsystem — ASLR, SMAP/SMEP, seccomp, MAC, and audit logging.

pub mod aslr;
pub mod audit;
pub mod mac;
pub mod seccomp;
pub mod smap_smep;

/// Initialize all security subsystems.
///
/// - Enables SMAP/SMEP in CR4.
/// - Loads the default MAC policy.
/// - Prepares the audit log (directories created lazily when VFS is ready).
pub fn init_security() {
    unsafe {
        smap_smep::enable_smap_smep();
    }
    mac::init_default_policy();
    audit::init_audit_log();
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use crate::testing::TestResult;

    fn test_security_init_does_not_panic() -> TestResult {
        // init_security is already called during boot; calling it again should
        // be idempotent (or at least not panic).
        super::init_security();
        TestResult::Pass
    }

    pub fn register_all() {
        super::aslr::tests::register_all();
        super::smap_smep::tests::register_all();
        super::seccomp::tests::register_all();
        super::mac::tests::register_all();
        super::audit::tests::register_all();
        crate::testing::register_test("security::init", test_security_init_does_not_panic);
    }
}
