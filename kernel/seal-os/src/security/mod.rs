// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Security subsystem — ASLR, SMAP/SMEP, seccomp, MAC, and audit logging.

pub mod aslr;
pub mod audit;
pub mod audit_runtime;
pub mod group;
pub mod kpti;
pub mod mac;
pub mod manifold_acl;
pub mod passwd;
pub mod retpoline;
pub mod seccomp;
pub mod shadow;
pub mod smap_smep;
pub mod topcrypt_guard;

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
    kpti::init();
    retpoline::init();
    emit_hardening_proof();
}

fn emit_hardening_proof() {
    let kpti = kpti::runtime_proof();
    let smap_supported = smap_smep::has_smep_smap();
    let smap_enabled = smap_smep::is_enabled();
    let smap_ok = !smap_supported || smap_enabled;
    let result = if kpti.passes() && smap_ok {
        "pass"
    } else {
        "fail"
    };
    crate::serial_println!(
        "[SECURITY] hardening proof version=1 kpti={} kernel_cr3={:#x} user_cr3={:#x} kpti_distinct={} user_lower_zero={} kernel_upper_mirrored={} smap_smep_supported={} smap_smep_enabled={} user_access_faults={} result={}",
        if kpti.passes() { 1 } else { 0 },
        kpti.kernel_cr3,
        kpti.user_cr3,
        if kpti.distinct_roots { 1 } else { 0 },
        if kpti.user_lower_half_empty { 1 } else { 0 },
        if kpti.kernel_upper_half_mirrored { 1 } else { 0 },
        if smap_supported { 1 } else { 0 },
        if smap_enabled { 1 } else { 0 },
        smap_smep::user_access_faults(),
        result
    );
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
        super::kpti::tests::register_all();
        super::smap_smep::tests::register_all();
        super::seccomp::tests::register_all();
        super::mac::tests::register_all();
        super::retpoline::tests::register_all();
        super::audit::tests::register_all();
        crate::testing::register_test("security::init", test_security_init_does_not_panic);
    }
}
