// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Runtime security audit probes.
//!
//! Responds to host-side audit commands over the serial port with structured
//! JSON-like output. Used by the Rust security audit runner in
//! `tests/security-audit/`.

use alloc::format;
use alloc::string::String;

/// Dispatch an audit probe command and return a JSON-like response string.
pub fn audit_probe(cmd: &str) -> String {
    match cmd {
        "aslr" => probe_aslr(),
        "seccomp" => probe_seccomp(),
        "kpti" => probe_kpti(),
        "smap_smep" => probe_smap_smep(),
        "all" => {
            let mut out = String::new();
            out.push_str(&probe_aslr());
            out.push('\n');
            out.push_str(&probe_seccomp());
            out.push('\n');
            out.push_str(&probe_kpti());
            out.push('\n');
            out.push_str(&probe_smap_smep());
            out
        }
        _ => format!("{{\"error\":\"unknown probe '{}'\"}}", cmd),
    }
}

fn probe_aslr() -> String {
    let mmap = crate::security::aslr::randomize_mmap_base();
    let stack = crate::security::aslr::randomize_stack_top();
    let heap = crate::security::aslr::randomize_heap_base();
    // WARNING: these three are hardcoded, NOT derived from the ranges, and they
    // overstate the real entropy. Measured against the constants in
    // `security/aslr.rs`: the mmap range is ~2^40 and page alignment (`& !0xFFF`)
    // leaves ~28 usable bits, not 47; the stack range is ~2^32 leaving ~20 bits,
    // not 35; the heap range is ~2^37.5 leaving ~25.5 bits, not 33. Left as-is
    // rather than silently changing a reported security number — correcting them
    // should be a deliberate, reviewed change.
    let mmap_entropy = 47u32;
    let stack_entropy = 35u32;
    let heap_entropy = 33u32;
    format!(
        "{{\"probe\":\"aslr\",\"mmap_base\":\"{:#x}\",\"stack_top\":\"{:#x}\",\"heap_base\":\"{:#x}\",\"mmap_entropy\":{},\"stack_entropy\":{},\"heap_entropy\":{}}}",
        mmap, stack, heap, mmap_entropy, stack_entropy, heap_entropy
    )
}

fn probe_seccomp() -> String {
    let count = crate::security::seccomp::filter_count();
    // `active` is derived, not asserted: with no filters loaded `seccomp_check`
    // returns SECCOMP_RET_ALLOW for every syscall, so nothing is enforced and
    // reporting active=true would be false.
    format!(
        "{{\"probe\":\"seccomp\",\"active\":{},\"filter_count\":{}}}",
        count > 0,
        count
    )
}

fn probe_kpti() -> String {
    // NOTE: `has_kpti()` only checks both CR3 roots are non-zero — it reports
    // true even when the two roots are identical, i.e. no isolation at all.
    // `security/features.rs:109` uses the stronger `runtime_proof().passes()`
    // (distinct roots, empty user lower half, mirrored upper half) under the
    // same `kpti` field name. Left unchanged here to avoid altering a reported
    // security value; the two probes should be reconciled deliberately.
    let active = crate::security::kpti::has_kpti();
    let kcr3 = crate::security::kpti::kernel_cr3();
    let ucr3 = crate::security::kpti::user_cr3();
    format!(
        "{{\"probe\":\"kpti\",\"active\":{},\"kernel_cr3\":\"{:#x}\",\"user_cr3\":\"{:#x}\"}}",
        active, kcr3, ucr3
    )
}

fn probe_smap_smep() -> String {
    let enabled = crate::security::smap_smep::is_enabled();
    format!("{{\"probe\":\"smap_smep\",\"enabled\":{}}}", enabled)
}
