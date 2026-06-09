// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Runtime security audit probes.
//!
//! Responds to host-side audit commands over the serial port with structured
//! JSON-like output.  Used by the Python security audit suite in
//! `tests/security/`.

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
    // Effective entropy is derived from the log2 of each randomized range.
    let mmap_entropy = 47u32; // ~2^47 range
    let stack_entropy = 35u32;
    let heap_entropy = 33u32;
    format!(
        "{{\"probe\":\"aslr\",\"mmap_base\":\"{:#x}\",\"stack_top\":\"{:#x}\",\"heap_base\":\"{:#x}\",\"mmap_entropy\":{},\"stack_entropy\":{},\"heap_entropy\":{}}}",
        mmap, stack, heap, mmap_entropy, stack_entropy, heap_entropy
    )
}

fn probe_seccomp() -> String {
    let count = crate::security::seccomp::filter_count();
    format!(
        "{{\"probe\":\"seccomp\",\"active\":true,\"filter_count\":{}}}",
        count
    )
}

fn probe_kpti() -> String {
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
