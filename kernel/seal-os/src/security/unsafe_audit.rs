// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Unsafe-code audit.
//!
//! `tests/unsafe-audit.fixture` is a checked-in census of every `unsafe`
//! block under `kernel/seal-os/src` and of which ones carry a `SAFETY:`
//! justification. The fixture is compiled into the image with `include_str!`,
//! so the boot log carries the census that was true for *this* build, and the
//! host-side image gate re-scans the source tree and compares.
//!
//! The scan rule lives in [`scan_source`] and is the single definition both
//! sides must implement: a site is any occurrence of the block-opening token
//! (the `unsafe` keyword followed by a space and a brace), and it is justified
//! when `SAFETY:` appears on that line or anywhere in the contiguous run of
//! `//` comment lines directly above it.
//!
//! The measured result is unflattering and is reported as measured: the
//! overwhelming majority of `unsafe` blocks in this kernel have no written
//! justification. The gate freezes that count so it can only fall.

use alloc::string::String;

/// The checked-in census, compiled into the image.
const FIXTURE: &str = include_str!("../../tests/unsafe-audit.fixture");

/// The block-opening token the census counts. Spelled in two pieces so that
/// this file, which is itself scanned, does not add phantom sites to its own
/// census.
const BLOCK_TOKEN: &str = concat!("unsa", "fe {");

/// Totals recorded by the fixture.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct AuditTotals {
    pub version: u32,
    pub total: u64,
    pub justified: u64,
    pub unjustified: u64,
    pub files: u64,
}

impl AuditTotals {
    /// The fixture is self-consistent and non-empty.
    pub fn passes(self) -> bool {
        self.version == 1
            && self.total > 0
            && self.files > 0
            && self.justified + self.unjustified == self.total
            && self.justified <= self.total
    }
}

/// Count unsafe-block sites and justified sites in one Rust source text.
///
/// This is the normative scan rule. A host checker that disagrees with this
/// function is measuring something else and the gate becomes meaningless.
pub fn scan_source(text: &str) -> (u64, u64) {
    let lines: alloc::vec::Vec<&str> = text.split('\n').collect();
    let mut total = 0u64;
    let mut justified = 0u64;
    for (i, line) in lines.iter().enumerate() {
        let sites = line.matches(BLOCK_TOKEN).count() as u64;
        if sites == 0 {
            continue;
        }
        total += sites;
        let mut documented = line.contains("SAFETY:");
        let mut j = i;
        while !documented && j > 0 {
            let above = lines[j - 1].trim_start();
            if !above.starts_with("//") {
                break;
            }
            documented = above.contains("SAFETY:");
            j -= 1;
        }
        if documented {
            justified += sites;
        }
    }
    (total, justified)
}

/// Parse the totals out of a fixture text.
pub fn parse_totals(fixture: &str) -> AuditTotals {
    let mut totals = AuditTotals::default();
    for line in fixture.split('\n') {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(key), Some(value)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Ok(n) = value.parse::<u64>() else {
            continue;
        };
        match key {
            "version" => totals.version = n as u32,
            "total" => totals.total = n,
            "justified" => totals.justified = n,
            "unjustified" => totals.unjustified = n,
            "files" => totals.files = n,
            _ => {}
        }
    }
    totals
}

/// Totals for the fixture baked into this image.
pub fn totals() -> AuditTotals {
    parse_totals(FIXTURE)
}

/// `[UNSAFE-AUDIT] proof version=1 ...`
///
/// `result=pass` means the compiled-in census is internally consistent, not
/// that the unsafe code is safe. `unjustified` is the number the host gate
/// ratchets against; `undocumented_permille` is there so nobody has to divide
/// two numbers to notice how bad it is.
pub fn unsafe_audit_proof_line() -> String {
    let t = totals();
    let pct = (t.unjustified * 1000).checked_div(t.total).unwrap_or(0);
    alloc::format!(
        "[UNSAFE-AUDIT] proof version=1 fixture=tests/unsafe-audit.fixture fixture_version={} blocks={} justified={} unjustified={} files={} undocumented_permille={} rule=safety-comment-above-block result={}",
        t.version,
        t.total,
        t.justified,
        t.unjustified,
        t.files,
        pct,
        if t.passes() { "pass" } else { "fail" }
    )
}

/// Emit the unsafe-audit proof line to the serial console.
pub fn emit_unsafe_audit_proof() {
    crate::serial_println!("{}", unsafe_audit_proof_line());
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn justified_src() -> String {
        alloc::format!(
            "fn a() {{\n    // SAFETY: the pointer is derived from a live\n    // allocation that outlives this block.\n    {} *p = 1; }}\n}}\n",
            BLOCK_TOKEN
        )
    }

    fn unjustified_src() -> String {
        alloc::format!(
            "fn a() {{\n    // Bump the counter.\n    {} *p = 1; }}\n}}\n",
            BLOCK_TOKEN
        )
    }

    /// The scanner must flag a block whose comment block says nothing about
    /// safety — otherwise the gate would pass on decoration.
    fn test_detects_unjustified_block() -> TestResult {
        let (total, justified) = scan_source(&unjustified_src());
        test_assert_eq!(total, 1);
        test_assert_eq!(justified, 0);
        TestResult::Pass
    }

    /// A block whose preceding comment run carries SAFETY: counts as justified.
    fn test_accepts_justified_block() -> TestResult {
        let (total, justified) = scan_source(&justified_src());
        test_assert_eq!(total, 1);
        test_assert_eq!(justified, 1);
        // A SAFETY: comment separated from the block by a non-comment line does
        // not carry over.
        let detached = alloc::format!(
            "// SAFETY: unrelated\nfn a() {{\n    {} *p = 1; }}\n}}\n",
            BLOCK_TOKEN
        );
        let (t, j) = scan_source(&detached);
        test_assert_eq!(t, 1);
        test_assert_eq!(j, 0);
        // `unsafe fn` is a declaration, not a block, and must not be counted.
        let (t, _) = scan_source("pub unsafe fn f() {\n    let x = 1;\n}\n");
        test_assert_eq!(t, 0);
        TestResult::Pass
    }

    /// The compiled-in fixture must be parseable and self-consistent.
    fn test_fixture_is_consistent() -> TestResult {
        let t = totals();
        test_assert!(
            t.passes(),
            "compiled-in unsafe-audit fixture is inconsistent"
        );
        test_assert_eq!(t.justified + t.unjustified, t.total);
        test_assert!(t.total >= t.files, "fewer blocks than files with blocks");
        // Negative control: a fixture whose parts do not add up must fail.
        let broken = parse_totals("version 1\ntotal 10\njustified 2\nunjustified 3\nfiles 1\n");
        test_assert!(!broken.passes(), "inconsistent fixture must not pass");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "security::unsafe_audit_unjustified",
            test_detects_unjustified_block,
        );
        crate::testing::register_test(
            "security::unsafe_audit_justified",
            test_accepts_justified_block,
        );
        crate::testing::register_test("security::unsafe_audit_fixture", test_fixture_is_consistent);
    }
}
