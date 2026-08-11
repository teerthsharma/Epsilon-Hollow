// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Path-prefix access policy — advisory, and default-ALLOW.
//!
//! Despite the `mac` module name this is not mandatory access control:
//! `check_file_permission` returns `true` when no rule matches the path, and
//! `true` when no policy has been loaded at all. It can only ever subtract
//! access from paths an explicit rule already covers.
//!
//! The default policy denies `/root` to non-root users and allows `/data` and `/tmp`.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use spin::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Permissions {
    pub const R: Self = Self {
        read: true,
        write: false,
        execute: false,
    };
    pub const W: Self = Self {
        read: false,
        write: true,
        execute: false,
    };
    pub const RW: Self = Self {
        read: true,
        write: true,
        execute: false,
    };
    pub const X: Self = Self {
        read: false,
        write: false,
        execute: true,
    };
    pub const RWX: Self = Self {
        read: true,
        write: true,
        execute: true,
    };
}

#[derive(Debug, Clone)]
pub enum MacRule {
    Allow { path: String, perms: Permissions },
    Deny { path: String, perms: Permissions },
}

#[derive(Debug, Clone)]
pub struct MacPolicy {
    pub rules: Vec<MacRule>,
}

impl MacPolicy {
    pub fn default_policy() -> Self {
        Self {
            rules: vec![
                MacRule::Deny {
                    path: String::from("/root"),
                    perms: Permissions::RWX,
                },
                MacRule::Allow {
                    path: String::from("/data"),
                    perms: Permissions::RWX,
                },
                MacRule::Allow {
                    path: String::from("/tmp"),
                    perms: Permissions::RWX,
                },
            ],
        }
    }
}

static GLOBAL_POLICY: Mutex<Option<MacPolicy>> = Mutex::new(None);

pub fn init_default_policy() {
    let mut guard = GLOBAL_POLICY.lock();
    *guard = Some(MacPolicy::default_policy());
}

pub fn load_policy(policy: MacPolicy) {
    let mut guard = GLOBAL_POLICY.lock();
    *guard = Some(policy);
}

/// Check whether `uid` is allowed to access `path` with `perms`.
/// Root (uid == 0) bypasses MAC.
///
/// `path` must already be canonical: single `/` separators, no trailing
/// slash, no `.`/`..` components. `Vfs::canonicalize_path` (vfs.rs) is
/// the single place that produces this form and is the only caller that
/// should ever reach this function with an un-normalized string —
/// otherwise a rule keyed on `/root` can be walked around with an extra
/// `/` (`//root/secret`) or a `..` the routing layer would still resolve
/// structurally. This function does not re-normalize; it only adds the
/// component-boundary check below, which is the other half of that same
/// bug class (over-matching instead of under-matching).
pub fn check_file_permission(uid: u32, path: &str, perms: Permissions) -> bool {
    if uid == 0 {
        return true;
    }

    let guard = GLOBAL_POLICY.lock();
    let policy = match guard.as_ref() {
        Some(p) => p,
        None => return true,
    };

    for rule in &policy.rules {
        match rule {
            MacRule::Deny {
                path: rule_path,
                perms: rule_perms,
            } => {
                if path_matches_rule(path, rule_path) && perms_overlap(perms, *rule_perms) {
                    return false;
                }
            }
            MacRule::Allow {
                path: rule_path,
                perms: rule_perms,
            } => {
                if path_matches_rule(path, rule_path) && perms_overlap(perms, *rule_perms) {
                    return true;
                }
            }
        }
    }

    true
}

/// A rule's `path` matches only at a path-component boundary: `path`
/// equals `rule_path` exactly, or the byte right after `rule_path` in
/// `path` is `/`. Plain `starts_with` over-matches — `/rootkit` would
/// trip a `/root` rule the policy never meant to cover it with. `"/"` as
/// a rule path is the one case with no boundary byte left to test after
/// stripping the whole string, so it matches every absolute path.
fn path_matches_rule(path: &str, rule_path: &str) -> bool {
    if rule_path == "/" {
        return true;
    }
    match path.strip_prefix(rule_path) {
        Some(rest) => rest.is_empty() || rest.starts_with('/'),
        None => false,
    }
}

fn perms_overlap(a: Permissions, b: Permissions) -> bool {
    (a.read && b.read) || (a.write && b.write) || (a.execute && b.execute)
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_mac_root_bypass() -> TestResult {
        init_default_policy();
        test_assert!(check_file_permission(0, "/root/secret", Permissions::R));
        test_assert!(check_file_permission(0, "/root/secret", Permissions::W));
        TestResult::Pass
    }

    fn test_mac_deny_root_for_user() -> TestResult {
        init_default_policy();
        test_assert!(!check_file_permission(1000, "/root/secret", Permissions::R));
        test_assert!(!check_file_permission(1000, "/root", Permissions::W));
        TestResult::Pass
    }

    fn test_mac_allow_data_and_tmp() -> TestResult {
        init_default_policy();
        test_assert!(check_file_permission(1000, "/data/file", Permissions::R));
        test_assert!(check_file_permission(1000, "/tmp/foo", Permissions::W));
        TestResult::Pass
    }

    /// `/rootkit` shares a byte-prefix with the `/root` `Deny` rule but is
    /// not inside it — plain `starts_with` used to deny it anyway. This
    /// is the over-matching half of the same defect class as
    /// `vfs::devfoo_is_not_inside_dev_mount`; assumes canonical input,
    /// same as every other case here.
    fn test_mac_rootkit_not_denied_by_root_rule() -> TestResult {
        init_default_policy();
        test_assert!(check_file_permission(1000, "/rootkit", Permissions::R));
        TestResult::Pass
    }

    /// `"/"` as a rule path has no boundary byte left after stripping —
    /// it must match every absolute path, not just the empty remainder.
    fn test_mac_root_rule_path_matches_everything() -> TestResult {
        load_policy(MacPolicy {
            rules: vec![MacRule::Deny {
                path: String::from("/"),
                perms: Permissions::RWX,
            }],
        });
        test_assert!(!check_file_permission(1000, "/anything/at/all", Permissions::R));
        test_assert!(!check_file_permission(1000, "/", Permissions::R));
        init_default_policy(); // restore default policy for tests that run after this one
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::mac_root_bypass", test_mac_root_bypass);
        crate::testing::register_test("security::mac_deny_root", test_mac_deny_root_for_user);
        crate::testing::register_test("security::mac_allow_data", test_mac_allow_data_and_tmp);
        crate::testing::register_test(
            "security::mac_rootkit_not_denied",
            test_mac_rootkit_not_denied_by_root_rule,
        );
        crate::testing::register_test(
            "security::mac_root_rule_matches_everything",
            test_mac_root_rule_path_matches_everything,
        );
    }
}
