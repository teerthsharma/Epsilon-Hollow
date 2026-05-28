// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Mandatory Access Control (MAC) — simple LSM framework.
//!
//! A default policy denies `/root` to non-root users and allows `/data` and `/tmp`.

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
                if path.starts_with(rule_path) && perms_overlap(perms, *rule_perms) {
                    return false;
                }
            }
            MacRule::Allow {
                path: rule_path,
                perms: rule_perms,
            } => {
                if path.starts_with(rule_path) && perms_overlap(perms, *rule_perms) {
                    return true;
                }
            }
        }
    }

    true
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

    pub fn register_all() {
        crate::testing::register_test("security::mac_root_bypass", test_mac_root_bypass);
        crate::testing::register_test("security::mac_deny_root", test_mac_deny_root_for_user);
        crate::testing::register_test("security::mac_allow_data", test_mac_allow_data_and_tmp);
    }
}
