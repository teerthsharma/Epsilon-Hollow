// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! seccomp BPF filter for syscall whitelisting/blacklisting.
//!
//! A simplified classic BPF evaluator.  Processes load a filter (array of
//! instructions) which is evaluated on every syscall entry.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;

/// A single BPF-like instruction for seccomp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeccompInsn {
    pub code: u16,
    pub jt: u8,
    pub jf: u8,
    pub k: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeccompError {
    InvalidFilter,
    NotLoaded,
}

// Standard seccomp return values
pub const SECCOMP_RET_KILL: u32 = 0x0000_0000;
pub const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
pub const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;

// Classic BPF opcodes (subset)
const BPF_LD_W_ABS: u16 = 0x20;
const BPF_JMP_JEQ: u16 = 0x05 | 0x10;       // 0x15
const BPF_RET: u16 = 0x06;

/// Per-task seccomp filters keyed by task ID.
static TASK_FILTERS: Mutex<BTreeMap<u64, Vec<SeccompInsn>>> = Mutex::new(BTreeMap::new());

/// Load a seccomp filter for the given task.
pub fn seccomp_load_filter(task_id: u64, filter: &[SeccompInsn]) -> Result<(), SeccompError> {
    if filter.is_empty() {
        return Err(SeccompError::InvalidFilter);
    }
    let mut map = TASK_FILTERS.lock();
    map.insert(task_id, filter.into());
    Ok(())
}

/// Remove a task's seccomp filter.
pub fn seccomp_unload_filter(task_id: u64) {
    let mut map = TASK_FILTERS.lock();
    map.remove(&task_id);
}

/// Evaluate the seccomp filter for `task_id` against `syscall_num`.
///
/// Returns one of `SECCOMP_RET_*` constants.
pub fn seccomp_check(task_id: u64, syscall_num: u64) -> u32 {
    let map = TASK_FILTERS.lock();
    let filter = match map.get(&task_id) {
        Some(f) => f,
        None => return SECCOMP_RET_ALLOW,
    };

    let mut pc = 0usize;
    let mut acc: u32 = 0;

    while pc < filter.len() {
        let insn = filter[pc];
        match insn.code {
            BPF_LD_W_ABS => {
                // Load syscall number from argument offset k (0 = syscall_num)
                acc = if insn.k == 0 {
                    syscall_num as u32
                } else {
                    0
                };
                pc += 1;
            }
            BPF_JMP_JEQ => {
                if acc == insn.k {
                    pc += insn.jt as usize + 1;
                } else {
                    pc += insn.jf as usize + 1;
                }
            }
            BPF_RET => {
                // Return value is in k, combined with accumulator if needed.
                // For simplicity we just return k as the action.
                return insn.k;
            }
            _ => {
                // Unknown instruction — deny for safety.
                return SECCOMP_RET_KILL;
            }
        }
    }

    // Reached end of filter without returning — deny.
    SECCOMP_RET_KILL
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use alloc::vec;
    use crate::testing::TestResult;
    use crate::test_assert_eq;

    fn test_seccomp_allow() -> TestResult {
        let filter = vec![
            SeccompInsn { code: BPF_LD_W_ABS, jt: 0, jf: 0, k: 0 },
            SeccompInsn { code: BPF_JMP_JEQ, jt: 0, jf: 1, k: 1 },
            SeccompInsn { code: BPF_RET, jt: 0, jf: 0, k: SECCOMP_RET_ALLOW },
            SeccompInsn { code: BPF_RET, jt: 0, jf: 0, k: SECCOMP_RET_KILL },
        ];
        seccomp_load_filter(99, &filter).unwrap();
        let action = seccomp_check(99, 1);
        test_assert_eq!(action, SECCOMP_RET_ALLOW);
        let action2 = seccomp_check(99, 2);
        test_assert_eq!(action2, SECCOMP_RET_KILL);
        seccomp_unload_filter(99);
        TestResult::Pass
    }

    fn test_seccomp_no_filter_allows() -> TestResult {
        let action = seccomp_check(999, 42);
        test_assert_eq!(action, SECCOMP_RET_ALLOW);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::seccomp_allow", test_seccomp_allow);
        crate::testing::register_test("security::seccomp_no_filter", test_seccomp_no_filter_allows);
    }
}
