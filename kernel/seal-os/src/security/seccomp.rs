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

// Actions Linux defines that this kernel has no implementation for. They are
// listed so `normalize_action` denies them by name rather than by accident.
pub const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
pub const SECCOMP_RET_TRAP: u32 = 0x0003_0000;
pub const SECCOMP_RET_USER_NOTIF: u32 = 0x7fc0_0000;
pub const SECCOMP_RET_TRACE: u32 = 0x7ff0_0000;
pub const SECCOMP_RET_LOG: u32 = 0x7ffc_0000;

/// Selects the action bits of a BPF return value. The low 16 bits carry data —
/// an errno for `SECCOMP_RET_ERRNO` — and are not part of the action.
pub const SECCOMP_RET_ACTION_FULL: u32 = 0xffff_0000;

/// Reduce a raw BPF `ret` value to one of the actions this kernel implements.
///
/// A filter's `k` field is an arbitrary u32, so every value must land somewhere.
/// This mapping fails closed: only ALLOW allows. That covers three cases that
/// were previously permitted by falling through a catch-all allow arm:
///
///   - `SECCOMP_RET_ERRNO | errno`, where the data bits made the value miss an
///     equality test against the bare constant. Masking to the action bits
///     first is what Linux does, and it is why the errno form now denies.
///   - TRAP, TRACE, USER_NOTIF and KILL_PROCESS, which this kernel cannot
///     deliver. A filter author asking for a trap gets a denial, never an allow.
///   - Any undefined action, which Linux also treats as fail-closed.
///
/// LOG is the one deliberate divergence from Linux, which treats it as
/// allow-and-log. There is no seccomp logging path here, so honouring it would
/// mean allowing on the strength of a side effect that never happens.
fn normalize_action(raw: u32) -> u32 {
    match raw & SECCOMP_RET_ACTION_FULL {
        SECCOMP_RET_ALLOW => SECCOMP_RET_ALLOW,
        SECCOMP_RET_ERRNO => SECCOMP_RET_ERRNO,
        _ => SECCOMP_RET_KILL,
    }
}

// Classic BPF opcodes (subset)
const BPF_LD_W_ABS: u16 = 0x20;
const BPF_JMP_JEQ: u16 = 0x05 | 0x10; // 0x15
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

/// Return the number of loaded seccomp filters.
pub fn filter_count() -> usize {
    let map = TASK_FILTERS.lock();
    map.len()
}

/// Evaluate the seccomp filter for `task_id` against `syscall_num`.
///
/// Returns exactly one of `SECCOMP_RET_ALLOW`, `SECCOMP_RET_ERRNO` or
/// `SECCOMP_RET_KILL` — never a filter-supplied value. See `normalize_action`
/// for how the other Linux actions and undefined values are folded into KILL.
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
                acc = if insn.k == 0 { syscall_num as u32 } else { 0 };
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
                // `k` is filter-supplied and unvalidated, so it is reduced to a
                // known action here rather than handed to the caller verbatim.
                return normalize_action(insn.k);
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
    use crate::test_assert_eq;
    use crate::testing::TestResult;
    use alloc::vec;

    fn test_seccomp_allow() -> TestResult {
        let filter = vec![
            SeccompInsn {
                code: BPF_LD_W_ABS,
                jt: 0,
                jf: 0,
                k: 0,
            },
            SeccompInsn {
                code: BPF_JMP_JEQ,
                jt: 0,
                jf: 1,
                k: 1,
            },
            SeccompInsn {
                code: BPF_RET,
                jt: 0,
                jf: 0,
                k: SECCOMP_RET_ALLOW,
            },
            SeccompInsn {
                code: BPF_RET,
                jt: 0,
                jf: 0,
                k: SECCOMP_RET_KILL,
            },
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

    /// Only ALLOW may allow. Every other action — the ones this kernel cannot
    /// deliver, and every undefined value — must reduce to KILL. A regression
    /// to the old "return `k` verbatim" behaviour fails here first, because
    /// each of these values would come back out unchanged and be read as an
    /// allow by the dispatcher's catch-all arm.
    fn test_unknown_action_denies() -> TestResult {
        for raw in [
            SECCOMP_RET_TRAP,
            SECCOMP_RET_TRACE,
            SECCOMP_RET_LOG,
            SECCOMP_RET_USER_NOTIF,
            SECCOMP_RET_KILL_PROCESS,
            0xdead_beef,
            0x0000_0001,
        ] {
            test_assert_eq!(normalize_action(raw), SECCOMP_RET_KILL);
        }
        // Data bits must not knock a defined action off its match arm.
        test_assert_eq!(
            normalize_action(SECCOMP_RET_ERRNO | 13),
            SECCOMP_RET_ERRNO
        );
        test_assert_eq!(normalize_action(SECCOMP_RET_ALLOW), SECCOMP_RET_ALLOW);

        // End to end, through a loaded filter: a TRAP-returning filter denies.
        let filter = vec![
            SeccompInsn {
                code: BPF_RET,
                jt: 0,
                jf: 0,
                k: SECCOMP_RET_TRAP,
            },
        ];
        seccomp_load_filter(98, &filter).unwrap();
        test_assert_eq!(seccomp_check(98, 1), SECCOMP_RET_KILL);
        seccomp_unload_filter(98);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::seccomp_allow", test_seccomp_allow);
        crate::testing::register_test("security::seccomp_no_filter", test_seccomp_no_filter_allows);
        crate::testing::register_test("security::seccomp_unknown_denies", test_unknown_action_denies);
    }
}
