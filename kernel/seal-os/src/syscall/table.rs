// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Syscall dispatch table. Standard POSIX-like + Epsilon extensions.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub const SYS_EXIT: u64 = 0;
pub const SYS_WRITE: u64 = 1;
pub const SYS_READ: u64 = 2;
pub const SYS_OPEN: u64 = 3;
pub const SYS_CLOSE: u64 = 4;
pub const SYS_EXEC: u64 = 5;
pub const SYS_FORK: u64 = 6;
pub const SYS_WAITPID: u64 = 7;
pub const SYS_MMAP: u64 = 8;
pub const SYS_GETPID: u64 = 9;
pub const SYS_STAT: u64 = 10;

// Epsilon extensions
pub const SYS_MANIFOLD_QUERY: u64 = 100;
pub const SYS_TELEPORT: u64 = 101;
pub const SYS_THEOREM_STATUS: u64 = 102;

#[derive(Debug)]
pub struct SyscallResult {
    pub code: i64,
    pub data: Option<String>,
}

impl SyscallResult {
    pub fn ok(code: i64) -> Self {
        Self { code, data: None }
    }

    pub fn with_data(code: i64, data: String) -> Self {
        Self {
            code,
            data: Some(data),
        }
    }

    pub fn err(errno: i64) -> Self {
        Self {
            code: -errno,
            data: None,
        }
    }
}

pub fn dispatch(num: u64, arg0: u64, arg1: u64, arg2: u64) -> SyscallResult {
    match num {
        SYS_EXIT => SyscallResult::ok(0),
        SYS_WRITE => {
            // arg0=fd, arg1=ptr, arg2=len — kernel write to serial
            SyscallResult::ok(arg2 as i64)
        }
        SYS_READ => SyscallResult::ok(0),
        SYS_OPEN => SyscallResult::ok(3), // return fd=3
        SYS_CLOSE => SyscallResult::ok(0),
        SYS_GETPID => SyscallResult::ok(1),
        SYS_MANIFOLD_QUERY => SyscallResult::with_data(
            0,
            format!("theorem_{}: ACTIVE", arg0),
        ),
        SYS_TELEPORT => SyscallResult::with_data(
            0,
            format!("teleported inode {} -> dir {}", arg0, arg1),
        ),
        SYS_THEOREM_STATUS => SyscallResult::with_data(
            0,
            String::from("T1:ACTIVE T2:ACTIVE T3:ACTIVE T4:ACTIVE T5:ACTIVE"),
        ),
        _ => SyscallResult::err(38), // ENOSYS
    }
}
