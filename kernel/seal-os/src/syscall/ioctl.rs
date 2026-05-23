// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Syscall ioctl dispatch.

use crate::fs::vfs::{with_vfs, VfsError, VfsNodeType};
use crate::syscall::table::{FILE_TABLE, SyscallResult};

pub fn dispatch_ioctl(fd: u64, request: u64, arg: u64) -> SyscallResult {
    let table = FILE_TABLE.lock();
    let entry = match table.get(&fd) {
        Some(e) => e,
        None => return SyscallResult::err(9), // EBADF
    };
    let handle = entry.handle;
    drop(table);

    let node = match with_vfs(|vfs| vfs.stat(handle)) {
        Ok(n) => n,
        Err(e) => return SyscallResult::err(vfs_error_to_errno(e)),
    };

    match node.node_type {
        VfsNodeType::CharDevice | VfsNodeType::BlockDevice => {
            let result = crate::drivers::ioctl::dispatch(node.major, node.minor, request, arg);
            if result < 0 {
                SyscallResult::err(-result as i64)
            } else {
                SyscallResult::ok(result)
            }
        }
        _ => SyscallResult::err(25), // ENOTTY
    }
}

fn vfs_error_to_errno(e: VfsError) -> i64 {
    match e {
        VfsError::NotFound => 2,
        VfsError::AlreadyExists => 17,
        VfsError::NotADirectory => 20,
        VfsError::PermissionDenied => 13,
        VfsError::InvalidPath => 22,
        VfsError::TooManySymlinks => 40,
        _ => 5,
    }
}
