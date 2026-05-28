// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Syscall helpers for pipe, dup, dup2, and brk.

use alloc::string::String;
use core::sync::atomic::Ordering;

use crate::fs::pipe::create_pipe;
use crate::syscall::table::{FdEntry, SyscallResult, FILE_TABLE, NEXT_FD};

pub const SYS_PIPE: u64 = 28;
pub const SYS_DUP: u64 = 29;
pub const SYS_DUP2: u64 = 30;
pub const SYS_BRK: u64 = 31;

/// SYS_PIPE — create a pipe and return two fds in the userspace array.
pub fn dispatch_pipe(arg0: u64) -> SyscallResult {
    let pipefd = arg0 as *mut [u64; 2];
    if pipefd.is_null() {
        return SyscallResult::err(22); // EINVAL
    }

    let (read_handle, write_handle) = create_pipe();

    let fd_read = NEXT_FD.fetch_add(1, Ordering::SeqCst);
    let fd_write = NEXT_FD.fetch_add(1, Ordering::SeqCst);

    let mut table = FILE_TABLE.lock();
    table.insert(
        fd_read,
        FdEntry {
            handle: read_handle,
            path: String::from("pipe:r"),
            offset: 0,
        },
    );
    table.insert(
        fd_write,
        FdEntry {
            handle: write_handle,
            path: String::from("pipe:w"),
            offset: 0,
        },
    );
    drop(table);

    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&fd_read.to_ne_bytes());
    buf[8..16].copy_from_slice(&fd_write.to_ne_bytes());
    unsafe {
        if crate::security::smap_smep::copy_to_user(pipefd as *mut u8, &buf).is_err() {
            return SyscallResult::err(14); // EFAULT
        }
    }

    SyscallResult::ok(0)
}

/// SYS_DUP — duplicate an open file descriptor.
pub fn dispatch_dup(arg0: u64) -> SyscallResult {
    let old_fd = arg0;
    let mut table = FILE_TABLE.lock();
    let entry = match table.get(&old_fd) {
        Some(e) => e.clone(),
        None => return SyscallResult::err(9), // EBADF
    };
    let new_fd = NEXT_FD.fetch_add(1, Ordering::SeqCst);
    table.insert(new_fd, entry);
    SyscallResult::ok(new_fd as i64)
}

/// SYS_DUP2 — duplicate an fd to a specific new fd.
pub fn dispatch_dup2(arg0: u64, arg1: u64) -> SyscallResult {
    let old_fd = arg0;
    let new_fd = arg1;
    let mut table = FILE_TABLE.lock();
    let entry = match table.get(&old_fd) {
        Some(e) => e.clone(),
        None => return SyscallResult::err(9), // EBADF
    };
    let _ = table.remove(&new_fd);
    table.insert(new_fd, entry);
    SyscallResult::ok(new_fd as i64)
}

/// SYS_BRK — adjust the program break.
pub fn dispatch_brk(arg0: u64) -> SyscallResult {
    let new_addr = arg0;
    let current_brk = crate::process::scheduler::current_brk_end();

    if new_addr == 0 {
        return SyscallResult::ok(current_brk as i64);
    }

    if new_addr > current_brk {
        let diff = new_addr - current_brk;
        let pages = ((diff + 4095) / 4096) as usize;
        if pages == 0 {
            crate::process::scheduler::set_current_brk_end(new_addr);
            return SyscallResult::ok(new_addr as i64);
        }
        let flags = x86_64::structures::paging::PageTableFlags::PRESENT
            | x86_64::structures::paging::PageTableFlags::WRITABLE
            | x86_64::structures::paging::PageTableFlags::USER_ACCESSIBLE;
        if let Some(pt) = crate::process::scheduler::current_page_table() {
            match crate::memory::mmap::mmap_user(pages, flags, pt) {
                Some(_virt) => {
                    crate::process::scheduler::set_current_brk_end(new_addr);
                    SyscallResult::ok(new_addr as i64)
                }
                None => SyscallResult::err(12), // ENOMEM
            }
        } else {
            SyscallResult::err(1) // EPERM
        }
    } else if new_addr < current_brk {
        // v1: shrinking not implemented — just update the limit.
        crate::process::scheduler::set_current_brk_end(new_addr);
        SyscallResult::ok(new_addr as i64)
    } else {
        SyscallResult::ok(current_brk as i64)
    }
}
