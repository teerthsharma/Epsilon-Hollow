// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Time-related syscall helpers (gettimeofday, settimeofday, watchdog).

use crate::syscall::table::SyscallResult;

#[repr(C)]
struct Timeval {
    tv_sec: u64,
    tv_usec: u64,
}

/// Copy the current time into a userspace `struct timeval`.
pub fn dispatch_gettimeofday(buf_ptr: u64) -> SyscallResult {
    let sec = crate::drivers::rtc::seconds_since_epoch();
    let usec = (crate::drivers::interrupts::ticks() % 1000) * 1000;
    let tv = Timeval { tv_sec: sec, tv_usec: usec };

    let user_ptr = buf_ptr as *mut u8;
    let bytes = unsafe {
        core::slice::from_raw_parts(
            &tv as *const Timeval as *const u8,
            core::mem::size_of::<Timeval>(),
        )
    };

    unsafe {
        if crate::security::smap_smep::copy_to_user(user_ptr, bytes).is_err() {
            return SyscallResult::err(14); // EFAULT
        }
    }

    SyscallResult::ok(0)
}

/// Stub: setting the system time is not permitted from userspace.
pub fn dispatch_settimeofday(_sec: u64, _usec: u64) -> SyscallResult {
    SyscallResult::err(1) // EPERM
}

/// Pet the kernel watchdog if `pet` is non-zero.
pub fn dispatch_watchdog(pet: u64) -> SyscallResult {
    if pet != 0 {
        crate::drivers::watchdog::pet();
    }
    SyscallResult::ok(0)
}
