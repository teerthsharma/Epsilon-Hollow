// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Ioctl dispatch framework for device drivers.

use alloc::collections::BTreeMap;
use spin::Mutex;

pub type IoctlHandler = fn(request: u64, arg: u64) -> i64;

static IOCTL_TABLE: Mutex<BTreeMap<u64, IoctlHandler>> = Mutex::new(BTreeMap::new());

fn encode_dev(major: u32, minor: u32) -> u64 {
    ((major as u64) << 32) | (minor as u64)
}

/// Register an ioctl handler for a device node.
pub fn register(major: u32, minor: u32, handler: IoctlHandler) {
    let key = encode_dev(major, minor);
    IOCTL_TABLE.lock().insert(key, handler);
}

/// Dispatch an ioctl to the registered handler for a device node.
pub fn dispatch(major: u32, minor: u32, request: u64, arg: u64) -> i64 {
    let key = encode_dev(major, minor);
    match IOCTL_TABLE.lock().get(&key).copied() {
        Some(handler) => handler(request, arg),
        None => -25, // ENOTTY
    }
}

/// Basic terminal ioctl handler.
fn tty_ioctl(request: u64, _arg: u64) -> i64 {
    match request {
        0x5401 => 0, // TCGETS — return dummy terminal info
        0x5402 => 0, // TCSETS — accept but do nothing
        0x541B => 0, // FIONREAD — return 0 for tty
        _ => -25,    // ENOTTY
    }
}

/// Pipe ioctl handler.
fn pipe_ioctl(request: u64, _arg: u64) -> i64 {
    match request {
        0x541B => 0, // FIONREAD — return 0 (no pipe buffer tracking yet)
        _ => -25,
    }
}

/// Register basic built-in ioctls.
pub fn init() {
    register(5, 1, tty_ioctl);   // /dev/console
    register(1, 3, pipe_ioctl);  // /dev/null
    register(1, 5, pipe_ioctl);  // /dev/zero
    register(1, 8, pipe_ioctl);  // /dev/random
}
