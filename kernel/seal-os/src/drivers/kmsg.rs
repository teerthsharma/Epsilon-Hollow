// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Kernel message ring buffer — dmesg equivalent.
//!
//! Stores up to 32 KiB of kernel log messages in a fixed-size `.bss` buffer.
//! Overwrites the oldest data when full.

use core::sync::atomic::{AtomicUsize, Ordering};

const KMSG_SIZE: usize = 32 * 1024;

static mut KMSG_BUF: [u8; KMSG_SIZE] = [0u8; KMSG_SIZE];
static KMSG_HEAD: AtomicUsize = AtomicUsize::new(0);
static KMSG_TAIL: AtomicUsize = AtomicUsize::new(0);
static KMSG_LEN: AtomicUsize = AtomicUsize::new(0);

/// Append a message to the kernel ring buffer.
pub fn kmsg_write(msg: &str) {
    for &b in msg.as_bytes() {
        let tail = KMSG_TAIL.load(Ordering::Relaxed);
        unsafe {
            KMSG_BUF[tail] = b;
        }
        let new_tail = (tail + 1) % KMSG_SIZE;
        let len = KMSG_LEN.load(Ordering::Relaxed);
        if len == KMSG_SIZE {
            let head = KMSG_HEAD.load(Ordering::Relaxed);
            KMSG_HEAD.store((head + 1) % KMSG_SIZE, Ordering::Relaxed);
        } else {
            KMSG_LEN.store(len + 1, Ordering::Relaxed);
        }
        KMSG_TAIL.store(new_tail, Ordering::Relaxed);
    }
}

/// Copy up to `buf.len()` bytes from the ring buffer into `buf`.
/// Returns the number of bytes copied.
pub fn kmsg_read(buf: &mut [u8]) -> usize {
    let head = KMSG_HEAD.load(Ordering::Relaxed);
    let len = KMSG_LEN.load(Ordering::Relaxed);
    let mut read = 0;
    let mut idx = head;
    while read < len && read < buf.len() {
        buf[read] = unsafe { KMSG_BUF[idx] };
        read += 1;
        idx = (idx + 1) % KMSG_SIZE;
    }
    read
}
