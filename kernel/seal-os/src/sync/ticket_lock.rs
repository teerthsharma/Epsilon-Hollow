// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! FIFO ticket lock — fair spin lock with strict arrival ordering.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A fair spin lock that services waiters in FIFO order.
pub struct TicketLock {
    next_ticket: AtomicUsize,
    now_serving: AtomicUsize,
}

impl TicketLock {
    /// Create a new unlocked `TicketLock`.
    pub const fn new() -> Self {
        Self {
            next_ticket: AtomicUsize::new(0),
            now_serving: AtomicUsize::new(0),
        }
    }

    /// Acquire the lock, spinning until this caller's ticket is served.
    pub fn lock(&self) -> TicketLockGuard<'_> {
        let my_ticket = self.next_ticket.fetch_add(1, Ordering::SeqCst);
        while self.now_serving.load(Ordering::SeqCst) != my_ticket {
            core::hint::spin_loop();
        }
        TicketLockGuard { lock: self }
    }

    /// Release the lock, allowing the next waiter to proceed.
    ///
    /// # Safety
    /// Caller must actually hold the lock.
    pub unsafe fn unlock(&self) {
        self.now_serving.fetch_add(1, Ordering::SeqCst);
    }
}

/// RAII guard for `TicketLock`.
pub struct TicketLockGuard<'a> {
    lock: &'a TicketLock,
}

impl<'a> Drop for TicketLockGuard<'a> {
    fn drop(&mut self) {
        unsafe { self.lock.unlock(); }
    }
}

unsafe impl Send for TicketLock {}
unsafe impl Sync for TicketLock {}
