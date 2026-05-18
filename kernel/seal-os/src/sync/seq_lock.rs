// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Sequence lock — optimized for read-mostly data (system ticks, stats).
//!
//! Readers sample a sequence counter before and after reading; if the counter
//! changed (or is odd, indicating a write in progress), the read is retried.

use core::sync::atomic::{AtomicUsize, Ordering};

/// A sequence lock providing concurrent readers and a single writer.
pub struct SeqLock {
    sequence: AtomicUsize,
}

impl SeqLock {
    /// Create a new `SeqLock` in unlocked (even) state.
    pub const fn new() -> Self {
        Self {
            sequence: AtomicUsize::new(0),
        }
    }

    /// Begin a read transaction. Returns the current sequence number.
    /// Readers should verify the returned value is even; if odd, a write is
    /// in progress.
    pub fn read_begin(&self) -> usize {
        self.sequence.load(Ordering::SeqCst)
    }

    /// Check whether the read should be retried because a writer ran.
    /// Returns `true` if the sequence changed or is odd.
    pub fn read_retry(&self, seq: usize) -> bool {
        let current = self.sequence.load(Ordering::SeqCst);
        current != seq || current & 1 != 0
    }

    /// Acquire the write lock by waiting for the sequence to become even,
    /// then incrementing it to odd.
    pub fn write_lock(&self) {
        loop {
            let seq = self.sequence.load(Ordering::SeqCst);
            if seq & 1 == 0 {
                if self
                    .sequence
                    .compare_exchange_weak(seq, seq + 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    break;
                }
            }
            core::hint::spin_loop();
        }
    }

    /// Release the write lock by incrementing the sequence to the next even
    /// number.
    pub fn write_unlock(&self) {
        self.sequence.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe impl Send for SeqLock {}
unsafe impl Sync for SeqLock {}
