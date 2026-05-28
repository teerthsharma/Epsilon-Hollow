// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Synchronization primitives — ticket lock, seq-lock, and TLB shootdown.

pub mod seq_lock;
pub mod ticket_lock;
pub mod tlb;

#[cfg(any(test, feature = "test-mode"))]
pub mod tests;

pub use seq_lock::SeqLock;
pub use ticket_lock::TicketLock;
