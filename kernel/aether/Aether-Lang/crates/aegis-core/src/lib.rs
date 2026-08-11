// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! AEGIS Core: TitanClock lock-free sharded memory allocator.
//!
//! Not a language crate — see [`memory::TitanClock`] for the allocator
//! itself, a highly parallel, sharded implementation of the Bio-Clock
//! algorithm used by the AEGIS/AETHER runtimes.

#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod memory;
