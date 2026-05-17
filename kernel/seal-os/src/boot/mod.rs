// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Boot infrastructure: Multiboot2 header + 32→64 trampoline.

pub mod multiboot2;

core::arch::global_asm!(include_str!("boot.S"), options(att_syntax));
