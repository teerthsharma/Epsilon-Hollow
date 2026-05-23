// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Signal syscall numbers — wired by master in syscall/table.rs.

pub const SYS_KILL: u64 = 25;
pub const SYS_SIGACTION: u64 = 26;
pub const SYS_SIGRETURN: u64 = 27;
