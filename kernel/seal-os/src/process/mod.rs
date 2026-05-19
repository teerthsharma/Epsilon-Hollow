// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Process management with manifold-based scheduling and real context switching.

pub mod context_switch;
pub mod elf;
pub mod scheduler;
pub mod task;
pub mod userspace;

use alloc::string::String;
use alloc::vec::Vec;

pub struct Process {
    pub name: String,
    pub pid: u64,
}

pub fn create(name: &str) -> Process {
    Process {
        name: String::from(name),
        pid: 1, // hardcoded for init
    }
}

impl Process {
    pub fn execve(&self, _path: &str, _argv: &[&str], _envp: &[&str]) {
        // Fallback to embedded emergency shell if the file is not found.
        // For now, we spawn it as a kernel task as a placeholder until the FS is fully wired.
        let shell_elf = userspace::EMERGENCY_SHELL_ELF;
        let _ = scheduler::spawn_user(
            "emergency_shell",
            5,
            shell_elf,
        );
    }
}
