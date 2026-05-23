// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Process management with manifold-based scheduling and real context switching.

pub mod context_switch;
pub mod elf;
pub mod hollow_asm;
pub mod scheduler;
pub mod signal;
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
        pid: 1,
    }
}

impl Process {
    pub fn execve(&self, path: &str, _argv: &[&str], _envp: &[&str]) {
        // Try to load real ELF from VFS
        let elf_data = Self::read_file_bytes(path);
        let data = match elf_data {
            Some(bytes) => {
                crate::serial_println!("[execve] Loading '{}' ({} bytes)", path, bytes.len());
                bytes
            }
            None => {
                crate::serial_println!("[execve] '{}' not found — falling back to emergency shell", path);
                userspace::EMERGENCY_SHELL_ELF.to_vec()
            }
        };

        let aslr_base = crate::security::aslr::randomize_mmap_base();
        match elf::load(&data, aslr_base) {
            Ok(_loaded) => {
                let name: &'static str = if path == "/bin/init" { "init" } else { "userspace" };
                let _ = scheduler::spawn_user(
                    name,
                    5,
                    &data,
                );
            }
            Err(e) => {
                crate::serial_println!("[execve] ELF load failed: {:?}", e);
                // Fallback to emergency shell
                let _ = scheduler::spawn_user(
                    "emergency_shell",
                    5,
                    userspace::EMERGENCY_SHELL_ELF,
                );
            }
        }
    }

    fn read_file_bytes(path: &str) -> Option<Vec<u8>> {
        use crate::fs::vfs::with_vfs;
        let handle = with_vfs(|vfs| vfs.lookup_follow(path)).ok()?;
        let mut buf = [0u8; 4096];
        let mut data = Vec::new();
        let mut offset = 0usize;
        loop {
            match with_vfs(|vfs| vfs.read(handle, &mut buf, offset as u64)) {
                Ok(0) => break,
                Ok(n) => {
                    data.extend_from_slice(&buf[..n]);
                    offset += n;
                }
                Err(_) => break,
            }
        }
        if data.is_empty() {
            None
        } else {
            Some(data)
        }
    }
}
