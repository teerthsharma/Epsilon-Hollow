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
        let boot_uid = crate::security::passwd::boot_uid().unwrap_or(1000);
        let boot_gid = crate::security::passwd::boot_gid().unwrap_or(1000);

        let (data, file_mode, file_uid, file_gid) =
            if let Some((bytes, stat)) = Self::read_file_bytes_and_stat(path) {
                crate::serial_println!("[execve] Loading '{}' ({} bytes)", path, bytes.len());
                (bytes, stat.mode, stat.uid, stat.gid)
            } else {
                crate::serial_println!(
                    "[execve] '{}' not found — falling back to emergency shell",
                    path
                );
                (userspace::EMERGENCY_SHELL_ELF.to_vec(), 0o755, 0, 0)
            };

        let name: &'static str = if path == "/bin/init" {
            "init"
        } else {
            "userspace"
        };
        match scheduler::spawn_user(
            name, 5, &data, file_mode, file_uid, file_gid, boot_uid, boot_gid,
        ) {
            Ok(id) if id != 0 => {
                crate::serial_println!("[execve] Spawned '{}' as task {}", name, id);
            }
            Ok(_) => {
                crate::serial_println!("[execve] Spawn of '{}' returned no task", name);
            }
            Err(e) => {
                crate::serial_println!("[execve] ELF load failed: {:?}", e);
                let _ = scheduler::spawn_user(
                    "emergency_shell",
                    5,
                    userspace::EMERGENCY_SHELL_ELF,
                    0o755,
                    0,
                    0,
                    boot_uid,
                    boot_gid,
                );
            }
        }
    }

    fn read_file_bytes_and_stat(path: &str) -> Option<(Vec<u8>, crate::fs::vfs::VfsNode)> {
        use crate::fs::vfs::with_vfs;
        let handle = with_vfs(|vfs| vfs.lookup_follow(path)).ok()?;
        let stat = with_vfs(|vfs| vfs.stat(handle)).ok()?;
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
            Some((data, stat))
        }
    }
}
