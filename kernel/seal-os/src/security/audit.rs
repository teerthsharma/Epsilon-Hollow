// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Audit logging subsystem.
//!
//! Events are formatted as JSON-like text and appended to `/var/log/audit.log`
//! via the VFS.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

use crate::fs::vfs::with_vfs;

#[derive(Debug, Clone)]
pub enum AuditEvent {
    Open { uid: u32, path: String, perms: String },
    Execve { uid: u32, path: String },
    Setuid { uid: u32, new_uid: u32 },
    Sudo { user: String, command: String, success: bool },
}

/// In-memory buffer for audit events before VFS flush.
static AUDIT_BUF: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Has the `/var/log` directory been ensured to exist?
static AUDIT_INIT: Mutex<bool> = Mutex::new(false);

pub fn audit_log(event: AuditEvent) {
    let time = crate::drivers::interrupts::ticks();
    let msg = match event {
        AuditEvent::Open { uid, path, perms } => {
            format!(
                "{{\"event\":\"open\",\"uid\":{},\"path\":\"{}\",\"perms\":\"{}\",\"time\":{}}}\n",
                uid, path, perms, time
            )
        }
        AuditEvent::Execve { uid, path } => {
            format!(
                "{{\"event\":\"execve\",\"uid\":{},\"path\":\"{}\",\"time\":{}}}\n",
                uid, path, time
            )
        }
        AuditEvent::Setuid { uid, new_uid } => {
            format!(
                "{{\"event\":\"setuid\",\"uid\":{},\"new_uid\":{},\"time\":{}}}\n",
                uid, new_uid, time
            )
        }
        AuditEvent::Sudo { user, command, success } => {
            format!(
                "{{\"event\":\"sudo\",\"user\":\"{}\",\"command\":\"{}\",\"success\":{},\"time\":{}}}\n",
                user, command, success, time
            )
        }
    };

    {
        let mut buf = AUDIT_BUF.lock();
        buf.extend_from_slice(msg.as_bytes());
    }

    flush_audit_log();
}

/// Ensure `/var/log` exists and flush buffered events to `/var/log/audit.log`.
fn flush_audit_log() {
    // If VFS is not yet initialized, keep events in the buffer.
    if !crate::fs::vfs::is_vfs_initialized() {
        return;
    }

    // Ensure directories exist (lazy init).
    {
        let mut init = AUDIT_INIT.lock();
        if !*init {
            let _ = with_vfs(|vfs| vfs.mkdir("/var"));
            let _ = with_vfs(|vfs| vfs.mkdir("/var/log"));
            *init = true;
        }
    }

    let data = {
        let mut buf = AUDIT_BUF.lock();
        if buf.is_empty() {
            return;
        }
        let v = buf.clone();
        buf.clear();
        v
    };

    // Try to append to existing file, otherwise create it.
    match with_vfs(|vfs| vfs.lookup_follow("/var/log/audit.log")) {
        Ok(handle) => {
            // Get current size to append.
            let size = with_vfs(|vfs| vfs.stat(handle))
                .map(|n| n.size)
                .unwrap_or(0);
            let _ = with_vfs(|vfs| vfs.write(handle, &data, size));
        }
        Err(_) => {
            match with_vfs(|vfs| vfs.create("/var/log/audit.log")) {
                Ok(handle) => {
                    let _ = with_vfs(|vfs| vfs.write(handle, &data, 0));
                }
                Err(_) => {
                    // Re-buffer for next attempt.
                    let mut buf = AUDIT_BUF.lock();
                    buf.extend_from_slice(&data);
                }
            }
        }
    }
}

/// Initialize the audit log subsystem. Safe to call before VFS is ready.
pub fn init_audit_log() {
    // Directories are created lazily on first flush.
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::test_assert;

    fn test_audit_event_format() -> TestResult {
        let msg = format!(
            "{{\"event\":\"open\",\"uid\":{},\"path\":\"{}\",\"perms\":\"{}\",\"time\":{}}}\n",
            0, "/etc/shadow", "r", 12345
        );
        test_assert!(msg.contains("\"event\":\"open\""));
        test_assert!(msg.contains("\"path\":\"/etc/shadow\""));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("security::audit_format", test_audit_event_format);
    }
}
