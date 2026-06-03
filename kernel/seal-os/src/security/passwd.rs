// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Password authentication via /etc/passwd.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

use crate::fs::vfs::with_vfs;

#[derive(Debug, Clone)]
pub struct UserEntry {
    pub username: String,
    pub uid: u32,
    pub gid: u32,
    pub home_dir: String,
    pub shell: String,
}

/// Boot-time authenticated user identity (set before scheduler is running).
/// Replaces the old global CURRENT_USER with a transient bootstrap record.
static BOOT_USER: Mutex<Option<(u32, u32)>> = Mutex::new(None);

pub fn set_boot_user(uid: u32, gid: u32) {
    *BOOT_USER.lock() = Some((uid, gid));
}

pub fn boot_uid() -> Option<u32> {
    BOOT_USER.lock().map(|(uid, _)| uid)
}

pub fn boot_gid() -> Option<u32> {
    BOOT_USER.lock().map(|(_, gid)| gid)
}

/// Set the authenticated user for the current task (and boot record).
pub fn set_current_user(user: UserEntry) {
    set_boot_user(user.uid, user.gid);
    crate::process::scheduler::set_current_uid(user.uid);
    crate::process::scheduler::set_current_gid(user.gid);
}

/// Get the current user from the running task, falling back to boot user.
pub fn get_current_user() -> Option<UserEntry> {
    let uid = crate::process::scheduler::current_uid();
    if uid != 0 {
        return get_user_by_uid(uid);
    }
    boot_uid().and_then(get_user_by_uid)
}

/// Re-export shadow-based login verification.
pub use super::shadow::verify_login;

fn read_passwd_file() -> Vec<u8> {
    with_vfs(|vfs| {
        let handle = vfs.lookup_follow("/etc/passwd").ok()?;
        let size = vfs.stat(handle).map(|n| n.size).unwrap_or(0) as usize;
        let mut buf = alloc::vec![0u8; size];
        vfs.read(handle, &mut buf, 0).ok()?;
        Some(buf)
    })
    .unwrap_or_default()
}

fn parse_passwd(data: &[u8]) -> Vec<UserEntry> {
    let mut entries = Vec::new();
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&[u8]> = line.split(|&b| b == b':').collect();
        // Support both old 6-field format (with hash) and new 5-field format.
        if parts.len() != 5 && parts.len() != 6 {
            continue;
        }
        let username = String::from_utf8_lossy(parts[0]).into_owned();
        let uid = core::str::from_utf8(parts[1])
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let gid = core::str::from_utf8(parts[2])
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let home_dir = String::from_utf8_lossy(parts[3]).into_owned();
        let shell = String::from_utf8_lossy(parts[4]).into_owned();
        entries.push(UserEntry {
            username,
            uid,
            gid,
            home_dir,
            shell,
        });
    }
    entries
}

const LOCKED_BOOTSTRAP_PASSWORD: &str = "__seal_locked_bootstrap_password__";

/// Create /etc/passwd and /etc/shadow with a locked bootstrap user if absent.
/// The historical `seal`/`seal` credential is deliberately not generated.
pub fn init_passwd() {
    let exists = with_vfs(|vfs| vfs.lookup_follow("/etc/passwd")).is_ok();
    if exists {
        return;
    }
    let _ = with_vfs(|vfs| vfs.mkdir("/etc"));
    let line = String::from("seal:1000:1000:/home/seal:/bin/shell\n");
    with_vfs(|vfs| {
        let handle = vfs.create("/etc/passwd").ok()?;
        vfs.write(handle, line.as_bytes(), 0).ok()?;
        Some(())
    });
    let shadow_line =
        crate::security::shadow::make_shadow_line("seal", LOCKED_BOOTSTRAP_PASSWORD, 1000);
    with_vfs(|vfs| {
        let handle = vfs.create("/etc/shadow").ok()?;
        vfs.write(handle, shadow_line.as_bytes(), 0).ok()?;
        Some(())
    });
}

/// Verify a username/password pair against /etc/passwd.
pub fn get_user_by_uid(uid: u32) -> Option<UserEntry> {
    let data = read_passwd_file();
    let entries = parse_passwd(&data);
    entries.into_iter().find(|e| e.uid == uid)
}

/// Look up a user by username.
pub fn get_user(username: &str) -> Option<UserEntry> {
    let data = read_passwd_file();
    let entries = parse_passwd(&data);
    entries.into_iter().find(|e| e.username == username)
}

/// Append a new user to /etc/passwd.
pub fn add_user(username: &str, password: &str, uid: u32, gid: u32) {
    let home_dir = format!("/home/{}", username);
    let shell = String::from("/bin/shell");
    let line = format!("{}:{}:{}:{}:{}\n", username, uid, gid, home_dir, shell);

    with_vfs(|vfs| {
        let handle = vfs.lookup_follow("/etc/passwd").ok()?;
        let size = vfs.stat(handle).map(|n| n.size).unwrap_or(0);
        vfs.write(handle, line.as_bytes(), size).ok()?;
        Some(())
    });

    // Also append to /etc/shadow with topological hash.
    let shadow_line = crate::security::shadow::make_shadow_line(username, password, uid);
    with_vfs(|vfs| {
        let handle = vfs.lookup_follow("/etc/shadow").ok()?;
        let size = vfs.stat(handle).map(|n| n.size).unwrap_or(0);
        vfs.write(handle, shadow_line.as_bytes(), size).ok()?;
        Some(())
    });
}
