// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! /etc/group — group database with topological member lookups.
//!
//! Format: `groupname:gid:member1,member2,...`

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::fs::vfs::with_vfs;

#[derive(Debug, Clone)]
pub struct GroupEntry {
    pub groupname: String,
    pub gid: u32,
    pub members: Vec<String>,
}

fn read_group_file() -> Vec<u8> {
    with_vfs(|vfs| {
        let handle = vfs.lookup_follow("/etc/group").ok()?;
        let size = vfs.stat(handle).map(|n| n.size).unwrap_or(0) as usize;
        let mut buf = alloc::vec![0u8; size];
        vfs.read(handle, &mut buf, 0).ok()?;
        Some(buf)
    })
    .unwrap_or_default()
}

fn parse_group(data: &[u8]) -> Vec<GroupEntry> {
    let mut entries = Vec::new();
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&[u8]> = line.split(|&b| b == b':').collect();
        if parts.len() != 3 {
            continue;
        }
        let groupname = String::from_utf8_lossy(parts[0]).into_owned();
        let gid = core::str::from_utf8(parts[1])
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let members_str = core::str::from_utf8(parts[2]).unwrap_or("");
        let members: Vec<String> = if members_str.is_empty() {
            Vec::new()
        } else {
            members_str.split(',').map(String::from).collect()
        };
        entries.push(GroupEntry {
            groupname,
            gid,
            members,
        });
    }
    entries
}

/// Look up a group by name.
pub fn getgrnam(name: &str) -> Option<GroupEntry> {
    let data = read_group_file();
    let entries = parse_group(&data);
    entries.into_iter().find(|e| e.groupname == name)
}

/// Look up a group by GID.
pub fn getgrgid(gid: u32) -> Option<GroupEntry> {
    let data = read_group_file();
    let entries = parse_group(&data);
    entries.into_iter().find(|e| e.gid == gid)
}

/// Return all GIDs of which `username` is a member.
pub fn groups_for_user(username: &str) -> Vec<u32> {
    let data = read_group_file();
    let entries = parse_group(&data);
    let mut groups = Vec::new();
    for entry in entries {
        if entry.members.iter().any(|m| m == username) {
            groups.push(entry.gid);
        }
    }
    groups
}

/// Return all GIDs of which the user with the given UID is a member.
/// (Look-up goes via /etc/passwd to resolve username first.)
pub fn groups_for_uid(uid: u32) -> Vec<u32> {
    if let Some(user) = crate::security::passwd::get_user_by_uid(uid) {
        groups_for_user(&user.username)
    } else {
        Vec::new()
    }
}

/// Create a default /etc/group if none exists.
pub fn init_group() {
    let exists = with_vfs(|vfs| vfs.lookup_follow("/etc/group")).is_ok();
    if exists {
        return;
    }
    let _ = with_vfs(|vfs| vfs.mkdir("/etc"));
    let line = "seal:1000:seal\n";
    with_vfs(|vfs| {
        let handle = vfs.create("/etc/group").ok()?;
        vfs.write(handle, line.as_bytes(), 0).ok()?;
        Some(())
    });
}
