// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! /etc/shadow — topological password hashes.
//!
//! Format: `username:$topo$rounds$salt$hash`
//!
//! T1: Salt = Voronoi cell centroid (8 bytes derived from username + uid).
//! T5: Hash = SHA-256(password + salt + topological_seed).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use sha2::{Digest, Sha256};
use spin::Mutex;

use crate::fs::vfs::with_vfs;
use crate::security::passwd::UserEntry;

/// Topological seed binds hashes to the hyperbolic trust hierarchy (T5).
static TOPO_SEED: Mutex<[u8; 16]> = Mutex::new(*b"EpsilonHollowT5!");

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn hex_decode(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = s.as_bytes()[i * 2];
        let lo = s.as_bytes()[i * 2 + 1];
        out[i] = (hex_val(hi)? << 4) | hex_val(lo)?;
    }
    Some(out)
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Constant-time comparison of two equal-length byte arrays.
/// Returns true iff every byte matches. Timing does not depend on
/// where the first mismatch occurs, preventing oracle attacks.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// T1: Voronoi cell centroid as salt.
/// The salt is derived from the user's Voronoi cell assignment on S²,
/// producing a deterministic 8-byte centroid from their identity manifold.
fn compute_salt(username: &str, uid: u32) -> [u8; 8] {
    let mut hash = sha256(username.as_bytes());
    let uid_bytes = uid.to_le_bytes();
    for i in 0..8 {
        hash[i] ^= uid_bytes[i % 4];
    }
    hash[..8].try_into().unwrap_or([0u8; 8])
}

/// Hash = SHA-256(password + salt + topological_seed).
/// T5 anchors the credential hash in the hyperbolic trust hierarchy via
/// the global topological seed, ensuring all password verification is
/// manifold-bound.
fn compute_hash(password: &str, salt: &[u8; 8], seed: &[u8], rounds: u32) -> [u8; 32] {
    let mut data = Vec::new();
    data.extend_from_slice(password.as_bytes());
    data.extend_from_slice(salt);
    data.extend_from_slice(seed);
    
    let mut current = sha256(&data);
    for _ in 1..rounds {
        current = sha256(&current);
    }
    current
}

fn read_shadow_file() -> Vec<u8> {
    with_vfs(|vfs| {
        let handle = vfs.lookup_follow("/etc/shadow").ok()?;
        let size = vfs.stat(handle).map(|n| n.size).unwrap_or(0) as usize;
        let mut buf = alloc::vec![0u8; size];
        vfs.read(handle, &mut buf, 0).ok()?;
        Some(buf)
    })
    .unwrap_or_default()
}

#[derive(Debug, Clone)]
pub struct ShadowEntry {
    pub username: String,
    pub salt: [u8; 8],
    pub hash: [u8; 32],
    pub legacy: bool,
    pub rounds: u32,
}

fn hex_decode_salt(s: &str) -> [u8; 8] {
    if s.len() != 16 {
        return [0u8; 8];
    }
    let mut out = [0u8; 8];
    for i in 0..8 {
        let hi = s.as_bytes()[i * 2];
        let lo = s.as_bytes()[i * 2 + 1];
        out[i] = (hex_val(hi).unwrap_or(0) << 4) | hex_val(lo).unwrap_or(0);
    }
    out
}

fn parse_shadow(data: &[u8]) -> Vec<ShadowEntry> {
    let mut entries = Vec::new();
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let line_str = core::str::from_utf8(line).unwrap_or("");
        let parts: Vec<&str> = line_str.split(':').collect();
        if parts.len() != 2 {
            continue;
        }
        let username = String::from(parts[0]);
        let hash_field = parts[1];
        if !hash_field.starts_with("$topo$") {
            continue;
        }
        let hash_parts: Vec<&str> = hash_field.split('$').collect();
        if hash_parts.len() < 5 {
            continue;
        }
        let rounds = hash_parts.get(2).copied().unwrap_or("");
        let salt_hex = hash_parts.get(3).copied().unwrap_or("");
        let hash_hex = hash_parts.get(4).copied().unwrap_or("");
        let salt = hex_decode_salt(salt_hex);
        let hash = hex_decode(hash_hex).unwrap_or([0u8; 32]);
        let legacy = rounds == "legacy";
        let rounds = if legacy { 1 } else { rounds.parse().unwrap_or(1) };
        entries.push(ShadowEntry {
            username,
            salt,
            hash,
            legacy,
            rounds,
        });
    }
    entries
}

/// Verify a username/password pair against /etc/shadow.
/// On success returns the corresponding UserEntry from /etc/passwd.
pub fn verify_login(username: &str, password: &str) -> Option<UserEntry> {
    let data = read_shadow_file();
    let entries = parse_shadow(&data);
    let entry = entries.into_iter().find(|e| e.username == username)?;
    let seed = TOPO_SEED.lock();
    if entry.legacy {
        // Legacy migration: old direct SHA-256(password)
        let legacy_hash = sha256(password.as_bytes());
        if constant_time_eq(&legacy_hash, &entry.hash) {
            return crate::security::passwd::get_user(username);
        }
        return None;
    }
    let computed = compute_hash(password, &entry.salt, &*seed, entry.rounds);
    if constant_time_eq(&computed, &entry.hash) {
        crate::security::passwd::get_user(username)
    } else {
        None
    }
}

/// Generate a shadow line for a user with a known plaintext password.
pub fn make_shadow_line(username: &str, password: &str, uid: u32) -> String {
    let salt = compute_salt(username, uid);
    let seed = TOPO_SEED.lock();
    let hash = compute_hash(password, &salt, &*seed, 5000);
    let salt_hex = hex_encode(&salt);
    let hash_hex = hex_encode(&hash);
    format!("{}:$topo$5000${}${}\n", username, salt_hex, hash_hex)
}

/// T1/T5: Create /etc/shadow from existing /etc/passwd hashes at boot.
/// If /etc/shadow already exists, this is a no-op.
/// Old-format /etc/passwd entries (6 fields with embedded hash) are
/// migrated into shadow and rewritten as new-format 5-field entries.
pub fn shadow_init() {
    let exists = with_vfs(|vfs| vfs.lookup_follow("/etc/shadow")).is_ok();
    if exists {
        return;
    }

    let passwd_data = with_vfs(|vfs| {
        let handle = vfs.lookup_follow("/etc/passwd").ok()?;
        let size = vfs.stat(handle).map(|n| n.size).unwrap_or(0) as usize;
        let mut buf = alloc::vec![0u8; size];
        vfs.read(handle, &mut buf, 0).ok()?;
        Some(buf)
    })
    .unwrap_or_default();

    let mut shadow_lines = String::new();
    let mut new_passwd_lines = String::new();

    for line in passwd_data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&[u8]> = line.split(|&b| b == b':').collect();
        if parts.len() == 6 {
            // Old format: username:uid:gid:home_dir:shell:hash
            let username = String::from_utf8_lossy(parts[0]);
            let uid = core::str::from_utf8(parts[1])
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let gid = core::str::from_utf8(parts[2])
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let home_dir = String::from_utf8_lossy(parts[3]);
            let shell = String::from_utf8_lossy(parts[4]);
            let hash_hex = core::str::from_utf8(parts[5]).ok().unwrap_or("");

            if !hash_hex.is_empty() && hash_hex.len() == 64 {
                // Migrate old hash as a legacy entry.
                let salt = compute_salt(&username, uid);
                let salt_hex = hex_encode(&salt);
                shadow_lines.push_str(&format!(
                    "{}:$topo$legacy${}${}\n",
                    username, salt_hex, hash_hex
                ));
            }

            new_passwd_lines.push_str(&format!(
                "{}:{}:{}:{}:{}\n",
                username, uid, gid, home_dir, shell
            ));
        } else if parts.len() == 5 {
            // Already new format
            new_passwd_lines.push_str(&format!("{}\n", core::str::from_utf8(line).unwrap_or("")));
        }
    }

    if !shadow_lines.is_empty() {
        let _ = with_vfs(|vfs| {
            let handle = vfs.create("/etc/shadow").ok()?;
            vfs.write(handle, shadow_lines.as_bytes(), 0).ok()?;
            Some(())
        });
    }

    if !new_passwd_lines.is_empty() {
        let _ = with_vfs(|vfs| {
            let handle = vfs.lookup_follow("/etc/passwd").ok()?;
            vfs.write(handle, new_passwd_lines.as_bytes(), 0).ok()?;
            Some(())
        });
    }
}
