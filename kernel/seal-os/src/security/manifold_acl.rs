// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological permission checking driven by T1–T5 theorems.
//!
//! T1: File permissions mapped to Voronoi cells on S².
//!      Owner / group / other = three spherical cells.
//! T2: Spectral anomaly detection on access patterns.
//! T3: Entropy of permission patterns.
//! T4: Governor ε adapts MAC strictness after anomalies.
//! T5: Hyperbolic trust hierarchy — distance from root = privilege level.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use sha2::{Digest, Sha256};
use spin::Mutex;

use crate::fs::vfs::VfsNode;

pub const PERM_READ: u16 = 0o4;
pub const PERM_WRITE: u16 = 0o2;
pub const PERM_EXEC: u16 = 0o1;

/// T4: When an anomaly is detected, non-owner access is denied for
/// this many ticks.
const ANOMALY_WINDOW: u64 = 100;

/// T4: Tick count at which the last anomaly was recorded.
static ANOMALY_TICK: AtomicU64 = AtomicU64::new(0);

/// T2: Recent access history for spectral anomaly detection.
const HISTORY_SIZE: usize = 16;

struct AccessRecord {
    path_hash: u64,
    tick_bucket: u64,
}

static ACCESS_HISTORY: Mutex<Vec<AccessRecord>> = Mutex::new(Vec::new());

/// Access decision returned by the topological checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessDecision {
    Allow,
    Deny,
}

/// T1: Map the subject to one of three Voronoi cells (owner / group / other)
/// on the S² permission manifold and return the applicable 3-bit permission
/// set for that cell.
fn t1_permission_bits(uid: u32, gid: u32, groups: &[u32], node: &VfsNode) -> u16 {
    if uid == node.uid {
        // T1: Owner cell — innermost privilege region.
        (node.mode >> 6) & 0o7
    } else if gid == node.gid || groups.contains(&node.gid) {
        // T1: Group cell — intermediate ring.
        (node.mode >> 3) & 0o7
    } else {
        // T1: Other cell — outer boundary.
        node.mode & 0o7
    }
}

/// Novelty check: true when neither this path nor this 1000-tick bucket appears
/// in the last 16 recorded accesses.
///
/// Despite the T2 label this is not spectral — it is a linear scan of a 16-entry
/// ring buffer comparing a SHA-256 path hash and a tick bucket. There is no
/// Laplacian, eigendecomposition, or contraction operator involved.
fn t2_detect_anomaly(path: &str, _uid: u32) -> bool {
    let current_tick = crate::drivers::interrupts::ticks();
    let tick_bucket = current_tick / 1000;
    let path_hash = hash_path(path);

    let mut history = ACCESS_HISTORY.lock();

    let normal = history
        .iter()
        .any(|r| r.path_hash == path_hash || r.tick_bucket == tick_bucket);

    history.push(AccessRecord {
        path_hash,
        tick_bucket,
    });
    if history.len() > HISTORY_SIZE {
        history.remove(0);
    }

    !normal
}

/// The first eight bytes of SHA-256 of `path`.
///
/// The multiply-add hash this replaces is a polynomial mod 2^64, which maps a
/// 1024-byte Thue–Morse word and its complement to one value; FNV-1a does the
/// same from 2048 bytes, since its low bits see only the inputs' low bits. A
/// new path that collides with a recorded one reads as seen.
fn hash_path(path: &str) -> u64 {
    let digest = Sha256::digest(path.as_bytes());
    u64::from_le_bytes(core::array::from_fn(|i| digest[i]))
}

/// T3: Shannon entropy of the 12 mode bits.
/// Low entropy  → rigid, predictable security posture.
/// High entropy → potential misconfiguration (many permission transitions).
fn t3_permission_entropy(mode: u16) -> f64 {
    let bits = mode & 0o777;
    let mut ones = 0u8;
    let mut zeros = 0u8;
    for i in 0..12 {
        if ((bits >> i) & 1) != 0 {
            ones += 1;
        } else {
            zeros += 1;
        }
    }
    let total = ones + zeros;
    if total == 0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &c in &[ones, zeros] {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total as f64;
        entropy -= p * libm::log2(p);
    }
    entropy
}

/// T4: Governor ε adapts MAC strictness.
/// After an anomaly is detected, deny all non-owner access for a fixed
/// window.  Also, if the scheduler's geometric governor reports a high
/// epsilon (manifold curvature spike), restrict access.
fn t4_epsilon_strictness(uid: u32, node_uid: u32) -> AccessDecision {
    let current_tick = crate::drivers::interrupts::ticks();
    let anomaly_tick = ANOMALY_TICK.load(Ordering::Relaxed);

    if anomaly_tick != 0
        && current_tick.saturating_sub(anomaly_tick) < ANOMALY_WINDOW
        && uid != node_uid
    {
        return AccessDecision::Deny;
    }

    let eps = crate::process::scheduler::governor_epsilon();
    if eps > 2.0 && uid != node_uid {
        return AccessDecision::Deny;
    }

    AccessDecision::Allow
}

/// T5: Hyperbolic distance from the origin (root).
/// Root (uid = 0) sits at distance 0.  Regular users live on the boundary
/// with distance growing logarithmically with uid.
fn t5_hyperbolic_distance(uid: u32) -> f64 {
    if uid == 0 {
        0.0
    } else {
        libm::log(1.0 + uid as f64)
    }
}

/// T5: Check whether the user's hyperbolic position permits access.
/// Sensitive files (root-owned with restricted mode) are protected from
/// users too far from the origin.
fn t5_check_distance(uid: u32, node: &VfsNode) -> AccessDecision {
    let distance = t5_hyperbolic_distance(uid);
    let max_distance = libm::log(1.0 + 65535.0);

    // Heuristic: root-owned files that are not world-readable/writable
    // are treated as sensitive and guarded by the hyperbolic boundary.
    let is_sensitive = node.uid == 0 && (node.mode & 0o077) != 0o077;

    if is_sensitive && distance > max_distance * 0.3 && uid != node.uid {
        return AccessDecision::Deny;
    }

    AccessDecision::Allow
}

/// Topological access check integrating T1–T5.
///
/// * `uid` / `gid` — real identity of the requesting task.
/// * `groups` — supplementary group list.
/// * `node` — target VFS node (mode, uid, gid).
/// * `requested` — permission bits being requested (PERM_READ, PERM_WRITE, PERM_EXEC).
/// * `path` — path string for T2 anomaly detection.
pub fn check_access(
    uid: u32,
    gid: u32,
    groups: &[u32],
    node: &VfsNode,
    requested: u16,
    path: &str,
) -> AccessDecision {
    // T4: Strict-mode first — deny if anomaly window is active.
    if t4_epsilon_strictness(uid, node.uid) == AccessDecision::Deny {
        return AccessDecision::Deny;
    }

    // T5: Hyperbolic trust hierarchy.
    if t5_check_distance(uid, node) == AccessDecision::Deny {
        return AccessDecision::Deny;
    }

    // T1: Voronoi cell permission mapping.
    let perm_bits = t1_permission_bits(uid, gid, groups, node);
    if perm_bits & requested != requested {
        return AccessDecision::Deny;
    }

    // T2: Spectral anomaly detection.
    if t2_detect_anomaly(path, uid) {
        let current_tick = crate::drivers::interrupts::ticks();
        ANOMALY_TICK.store(current_tick, Ordering::Relaxed);
    }

    // T3: Entropy check (informational — does not block, but high entropy
    // indicates a potentially misconfigured permission manifold).
    let _entropy = t3_permission_entropy(node.mode);

    AccessDecision::Allow
}

/// Manually trigger anomaly mode (for security events or testing).
pub fn trigger_anomaly() {
    let current_tick = crate::drivers::interrupts::ticks();
    ANOMALY_TICK.store(current_tick, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Tests -- run by the in-kernel harness (crate::testing), not `cargo test`.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::test_assert;
    use crate::testing::TestResult;
    use alloc::string::String;

    /// `/` then the first `n` Thue–Morse symbols over `a`/`b`, and the same
    /// path with every symbol complemented.
    fn thue_morse_paths(n: usize) -> (String, String) {
        let symbol = |i: usize, flip: u32| {
            if (i.count_ones() + flip) % 2 == 0 {
                'a'
            } else {
                'b'
            }
        };
        let path = |flip| {
            core::iter::once('/')
                .chain((0..n).map(|i| symbol(i, flip)))
                .collect()
        };
        (path(0), path(1))
    }

    /// Distinct Thue–Morse paths hash apart. A multiply-add hash mod 2^64
    /// maps the pair to one value from 1024 symbols on, FNV-1a from 2048.
    fn test_thue_morse_paths_hash_apart() -> TestResult {
        for n in [1024, 2048, 4096] {
            let (t, c) = thue_morse_paths(n);
            crate::serial_println!(
                "[manifold_acl] n={} hash_path {:#018x} vs {:#018x}",
                n,
                hash_path(&t),
                hash_path(&c)
            );
            test_assert!(
                hash_path(&t) != hash_path(&c),
                "two distinct Thue-Morse paths share a hash"
            );
        }
        TestResult::Pass
    }

    /// A new path is anomalous even when the history holds its Thue–Morse
    /// complement. The record's tick bucket is one no tick reaches, so only a
    /// path-hash match could make the new path read as seen.
    fn test_new_path_is_not_read_as_seen() -> TestResult {
        let (seen, new) = thue_morse_paths(2048);
        let saved = core::mem::take(&mut *ACCESS_HISTORY.lock());
        ACCESS_HISTORY.lock().push(AccessRecord {
            path_hash: hash_path(&seen),
            tick_bucket: u64::MAX,
        });
        let anomalous = t2_detect_anomaly(&new, 0);
        *ACCESS_HISTORY.lock() = saved;
        test_assert!(
            anomalous,
            "a new path matched a recorded path's hash and skipped the anomaly check"
        );
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "security::manifold_acl::thue_morse_paths_hash_apart",
            test_thue_morse_paths_hash_apart,
        );
        crate::testing::register_test(
            "security::manifold_acl::new_path_is_not_read_as_seen",
            test_new_path_is_not_read_as_seen,
        );
    }
}
