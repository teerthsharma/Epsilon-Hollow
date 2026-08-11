// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! O(1) directory entry hash table — open addressing with FxHash-style hasher.

use alloc::string::String;
use alloc::vec::Vec;
use core::hash::{BuildHasher, Hasher};

/// Simple FxHash-style 64-bit hasher with per-table seed.
struct FxHasher64 {
    state: u64,
}

impl FxHasher64 {
    const K: u64 = 0x517cc1b727220a95;
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Hasher for FxHasher64 {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state = self.state.wrapping_mul(Self::K).wrapping_add(b as u64);
        }
    }
    fn finish(&self) -> u64 {
        self.state
    }
}

#[allow(dead_code)] // REASON: FxHash hasher reserved for future O(1) directory entry lookup
#[derive(Clone, Copy)]
struct FxBuildHasher {
    seed: u64,
}

impl FxBuildHasher {
    #[allow(dead_code)] // REASON: constructor for future seeded hasher instantiation
    fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher64;
    fn build_hasher(&self) -> FxHasher64 {
        FxHasher64::new(self.seed)
    }
}

fn hash_name(name: &str, seed: u64) -> u64 {
    let mut h = FxHasher64::new(seed);
    h.write(name.as_bytes());
    h.finish()
}

fn hash_dir_key(parent: u64, name_hash: u64, seed: u64) -> u64 {
    let mut h = FxHasher64::new(seed);
    h.write(&parent.to_le_bytes());
    h.write(&name_hash.to_le_bytes());
    h.finish()
}

enum Bucket {
    Empty,
    Tombstone,
    Occupied {
        parent: u64,
        name_hash: u64,
        name: String,
        inode_id: u64,
    },
}

/// Directory entry table with average O(1) lookup/insert/remove under its load target.
/// Growth and directory enumeration are linear in table capacity.
pub struct DirHash {
    buckets: Vec<Bucket>,
    seed: u64,
    len: usize,
    cap_mask: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DirLookupProbe {
    pub probes: usize,
    pub probe_bound: usize,
    pub found: bool,
}

impl DirHash {
    pub fn new(seed: u64) -> Self {
        let cap = 16usize;
        let mut buckets = Vec::with_capacity(cap);
        for _ in 0..cap {
            buckets.push(Bucket::Empty);
        }
        Self {
            buckets,
            seed,
            len: 0,
            cap_mask: cap - 1,
        }
    }

    fn probe(&self, parent: u64, name: &str) -> (usize, bool, usize) {
        let name_hash = hash_name(name, self.seed);
        let hash = hash_dir_key(parent, name_hash, self.seed);
        let mut idx = (hash as usize) & self.cap_mask;
        let mut dist = 0usize;
        loop {
            let probes = dist + 1;
            match &self.buckets[idx] {
                Bucket::Empty => return (idx, false, probes),
                Bucket::Tombstone => {}
                Bucket::Occupied {
                    parent: p, name: n, ..
                } => {
                    if *p == parent && n == name {
                        return (idx, true, probes);
                    }
                }
            }
            idx = (idx + 1) & self.cap_mask;
            dist += 1;
            if dist > self.cap_mask {
                break;
            }
        }
        (0, false, self.buckets.len())
    }

    fn grow(&mut self) {
        let new_cap = self.buckets.len() * 2;
        let mut new_buckets = Vec::with_capacity(new_cap);
        for _ in 0..new_cap {
            new_buckets.push(Bucket::Empty);
        }
        let old = core::mem::replace(&mut self.buckets, new_buckets);
        self.cap_mask = new_cap - 1;
        self.len = 0;
        for bucket in old {
            if let Bucket::Occupied {
                parent,
                name_hash,
                name,
                inode_id,
            } = bucket
            {
                self.insert_unchecked(parent, name_hash, name, inode_id);
            }
        }
    }

    fn insert_unchecked(&mut self, parent: u64, name_hash: u64, name: String, inode_id: u64) {
        let hash = hash_dir_key(parent, name_hash, self.seed);
        let mut idx = (hash as usize) & self.cap_mask;
        // Remember the first reusable (tombstoned) slot seen along the probe
        // chain, but keep scanning past it: if the same (parent, name) key
        // is already occupied further along the chain, that entry must be
        // overwritten in place rather than shadowed by a second insert. A
        // blind first-fit here would double-count `len` and leave the old
        // entry as a stale duplicate reachable by `probe`/`entries_in_dir`.
        let mut first_reusable: Option<usize> = None;
        loop {
            match &self.buckets[idx] {
                Bucket::Empty => {
                    let target = first_reusable.unwrap_or(idx);
                    self.buckets[target] = Bucket::Occupied {
                        parent,
                        name_hash,
                        name,
                        inode_id,
                    };
                    self.len += 1;
                    return;
                }
                Bucket::Tombstone => {
                    if first_reusable.is_none() {
                        first_reusable = Some(idx);
                    }
                }
                Bucket::Occupied { parent: p, name: n, .. } if *p == parent && *n == name => {
                    // Same key already present: replace its value in place.
                    // `len` is unchanged and no duplicate slot is created.
                    self.buckets[idx] = Bucket::Occupied {
                        parent,
                        name_hash,
                        name,
                        inode_id,
                    };
                    return;
                }
                Bucket::Occupied { .. } => {
                    // Occupied bucket with hash collision; continue probing
                }
            }
            idx = (idx + 1) & self.cap_mask;
        }
    }

    pub fn insert(&mut self, parent: u64, name: &str, inode_id: u64) {
        if self.len * 2 > self.buckets.len() {
            self.grow();
        }
        let name_hash = hash_name(name, self.seed);
        self.insert_unchecked(parent, name_hash, String::from(name), inode_id);
    }

    pub fn lookup(&self, parent: u64, name: &str) -> Option<u64> {
        let (idx, found, _probes) = self.probe(parent, name);
        if found {
            match &self.buckets[idx] {
                Bucket::Occupied { inode_id, .. } => Some(*inode_id),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn lookup_with_probe(&self, parent: u64, name: &str) -> (Option<u64>, DirLookupProbe) {
        let (idx, found, probes) = self.probe(parent, name);
        let inode_id = if found {
            match &self.buckets[idx] {
                Bucket::Occupied { inode_id, .. } => Some(*inode_id),
                _ => None,
            }
        } else {
            None
        };
        (
            inode_id,
            DirLookupProbe {
                probes,
                probe_bound: self.buckets.len(),
                found,
            },
        )
    }

    pub fn remove(&mut self, parent: u64, name: &str) -> Option<u64> {
        let (idx, found, _probes) = self.probe(parent, name);
        if found {
            let old = core::mem::replace(&mut self.buckets[idx], Bucket::Tombstone);
            self.len -= 1;
            match old {
                Bucket::Occupied { inode_id, .. } => Some(inode_id),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn contains_dir(&self, _dir_id: u64) -> bool {
        // A directory exists if it has an entry in any bucket as a parent,
        // OR if it's the root (dir_id == 0). For root, we handle outside.
        // This is an approximation; the real check is via InodeSlab.
        true
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Collect all entries in a directory (for readdir fallback / testing).
    pub fn entries_in_dir(&self, parent: u64) -> Vec<(String, u64)> {
        let mut out = Vec::new();
        for bucket in &self.buckets {
            if let Bucket::Occupied {
                parent: p,
                name,
                inode_id,
                ..
            } = bucket
            {
                if *p == parent {
                    out.push((name.clone(), *inode_id));
                }
            }
        }
        out
    }

    /// Count tombstoned slots. Test-only: used to confirm that `insert`
    /// reuses a vacated slot rather than leaking it as permanently dead.
    #[cfg(any(test, feature = "test-mode"))]
    fn tombstone_count(&self) -> usize {
        self.buckets
            .iter()
            .filter(|b| matches!(b, Bucket::Tombstone))
            .count()
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_insert_lookup() -> TestResult {
        let mut dh = DirHash::new(0x1234);
        dh.insert(0, "hello.txt", 42);
        test_assert_eq!(dh.lookup(0, "hello.txt"), Some(42));
        test_assert_eq!(dh.lookup(0, "missing"), None);
        test_assert_eq!(dh.lookup(1, "hello.txt"), None);
        TestResult::Pass
    }

    fn test_remove() -> TestResult {
        let mut dh = DirHash::new(0x1234);
        dh.insert(0, "a", 1);
        dh.insert(0, "b", 2);
        test_assert_eq!(dh.remove(0, "a"), Some(1));
        test_assert_eq!(dh.lookup(0, "a"), None);
        test_assert_eq!(dh.lookup(0, "b"), Some(2));
        TestResult::Pass
    }

    /// Inserting an already-present (parent, name) key must overwrite the
    /// existing entry in place, not append a second occupied bucket.
    fn test_insert_duplicate_replaces() -> TestResult {
        let mut dh = DirHash::new(0x1234);
        dh.insert(0, "dup.txt", 1);
        test_assert_eq!(dh.len(), 1);
        dh.insert(0, "dup.txt", 2);
        test_assert_eq!(dh.len(), 1);
        test_assert_eq!(dh.lookup(0, "dup.txt"), Some(2));
        TestResult::Pass
    }

    /// A slot vacated by `remove` is reused by a later `insert` instead of
    /// being permanently leaked. Re-inserting the removed key lands its
    /// probe chain on that key's own former slot at distance 0, so the
    /// tombstone count returning to zero proves reuse rather than a fresh
    /// empty slot being consumed instead.
    fn test_insert_reuses_tombstone() -> TestResult {
        let mut dh = DirHash::new(0x1234);
        dh.insert(0, "a", 1);
        dh.insert(0, "b", 2);
        test_assert_eq!(dh.remove(0, "a"), Some(1));
        test_assert_eq!(dh.tombstone_count(), 1);
        dh.insert(0, "a", 99);
        test_assert_eq!(dh.tombstone_count(), 0);
        test_assert_eq!(dh.len(), 2);
        test_assert_eq!(dh.lookup(0, "a"), Some(99));
        test_assert_eq!(dh.lookup(0, "b"), Some(2));
        TestResult::Pass
    }

    /// Two distinct keys whose base hash bucket collides must each get
    /// their own slot via probing, rather than the second overwriting the
    /// first (which would only be correct for a true key match).
    fn test_insert_collision_gets_own_slot() -> TestResult {
        let dh = DirHash::new(0x1234);
        let mut found: Option<(String, String)> = None;
        'search: for i in 0..64u32 {
            for j in (i + 1)..64u32 {
                let a = alloc::format!("k{}", i);
                let b = alloc::format!("k{}", j);
                let ha = hash_name(&a, dh.seed);
                let hb = hash_name(&b, dh.seed);
                let ba = (hash_dir_key(0, ha, dh.seed) as usize) & dh.cap_mask;
                let bb = (hash_dir_key(0, hb, dh.seed) as usize) & dh.cap_mask;
                if ba == bb {
                    found = Some((a, b));
                    break 'search;
                }
            }
        }
        let (a, b) = match found {
            Some(pair) => pair,
            None => return TestResult::Fail("no colliding name pair found among 64 candidates"),
        };
        let mut dh = dh;
        dh.insert(0, &a, 10);
        dh.insert(0, &b, 20);
        test_assert_eq!(dh.len(), 2);
        test_assert_eq!(dh.lookup(0, &a), Some(10));
        test_assert_eq!(dh.lookup(0, &b), Some(20));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("dir_hash::insert_lookup", test_insert_lookup);
        crate::testing::register_test("dir_hash::remove", test_remove);
        crate::testing::register_test(
            "dir_hash::insert_duplicate_replaces",
            test_insert_duplicate_replaces,
        );
        crate::testing::register_test(
            "dir_hash::insert_reuses_tombstone",
            test_insert_reuses_tombstone,
        );
        crate::testing::register_test(
            "dir_hash::insert_collision_gets_own_slot",
            test_insert_collision_gets_own_slot,
        );
    }
}
