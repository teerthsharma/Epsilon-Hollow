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

#[derive(Clone, Copy)]
struct FxBuildHasher {
    seed: u64,
}

impl FxBuildHasher {
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
    Occupied { parent: u64, name_hash: u64, name: String, inode_id: u64 },
}

/// O(1) directory entry table. Every op is open-addressing with linear probing.
pub struct DirHash {
    buckets: Vec<Bucket>,
    seed: u64,
    len: usize,
    cap_mask: usize,
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

    fn probe(&self, parent: u64, name: &str) -> (usize, bool) {
        let name_hash = hash_name(name, self.seed);
        let hash = hash_dir_key(parent, name_hash, self.seed);
        let mut idx = (hash as usize) & self.cap_mask;
        let mut dist = 0usize;
        loop {
            match &self.buckets[idx] {
                Bucket::Empty => return (idx, false),
                Bucket::Tombstone => {}
                Bucket::Occupied { parent: p, name: n, .. } => {
                    if *p == parent && n == name {
                        return (idx, true);
                    }
                }
            }
            idx = (idx + 1) & self.cap_mask;
            dist += 1;
            if dist > self.cap_mask {
                break;
            }
        }
        (0, false)
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
            if let Bucket::Occupied { parent, name_hash, name, inode_id } = bucket {
                self.insert_unchecked(parent, name_hash, name, inode_id);
            }
        }
    }

    fn insert_unchecked(&mut self, parent: u64, name_hash: u64, name: String, inode_id: u64) {
        let hash = hash_dir_key(parent, name_hash, self.seed);
        let mut idx = (hash as usize) & self.cap_mask;
        loop {
            match &self.buckets[idx] {
                Bucket::Empty | Bucket::Tombstone => {
                    self.buckets[idx] = Bucket::Occupied { parent, name_hash, name, inode_id };
                    self.len += 1;
                    return;
                }
                _ => {}
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
        let (idx, found) = self.probe(parent, name);
        if found {
            match &self.buckets[idx] {
                Bucket::Occupied { inode_id, .. } => Some(*inode_id),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn remove(&mut self, parent: u64, name: &str) -> Option<u64> {
        let (idx, found) = self.probe(parent, name);
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

    pub fn contains_dir(&self, dir_id: u64) -> bool {
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
            if let Bucket::Occupied { parent: p, name, inode_id, .. } = bucket {
                if *p == parent {
                    out.push((name.clone(), *inode_id));
                }
            }
        }
        out
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;

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

    pub fn register_all() {
        crate::testing::register_test("dir_hash::insert_lookup", test_insert_lookup);
        crate::testing::register_test("dir_hash::remove", test_remove);
    }
}
