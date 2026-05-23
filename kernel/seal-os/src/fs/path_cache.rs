// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! O(1) hot-path resolution cache — CLOCK (second-chance) eviction.

use alloc::vec::Vec;

const DEFAULT_CAPACITY: usize = 8;

struct CacheEntry {
    path_hash: u64,
    inode_id: u64,
    gen_at_insert: u64,
    accessed: bool,
}

pub struct PathCache {
    entries: Vec<Option<CacheEntry>>,
    generation: u64,
    clock_hand: usize,
    capacity: usize,
    pub hits: u64,
    pub misses: u64,
}

fn hash_path(path: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in path.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl PathCache {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut entries = Vec::new();
        if capacity > 0 {
            entries.resize_with(capacity, || None);
        }
        Self {
            entries,
            generation: 1,
            clock_hand: 0,
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, path: &str) -> Option<u64> {
        if self.capacity == 0 || self.entries.is_empty() {
            self.misses += 1;
            return None;
        }
        let h = hash_path(path);
        let start = (h as usize) % self.capacity;
        for i in 0..self.capacity {
            let idx = (start + i) % self.capacity;
            if let Some(ref mut e) = self.entries[idx] {
                if e.path_hash == h && e.gen_at_insert == self.generation {
                    e.accessed = true;
                    self.hits += 1;
                    return Some(e.inode_id);
                }
            }
        }
        self.misses += 1;
        None
    }

    pub fn insert(&mut self, path: &str, inode_id: u64) {
        if self.capacity == 0 || self.entries.is_empty() {
            return;
        }
        let h = hash_path(path);
        let start = (h as usize) % self.capacity;
        for i in 0..self.capacity {
            let idx = (start + i) % self.capacity;
            match &mut self.entries[idx] {
                Some(e) if e.path_hash == h && e.gen_at_insert == self.generation => {
                    e.inode_id = inode_id;
                    e.accessed = true;
                    return;
                }
                slot @ None => {
                    *slot = Some(CacheEntry {
                        path_hash: h,
                        inode_id,
                        gen_at_insert: self.generation,
                        accessed: true,
                    });
                    return;
                }
                _ => {}
            }
        }
        // No empty slot — CLOCK eviction
        loop {
            let idx = self.clock_hand % self.capacity;
            self.clock_hand += 1;
            if let Some(ref mut e) = self.entries[idx] {
                if e.gen_at_insert != self.generation {
                    self.entries[idx] = Some(CacheEntry {
                        path_hash: h,
                        inode_id,
                        gen_at_insert: self.generation,
                        accessed: true,
                    });
                    return;
                }
                if e.accessed {
                    e.accessed = false;
                } else {
                    self.entries[idx] = Some(CacheEntry {
                        path_hash: h,
                        inode_id,
                        gen_at_insert: self.generation,
                        accessed: true,
                    });
                    return;
                }
            } else {
                self.entries[idx] = Some(CacheEntry {
                    path_hash: h,
                    inode_id,
                    gen_at_insert: self.generation,
                    accessed: true,
                });
                return;
            }
        }
    }

    pub fn bump_generation(&mut self) {
        self.generation += 1;
    }

    pub fn clear(&mut self) {
        self.generation += 1;
        for e in &mut self.entries {
            *e = None;
        }
        self.clock_hand = 0;
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;

    fn test_hit_miss() -> TestResult {
        let mut pc = PathCache::with_capacity(4);
        pc.insert("/a/b", 1);
        test_assert_eq!(pc.get("/a/b"), Some(1));
        test_assert_eq!(pc.get("/x"), None);
        test_assert_eq!(pc.hits, 1);
        test_assert_eq!(pc.misses, 1);
        TestResult::Pass
    }

    fn test_invalidation() -> TestResult {
        let mut pc = PathCache::with_capacity(4);
        pc.insert("/a", 1);
        pc.bump_generation();
        test_assert_eq!(pc.get("/a"), None);
        TestResult::Pass
    }

    fn test_eviction() -> TestResult {
        let mut pc = PathCache::with_capacity(2);
        pc.insert("/a", 1);
        pc.insert("/b", 2);
        pc.insert("/c", 3);
        let mut found = 0;
        if pc.get("/a").is_some() { found += 1; }
        if pc.get("/b").is_some() { found += 1; }
        if pc.get("/c").is_some() { found += 1; }
        test_assert!(found >= 2, "eviction lost too many entries");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("path_cache::hit_miss", test_hit_miss);
        crate::testing::register_test("path_cache::invalidation", test_invalidation);
        crate::testing::register_test("path_cache::eviction", test_eviction);
    }
}
