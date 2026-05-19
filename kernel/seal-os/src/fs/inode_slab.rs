// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! O(1) inode allocator — Vec-indexed slab with generation counters.

use alloc::vec::Vec;
use super::manifold_fs::{Inode, InodeKind, InodeMetadata};

struct SlotState {
    generation: u32,
    inode: Option<Inode>,
    next_free: Option<u32>,
}

pub struct InodeSlab {
    slots: Vec<SlotState>,
    free_head: Option<u32>,
}

impl InodeSlab {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
        }
    }

    /// Allocate a slot, assign a fresh inode_id, and return it.
    pub fn alloc(&mut self, mut inode: Inode) -> u64 {
        if let Some(idx) = self.free_head {
            let idx = idx as usize;
            let slot = &mut self.slots[idx];
            self.free_head = slot.next_free;
            slot.inode = Some(inode);
            let id = ((slot.generation as u64) << 32) | (idx as u64);
            if let Some(ref mut i) = slot.inode {
                i.id = id;
            }
            id
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(SlotState {
                generation: 1,
                inode: Some(inode),
                next_free: None,
            });
            let id = ((1u64) << 32) | (idx as u64);
            if let Some(ref mut i) = self.slots[idx as usize].inode {
                i.id = id;
            }
            id
        }
    }

    /// Free a slot by id. Returns the evicted inode if the id was valid.
    pub fn free(&mut self, id: u64) -> Option<Inode> {
        let idx = (id & 0xFFFF_FFFF) as usize;
        let gen = (id >> 32) as u32;
        let slot = self.slots.get_mut(idx)?;
        if slot.generation != gen || slot.inode.is_none() {
            return None;
        }
        slot.generation = slot.generation.wrapping_add(1);
        let inode = slot.inode.take();
        slot.next_free = self.free_head;
        self.free_head = Some(idx as u32);
        inode
    }

    pub fn get(&self, id: u64) -> Option<&Inode> {
        let idx = (id & 0xFFFF_FFFF) as usize;
        let gen = (id >> 32) as u32;
        let slot = self.slots.get(idx)?;
        if slot.generation != gen {
            return None;
        }
        slot.inode.as_ref()
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Inode> {
        let idx = (id & 0xFFFF_FFFF) as usize;
        let gen = (id >> 32) as u32;
        let slot = self.slots.get_mut(idx)?;
        if slot.generation != gen {
            return None;
        }
        slot.inode.as_mut()
    }

    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.inode.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &Inode> {
        self.slots.iter().filter_map(|s| s.inode.as_ref())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Inode> {
        self.slots.iter_mut().filter_map(|s| s.inode.as_mut())
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;
    use alloc::string::String;
    use alloc::vec;

    fn dummy_inode(id: u64, name: &str) -> Inode {
        Inode {
            id,
            name: String::from(name),
            kind: InodeKind::File,
            payload: super::super::encoder::encode_text(name),
            data: vec![],
            metadata: InodeMetadata {
                created_ms: 0,
                modified_ms: 0,
                original_size: 0,
                permissions: 0o644,
            },
            voronoi_cell: 0,
            cluster_id: 0,
            parent: 0,
            sibling_next: None,
            sibling_prev: None,
            dir_first_child: None,
        }
    }

    fn test_alloc_and_get() -> TestResult {
        let mut slab = InodeSlab::new();
        let id = slab.alloc(dummy_inode(0, "a"));
        test_assert!(id != 0, "alloc returned zero id");
        test_assert_eq!(slab.get(id).unwrap().name, "a");
        TestResult::Pass
    }

    fn test_free_invalidates() -> TestResult {
        let mut slab = InodeSlab::new();
        let id = slab.alloc(dummy_inode(0, "a"));
        slab.free(id);
        test_assert!(slab.get(id).is_none(), "stale id should be invalid");
        TestResult::Pass
    }

    fn test_free_reuse_generation() -> TestResult {
        let mut slab = InodeSlab::new();
        let id1 = slab.alloc(dummy_inode(0, "a"));
        let slot = (id1 & 0xFFFF_FFFF) as usize;
        slab.free(id1);
        let id2 = slab.alloc(dummy_inode(0, "b"));
        test_assert_eq!((id2 & 0xFFFF_FFFF) as usize, slot);
        test_assert!((id2 >> 32) != (id1 >> 32), "generation should increment");
        test_assert!(slab.get(id1).is_none(), "old generation invalid");
        test_assert_eq!(slab.get(id2).unwrap().name, "b");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("inode_slab::alloc_and_get", test_alloc_and_get);
        crate::testing::register_test("inode_slab::free_invalidates", test_free_invalidates);
        crate::testing::register_test("inode_slab::free_reuse_generation", test_free_reuse_generation);
    }
}
