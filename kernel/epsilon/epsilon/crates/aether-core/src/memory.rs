// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//! AEGIS Memory Substrate: The Manifold Heap
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//!
//! "Memory is not a bucket; it is a topological space."
//!
//! This module implements the Manifold Garbage Collector, a biologically inspired
//! memory model that treats unused objects as "entropy" to be reclaimed.
//!
//! Key Components:
//! 1. ManifoldHeap: A spatial tree (Octree-like) organizing objects into blocks.
//! 2. Entropy Regulation: O(N/8) scan over blocks with Chebyshev-guarded pruning.
//!    The spatial tree enables future O(log N) branch-skip optimization when
//!    aggregate liveness statistics are maintained per-node.
//! 3. Chebyshev's Guard: Statistical safety — objects within k·σ of mean liveness
//!    are protected from collection.
//!
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use core::marker::PhantomData;
use libm::sqrt;

/// A Geometric Cell (Gc) handle.
/// Represents a reference to an object in the ManifoldHeap.
/// Unlike standard pointers, this is a topological index.
///
/// We implement Copy/Clone manually to avoid implicit T: Copy bound.
#[derive(Debug, PartialOrd, Ord)]
pub struct Gc<T> {
    pub(crate) index: usize,
    pub(crate) generation: u32,
    _marker: PhantomData<fn() -> T>, // Covariant, implies no ownership logic for drop
}

impl<T> Copy for Gc<T> {}

impl<T> Clone for Gc<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for Gc<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Gc<T> {}

impl<T> core::hash::Hash for Gc<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> Gc<T> {
    pub fn new(index: usize, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// Returns the slot index of this handle within the heap.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the generation counter used to detect stale handles.
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

/// Metadata for an object in the heap.
#[derive(Debug, Clone, Copy)]
pub struct ObjectHeader {
    /// Is this object currently reachable?
    pub marked: bool,
    /// Generation count to detect stale handles
    pub generation: u32,
}

/// A slot in the ManifoldHeap.
/// Note: Liveness is now stored in the SpatialBlock for SIMD access.
#[derive(Debug, Clone)]
pub enum HeapSlot<T> {
    /// `generation` is the last generation this slot handed out (0 = never
    /// used); the next occupant gets `generation + 1`. A slot freed at
    /// `u32::MAX` is retired: it is never linked into the free list again.
    Free {
        next_free: usize,
        generation: u32,
    },
    Occupied {
        header: ObjectHeader,
        data: T,
    },
}

/// A Spatial Block acting as a leaf in the memory tree.
/// Contains contiguous arrays for SIMD optimization.
/// Size N=8.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct SpatialBlock<T> {
    /// Liveness scores [f64; 8]
    pub liveness: [f64; 8],
    /// The actual data slots
    pub slots: [HeapSlot<T>; 8],
    /// Mask or counter of occupied slots (optional but useful)
    pub occupied_mask: u8,
}

impl<T> Default for SpatialBlock<T> {
    fn default() -> Self {
        Self {
            liveness: [0.0; 8],
            slots: core::array::from_fn(|_| HeapSlot::Free {
                next_free: usize::MAX,
                generation: 0,
            }),
            occupied_mask: 0,
        }
    }
}

impl<T> SpatialBlock<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Internal node of the Spatial Tree.
/// Aggregates statistics of its children.
#[derive(Debug, Clone)]
pub struct SpatialNode {
    /// Indices of children.
    /// If `is_leaf_parent` is true, these are indices into `blocks`.
    /// Otherwise, indices into `nodes`.
    /// None indicates empty branch.
    pub children: [Option<usize>; 8],

    /// Aggregate Mean Liveness of this branch
    pub mean_liveness: f64,
    /// Max Liveness in this branch (for quick "is hot" checks)
    pub max_liveness: f64,

    /// Does this node point to Blocks (true) or Nodes (false)?
    pub is_leaf_parent: bool,
}

impl SpatialNode {
    pub fn new(is_leaf_parent: bool) -> Self {
        Self {
            children: [None; 8],
            mean_liveness: 0.0,
            max_liveness: 0.0,
            is_leaf_parent,
        }
    }
}

/// Configuration for Memory Behavior
#[derive(Debug, Clone, Copy)]
pub enum MemoryMode {
    Consumer,
    Datacenter,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub mode: MemoryMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: MemoryMode::Consumer,
        }
    }
}

/// The Manifold Allocator.
/// Manages memory as a dense topological substrate using a Spatial Tree.
pub struct ManifoldHeap<T> {
    /// Leaf Blocks
    pub blocks: Vec<SpatialBlock<T>>,
    /// Tree Nodes
    pub nodes: Vec<SpatialNode>,
    /// Root Node Index
    pub root_idx: usize,

    /// Head of the free list (Global index)
    /// Index = block_idx * 8 + slot_idx
    free_head: Option<usize>,

    /// Active objects count
    active_count: usize,
    /// Global entropy counter
    entropy_counter: usize,

    pub config: Config,
}

impl<T> Default for ManifoldHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ManifoldHeap<T> {
    /// Create an empty manifold heap with one root node.
    pub fn new() -> Self {
        let mut heap = Self {
            blocks: Vec::new(),
            nodes: Vec::new(),
            root_idx: 0,
            free_head: None,
            active_count: 0,
            entropy_counter: 0,
            config: Config::default(),
        };
        // Initialize with one root node that is a leaf parent
        heap.nodes.push(SpatialNode::new(true));
        heap
    }

    /// Helper to decompose global index into (block, offset)
    fn resolve_index(index: usize) -> (usize, usize) {
        (index / 8, index % 8)
    }

    /// Allocate a new object.
    pub fn alloc(&mut self, data: T) -> Gc<T> {
        self.entropy_counter += 1;

        let (block_idx, slot_idx, last_generation) = if let Some(head) = self.free_head {
            let (b, s) = Self::resolve_index(head);
            let mut last_generation = 0;
            // Verify and update free_head
            if b < self.blocks.len() {
                if let HeapSlot::Free {
                    next_free,
                    generation,
                } = &self.blocks[b].slots[s]
                {
                    if *next_free == usize::MAX {
                        self.free_head = None;
                    } else {
                        self.free_head = Some(*next_free);
                    }
                    last_generation = *generation;
                } else {
                    return Gc::new(0, 0); // Corrupt free-list; return a dead handle
                }
            }
            (b, s, last_generation)
        } else {
            // Bump allocation
            let next_blk_idx = self.blocks.len();
            self.blocks.push(SpatialBlock::new());
            self.link_block_to_tree(next_blk_idx);

            for i in 1..7 {
                self.blocks[next_blk_idx].slots[i] = HeapSlot::Free {
                    next_free: next_blk_idx * 8 + i + 1,
                    generation: 0,
                };
            }
            self.blocks[next_blk_idx].slots[7] = HeapSlot::Free {
                next_free: usize::MAX,
                generation: 0,
            };

            self.free_head = Some(next_blk_idx * 8 + 1);
            (next_blk_idx, 0, 0)
        };

        // Cannot overflow: a slot freed at u32::MAX is retired, never re-listed.
        let generation = last_generation + 1;
        self.blocks[block_idx].slots[slot_idx] = HeapSlot::Occupied {
            header: ObjectHeader {
                marked: false,
                generation,
            },
            data,
        };
        self.blocks[block_idx].liveness[slot_idx] = 1.0;
        self.blocks[block_idx].occupied_mask |= 1 << slot_idx;

        self.active_count += 1;

        Gc::new(block_idx * 8 + slot_idx, generation)
    }

    fn link_block_to_tree(&mut self, block_idx: usize) {
        let needed_node_idx = block_idx / 8;
        if needed_node_idx >= self.nodes.len() {
            self.nodes.push(SpatialNode::new(true));
        }

        let node_idx = needed_node_idx;
        let child_slot = block_idx % 8;
        self.nodes[node_idx].children[child_slot] = Some(block_idx);
    }

    /// Access mutably. Heats up object.
    pub fn get_mut(&mut self, handle: Gc<T>) -> Option<&mut T> {
        let (b, s) = Self::resolve_index(handle.index);

        if b >= self.blocks.len() {
            return None;
        }

        let block = &mut self.blocks[b];
        match &mut block.slots[s] {
            HeapSlot::Occupied { header, data } => {
                if header.generation != handle.generation {
                    return None;
                }
                // Heat up - split borrow of block works here
                block.liveness[s] = (block.liveness[s] + 1.0).min(10.0);
                Some(data)
            }
            _ => None,
        }
    }

    /// Access immutably (Peek). Does NOT update liveness to avoid &mut borrow.
    /// This fixes autograd multiple borrow issues.
    pub fn get(&self, handle: Gc<T>) -> Option<&T> {
        let (b, s) = Self::resolve_index(handle.index);
        if b >= self.blocks.len() {
            return None;
        }

        match &self.blocks[b].slots[s] {
            HeapSlot::Occupied { header, data } => {
                if header.generation != handle.generation {
                    return None;
                }
                Some(data)
            }
            _ => None,
        }
    }

    pub fn touch(&mut self, handle: Gc<T>) {
        let (b, s) = Self::resolve_index(handle.index);
        if b < self.blocks.len() {
            let block = &mut self.blocks[b];
            if let HeapSlot::Occupied { header, .. } = &mut block.slots[s] {
                if header.generation == handle.generation {
                    block.liveness[s] = (block.liveness[s] + 0.5).min(10.0);
                }
            }
        }
    }

    pub fn mark(&mut self, handle: Gc<T>) {
        let (b, s) = Self::resolve_index(handle.index);
        if b < self.blocks.len() {
            let block = &mut self.blocks[b];
            if let HeapSlot::Occupied { header, .. } = &mut block.slots[s] {
                if header.generation == handle.generation {
                    header.marked = true;
                    block.liveness[s] = (block.liveness[s] + 2.0).min(10.0);
                }
            }
        }
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn capacity(&self) -> usize {
        self.blocks.len() * 8
    }
}

/// Chebyshev Guard logic.
pub struct ChebyshevGuard {
    mean: f64,
    std_dev: f64,
    k: f64,
}

impl ChebyshevGuard {
    /// Mean and standard deviation of the occupied slots' liveness.
    ///
    /// The ceiling `AetherVerified.Chebyshev` proves — at most `n / k^2` scores
    /// at or below `mu - k sigma` — needs `sigma^2 * n = sum (x - mu)^2` over the
    /// same `x` the guard then judges, and `sigma > 0`. The variance is therefore
    /// two-pass: the one-pass `sum_sq / n - mu^2` cancels, reading sigma 2.98e-8
    /// for five scores whose exact sigma is 4.99e-8 and pruning two of them
    /// against a ceiling of one.
    pub fn calculate<T>(heap: &ManifoldHeap<T>) -> Self {
        let live = || {
            heap.blocks.iter().flat_map(|block| {
                (0..8)
                    .filter(move |&i| block.occupied_mask & (1 << i) != 0)
                    .map(move |i| block.liveness[i])
            })
        };
        let count = live().count() as f64;
        if count == 0.0 {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                k: 2.0,
            };
        }

        let mean = live().sum::<f64>() / count;
        // from teerthsharma/sigmoid sigmoid/telemetry.py:92: sigma is x.std() of the scores judged
        let variance = live().map(|x| (x - mean) * (x - mean)).sum::<f64>() / count;

        Self {
            mean,
            std_dev: sqrt(variance),
            k: 2.0,
        }
    }

    pub fn is_safe(&self, liveness: f64) -> bool {
        // No spread to measure against (sigmoid/telemetry.py:93): keep all.
        if liveness >= self.mean || self.std_dev <= 0.0 || self.std_dev.is_nan() {
            return true;
        }
        let boundary = self.mean - (self.k * self.std_dev);
        liveness > boundary
    }
}

impl<T> ManifoldHeap<T> {
    /// Regulation pass: mark-trace-sweep with Chebyshev-guarded pruning.
    pub fn regulate_entropy<F>(&mut self, tracer: F) -> usize
    where
        F: Fn(&mut Self),
    {
        // 0. Reset Marks
        for block in &mut self.blocks {
            for slot in &mut block.slots {
                if let HeapSlot::Occupied { header, .. } = slot {
                    header.marked = false;
                }
            }
        }

        // 1. Trace
        tracer(self);

        // 2. Calc Stats
        let guard = ChebyshevGuard::calculate(self);

        let mut pruned = 0;
        let num_blocks = self.blocks.len();
        let mut new_free_head = self.free_head;

        for b_idx in 0..num_blocks {
            let block = &mut self.blocks[b_idx];

            for s_idx in 0..8 {
                if (block.occupied_mask & (1 << s_idx)) == 0 {
                    continue;
                }

                // The guard describes this pass's liveness before decay, so that
                // is the value it judges; 0.95 x against the undecayed mean put
                // a uniform heap entirely past the boundary.
                let is_safe = guard.is_safe(block.liveness[s_idx]);
                block.liveness[s_idx] *= 0.95;

                let should_prune;
                let mut generation = 0;

                if let HeapSlot::Occupied { header, .. } = &mut block.slots[s_idx] {
                    generation = header.generation;
                    let is_marked = header.marked;

                    if is_marked {
                        block.liveness[s_idx] += 0.1;
                        should_prune = false;
                    } else {
                        should_prune = !is_safe;
                    }
                } else {
                    should_prune = false;
                }

                if should_prune {
                    block.occupied_mask &= !(1 << s_idx);
                    if generation == u32::MAX {
                        // Every generation is spent; reuse would alias old handles.
                        block.slots[s_idx] = HeapSlot::Free {
                            next_free: usize::MAX,
                            generation,
                        };
                    } else {
                        block.slots[s_idx] = HeapSlot::Free {
                            next_free: new_free_head.unwrap_or(usize::MAX),
                            generation,
                        };
                        new_free_head = Some(b_idx * 8 + s_idx);
                    }

                    pruned += 1;
                }
            }
        }

        self.active_count -= pruned;
        self.free_head = new_free_head;
        self.entropy_counter = 0;

        pruned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifold_allocation() {
        let mut heap = ManifoldHeap::<i32>::new();
        let a = heap.alloc(10);
        let b = heap.alloc(20);

        assert_eq!(*heap.get(a).unwrap(), 10);
        assert_eq!(*heap.get(b).unwrap(), 20);
        assert_eq!(heap.active_count(), 2);
    }

    #[test]
    fn test_spatial_clustering() {
        let mut heap = ManifoldHeap::<i32>::new();
        let mut handles = Vec::new();
        for i in 0..8 {
            handles.push(heap.alloc(i));
        }

        let h9 = heap.alloc(99);
        let (b1, _) = ManifoldHeap::<i32>::resolve_index(h9.index);
        assert_eq!(b1, 1);
        assert_eq!(heap.blocks.len(), 2);
    }

    /// Allocate 16 objects and heat all but the first, so the next
    /// regulation pass prunes exactly that one. Returns its handle.
    fn sixteen_with_cold_first(heap: &mut ManifoldHeap<i32>) -> Gc<i32> {
        let handles: Vec<_> = (0..16).map(|i| heap.alloc(i)).collect();
        for h in &handles[1..] {
            for _ in 0..8 {
                heap.touch(*h);
            }
        }
        handles[0]
    }

    #[test]
    fn test_stale_handle_reads_none_after_slot_reuse() {
        let mut heap = ManifoldHeap::<i32>::new();
        let cold = sixteen_with_cold_first(&mut heap);
        assert_eq!(heap.regulate_entropy(|_| {}), 1);

        let fresh = heap.alloc(999);
        assert_eq!(fresh.index(), cold.index(), "slot was not reused");
        assert_ne!(fresh.generation(), cold.generation());
        assert_eq!(heap.get(cold), None);
        assert_eq!(heap.get_mut(cold), None);
        assert_eq!(heap.get(fresh), Some(&999));
    }

    #[test]
    fn test_slot_at_max_generation_is_retired() {
        let mut heap = ManifoldHeap::<i32>::new();
        let cold = sixteen_with_cold_first(&mut heap);
        let (b, s) = ManifoldHeap::<i32>::resolve_index(cold.index());
        if let HeapSlot::Occupied { header, .. } = &mut heap.blocks[b].slots[s] {
            header.generation = u32::MAX;
        }
        assert_eq!(heap.regulate_entropy(|_| {}), 1);

        // Any next generation would wrap and alias an old handle, so the slot
        // must never be handed out again.
        for i in 0..64 {
            assert_ne!(
                heap.alloc(i).index(),
                cold.index(),
                "exhausted slot was reused"
            );
        }
    }

    #[test]
    fn a_pass_prunes_no_more_than_the_chebyshev_ceiling() {
        // AetherVerified.Chebyshev bounds |{x <= mu - k sigma}| by n / k^2 given
        // sigma > 0 and sigma^2 * n = sum (x - mu)^2 over the very x judged. k = 2
        // and no object is marked, so each pass may prune at most n / 4.
        fn heap_at(liveness: &[f64]) -> ManifoldHeap<usize> {
            let mut heap = ManifoldHeap::new();
            for (i, &x) in liveness.iter().enumerate() {
                heap.alloc(i);
                heap.blocks[i / 8].liveness[i % 8] = x;
            }
            heap
        }
        let cases: [(&str, &[f64]); 3] = [
            // 64 at one liveness sit at their mean: none may go.
            ("uniform", &[1.0; 64]),
            // sum_sq / n - mu^2 reads sigma 2.98e-8; the exact sigma is 4.99e-8.
            (
                "cancelling",
                &[
                    2.1364115047690455,
                    2.1364115228011884,
                    2.1364114091048214,
                    2.1364115011049716,
                    2.1364114084786783,
                ],
            ),
            // (x - mu)^2 underflows, so sigma reads 0 over a real spread.
            ("underflowing", &[1e-170, 2e-170]),
        ];
        for (name, liveness) in cases {
            let mut heap = heap_at(liveness);
            for pass in 0..4 {
                let live = heap.active_count();
                let pruned = heap.regulate_entropy(|_| {});
                assert!(
                    pruned * 4 <= live,
                    "{name}, pass {pass}: pruned {pruned} of {live}"
                );
            }
        }
    }

    #[test]
    fn test_simd_alignment() {
        use core::mem::align_of;
        assert_eq!(align_of::<SpatialBlock<i32>>(), 64);
    }
}
