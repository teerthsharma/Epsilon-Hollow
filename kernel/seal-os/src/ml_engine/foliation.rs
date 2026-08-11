// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Foliation — kernel-managed paged KV cache for LLM inference.
//!
//! # The structure
//!
//! The set of live sequences is modelled as a **foliation** of token-stream
//! space. Two sequences that agree on a block-aligned prefix lie on the same
//! *leaf*; a leaf is therefore an equivalence class of the prefix relation and
//! the leaf set is the quotient of token-stream space by that relation. A
//! **plaque** is one KV block: the piece of a leaf that is actually resident in
//! physical memory (standard foliation vocabulary — a leaf is locally a stack
//! of plaques).
//!
//! Prefix sharing is not a hash table bolted onto an allocator. A sequence's
//! block table *is* its path from the root leaf down the foliation. Appending a
//! block means `descend(current_leaf, block_key)`; if that child already exists
//! the sequence lands on the same leaf as everyone else who wrote those tokens
//! and therefore on the same plaque. Sharing is the quotient map, and the
//! reference count of a plaque is the cardinality of the fibre over it.
//!
//! # Metadata persists, memory does not
//!
//! Leaf metadata (parent, key, depth, entrant count) lives in a leaf arena and
//! survives eviction. Only the *plaque* — the 4 KiB frame — is reclaimed. The
//! foliation is therefore a persistent model of the workload while residency is
//! transient, which is what lets a structural retention signal accumulate across
//! evictions instead of resetting every time a block is dropped.
//!
//! # Eviction
//!
//! Residency is constrained to be a connected rooted subtree of the foliation.
//! The only admissible eviction is an elementary collapse of a *free face*: a
//! resident leaf with reference count zero and no resident children. Every
//! policy in this module operates on that same candidate set, so LRU and the
//! foliation policy differ only in victim choice, never in what they are allowed
//! to touch.
//!
//! Within the frontier the foliation policy ranks by
//! `(entrants, -depth, last_use)`: fewest distinct sequences that ever entered
//! the leaf first, then deepest, then oldest. `entrants` is the multiplicity of
//! the leaf's H0 bar over the trace — a persistence proxy, and honestly also a
//! frequency counter. Depth is codimension in the foliation: root-adjacent
//! leaves are shared prompt prefixes, deep leaves are per-sequence decode tails.
//!
//! # Complexity (all bounds are compile-time or construction-time constants,
//! independent of live sequence count, token count, and installed RAM)
//!
//! | operation                    | bound                              |
//! |------------------------------|------------------------------------|
//! | append token (mid-block)     | O(1)                               |
//! | block seal / descend         | O(MAX_CHILDREN) = O(32)            |
//! | admission (free plaque)      | O(1) free-list pop                 |
//! | admission (needs eviction)   | O(pool_blocks) frontier scan       |
//! | leaf-arena GC                | O(leaf_arena) scan                 |
//! | logical block -> frame       | O(1) indexed                       |
//! | release                      | O(MAX_SEQ_BLOCKS) = O(16)          |

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use x86_64::PhysAddr;

use crate::memory::topo_ram::{self, ZoneHint};

/// Tokens per KV block. PagedAttention block granularity.
pub const BLOCK_TOKENS: usize = 8;
/// Maximum fan-out of a foliation leaf.
///
/// ponytail: fixed fan-out with a linear child scan. Ceiling is 32 distinct
/// continuations per prefix; past that `descend` refuses to share and reports
/// `children_full`. Upgrade path is an open-addressed key->child map per leaf,
/// which trades 3x metadata for unbounded fan-out.
pub const MAX_CHILDREN: usize = 32;
/// Maximum blocks in one sequence's block table.
pub const MAX_SEQ_BLOCKS: usize = 16;
/// Bytes backing one plaque (one physical frame).
pub const PLAQUE_BYTES: usize = 4096;

const NONE: u16 = u16::MAX;
const ROOT: u16 = 0;

/// Eviction policy over the frontier of the resident foliation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Policy {
    /// Persistence-and-depth ranked: `(entrants, -depth, last_use)`.
    Foliation,
    /// Recency only. Baseline.
    Lru,
    /// Uniform over the frontier. Same-budget null model.
    Random,
    /// Belady optimum over a supplied future key trace. Oracle, not runnable
    /// online.
    Belady,
}

/// Why a KV-cache operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FoliationError {
    /// No such sequence id, or the sequence was already released.
    NoSuchSeq,
    /// Sequence reached its declared block budget.
    BudgetExceeded,
    /// Every resident plaque is referenced; nothing may be collapsed.
    Exhausted,
    /// Leaf arena is full and no dead leaf could be reclaimed.
    LeafArenaFull,
    /// Prefix already has `MAX_CHILDREN` distinct continuations.
    ChildrenFull,
    /// Plaque is still referenced by a live sequence.
    StillReferenced,
    /// Sequence table is full.
    TooManySeqs,
}

/// One leaf of the foliation. Metadata outlives plaque residency.
struct Leaf {
    used: bool,
    parent: u16,
    key: u64,
    depth: u16,
    child: [u16; MAX_CHILDREN],
    nchild: u8,
    /// Plaque slot, or `NONE` when the leaf is modelled but not resident.
    slot: u16,
    /// Live sequences whose block table contains this leaf.
    refcount: u16,
    /// Resident children — the collapse guard.
    resident_children: u16,
    /// Distinct descents that ever entered this leaf. H0 bar multiplicity.
    entrants: u32,
    last_use: u64,
}

impl Leaf {
    const fn blank() -> Self {
        Self {
            used: false,
            parent: NONE,
            key: 0,
            depth: 0,
            child: [NONE; MAX_CHILDREN],
            nchild: 0,
            slot: NONE,
            refcount: 0,
            resident_children: 0,
            entrants: 0,
            last_use: 0,
        }
    }
}

/// A resident KV block: one physical frame bound to one leaf.
struct Plaque {
    leaf: u16,
    frame: Option<PhysAddr>,
}

/// A live inference sequence. Its block table is its path down the foliation.
struct Seq {
    active: bool,
    budget_blocks: u16,
    blocks: [u16; MAX_SEQ_BLOCKS],
    nblocks: u16,
    pending: [u32; BLOCK_TOKENS],
    fill: u8,
    key: u64,
    leaf: u16,
    hits: u32,
    admits: u32,
}

impl Seq {
    const fn blank() -> Self {
        Self {
            active: false,
            budget_blocks: 0,
            blocks: [NONE; MAX_SEQ_BLOCKS],
            nblocks: 0,
            pending: [0; BLOCK_TOKENS],
            fill: 0,
            key: 0,
            leaf: ROOT,
            hits: 0,
            admits: 0,
        }
    }
}

/// Counters, every one measured at runtime.
#[derive(Clone, Copy, Default)]
pub struct FoliationStats {
    /// Block seals attempted.
    pub descents: u64,
    /// Seals that landed on an already-resident plaque.
    pub shared: u64,
    /// Seals that needed a fresh plaque.
    pub admits: u64,
    /// Plaques collapsed to make room.
    pub evictions: u64,
    /// Physical frames obtained from `topo_ram`.
    pub frames_backed: u64,
    /// Physical frames returned to `topo_ram`.
    pub frames_freed: u64,
    /// Frame allocations that failed.
    pub frames_failed: u64,
    /// Dead leaves reclaimed from the arena.
    pub leaf_gc: u64,
    /// Evictions that touched a referenced plaque. Must stay zero.
    pub referenced_evictions: u64,
    /// Appends refused for exceeding the sequence budget.
    pub refused_budget: u64,
    /// Admissions refused because the frontier was empty.
    pub refused_exhaustion: u64,
    /// Explicit frees refused because the plaque was still referenced.
    pub refused_referenced_free: u64,
    /// Descents that could not share because fan-out was saturated.
    pub children_full: u64,
}

/// Kernel-side paged KV cache over a prefix foliation.
pub struct Foliation {
    leaves: Vec<Leaf>,
    plaques: Vec<Plaque>,
    free_slots: Vec<u16>,
    seqs: Vec<Seq>,
    policy: Policy,
    tick: u64,
    rng: u64,
    /// Future key trace, `Policy::Belady` only.
    oracle: Vec<u64>,
    stats: FoliationStats,
}

/// Fold a block's tokens into the running prefix key.
///
/// The key of a block depends only on the whole token prefix, never on
/// residency or policy, so the same trace produces the same key sequence under
/// every policy. That is what makes the policy comparison and the Belady oracle
/// well defined.
pub fn fold_key(prev: u64, tokens: &[u32]) -> u64 {
    let mut h = prev ^ 0x9e37_79b9_7f4a_7c15;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
        h ^= h >> 29;
    }
    h
}

impl Foliation {
    /// Create a cache with `pool_blocks` plaques and a `leaf_arena` leaf budget.
    pub fn new(pool_blocks: usize, leaf_arena: usize, max_seqs: usize, policy: Policy) -> Self {
        let mut leaves = Vec::with_capacity(leaf_arena.max(1));
        for _ in 0..leaf_arena.max(1) {
            leaves.push(Leaf::blank());
        }
        leaves[ROOT as usize].used = true;
        leaves[ROOT as usize].parent = NONE;

        let mut plaques = Vec::with_capacity(pool_blocks);
        let mut free_slots = Vec::with_capacity(pool_blocks);
        for i in 0..pool_blocks {
            plaques.push(Plaque {
                leaf: NONE,
                frame: None,
            });
            free_slots.push((pool_blocks - 1 - i) as u16);
        }

        let mut seqs = Vec::with_capacity(max_seqs);
        for _ in 0..max_seqs {
            seqs.push(Seq::blank());
        }

        Self {
            leaves,
            plaques,
            free_slots,
            seqs,
            policy,
            tick: 0,
            rng: 0x2545_F491_4F6C_DD1D,
            oracle: Vec::new(),
            stats: FoliationStats::default(),
        }
    }

    /// Install the future key trace consumed by `Policy::Belady`.
    pub fn set_oracle(&mut self, trace: Vec<u64>) {
        self.oracle = trace;
    }

    /// Snapshot the counters.
    pub fn stats(&self) -> FoliationStats {
        self.stats
    }

    /// Configured policy.
    pub fn policy(&self) -> Policy {
        self.policy
    }

    /// Resident plaque count.
    pub fn resident(&self) -> usize {
        self.plaques.len() - self.free_slots.len()
    }

    // -- sequence lifecycle -------------------------------------------------

    /// Open a sequence with a hard block budget.
    pub fn seq_create(&mut self, budget_blocks: u16) -> Result<usize, FoliationError> {
        let budget = budget_blocks.min(MAX_SEQ_BLOCKS as u16);
        for (id, s) in self.seqs.iter_mut().enumerate() {
            if !s.active {
                *s = Seq::blank();
                s.active = true;
                s.budget_blocks = budget;
                return Ok(id);
            }
        }
        Err(FoliationError::TooManySeqs)
    }

    /// Append one token. Sealing a full block descends the foliation, which is
    /// where sharing happens.
    ///
    /// `fill` is strictly below `BLOCK_TOKENS` on entry and on every return,
    /// including the error returns: a refused seal rolls the token back out of
    /// the pending buffer. That invariant is what makes the `pending[fill]`
    /// store below in range without a second bounds test, and it is the reason
    /// a refusal cannot be converted into an out-of-range store by a caller
    /// that retries.
    pub fn seq_append(&mut self, id: usize, token: u32) -> Result<u16, FoliationError> {
        if self.seqs.get(id).map(|s| s.active) != Some(true) {
            return Err(FoliationError::NoSuchSeq);
        }
        if self.seqs[id].fill == 0 && self.seqs[id].nblocks >= self.seqs[id].budget_blocks {
            self.stats.refused_budget += 1;
            return Err(FoliationError::BudgetExceeded);
        }
        let fill = self.seqs[id].fill as usize;
        self.seqs[id].pending[fill] = token;
        self.seqs[id].fill += 1;
        if self.seqs[id].fill as usize == BLOCK_TOKENS {
            if let Err(e) = self.seal_block(id) {
                // The block was not sealed, so this token was never committed.
                // Restore the buffer to its state on entry and report the
                // refusal: leaving `fill` at `BLOCK_TOKENS` would make the next
                // append store one past the end of `pending`.
                self.seqs[id].pending[fill] = 0;
                self.seqs[id].fill = fill as u8;
                return Err(e);
            }
        }
        Ok(self.seqs[id].nblocks)
    }

    /// Frame backing logical block `idx` of a live sequence.
    ///
    /// O(1): two indexed loads, no scan. This is the block-table lookup.
    pub fn seq_frame(&self, id: usize, idx: usize) -> Option<PhysAddr> {
        let s = self.seqs.get(id)?;
        if !s.active || idx >= s.nblocks as usize {
            return None;
        }
        let leaf = s.blocks[idx];
        let slot = self.leaves.get(leaf as usize)?.slot;
        if slot == NONE {
            return None;
        }
        self.plaques[slot as usize].frame
    }

    /// Leaf id backing logical block `idx`.
    pub fn seq_leaf(&self, id: usize, idx: usize) -> Option<u16> {
        let s = self.seqs.get(id)?;
        if !s.active || idx >= s.nblocks as usize {
            return None;
        }
        Some(s.blocks[idx])
    }

    /// Blocks sealed, blocks shared on entry, blocks admitted.
    pub fn seq_counts(&self, id: usize) -> Option<(u16, u32, u32)> {
        let s = self.seqs.get(id)?;
        if !s.active {
            return None;
        }
        Some((s.nblocks, s.hits, s.admits))
    }

    /// Drop a sequence's references. Plaques stay resident — that is the cache.
    /// A block shared with another live sequence keeps a positive refcount.
    pub fn seq_release(&mut self, id: usize) -> Result<u16, FoliationError> {
        if self.seqs.get(id).map(|s| s.active) != Some(true) {
            return Err(FoliationError::NoSuchSeq);
        }
        let n = self.seqs[id].nblocks;
        for i in 0..n as usize {
            let leaf = self.seqs[id].blocks[i];
            if leaf != NONE {
                let l = &mut self.leaves[leaf as usize];
                l.refcount = l.refcount.saturating_sub(1);
            }
        }
        self.seqs[id] = Seq::blank();
        Ok(n)
    }

    /// Reference count of a leaf.
    pub fn leaf_refcount(&self, leaf: u16) -> u16 {
        self.leaves
            .get(leaf as usize)
            .map(|l| l.refcount)
            .unwrap_or(0)
    }

    /// Distinct descents that ever entered a leaf.
    pub fn leaf_entrants(&self, leaf: u16) -> u32 {
        self.leaves
            .get(leaf as usize)
            .map(|l| l.entrants)
            .unwrap_or(0)
    }

    /// True when the leaf currently holds a plaque.
    pub fn leaf_resident(&self, leaf: u16) -> bool {
        self.leaves
            .get(leaf as usize)
            .map(|l| l.slot != NONE)
            .unwrap_or(false)
    }

    /// Negative control: explicitly collapse a plaque. Refused while the leaf
    /// is still referenced by any live sequence.
    pub fn force_collapse(&mut self, leaf: u16) -> Result<(), FoliationError> {
        let l = self
            .leaves
            .get(leaf as usize)
            .ok_or(FoliationError::NoSuchSeq)?;
        if l.slot == NONE {
            return Err(FoliationError::NoSuchSeq);
        }
        if l.refcount > 0 || l.resident_children > 0 {
            self.stats.refused_referenced_free += 1;
            return Err(FoliationError::StillReferenced);
        }
        self.collapse(leaf);
        Ok(())
    }

    /// Verify the residency invariant: the resident set is a connected subtree
    /// rooted at the root leaf, and no resident leaf is referenced-but-absent.
    /// Returns the number of violations. O(pool_blocks).
    pub fn collapse_violations(&self) -> usize {
        let mut bad = 0;
        for p in &self.plaques {
            if p.leaf == NONE {
                continue;
            }
            let l = &self.leaves[p.leaf as usize];
            if l.parent != ROOT && l.parent != NONE && self.leaves[l.parent as usize].slot == NONE {
                bad += 1;
            }
        }
        bad
    }

    /// Release every plaque and return the frames. Used at teardown so the
    /// proof can show frames_freed == frames_backed.
    pub fn teardown(&mut self) {
        for i in 0..self.plaques.len() {
            if let Some(frame) = self.plaques[i].frame.take() {
                topo_ram::free_frames(frame, 1);
                self.stats.frames_freed += 1;
            }
            self.plaques[i].leaf = NONE;
        }
    }

    // -- foliation mechanics ------------------------------------------------

    fn seal_block(&mut self, id: usize) -> Result<(), FoliationError> {
        let parent = self.seqs[id].leaf;
        let prev_key = self.seqs[id].key;
        let tokens = self.seqs[id].pending;
        let key = fold_key(prev_key, &tokens[..]);

        self.tick += 1;
        self.stats.descents += 1;

        let child = self.find_child(parent, key);
        let fresh = child.is_none();
        let leaf = match child {
            Some(c) => c,
            None => {
                let c = self.new_leaf(parent, key)?;
                self.link_child(parent, c)?;
                c
            }
        };

        if self.leaves[leaf as usize].slot == NONE {
            // Leaf is modelled but its plaque was collapsed: re-admit.
            if let Err(e) = self.admit(leaf) {
                if fresh {
                    // This seal invented the leaf and admission then failed, so
                    // nothing references it. Retract it. Left linked it would
                    // consume one of the parent's `MAX_CHILDREN` slots until an
                    // arena GC happened to collect it, so a run of refused
                    // appends would saturate the prefix's fan-out and turn a
                    // transient refusal into a permanent one.
                    self.unlink_child(parent, leaf);
                    self.leaves[leaf as usize] = Leaf::blank();
                }
                return Err(e);
            }
            self.seqs[id].admits += 1;
            self.stats.admits += 1;
        } else {
            self.seqs[id].hits += 1;
            self.stats.shared += 1;
        }

        {
            let tick = self.tick;
            let l = &mut self.leaves[leaf as usize];
            l.refcount += 1;
            l.entrants += 1;
            l.last_use = tick;
        }

        let n = self.seqs[id].nblocks as usize;
        self.seqs[id].blocks[n] = leaf;
        self.seqs[id].nblocks += 1;
        self.seqs[id].key = key;
        self.seqs[id].leaf = leaf;
        self.seqs[id].fill = 0;
        self.seqs[id].pending = [0; BLOCK_TOKENS];
        Ok(())
    }

    /// O(MAX_CHILDREN) bounded scan.
    fn find_child(&self, parent: u16, key: u64) -> Option<u16> {
        let p = &self.leaves[parent as usize];
        for i in 0..p.nchild as usize {
            let c = p.child[i];
            if c != NONE && self.leaves[c as usize].key == key {
                return Some(c);
            }
        }
        None
    }

    fn link_child(&mut self, parent: u16, child: u16) -> Result<(), FoliationError> {
        let p = &mut self.leaves[parent as usize];
        if p.nchild as usize >= MAX_CHILDREN {
            return Err(FoliationError::ChildrenFull);
        }
        p.child[p.nchild as usize] = child;
        p.nchild += 1;
        Ok(())
    }

    fn unlink_child(&mut self, parent: u16, child: u16) {
        let p = &mut self.leaves[parent as usize];
        let n = p.nchild as usize;
        for i in 0..n {
            if p.child[i] == child {
                p.child[i] = p.child[n - 1];
                p.child[n - 1] = NONE;
                p.nchild -= 1;
                return;
            }
        }
    }

    fn new_leaf(&mut self, parent: u16, key: u64) -> Result<u16, FoliationError> {
        if self.leaves[parent as usize].nchild as usize >= MAX_CHILDREN {
            self.stats.children_full += 1;
            return Err(FoliationError::ChildrenFull);
        }
        let depth = self.leaves[parent as usize].depth + 1;
        let idx = match self.free_leaf() {
            Some(i) => i,
            None => {
                let i = self.gc_leaf().ok_or(FoliationError::LeafArenaFull)?;
                self.stats.leaf_gc += 1;
                i
            }
        };
        let l = &mut self.leaves[idx as usize];
        *l = Leaf::blank();
        l.used = true;
        l.parent = parent;
        l.key = key;
        l.depth = depth;
        Ok(idx)
    }

    fn free_leaf(&self) -> Option<u16> {
        // ponytail: linear scan of the leaf arena for a free slot, bounded by
        // the construction-time arena size. Upgrade path is an explicit free
        // list, which is 8 bytes per leaf and O(1) — not worth it until the
        // arena outgrows a few hundred entries.
        self.leaves
            .iter()
            .position(|l| !l.used)
            .map(|i| i as u16)
            .filter(|&i| i != ROOT)
    }

    /// Reclaim a dead leaf: not resident, no children, no references. Picks the
    /// weakest bar (fewest entrants) so persistent prefixes outlive noise.
    fn gc_leaf(&mut self) -> Option<u16> {
        let mut best: Option<(u32, u16)> = None;
        for (i, l) in self.leaves.iter().enumerate() {
            if i == ROOT as usize || !l.used || l.slot != NONE || l.nchild > 0 || l.refcount > 0 {
                continue;
            }
            if best.map(|(e, _)| l.entrants < e).unwrap_or(true) {
                best = Some((l.entrants, i as u16));
            }
        }
        let (_, idx) = best?;
        let parent = self.leaves[idx as usize].parent;
        if parent != NONE {
            self.unlink_child(parent, idx);
        }
        self.leaves[idx as usize] = Leaf::blank();
        Some(idx)
    }

    fn admit(&mut self, leaf: u16) -> Result<(), FoliationError> {
        if self.free_slots.is_empty() {
            let victim = match self.pick_victim() {
                Some(v) => v,
                None => {
                    self.stats.refused_exhaustion += 1;
                    return Err(FoliationError::Exhausted);
                }
            };
            if self.leaves[victim as usize].refcount > 0 {
                self.stats.referenced_evictions += 1;
            }
            self.collapse(victim);
            self.stats.evictions += 1;
        }
        let slot = self.free_slots.pop().ok_or(FoliationError::Exhausted)?;
        let cell = (self.leaves[leaf as usize].key % 8) as usize;
        let frame = match topo_ram::proof_hint(ZoneHint::Low, cell) {
            Some(hint) => topo_ram::alloc_frames(1, ZoneHint::Low, Some(&hint)),
            None => topo_ram::alloc_frames(1, ZoneHint::Low, None),
        };
        if frame.is_some() {
            self.stats.frames_backed += 1;
        } else {
            self.stats.frames_failed += 1;
        }
        self.plaques[slot as usize] = Plaque { leaf, frame };
        self.leaves[leaf as usize].slot = slot;
        let parent = self.leaves[leaf as usize].parent;
        if parent != NONE {
            self.leaves[parent as usize].resident_children += 1;
        }
        Ok(())
    }

    fn collapse(&mut self, leaf: u16) {
        let slot = self.leaves[leaf as usize].slot;
        if slot == NONE {
            return;
        }
        if let Some(frame) = self.plaques[slot as usize].frame.take() {
            topo_ram::free_frames(frame, 1);
            self.stats.frames_freed += 1;
        }
        self.plaques[slot as usize].leaf = NONE;
        self.free_slots.push(slot);
        self.leaves[leaf as usize].slot = NONE;
        let parent = self.leaves[leaf as usize].parent;
        if parent != NONE {
            let p = &mut self.leaves[parent as usize];
            p.resident_children = p.resident_children.saturating_sub(1);
        }
    }

    /// Choose a victim from the frontier of the resident foliation.
    ///
    /// The candidate set — resident, unreferenced, no resident children — is
    /// identical for every policy. Only the ranking differs.
    ///
    /// ponytail: O(pool_blocks) linear scan of resident plaques on the eviction
    /// path. pool_blocks is fixed at construction, so this does not grow with
    /// sequences, tokens, or RAM. Upgrade path is a bucketed priority queue
    /// keyed on (entrants, depth) — both are small integers — giving O(1) pop
    /// at the cost of maintaining bucket membership on every refcount change.
    fn pick_victim(&mut self) -> Option<u16> {
        let mut best: Option<(u16, [u64; 3])> = None;
        let mut candidates = 0u64;
        let mut reservoir = NONE;
        for p in &self.plaques {
            if p.leaf == NONE {
                continue;
            }
            let l = &self.leaves[p.leaf as usize];
            if l.refcount > 0 || l.resident_children > 0 {
                continue;
            }
            candidates += 1;
            let rank = match self.policy {
                Policy::Foliation => [l.entrants as u64, u64::MAX - l.depth as u64, l.last_use],
                Policy::Lru => [l.last_use, 0, 0],
                Policy::Random => [0, 0, 0],
                Policy::Belady => [u64::MAX - self.next_use(l.key), l.last_use, 0],
            };
            if self.policy == Policy::Random {
                // Reservoir sample so the null model is uniform over the same
                // candidate set, not biased by scan order.
                self.rng = self
                    .rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if (self.rng >> 33) % candidates == 0 {
                    reservoir = p.leaf;
                }
                continue;
            }
            if best.map(|(_, b)| rank < b).unwrap_or(true) {
                best = Some((p.leaf, rank));
            }
        }
        if self.policy == Policy::Random {
            return if reservoir == NONE {
                None
            } else {
                Some(reservoir)
            };
        }
        best.map(|(leaf, _)| leaf)
    }

    /// Index of the next descent that uses `key`, or `u64::MAX` if never.
    ///
    /// ponytail: O(remaining trace) forward scan. Belady is an offline oracle
    /// used only inside the boot benchmark to bound how much of the LRU gap a
    /// realizable policy could close; it is never on a runtime path.
    fn next_use(&self, key: u64) -> u64 {
        let from = self.stats.descents as usize;
        for (off, k) in self.oracle.iter().skip(from).enumerate() {
            if *k == key {
                return (from + off) as u64;
            }
        }
        u64::MAX
    }
}

// ---------------------------------------------------------------------------
// Embedded workload
// ---------------------------------------------------------------------------

const HOT_PREFIX_BLOCKS: usize = 4;
const COLD_PREFIX_BLOCKS: usize = 4;
const TAIL_BLOCKS: usize = 3;
const ROUNDS: usize = 6;
const COLD_PER_ROUND: usize = 4;
const BENCH_POOL_BLOCKS: usize = 24;
const BENCH_LEAF_ARENA: usize = 256;
const BENCH_MAX_SEQS: usize = 8;

/// One request: the token stream a sequence will append.
fn build_trace() -> Vec<Vec<u32>> {
    let mut trace = Vec::new();
    let mut req = 0u32;
    for round in 0..ROUNDS {
        // Hot request: shared system prompt + a fresh decode tail.
        let mut tokens = Vec::new();
        for j in 0..HOT_PREFIX_BLOCKS * BLOCK_TOKENS {
            tokens.push(1000 + j as u32);
        }
        for j in 0..TAIL_BLOCKS * BLOCK_TOKENS {
            tokens.push(50_000 + req * 100 + j as u32);
        }
        trace.push(tokens);
        req += 1;

        // Cold burst: unique prefix, unique tail, never reused.
        for c in 0..COLD_PER_ROUND {
            let cid = (round * COLD_PER_ROUND + c) as u32;
            let mut tokens = Vec::new();
            for j in 0..COLD_PREFIX_BLOCKS * BLOCK_TOKENS {
                tokens.push(20_000 + cid * 100 + j as u32);
            }
            for j in 0..TAIL_BLOCKS * BLOCK_TOKENS {
                tokens.push(50_000 + req * 100 + j as u32);
            }
            trace.push(tokens);
            req += 1;
        }
    }
    trace
}

/// The block-key sequence the trace produces, in descent order. Policy
/// independent by construction — `fold_key` sees only tokens.
fn trace_keys(trace: &[Vec<u32>]) -> Vec<u64> {
    let mut keys = Vec::new();
    for tokens in trace {
        let mut key = 0u64;
        for chunk in tokens.chunks(BLOCK_TOKENS) {
            if chunk.len() < BLOCK_TOKENS {
                break;
            }
            key = fold_key(key, chunk);
            keys.push(key);
        }
    }
    keys
}

struct Replay {
    hit_bp: u64,
    evictions: u64,
    descents: u64,
    shared: u64,
    collapse_violations: usize,
    referenced_evictions: u64,
    frames_backed: u64,
    frames_freed: u64,
    frames_failed: u64,
}

fn replay(policy: Policy, trace: &[Vec<u32>], keys: &[u64]) -> Replay {
    let mut fol = Foliation::new(BENCH_POOL_BLOCKS, BENCH_LEAF_ARENA, BENCH_MAX_SEQS, policy);
    if policy == Policy::Belady {
        fol.set_oracle(keys.to_vec());
    }
    for tokens in trace {
        let budget = (tokens.len() / BLOCK_TOKENS) as u16;
        let id = match fol.seq_create(budget) {
            Ok(id) => id,
            Err(_) => continue,
        };
        for &t in tokens {
            let _ = fol.seq_append(id, t);
        }
        let _ = fol.seq_release(id);
    }
    let violations = fol.collapse_violations();
    let s = fol.stats();
    fol.teardown();
    let after = fol.stats();
    Replay {
        hit_bp: (s.shared * 10_000).checked_div(s.descents).unwrap_or(0),
        evictions: s.evictions,
        descents: s.descents,
        shared: s.shared,
        collapse_violations: violations,
        referenced_evictions: s.referenced_evictions,
        frames_backed: s.frames_backed,
        frames_freed: after.frames_freed,
        frames_failed: s.frames_failed,
    }
}

/// Two sequences with an identical prefix must land on identical plaques, and
/// releasing one must leave the other's blocks intact.
///
/// Returns `(shared_blocks, refcount_after_partial_free, survivors_resident,
/// frames_identical)`.
fn share_and_refcount_probe() -> (u64, u16, u64, bool) {
    let mut fol = Foliation::new(16, 64, 4, Policy::Foliation);
    let prefix: Vec<u32> = (0..HOT_PREFIX_BLOCKS * BLOCK_TOKENS)
        .map(|j| 7000 + j as u32)
        .collect();

    let a = match fol.seq_create(8) {
        Ok(id) => id,
        Err(_) => return (0, 0, 0, false),
    };
    for &t in &prefix {
        let _ = fol.seq_append(a, t);
    }
    let b = match fol.seq_create(8) {
        Ok(id) => id,
        Err(_) => return (0, 0, 0, false),
    };
    for &t in &prefix {
        let _ = fol.seq_append(b, t);
    }

    let mut identical = true;
    let mut shared = 0u64;
    for i in 0..HOT_PREFIX_BLOCKS {
        match (fol.seq_leaf(a, i), fol.seq_leaf(b, i)) {
            (Some(la), Some(lb)) if la == lb => {
                shared += 1;
                if fol.seq_frame(a, i) != fol.seq_frame(b, i) {
                    identical = false;
                }
            }
            _ => identical = false,
        }
    }

    let _ = fol.seq_release(a);
    let mut refcount_after = 0u16;
    let mut survivors = 0u64;
    for i in 0..HOT_PREFIX_BLOCKS {
        if let Some(leaf) = fol.seq_leaf(b, i) {
            refcount_after = fol.leaf_refcount(leaf);
            if fol.leaf_resident(leaf) && fol.seq_frame(b, i).is_some() {
                survivors += 1;
            }
        }
    }
    let _ = fol.seq_release(b);
    fol.teardown();
    (shared, refcount_after, survivors, identical)
}

/// Drive `seq_append` past its first refusal.
///
/// The loop deliberately does not stop at the first `Err`. A refusal is only
/// worth anything if the sequence survives it, so every later append must be
/// refused too — one absorbed token after a refusal means the buffer kept a
/// token the cache never sealed. Returns `(refused_with, none_absorbed_after)`.
fn refuse_and_continue(
    fol: &mut Foliation,
    id: usize,
    base: u32,
    tokens: usize,
    expect: FoliationError,
) -> (bool, bool) {
    let mut refused = false;
    let mut absorbed_after_refusal = false;
    for j in 0..tokens {
        match fol.seq_append(id, base + j as u32) {
            Ok(_) => absorbed_after_refusal |= refused,
            Err(e) => refused |= e == expect,
        }
    }
    (refused, !absorbed_after_refusal)
}

/// Negative controls. Returns `(budget_refused, exhaustion_refused,
/// referenced_free_refused)`.
fn refusal_probe() -> (bool, bool, bool) {
    // Budget: a sequence declaring 2 blocks may not seal a third.
    let mut fol = Foliation::new(8, 32, 2, Policy::Foliation);
    let budget_refused = match fol.seq_create(2) {
        Ok(id) => {
            let (refused, held) = refuse_and_continue(
                &mut fol,
                id,
                900,
                3 * BLOCK_TOKENS,
                FoliationError::BudgetExceeded,
            );
            let _ = fol.seq_release(id);
            refused && held
        }
        Err(_) => false,
    };
    fol.teardown();

    // Exhaustion: every plaque referenced by the live sequence, so the frontier
    // is empty and admission must be refused rather than evicting live state.
    let mut fol = Foliation::new(3, 32, 2, Policy::Foliation);
    let exhaustion_refused = match fol.seq_create(MAX_SEQ_BLOCKS as u16) {
        Ok(id) => {
            let (refused, held) = refuse_and_continue(
                &mut fol,
                id,
                4000,
                6 * BLOCK_TOKENS,
                FoliationError::Exhausted,
            );
            let _ = fol.seq_release(id);
            refused && held
        }
        Err(_) => false,
    };
    fol.teardown();

    // Referenced free: collapsing a plaque a live sequence still holds.
    let mut fol = Foliation::new(8, 32, 2, Policy::Foliation);
    let referenced_free_refused = match fol.seq_create(4) {
        Ok(id) => {
            for j in 0..(2 * BLOCK_TOKENS) {
                let _ = fol.seq_append(id, 300 + j as u32);
            }
            let refused = match fol.seq_leaf(id, 0) {
                Some(leaf) => fol.force_collapse(leaf) == Err(FoliationError::StillReferenced),
                None => false,
            };
            let _ = fol.seq_release(id);
            refused
        }
        Err(_) => false,
    };
    fol.teardown();

    (budget_refused, exhaustion_refused, referenced_free_refused)
}

/// Run the embedded workload through the real manager and emit the boot proof.
///
/// Every field is measured during this call. Nothing is asserted that was not
/// executed.
pub fn foliation_proof_line() -> String {
    let trace = build_trace();
    let keys = trace_keys(&trace);
    let tokens: usize = trace.iter().map(|t| t.len()).sum();

    let fo = replay(Policy::Foliation, &trace, &keys);
    let lru = replay(Policy::Lru, &trace, &keys);
    let rnd = replay(Policy::Random, &trace, &keys);
    let opt = replay(Policy::Belady, &trace, &keys);

    let (shared_blocks, refcount_after, survivors, frames_identical) = share_and_refcount_probe();
    let (budget_refused, exhaustion_refused, referenced_free_refused) = refusal_probe();

    // Fraction of the LRU -> Belady headroom the foliation policy closed, in
    // basis points. Negative means the policy lost to LRU.
    let gap = opt.hit_bp as i64 - lru.hit_bp as i64;
    let gained = fo.hit_bp as i64 - lru.hit_bp as i64;
    let gap_closed_bp = if gap > 0 { gained * 10_000 / gap } else { 0 };

    let memory_ok = fo.frames_failed == 0
        && fo.frames_backed > 0
        && fo.frames_freed == fo.frames_backed
        && lru.frames_freed == lru.frames_backed;
    let sharing_ok = shared_blocks == HOT_PREFIX_BLOCKS as u64 && frames_identical;
    let refcount_ok = refcount_after == 1 && survivors == HOT_PREFIX_BLOCKS as u64;
    let safety_ok = fo.referenced_evictions == 0
        && lru.referenced_evictions == 0
        && rnd.referenced_evictions == 0
        && fo.collapse_violations == 0
        && lru.collapse_violations == 0;
    let refusals_ok = budget_refused && exhaustion_refused && referenced_free_refused;
    // The offline optimum must dominate every realizable policy on the same
    // candidate set. If it does not, the benchmark is measuring something else.
    let oracle_sane = opt.hit_bp >= fo.hit_bp && opt.hit_bp >= lru.hit_bp;
    let trace_ok = fo.descents == lru.descents
        && fo.descents == rnd.descents
        && fo.descents == opt.descents
        && fo.descents as usize == keys.len();

    let result = if memory_ok
        && sharing_ok
        && refcount_ok
        && safety_ok
        && refusals_ok
        && oracle_sane
        && trace_ok
    {
        "pass"
    } else {
        "fail"
    };

    format!(
        "[KVPOLICY] proof version=1 subsystem=foliation block_tokens={} pool_blocks={} leaf_arena={} \
requests={} tokens={} descents={} trace_keys={} \
blocks_admitted={} frames_backed={} frames_freed={} frames_failed={} \
shared_descents={} bytes_saved={} \
probe_shared_blocks={} probe_frames_identical={} probe_refcount_after_partial_free={} probe_survivors_resident={} \
evictions_foliation={} evictions_lru={} evictions_random={} \
hit_bp_foliation={} hit_bp_lru={} hit_bp_random={} hit_bp_belady={} gap_closed_bp={} \
referenced_evictions={} collapse_violations={} \
refused_budget={} refused_exhaustion={} refused_referenced_free={} \
complexity=descend<={}_children,evict<={}_plaques,lookup=O(1)_indexed \
result={}",
        BLOCK_TOKENS,
        BENCH_POOL_BLOCKS,
        BENCH_LEAF_ARENA,
        trace.len(),
        tokens,
        fo.descents,
        keys.len(),
        fo.descents - fo.shared,
        fo.frames_backed,
        fo.frames_freed,
        fo.frames_failed,
        fo.shared,
        fo.shared * PLAQUE_BYTES as u64,
        shared_blocks,
        if frames_identical { 1 } else { 0 },
        refcount_after,
        survivors,
        fo.evictions,
        lru.evictions,
        rnd.evictions,
        fo.hit_bp,
        lru.hit_bp,
        rnd.hit_bp,
        opt.hit_bp,
        gap_closed_bp,
        fo.referenced_evictions + lru.referenced_evictions + rnd.referenced_evictions,
        fo.collapse_violations + lru.collapse_violations,
        if budget_refused { 1 } else { 0 },
        if exhaustion_refused { 1 } else { 0 },
        if referenced_free_refused { 1 } else { 0 },
        MAX_CHILDREN,
        BENCH_POOL_BLOCKS,
        result
    )
}

/// Print the boot proof to serial.
pub fn emit_boot_proof() {
    crate::serial_println!("{}", foliation_proof_line());
}

// ---------------------------------------------------------------------------
// Seal ABI surface
// ---------------------------------------------------------------------------

const ABI_POOL_BLOCKS: usize = 64;
const ABI_LEAF_ARENA: usize = 512;
const ABI_MAX_SEQS: usize = 32;

static GLOBAL: spin::Mutex<Option<Foliation>> = spin::Mutex::new(None);

/// Run `f` against the global KV cache, creating it on first use.
pub fn with_global<R>(f: impl FnOnce(&mut Foliation) -> R) -> R {
    let mut guard = GLOBAL.lock();
    if guard.is_none() {
        *guard = Some(Foliation::new(
            ABI_POOL_BLOCKS,
            ABI_LEAF_ARENA,
            ABI_MAX_SEQS,
            Policy::Foliation,
        ));
    }
    f(guard.as_mut().expect("foliation initialised above"))
}

/// Map a refusal to an errno for the syscall layer.
pub fn errno(e: FoliationError) -> i64 {
    match e {
        FoliationError::NoSuchSeq => 2,        // ENOENT
        FoliationError::BudgetExceeded => 27,  // EFBIG
        FoliationError::Exhausted => 12,       // ENOMEM
        FoliationError::LeafArenaFull => 12,   // ENOMEM
        FoliationError::ChildrenFull => 28,    // ENOSPC
        FoliationError::StillReferenced => 16, // EBUSY
        FoliationError::TooManySeqs => 11,     // EAGAIN
    }
}

/// Human-readable global counters for `SYS_KV_POLICY_STATS`.
pub fn global_stats_line() -> String {
    with_global(|f| {
        let s = f.stats();
        format!(
            "policy=foliation pool_blocks={} resident={} descents={} shared={} admits={} \
evictions={} frames_backed={} frames_freed={} leaf_gc={} children_full={} \
refused_budget={} refused_exhaustion={} refused_referenced_free={}",
            ABI_POOL_BLOCKS,
            f.resident(),
            s.descents,
            s.shared,
            s.admits,
            s.evictions,
            s.frames_backed,
            s.frames_freed,
            s.leaf_gc,
            s.children_full,
            s.refused_budget,
            s.refused_exhaustion,
            s.refused_referenced_free,
        )
    })
}

/// Per-sequence counters for `SYS_KV_SEQ_STATS`.
pub fn seq_stats_line(id: usize) -> Option<String> {
    with_global(|f| {
        let (blocks, shared, admitted) = f.seq_counts(id)?;
        Some(format!(
            "seq={} blocks={} shared_on_entry={} admitted={} bytes_shared={}",
            id,
            blocks,
            shared,
            admitted,
            shared as usize * PLAQUE_BYTES
        ))
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn prefix_tokens(base: u32, blocks: usize) -> Vec<u32> {
        (0..blocks * BLOCK_TOKENS)
            .map(|j| base + j as u32)
            .collect()
    }

    /// Identical prefixes must land on identical leaves and identical frames.
    fn test_prefix_sharing_dedupes() -> TestResult {
        let mut fol = Foliation::new(16, 64, 4, Policy::Foliation);
        let toks = prefix_tokens(11_000, 3);
        let a = fol.seq_create(8).unwrap_or(usize::MAX);
        let b = fol.seq_create(8).unwrap_or(usize::MAX);
        test_assert!(a != usize::MAX && b != usize::MAX);
        for &t in &toks {
            let _ = fol.seq_append(a, t);
        }
        let resident_after_a = fol.resident();
        for &t in &toks {
            let _ = fol.seq_append(b, t);
        }
        test_assert!(
            fol.resident() == resident_after_a,
            "sharing allocated new plaques"
        );
        for i in 0..3 {
            test_assert!(fol.seq_leaf(a, i) == fol.seq_leaf(b, i), "leaf divergence");
            test_assert!(
                fol.seq_frame(a, i) == fol.seq_frame(b, i),
                "frame divergence"
            );
        }
        let s = fol.stats();
        test_assert!(s.shared == 3, "second sequence should share all 3 blocks");
        fol.teardown();
        TestResult::Pass
    }

    /// A shared block freed by one holder must survive for the others.
    fn test_refcount_survives_partial_free() -> TestResult {
        let mut fol = Foliation::new(16, 64, 4, Policy::Foliation);
        let toks = prefix_tokens(12_000, 2);
        let a = fol.seq_create(8).unwrap_or(usize::MAX);
        let b = fol.seq_create(8).unwrap_or(usize::MAX);
        test_assert!(a != usize::MAX && b != usize::MAX);
        for &t in &toks {
            let _ = fol.seq_append(a, t);
        }
        for &t in &toks {
            let _ = fol.seq_append(b, t);
        }
        let leaf = fol.seq_leaf(b, 0).unwrap_or(NONE);
        test_assert!(leaf != NONE);
        test_assert!(fol.leaf_refcount(leaf) == 2, "two holders");
        let frame_before = fol.seq_frame(b, 0);
        let _ = fol.seq_release(a);
        test_assert!(fol.leaf_refcount(leaf) == 1, "refcount after partial free");
        test_assert!(fol.leaf_resident(leaf), "shared block was dropped early");
        test_assert!(
            fol.seq_frame(b, 0) == frame_before,
            "frame moved under survivor"
        );
        fol.teardown();
        TestResult::Pass
    }

    /// Eviction must never take a referenced plaque, and must preserve the
    /// connected-subtree invariant.
    fn test_eviction_never_frees_referenced() -> TestResult {
        let mut fol = Foliation::new(6, 64, 8, Policy::Foliation);
        // Sequence `live` holds 3 blocks for the whole test.
        let live = fol.seq_create(4).unwrap_or(usize::MAX);
        test_assert!(live != usize::MAX);
        for &t in &prefix_tokens(13_000, 3) {
            let _ = fol.seq_append(live, t);
        }
        let held: Vec<u16> = (0..3).filter_map(|i| fol.seq_leaf(live, i)).collect();
        test_assert_eq!(held.len(), 3);

        // Drive churn through the remaining 3 plaques.
        for k in 0..6u32 {
            if let Ok(id) = fol.seq_create(3) {
                for &t in &prefix_tokens(14_000 + k * 1000, 3) {
                    let _ = fol.seq_append(id, t);
                }
                let _ = fol.seq_release(id);
            }
        }
        for leaf in &held {
            test_assert!(
                fol.leaf_refcount(*leaf) >= 1,
                "held block lost its reference"
            );
            test_assert!(fol.leaf_resident(*leaf), "held block was evicted");
        }
        let s = fol.stats();
        test_assert!(s.referenced_evictions == 0, "evicted a referenced plaque");
        test_assert!(fol.collapse_violations() == 0, "residency is not a subtree");
        let _ = fol.seq_release(live);
        fol.teardown();
        TestResult::Pass
    }

    /// Block tables must still resolve after the plaque pool is fragmented by
    /// interleaved release and eviction.
    fn test_block_table_after_fragmentation() -> TestResult {
        let mut fol = Foliation::new(8, 64, 8, Policy::Foliation);
        let mut ids = Vec::new();
        for k in 0..3u32 {
            if let Ok(id) = fol.seq_create(2) {
                for &t in &prefix_tokens(15_000 + k * 1000, 2) {
                    let _ = fol.seq_append(id, t);
                }
                ids.push(id);
            }
        }
        test_assert_eq!(ids.len(), 3);
        // Free the middle sequence, then churn so its slots are recycled.
        let _ = fol.seq_release(ids[1]);
        for k in 0..4u32 {
            if let Ok(id) = fol.seq_create(2) {
                for &t in &prefix_tokens(16_000 + k * 1000, 2) {
                    let _ = fol.seq_append(id, t);
                }
                let _ = fol.seq_release(id);
            }
        }
        // Surviving sequences must still map to live, distinct frames.
        for &id in &[ids[0], ids[2]] {
            for i in 0..2 {
                test_assert!(fol.seq_frame(id, i).is_some(), "block table lost a frame");
            }
            test_assert!(
                fol.seq_frame(id, 0) != fol.seq_frame(id, 1),
                "aliased blocks"
            );
        }
        test_assert_eq!(fol.collapse_violations(), 0);
        let _ = fol.seq_release(ids[0]);
        let _ = fol.seq_release(ids[2]);
        fol.teardown();
        TestResult::Pass
    }

    /// Capacity exhaustion and budget overrun must be refused, not absorbed.
    fn test_refusals() -> TestResult {
        let (budget, exhaustion, referenced) = refusal_probe();
        test_assert!(budget, "budget overrun was not refused");
        test_assert!(exhaustion, "capacity exhaustion was not refused");
        test_assert!(referenced, "freeing a referenced plaque was not refused");
        TestResult::Pass
    }

    /// A refused seal must leave the sequence able to take the next append.
    ///
    /// Both refusal paths are driven past the first `Err`: capacity exhaustion,
    /// and a prefix whose fan-out is saturated at the shipped ABI geometry. The
    /// pending buffer holds `BLOCK_TOKENS` tokens, so a refusal that left it
    /// full would make the next append store one past its end.
    fn test_refused_seal_stays_appendable() -> TestResult {
        // Exhaustion: three plaques, all referenced by the one live sequence,
        // so the fourth seal has no free face to collapse.
        let mut fol = Foliation::new(3, 32, 2, Policy::Foliation);
        let id = fol.seq_create(MAX_SEQ_BLOCKS as u16).unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        let mut first_refusal = usize::MAX;
        for j in 0..(6 * BLOCK_TOKENS) {
            match fol.seq_append(id, 4000 + j as u32) {
                Ok(_) => test_assert!(
                    first_refusal == usize::MAX,
                    "a token was absorbed after the sequence had been refused"
                ),
                Err(e) => {
                    test_assert!(e == FoliationError::Exhausted, "wrong refusal");
                    if first_refusal == usize::MAX {
                        first_refusal = j;
                    }
                }
            }
        }
        test_assert_eq!(first_refusal, 4 * BLOCK_TOKENS - 1);
        test_assert_eq!(fol.seq_counts(id).map(|c| c.0), Some(3u16));
        let _ = fol.seq_release(id);
        fol.teardown();

        // Fan-out: saturate the root leaf's children at the shipped ABI
        // geometry, where neither the pool nor the leaf arena is the binding
        // constraint, then seal one more distinct block off the root.
        let mut fol = Foliation::new(
            ABI_POOL_BLOCKS,
            ABI_LEAF_ARENA,
            ABI_MAX_SEQS,
            Policy::Foliation,
        );
        for round in 0..MAX_CHILDREN as u32 {
            let sid = fol.seq_create(1).unwrap_or(usize::MAX);
            test_assert!(sid != usize::MAX);
            for j in 0..BLOCK_TOKENS {
                let _ = fol.seq_append(sid, 60_000 + round * 100 + j as u32);
            }
            let _ = fol.seq_release(sid);
        }
        test_assert_eq!(fol.stats().children_full, 0);
        test_assert_eq!(fol.stats().descents, MAX_CHILDREN as u64);
        let sid = fol.seq_create(1).unwrap_or(usize::MAX);
        test_assert!(sid != usize::MAX);
        let mut refusals = 0u64;
        for j in 0..(2 * BLOCK_TOKENS) {
            if let Err(e) = fol.seq_append(sid, 90_000 + j as u32) {
                test_assert!(
                    e == FoliationError::ChildrenFull,
                    "wrong refusal at saturated fan-out"
                );
                refusals += 1;
            }
        }
        // One refusal for the seal attempt, then one for every later token.
        test_assert_eq!(refusals, BLOCK_TOKENS as u64 + 1);
        test_assert_eq!(fol.stats().children_full, refusals);
        test_assert_eq!(fol.seq_counts(sid).map(|c| c.0), Some(0u16));
        let _ = fol.seq_release(sid);
        fol.teardown();
        TestResult::Pass
    }

    /// A refused admission must not consume one of the prefix's child slots.
    /// Otherwise repeated refusals ratchet the fan-out to `MAX_CHILDREN` and a
    /// transient capacity refusal becomes a permanent one for that prefix.
    fn test_refused_admission_leaves_no_leaf() -> TestResult {
        let mut fol = Foliation::new(3, 64, 2, Policy::Foliation);
        let id = fol.seq_create(MAX_SEQ_BLOCKS as u16).unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        // Three blocks fill the pool and stay referenced; every seal after that
        // is refused, each with a distinct key, so each would invent a leaf.
        for j in 0..(3 * BLOCK_TOKENS) {
            test_assert!(fol.seq_append(id, 7100 + j as u32).is_ok());
        }
        let held = fol.seq_leaf(id, 2).unwrap_or(NONE);
        test_assert!(held != NONE);
        for j in 0..(20 * BLOCK_TOKENS) {
            let r = fol.seq_append(id, 7500 + j as u32);
            if j < BLOCK_TOKENS - 1 {
                // Still filling the buffer, so no seal is attempted yet.
                test_assert!(r == Ok(3), "a partial block was refused");
            } else {
                test_assert!(
                    r == Err(FoliationError::Exhausted),
                    "refusal changed shape, so the refused leaves accumulated"
                );
            }
        }
        test_assert_eq!(fol.stats().children_full, 0);
        test_assert_eq!(fol.stats().leaf_gc, 0);
        test_assert_eq!(fol.seq_counts(id).map(|c| c.0), Some(3u16));
        test_assert!(fol.leaf_resident(held), "a refused seal disturbed residency");
        test_assert_eq!(fol.collapse_violations(), 0);
        let _ = fol.seq_release(id);
        fol.teardown();
        TestResult::Pass
    }

    /// Policy comparison on one fixed trace. Asserts only what must hold
    /// structurally — the trace is identical across policies and the offline
    /// optimum dominates — never that the foliation policy wins.
    fn test_policy_vs_lru_on_fixed_trace() -> TestResult {
        let trace = build_trace();
        let keys = trace_keys(&trace);
        let fo = replay(Policy::Foliation, &trace, &keys);
        let lru = replay(Policy::Lru, &trace, &keys);
        let opt = replay(Policy::Belady, &trace, &keys);
        test_assert!(fo.descents == lru.descents, "policies saw different traces");
        test_assert!(
            fo.descents as usize == keys.len(),
            "key trace desynchronised"
        );
        test_assert!(opt.hit_bp >= fo.hit_bp, "oracle below foliation policy");
        test_assert!(opt.hit_bp >= lru.hit_bp, "oracle below LRU");
        test_assert_eq!(fo.collapse_violations, 0);
        test_assert_eq!(lru.collapse_violations, 0);
        test_assert_eq!(fo.referenced_evictions, 0);
        test_assert_eq!(lru.referenced_evictions, 0);
        TestResult::Pass
    }

    /// The proof must actually run and report `result=pass`.
    fn test_proof_line_passes() -> TestResult {
        let line = foliation_proof_line();
        test_assert!(
            line.starts_with("[KVPOLICY] proof version=1"),
            "bad proof prefix"
        );
        test_assert!(line.ends_with("result=pass"), "proof did not pass");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "foliation::prefix_sharing_dedupes",
            test_prefix_sharing_dedupes,
        );
        crate::testing::register_test(
            "foliation::refcount_survives_partial_free",
            test_refcount_survives_partial_free,
        );
        crate::testing::register_test(
            "foliation::eviction_never_frees_referenced",
            test_eviction_never_frees_referenced,
        );
        crate::testing::register_test(
            "foliation::block_table_after_fragmentation",
            test_block_table_after_fragmentation,
        );
        crate::testing::register_test("foliation::refusals", test_refusals);
        crate::testing::register_test(
            "foliation::refused_seal_stays_appendable",
            test_refused_seal_stays_appendable,
        );
        crate::testing::register_test(
            "foliation::refused_admission_leaves_no_leaf",
            test_refused_admission_leaves_no_leaf,
        );
        crate::testing::register_test(
            "foliation::policy_vs_lru_on_fixed_trace",
            test_policy_vs_lru_on_fixed_trace,
        );
        crate::testing::register_test("foliation::proof_line_passes", test_proof_line_passes);
    }
}
