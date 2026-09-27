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
/// live continuations per prefix: a full prefix first reclaims a continuation
/// no live sequence uses, and only when all 32 are live does `descend` refuse
/// and report `children_full`. Upgrade path is an open-addressed key->child
/// map per leaf, which trades 3x metadata for unbounded fan-out.
pub const MAX_CHILDREN: usize = 32;
/// Maximum blocks in one sequence's block table.
pub const MAX_SEQ_BLOCKS: usize = 16;
/// Bytes backing one plaque (one physical frame).
pub const PLAQUE_BYTES: usize = 4096;

const NONE: u16 = u16::MAX;
const ROOT: u16 = 0;
/// Owner of the sequences the kernel opens for its own proofs and tests.
/// `scheduler::current_task_id()` reports 0 outside any task.
const KERNEL: u64 = 0;

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
    /// Depth, then recency: the foliation ranking without `entrants`.
    /// Same-budget locality-only null — root-adjacent blocks, where sinks and
    /// shared prompts sit, outlive deep ones; within a depth, recent outlives
    /// old.
    Locality,
    /// The LRU or the foliation ranking, whichever has been choosing the
    /// better victims on this workload.
    ///
    /// Every eviction where the two rankings pick different victims opens a
    /// duel. The duel is settled by whichever of the two leaves is requested
    /// again first: that one was the worse victim, which is Belady's criterion
    /// applied to the two candidates. A saturating score of settled duels picks
    /// the ranking for the next eviction.
    ///
    /// It starts under the foliation ranking, and ties go to it, because that
    /// is the side whose mistakes surface fastest. The foliation ranking errs
    /// by evicting a recent block LRU would have kept, and that duel settles
    /// as soon as the block returns — within about one pool of reuse, or LRU
    /// would not have been right. LRU errs by evicting an old block the
    /// foliation ranking would have kept, and that duel settles only after a
    /// reuse distance longer than the pool. Started under LRU instead, the
    /// boot trace loses its hot prefix's first return while the evidence is
    /// in flight: 761 bp against 952 at the headline pool size.
    Adaptive,
}

/// Open duels an adaptive cache remembers. A duel still open when its slot is
/// reused expires without a verdict.
///
/// ponytail: fixed ring, scanned linearly on every descent. Ceiling is a
/// reuse distance of about `DUEL_SLOTS` disputed evictions — a block that
/// returns later than that can no longer vote. Upgrade path is a per-leaf open-
/// duel mark, which makes the common descent O(1).
const DUEL_SLOTS: usize = 32;
/// Saturation of the duel score, in settled duels. After a phase change the
/// ranking flips within `PSEL_MAX + 1` verdicts against it.
const PSEL_MAX: i32 = 8;

/// One disputed eviction awaiting a verdict. `new_leaf` drops every duel
/// naming a leaf slot it hands to a new block, so a reclaimed slot cannot
/// settle a duel its new block was never part of; the keys are a second check.
#[derive(Clone, Copy)]
struct Duel {
    /// Leaf the active ranking evicted; `NONE` marks a free slot.
    evicted: u16,
    evicted_key: u64,
    /// Leaf the other ranking would have evicted instead.
    spared: u16,
    spared_key: u64,
    /// The LRU ranking chose the eviction.
    by_lru: bool,
}

const NO_DUEL: Duel = Duel {
    evicted: NONE,
    evicted_key: 0,
    spared: NONE,
    spared_key: 0,
    by_lru: false,
};

/// Why a KV-cache operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FoliationError {
    /// No such sequence id, or the sequence was already released.
    NoSuchSeq,
    /// Sequence reached its declared block budget.
    BudgetExceeded,
    /// Out of memory for a plaque: either every resident plaque is referenced
    /// so nothing may be collapsed, or no physical frame could be obtained to
    /// back the block. `frames_failed` separates the two after the fact.
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
    /// The block's tokens. `key` only narrows the search; these decide
    /// whether a descent may share this leaf.
    tokens: [u32; BLOCK_TOKENS],
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
            tokens: [0; BLOCK_TOKENS],
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

/// Victim rank of `l` under a stateless ranking; the lowest rank is evicted.
/// `Policy::Belady` needs the oracle and is ranked in `pick_victim`.
fn rank_of(policy: Policy, l: &Leaf) -> [u64; 3] {
    match policy {
        Policy::Foliation => [l.entrants as u64, u64::MAX - l.depth as u64, l.last_use],
        Policy::Lru => [l.last_use, 0, 0],
        // from https://github.com/triton-lang/kernels/pull/22: sink + local window, as a null
        Policy::Locality => [u64::MAX - l.depth as u64, l.last_use, 0],
        Policy::Random | Policy::Belady | Policy::Adaptive => [0, 0, 0],
    }
}

/// A resident KV block: one physical frame bound to one leaf.
struct Plaque {
    /// Owning leaf, or `NONE` when the slot is free.
    leaf: u16,
    /// `Some` for every plaque with an owning leaf, and `None` only while the
    /// slot sits on the free list. `admit` refuses rather than publish a leaf
    /// it could not back, so residency implies a frame.
    frame: Option<PhysAddr>,
}

/// A live inference sequence. Its block table is its path down the foliation.
struct Seq {
    active: bool,
    /// Task that opened the sequence. Only it may append to, release, or read
    /// the counters of this sequence.
    owner: u64,
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
            owner: 0,
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
    /// Plaques collapsed while a live sequence still referenced them. Counted
    /// in `collapse`, which every removal passes through, so it does not
    /// depend on the victim filter it is meant to check. Must stay zero;
    /// tearing down under a live sequence raises it.
    pub referenced_evictions: u64,
    /// Appends refused for exceeding the sequence budget.
    pub refused_budget: u64,
    /// Admissions refused because the frontier was empty.
    pub refused_exhaustion: u64,
    /// Explicit frees refused because the plaque was still referenced.
    pub refused_referenced_free: u64,
    /// Descents that could not share because fan-out was saturated.
    pub children_full: u64,
    /// TSC cycles spent choosing eviction victims: the O(pool_blocks) frontier
    /// scan, measured around every call, whatever the policy.
    pub scan_cycles: u64,
    /// `Policy::Adaptive`: evictions the two rankings disputed.
    pub duels: u64,
    /// `Policy::Adaptive`: duels settled for the LRU ranking.
    pub duels_lru: u64,
    /// `Policy::Adaptive`: duels settled for the foliation ranking.
    pub duels_foliation: u64,
    /// `Policy::Adaptive`: TSC cycles spent settling duels, measured around
    /// every descent.
    pub duel_cycles: u64,
}

/// Time-stamp counter. A cost measure for the proof and the stats line, never
/// an input to any decision.
fn tsc() -> u64 {
    // SAFETY: RDTSC reads a counter and has no memory effects; every x86_64
    // CPU implements it.
    unsafe { core::arch::x86_64::_rdtsc() }
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
    /// `Policy::Adaptive` only: open duels, the next slot to reuse, and the
    /// duel score (positive favours LRU).
    duels: [Duel; DUEL_SLOTS],
    duel_next: usize,
    psel: i32,
    stats: FoliationStats,
}

/// Fold a block's tokens into the running prefix key.
///
/// The key of a block depends only on the whole token prefix, never on
/// residency or policy, so the same trace produces the same key sequence under
/// every policy. That is what makes the policy comparison and the Belady oracle
/// well defined.
///
/// A key is a 64-bit digest, not an identity: distinct token blocks can share
/// one (`COLLIDE_A`/`COLLIDE_B` below), which is why `find_child` compares the
/// tokens themselves before sharing a plaque.
pub const fn fold_key(prev: u64, tokens: &[u32]) -> u64 {
    let mut h = prev ^ 0x9e37_79b9_7f4a_7c15;
    let mut i = 0;
    while i < tokens.len() {
        h ^= tokens[i] as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
        h ^= h >> 29;
        i += 1;
    }
    h
}

/// Two different blocks with the same `fold_key` off the root
/// (`0xe0e00501162145bc`), found by meet-in-the-middle: each token step is a
/// bijection on the running key, so the last two steps can be inverted from
/// the target. Kept as a known collision so the sharing test exercises one.
const COLLIDE_A: [u32; BLOCK_TOKENS] = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007];
const COLLIDE_B: [u32; BLOCK_TOKENS] = [
    1000,
    1001,
    1002,
    1003,
    1004,
    2_191_639_884,
    3_946_364_168,
    3_153_004_709,
];
const _: () = assert!(fold_key(0, &COLLIDE_A) == fold_key(0, &COLLIDE_B));

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
            rng: null_seed(0),
            oracle: Vec::new(),
            duels: [NO_DUEL; DUEL_SLOTS],
            duel_next: 0,
            psel: 0,
            stats: FoliationStats::default(),
        }
    }

    /// The ranking the next eviction applies: the configured policy, or for
    /// `Policy::Adaptive` LRU once it leads on duels and the foliation ranking
    /// otherwise.
    pub fn ranking(&self) -> Policy {
        match self.policy {
            Policy::Adaptive if self.psel > 0 => Policy::Lru,
            Policy::Adaptive => Policy::Foliation,
            p => p,
        }
    }

    /// Switch the eviction policy of a live cache. Only victim choice changes:
    /// every policy works on the same candidate set, so residency, references
    /// and backing are untouched. Duel history is dropped. `Policy::Belady` is
    /// refused, since it needs a future trace no live cache has, and
    /// `Policy::Random`, which is a null model and not a policy to serve under.
    pub fn set_policy(&mut self, policy: Policy) -> bool {
        if matches!(policy, Policy::Belady | Policy::Random) {
            return false;
        }
        self.policy = policy;
        self.duels = [NO_DUEL; DUEL_SLOTS];
        self.duel_next = 0;
        self.psel = 0;
        true
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

    /// Whether `id` names a live sequence opened by `owner`.
    ///
    /// Sequence ids are small indices into one table shared by every task, so
    /// without this any task could append to, release, or read another task's
    /// sequence by guessing an id. A foreign id is refused as `NoSuchSeq`,
    /// exactly as an unused one is, so the refusal does not reveal that the
    /// sequence exists — the rule `syscall::table::fd_lookup` applies to fds.
    fn owned(&self, id: usize, owner: u64) -> bool {
        self.seqs.get(id).map(|s| s.active && s.owner == owner) == Some(true)
    }

    /// Open a sequence with a hard block budget, owned by task `owner`.
    pub fn seq_create(&mut self, budget_blocks: u16, owner: u64) -> Result<usize, FoliationError> {
        let budget = budget_blocks.min(MAX_SEQ_BLOCKS as u16);
        for (id, s) in self.seqs.iter_mut().enumerate() {
            if !s.active {
                *s = Seq::blank();
                s.active = true;
                s.owner = owner;
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
    pub fn seq_append(&mut self, id: usize, owner: u64, token: u32) -> Result<u16, FoliationError> {
        if !self.owned(id, owner) {
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
    pub fn seq_counts(&self, id: usize, owner: u64) -> Option<(u16, u32, u32)> {
        if !self.owned(id, owner) {
            return None;
        }
        let s = &self.seqs[id];
        Some((s.nblocks, s.hits, s.admits))
    }

    /// Drop a sequence's references. Plaques stay resident — that is the cache.
    /// A block shared with another live sequence keeps a positive refcount.
    pub fn seq_release(&mut self, id: usize, owner: u64) -> Result<u16, FoliationError> {
        if !self.owned(id, owner) {
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

    /// Release every sequence `owner` opened. Returns how many there were.
    pub fn release_owner(&mut self, owner: u64) -> usize {
        (0..self.seqs.len())
            .filter(|&id| self.seq_release(id, owner).is_ok())
            .count()
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
    /// rooted at the root leaf, and every block of every live sequence is
    /// resident. Returns the number of violations, one per orphaned plaque
    /// plus one per live block without a plaque.
    /// O(pool_blocks + max_seqs * MAX_SEQ_BLOCKS).
    pub fn collapse_violations(&self) -> usize {
        let mut bad = 0;
        for s in self.seqs.iter().filter(|s| s.active) {
            for &leaf in &s.blocks[..s.nblocks as usize] {
                if self.leaves[leaf as usize].slot == NONE {
                    bad += 1;
                }
            }
        }
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
    ///
    /// This is an elementary collapse of every resident leaf, not a shortcut
    /// that only reclaims frames: dropping the frames while the leaves still
    /// pointed at their slots would leave `leaf_resident` true for a block
    /// `seq_frame` can no longer resolve, and would leave the pool reporting
    /// itself full.
    pub fn teardown(&mut self) {
        for i in 0..self.plaques.len() {
            let leaf = self.plaques[i].leaf;
            if leaf != NONE {
                self.collapse(leaf);
            }
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

        let child = self.find_child(parent, key, &tokens);
        let fresh = child.is_none();
        let leaf = match child {
            Some(c) => c,
            None => {
                let c = self.new_leaf(parent, key, tokens)?;
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
        if self.policy == Policy::Adaptive {
            let t0 = tsc();
            self.settle_duels(leaf);
            self.stats.duel_cycles += tsc().wrapping_sub(t0);
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
    ///
    /// A key match alone is not a prefix match: `fold_key` is a 64-bit digest
    /// and collides. A child is shared only when its stored tokens equal the
    /// block being sealed; a colliding block falls through and gets its own
    /// leaf, so a collision costs a missed share, never a wrong plaque.
    fn find_child(&self, parent: u16, key: u64, tokens: &[u32; BLOCK_TOKENS]) -> Option<u16> {
        let p = &self.leaves[parent as usize];
        for i in 0..p.nchild as usize {
            let c = p.child[i];
            if c != NONE
                && self.leaves[c as usize].key == key
                && self.leaves[c as usize].tokens == *tokens
            {
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

    fn new_leaf(
        &mut self,
        parent: u16,
        key: u64,
        tokens: [u32; BLOCK_TOKENS],
    ) -> Result<u16, FoliationError> {
        if self.leaves[parent as usize].nchild as usize >= MAX_CHILDREN
            && !self.reclaim_dead_child(parent)
        {
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
        // The slot takes a new block, so a duel naming it can no longer be
        // settled by it. Its key is a digest that collides (`COLLIDE_A`/`_B`),
        // so the key alone cannot tell the new block from the old one.
        for d in self.duels.iter_mut() {
            if d.evicted == idx || d.spared == idx {
                *d = NO_DUEL;
            }
        }
        let l = &mut self.leaves[idx as usize];
        *l = Leaf::blank();
        l.used = true;
        l.parent = parent;
        l.key = key;
        l.tokens = tokens;
        l.depth = depth;
        Ok(idx)
    }

    /// Free one of `parent`'s child slots held by a continuation no live
    /// sequence uses. Returns false when every child is live.
    ///
    /// A sequence's block table is a path from the root, so a child with no
    /// references heads a subtree with none: the whole subtree is collapsed
    /// and its leaves blanked. Without this a prefix that ever had
    /// `MAX_CHILDREN` continuations refused every new one for the rest of the
    /// boot, because the leaf GC runs only on a full arena and never takes a
    /// resident leaf. Prefers a child already out of the pool, then the least
    /// recently used.
    fn reclaim_dead_child(&mut self, parent: u16) -> bool {
        let p = &self.leaves[parent as usize];
        let Some(top) = p.child[..p.nchild as usize]
            .iter()
            .copied()
            .filter(|&c| self.leaves[c as usize].refcount == 0)
            .min_by_key(|&c| {
                let l = &self.leaves[c as usize];
                (l.slot != NONE, l.last_use)
            })
        else {
            return false;
        };
        self.unlink_child(parent, top);
        let mut stack = Vec::new();
        stack.push(top);
        while let Some(x) = stack.pop() {
            let l = &self.leaves[x as usize];
            stack.extend_from_slice(&l.child[..l.nchild as usize]);
            if l.slot != NONE {
                self.collapse(x);
                self.stats.evictions += 1;
            }
            self.leaves[x as usize] = Leaf::blank();
            self.stats.leaf_gc += 1;
        }
        true
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

    /// Bind a plaque to `leaf`, evicting a free face first if the pool is full.
    ///
    /// Either the leaf comes out resident with a frame behind it, or nothing
    /// about the leaf changes and the caller gets an error. There is no third
    /// outcome, which is what keeps `leaf_resident` and `seq_frame` from
    /// disagreeing.
    fn admit(&mut self, leaf: u16) -> Result<(), FoliationError> {
        if self.free_slots.is_empty() {
            let t0 = tsc();
            let picked = self.pick_victim();
            self.stats.scan_cycles += tsc().wrapping_sub(t0);
            let victim = match picked {
                Some(v) => v,
                None => {
                    self.stats.refused_exhaustion += 1;
                    return Err(FoliationError::Exhausted);
                }
            };
            self.collapse(victim);
            self.stats.evictions += 1;
        }
        let slot = self.free_slots.pop().ok_or(FoliationError::Exhausted)?;
        let cell = (self.leaves[leaf as usize].key % 8) as usize;
        let allocation = match topo_ram::proof_hint(ZoneHint::Low, cell) {
            Some(hint) => topo_ram::alloc_frames(1, ZoneHint::Low, Some(&hint)),
            None => topo_ram::alloc_frames(1, ZoneHint::Low, None),
        };
        let frame = match allocation {
            Some(f) => f,
            None => {
                // No physical memory behind this plaque, so the leaf must not
                // become resident. Recording it as resident anyway would make
                // `leaf_resident` report true while `seq_frame` returns `None`,
                // and `None` is what a caller reads as "no such block": the
                // append would succeed, every later lookup of that block would
                // silently miss, and the pool slot would stay consumed until
                // eviction. Refusing hands the slot back and reports out of
                // memory, which the caller may retry.
                self.stats.frames_failed += 1;
                self.free_slots.push(slot);
                return Err(FoliationError::Exhausted);
            }
        };
        self.stats.frames_backed += 1;
        self.plaques[slot as usize] = Plaque {
            leaf,
            frame: Some(frame),
        };
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
        if self.leaves[leaf as usize].refcount > 0 {
            self.stats.referenced_evictions += 1;
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
        let ranking = self.ranking();
        // `Policy::Adaptive` also ranks every candidate under the ranking it
        // is not applying, so a disagreement can be recorded as a duel.
        let rival = match (self.policy, ranking) {
            (Policy::Adaptive, Policy::Lru) => Some(Policy::Foliation),
            (Policy::Adaptive, _) => Some(Policy::Lru),
            _ => None,
        };
        let mut best: Option<(u16, [u64; 3])> = None;
        let mut rival_best: Option<(u16, [u64; 3])> = None;
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
            let rank = match ranking {
                Policy::Belady => [u64::MAX - self.next_use(l.key), l.last_use, 0],
                r => rank_of(r, l),
            };
            if let Some(r) = rival {
                let other = rank_of(r, l);
                if rival_best.map(|(_, b)| other < b).unwrap_or(true) {
                    rival_best = Some((p.leaf, other));
                }
            }
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
        if let (Some((victim, _)), Some((spared, _))) = (best, rival_best) {
            if victim != spared {
                self.duels[self.duel_next] = Duel {
                    evicted: victim,
                    evicted_key: self.leaves[victim as usize].key,
                    spared,
                    spared_key: self.leaves[spared as usize].key,
                    by_lru: ranking == Policy::Lru,
                };
                self.duel_next = (self.duel_next + 1) % DUEL_SLOTS;
                self.stats.duels += 1;
            }
        }
        best.map(|(leaf, _)| leaf)
    }

    /// `leaf` is being requested: settle every open duel it was part of.
    ///
    /// Of the two leaves in a duel, the one requested first was the worse
    /// victim. If it is the evicted one, the ranking that evicted it lost; if
    /// it is the spared one, that ranking won.
    fn settle_duels(&mut self, leaf: u16) {
        let key = self.leaves[leaf as usize].key;
        for i in 0..DUEL_SLOTS {
            let d = self.duels[i];
            let lru_won = if d.evicted == NONE {
                continue;
            } else if d.evicted == leaf && d.evicted_key == key {
                !d.by_lru
            } else if d.spared == leaf && d.spared_key == key {
                d.by_lru
            } else {
                continue;
            };
            self.duels[i] = NO_DUEL;
            if lru_won {
                self.psel = (self.psel + 1).min(PSEL_MAX);
                self.stats.duels_lru += 1;
            } else {
                self.psel = (self.psel - 1).max(-PSEL_MAX);
                self.stats.duels_foliation += 1;
            }
        }
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

// The boot trace admits more distinct blocks between two reads of its hot
// prefix than the pool holds, so recency has always evicted the prefix by the
// time it returns: LRU's 0 on it is a property of the trace.
const _: () =
    assert!(TAIL_BLOCKS + COLD_PER_ROUND * (COLD_PREFIX_BLOCKS + TAIL_BLOCKS) > BENCH_POOL_BLOCKS);

const CHAT_CONVERSATIONS: u32 = 16;
const CHAT_LIVE: u32 = 4;
/// Every conversation live at once: reuse distance then exceeds the pool and
/// the chat result reverses.
const CHAT_LIVE_WIDE: u32 = 16;
const CHAT_TURNS: u32 = 6;
const CHAT_SYSTEM_BLOCKS: usize = 2;

/// Pool sizes the proof replays every trace at, bracketing
/// `BENCH_POOL_BLOCKS`. The smallest holds the longest request, so no
/// admission is refused for capacity and every point measures victim choice
/// alone.
const SWEEP_POOLS: [usize; 6] = [8, 12, 16, 24, 32, 48];
/// Index of `BENCH_POOL_BLOCKS` in `SWEEP_POOLS`: the headline column.
const HEADLINE_AT: usize = 3;
const _: () = assert!(SWEEP_POOLS[HEADLINE_AT] == BENCH_POOL_BLOCKS);
/// Policies the sweep replays. Belady is last and bounds every row before it.
const SWEPT: [Policy; 5] = [
    Policy::Foliation,
    Policy::Lru,
    Policy::Locality,
    Policy::Adaptive,
    Policy::Belady,
];
const SW_FOL: usize = 0;
const SW_LRU: usize = 1;
const SW_LOC: usize = 2;
const SW_ADAPTIVE: usize = 3;
const SW_BELADY: usize = SWEPT.len() - 1;
const _: () = assert!(
    HOT_PREFIX_BLOCKS + TAIL_BLOCKS <= SWEEP_POOLS[0]
        && COLD_PREFIX_BLOCKS + TAIL_BLOCKS <= SWEEP_POOLS[0]
        && CHAT_SYSTEM_BLOCKS + CHAT_TURNS as usize <= SWEEP_POOLS[0]
);

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

/// Multi-turn chat: `live` conversations served round robin, each turn
/// resending a shared system prompt and the conversation so far plus one new
/// block. A conversation ends after `CHAT_TURNS` turns and the next takes its
/// slot. Reuse follows recency — the block a conversation wrote last is the
/// first one its next turn re-reads past the prompt — so this is the request
/// shape the boot trace is not.
// from https://github.com/NVIDIA/NeMo-Relay/pull/481: a stable scaffold under varying turns
fn build_chat_trace(live: u32) -> Vec<Vec<u32>> {
    let mut trace = Vec::new();
    let mut started = live;
    // (conversation, turns already served)
    let mut live: Vec<(u32, u32)> = (0..live).map(|c| (c, 0)).collect();
    while !live.is_empty() {
        let mut next = Vec::new();
        for (conv, done) in live {
            let mut tokens: Vec<u32> = (0..CHAT_SYSTEM_BLOCKS * BLOCK_TOKENS)
                .map(|j| 1000 + j as u32)
                .collect();
            for turn in 0..=done {
                for j in 0..BLOCK_TOKENS as u32 {
                    tokens.push(100_000 + conv * 1000 + turn * 10 + j);
                }
            }
            trace.push(tokens);
            if done + 1 < CHAT_TURNS {
                next.push((conv, done + 1));
            } else if started < CHAT_CONVERSATIONS {
                next.push((started, 0));
                started += 1;
            }
        }
        live = next;
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

/// Seeds the `Policy::Random` null is replayed under in the boot proof.
const NULL_SEEDS: u64 = 32;

/// Seed `i` of the random null. Seed 0 is the generator's construction seed.
const fn null_seed(i: u64) -> u64 {
    0x2545_F491_4F6C_DD1D ^ i.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

struct Replay {
    /// Digest of which leaves were resident at the end of the trace. Two runs
    /// with different digests made different victim choices.
    resident_digest: u64,
    hit_bp: u64,
    evictions: u64,
    descents: u64,
    shared: u64,
    collapse_violations: usize,
    referenced_evictions: u64,
    frames_backed: u64,
    frames_freed: u64,
    frames_failed: u64,
    refused_exhaustion: u64,
    /// Cycles in the victim scan.
    scan_cycles: u64,
    /// `Policy::Adaptive`: cycles settling duels, duels opened, and duels
    /// settled for LRU and for the foliation ranking.
    duel_cycles: u64,
    duels: [u64; 3],
}

/// Replay `trace` under `policy` at the headline pool size.
fn replay(policy: Policy, trace: &[Vec<u32>], keys: &[u64], seed: u64) -> Replay {
    replay_at(BENCH_POOL_BLOCKS, policy, trace, keys, seed)
}

/// Replay `trace` under `policy` with `pool` plaques. `seed` drives
/// `Policy::Random` only.
fn replay_at(pool: usize, policy: Policy, trace: &[Vec<u32>], keys: &[u64], seed: u64) -> Replay {
    let mut fol = Foliation::new(pool, BENCH_LEAF_ARENA, BENCH_MAX_SEQS, policy);
    fol.rng = seed;
    if policy == Policy::Belady {
        fol.set_oracle(keys.to_vec());
    }
    for tokens in trace {
        let budget = (tokens.len() / BLOCK_TOKENS) as u16;
        let id = match fol.seq_create(budget, KERNEL) {
            Ok(id) => id,
            Err(_) => continue,
        };
        for &t in tokens {
            let _ = fol.seq_append(id, KERNEL, t);
        }
        let _ = fol.seq_release(id, KERNEL);
    }
    let violations = fol.collapse_violations();
    let resident_digest = (0..fol.leaves.len() as u32)
        .filter(|&i| fol.leaves[i as usize].slot != NONE)
        .fold(0, |h, i| fold_key(h, &[i]));
    let s = fol.stats();
    fol.teardown();
    let after = fol.stats();
    Replay {
        resident_digest,
        hit_bp: (s.shared * 10_000).checked_div(s.descents).unwrap_or(0),
        evictions: s.evictions,
        descents: s.descents,
        shared: s.shared,
        collapse_violations: violations,
        referenced_evictions: s.referenced_evictions,
        frames_backed: s.frames_backed,
        frames_freed: after.frames_freed,
        frames_failed: s.frames_failed,
        refused_exhaustion: s.refused_exhaustion,
        scan_cycles: s.scan_cycles,
        duel_cycles: s.duel_cycles,
        duels: [s.duels, s.duels_lru, s.duels_foliation],
    }
}

/// Hit rate of `policy` on one trace at every `SWEEP_POOLS` size, and whether
/// every one of those replays held the cache invariants with no admission
/// refused.
fn sweep(policy: Policy, trace: &[Vec<u32>], keys: &[u64]) -> ([u64; SWEEP_POOLS.len()], bool) {
    let mut hits = [0u64; SWEEP_POOLS.len()];
    let mut ok = true;
    for (i, &pool) in SWEEP_POOLS.iter().enumerate() {
        let r = replay_at(pool, policy, trace, keys, null_seed(0));
        hits[i] = r.hit_bp;
        ok &= r.descents as usize == keys.len()
            && r.refused_exhaustion == 0
            && r.frames_failed == 0
            && r.frames_freed == r.frames_backed
            && r.referenced_evictions == 0
            && r.collapse_violations == 0;
    }
    (hits, ok)
}

/// `a,b,c` — the proof's list encoding.
fn csv(values: &[u64]) -> String {
    values
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join(",")
}

/// A cycle cost of `policy` on `trace`, the lower of two identical replays.
/// Hit counts are deterministic and cycle counts are not: under QEMU the first
/// replay also pays translation, and a host that deschedules the vCPU
/// stretches whichever replay it lands in.
fn cycle_cost(policy: Policy, trace: &[Vec<u32>], keys: &[u64], cost: fn(&Replay) -> u64) -> u64 {
    (0..2)
        .map(|_| cost(&replay(policy, trace, keys, null_seed(0))))
        .min()
        .unwrap_or(0)
}

/// Victim-scan cycles per eviction.
fn per_eviction(r: &Replay) -> u64 {
    r.scan_cycles.checked_div(r.evictions).unwrap_or(0)
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

    let a = match fol.seq_create(8, KERNEL) {
        Ok(id) => id,
        Err(_) => return (0, 0, 0, false),
    };
    for &t in &prefix {
        let _ = fol.seq_append(a, KERNEL, t);
    }
    let b = match fol.seq_create(8, KERNEL) {
        Ok(id) => id,
        Err(_) => return (0, 0, 0, false),
    };
    for &t in &prefix {
        let _ = fol.seq_append(b, KERNEL, t);
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

    let _ = fol.seq_release(a, KERNEL);
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
    let _ = fol.seq_release(b, KERNEL);
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
        match fol.seq_append(id, KERNEL, base + j as u32) {
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
    let budget_refused = match fol.seq_create(2, KERNEL) {
        Ok(id) => {
            let (refused, held) = refuse_and_continue(
                &mut fol,
                id,
                900,
                3 * BLOCK_TOKENS,
                FoliationError::BudgetExceeded,
            );
            let _ = fol.seq_release(id, KERNEL);
            refused && held
        }
        Err(_) => false,
    };
    fol.teardown();

    // Exhaustion: every plaque referenced by the live sequence, so the frontier
    // is empty and admission must be refused rather than evicting live state.
    let mut fol = Foliation::new(3, 32, 2, Policy::Foliation);
    let exhaustion_refused = match fol.seq_create(MAX_SEQ_BLOCKS as u16, KERNEL) {
        Ok(id) => {
            let (refused, held) = refuse_and_continue(
                &mut fol,
                id,
                4000,
                6 * BLOCK_TOKENS,
                FoliationError::Exhausted,
            );
            let _ = fol.seq_release(id, KERNEL);
            refused && held
        }
        Err(_) => false,
    };
    fol.teardown();

    // Referenced free: collapsing a plaque a live sequence still holds.
    let mut fol = Foliation::new(8, 32, 2, Policy::Foliation);
    let referenced_free_refused = match fol.seq_create(4, KERNEL) {
        Ok(id) => {
            for j in 0..(2 * BLOCK_TOKENS) {
                let _ = fol.seq_append(id, KERNEL, 300 + j as u32);
            }
            let refused = match fol.seq_leaf(id, 0) {
                Some(leaf) => fol.force_collapse(leaf) == Err(FoliationError::StillReferenced),
                None => false,
            };
            let _ = fol.seq_release(id, KERNEL);
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

    let fo = replay(Policy::Foliation, &trace, &keys, null_seed(0));
    let lru = replay(Policy::Lru, &trace, &keys, null_seed(0));
    let rnd = replay(Policy::Random, &trace, &keys, null_seed(0));
    let opt = replay(Policy::Belady, &trace, &keys, null_seed(0));
    let loc = replay(Policy::Locality, &trace, &keys, null_seed(0));

    // The random null is a distribution over seeds, not the single draw in
    // `rnd` (seed 0, kept so `hit_bp_random` stays comparable across proof
    // runs). Every seed runs at the same budget on the same trace and must
    // hold the same safety and memory invariants.
    let mut null_beaten = 0u64;
    let mut null_min = u64::MAX;
    let mut null_max = 0u64;
    let mut null_ok = true;
    let mut digests = Vec::new();
    for i in 0..NULL_SEEDS {
        let r = replay(Policy::Random, &trace, &keys, null_seed(i));
        null_beaten += u64::from(fo.hit_bp > r.hit_bp);
        null_min = null_min.min(r.hit_bp);
        null_max = null_max.max(r.hit_bp);
        null_ok &= r.referenced_evictions == 0
            && r.collapse_violations == 0
            && r.descents == fo.descents
            && r.frames_freed == r.frames_backed;
        digests.push(r.resident_digest);
    }
    digests.sort_unstable();
    digests.dedup();

    let (shared_blocks, refcount_after, survivors, frames_identical) = share_and_refcount_probe();
    let (budget_refused, exhaustion_refused, referenced_free_refused) = refusal_probe();

    // A second request shape, whose reuse follows recency. Its margins are
    // recorded, not gated; every replay of it must hold the same invariants
    // the boot trace does.
    let chat = build_chat_trace(CHAT_LIVE);
    let chat_keys = trace_keys(&chat);
    let chat_fo = replay(Policy::Foliation, &chat, &chat_keys, null_seed(0));
    let chat_lru = replay(Policy::Lru, &chat, &chat_keys, null_seed(0));
    let chat_loc = replay(Policy::Locality, &chat, &chat_keys, null_seed(0));
    let chat_opt = replay(Policy::Belady, &chat, &chat_keys, null_seed(0));
    let chat_holds = |r: &Replay| {
        r.descents as usize == chat_keys.len()
            && r.frames_failed == 0
            && r.frames_freed == r.frames_backed
            && r.hit_bp <= chat_opt.hit_bp
    };
    let mut chat_ok = chat_holds(&chat_fo)
        && chat_holds(&chat_lru)
        && chat_holds(&chat_loc)
        && chat_holds(&chat_opt);
    let mut chat_referenced = chat_fo.referenced_evictions
        + chat_lru.referenced_evictions
        + chat_loc.referenced_evictions
        + chat_opt.referenced_evictions;
    let mut chat_violations = chat_fo.collapse_violations
        + chat_lru.collapse_violations
        + chat_loc.collapse_violations
        + chat_opt.collapse_violations;
    let mut chat_beaten = 0u64;
    for i in 0..NULL_SEEDS {
        let r = replay(Policy::Random, &chat, &chat_keys, null_seed(i));
        chat_beaten += u64::from(chat_fo.hit_bp > r.hit_bp);
        chat_ok &= chat_holds(&r);
        chat_referenced += r.referenced_evictions;
        chat_violations += r.collapse_violations;
    }

    // Pool-size sweep: every trace, every realizable policy and the oracle, at
    // every `SWEEP_POOLS` size. Which policy wins follows whether reuse
    // distance exceeds the pool, so one pool size cannot carry a policy claim.
    // Recorded, not gated, except that the oracle bounds every point and the
    // headline replays above must be the sweep's own column at
    // `BENCH_POOL_BLOCKS`.
    let wide = build_chat_trace(CHAT_LIVE_WIDE);
    let wide_keys = trace_keys(&wide);
    let traces = [
        ("boot", &trace, &keys),
        ("chat", &chat, &chat_keys),
        ("chat16", &wide, &wide_keys),
    ];
    let mut table = [[[0u64; SWEEP_POOLS.len()]; SWEPT.len()]; 3];
    let mut sweep_ok = true;
    let mut sweep_fields = String::new();
    for ((name, tr, ks), rows) in traces.iter().zip(table.iter_mut()) {
        for (row, &policy) in rows.iter_mut().zip(&SWEPT) {
            let (hits, ok) = sweep(policy, tr, ks);
            *row = hits;
            sweep_ok &= ok;
            sweep_fields.push_str(&format!(
                " sweep_{}_{}={}",
                name,
                policy_tag(policy),
                csv(row)
            ));
        }
        let belady = rows[SW_BELADY];
        sweep_ok &= rows
            .iter()
            .all(|row| row.iter().zip(&belady).all(|(h, b)| h <= b));
    }
    let col = |trace: usize, row: usize| table[trace][row][HEADLINE_AT];
    sweep_ok &= [fo.hit_bp, lru.hit_bp, loc.hit_bp, opt.hit_bp]
        == [
            col(0, SW_FOL),
            col(0, SW_LRU),
            col(0, SW_LOC),
            col(0, SW_BELADY),
        ]
        && [
            chat_fo.hit_bp,
            chat_lru.hit_bp,
            chat_loc.hit_bp,
            chat_opt.hit_bp,
        ] == [
            col(1, SW_FOL),
            col(1, SW_LRU),
            col(1, SW_LOC),
            col(1, SW_BELADY),
        ];
    let scan_fo = cycle_cost(Policy::Foliation, &trace, &keys, per_eviction);
    let scan_lru = cycle_cost(Policy::Lru, &trace, &keys, per_eviction);

    // The adaptive policy against the random null, per seed: wherever the
    // LRU or the foliation ranking beats a seed at the headline pool size,
    // the adaptive policy has to beat it too.
    let mut adaptive_beaten = [0u64; 3];
    let mut adaptive_regressions = 0u64;
    for (t, (_, tr, ks)) in traces.iter().enumerate() {
        let (fo_t, lru_t, ad_t) = (col(t, SW_FOL), col(t, SW_LRU), col(t, SW_ADAPTIVE));
        for i in 0..NULL_SEEDS {
            let r = replay(Policy::Random, tr, ks, null_seed(i)).hit_bp;
            adaptive_beaten[t] += u64::from(ad_t > r);
            adaptive_regressions += u64::from((fo_t > r || lru_t > r) && ad_t <= r);
        }
    }
    let ad = replay(Policy::Adaptive, &trace, &keys, null_seed(0));
    let scan_ad = cycle_cost(Policy::Adaptive, &trace, &keys, per_eviction);
    let duel_ad = cycle_cost(Policy::Adaptive, &trace, &keys, |r| {
        r.duel_cycles.checked_div(r.descents).unwrap_or(0)
    });

    // Fraction of the LRU -> Belady headroom the foliation policy closed, in
    // basis points. Negative means the policy lost to LRU.
    let gap = opt.hit_bp as i64 - lru.hit_bp as i64;
    let gained = fo.hit_bp as i64 - lru.hit_bp as i64;
    let gap_closed_bp = if gap > 0 { gained * 10_000 / gap } else { 0 };

    let memory_ok = fo.frames_failed == 0
        && fo.frames_backed > 0
        && fo.frames_freed == fo.frames_backed
        && lru.frames_freed == lru.frames_backed
        && loc.frames_freed == loc.frames_backed;
    let sharing_ok = shared_blocks == HOT_PREFIX_BLOCKS as u64 && frames_identical;
    let refcount_ok = refcount_after == 1 && survivors == HOT_PREFIX_BLOCKS as u64;
    let safety_ok = fo.referenced_evictions == 0
        && lru.referenced_evictions == 0
        && rnd.referenced_evictions == 0
        && loc.referenced_evictions == 0
        && fo.collapse_violations == 0
        && lru.collapse_violations == 0
        && loc.collapse_violations == 0
        && chat_referenced == 0
        && chat_violations == 0;
    let refusals_ok = budget_refused && exhaustion_refused && referenced_free_refused;
    // The offline optimum must dominate every realizable policy on the same
    // candidate set. If it does not, the benchmark is measuring something else.
    let oracle_sane =
        opt.hit_bp >= fo.hit_bp && opt.hit_bp >= lru.hit_bp && opt.hit_bp >= loc.hit_bp;
    let trace_ok = null_ok
        && chat_ok
        && fo.descents == lru.descents
        && fo.descents == rnd.descents
        && fo.descents == opt.descents
        && fo.descents == loc.descents
        && fo.descents as usize == keys.len();

    let result = if memory_ok
        && sharing_ok
        && refcount_ok
        && safety_ok
        && refusals_ok
        && oracle_sane
        && trace_ok
        && sweep_ok
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
hit_bp_foliation={} hit_bp_lru={} hit_bp_random={} hit_bp_locality={} hit_bp_belady={} gap_closed_bp={} \
random_seeds={} hit_bp_random_min={} hit_bp_random_max={} random_distinct_outcomes={} \
foliation_beats_random={}/{} \
chat_requests={} chat_descents={} chat_hit_bp_foliation={} chat_hit_bp_lru={} chat_hit_bp_locality={} \
chat_hit_bp_belady={} chat_foliation_beats_random={}/{} \
sweep_pools={}{} \
scan_cycles_per_eviction_foliation={} scan_cycles_per_eviction_lru={} \
adaptive_beats_random={}/{} chat_adaptive_beats_random={}/{} chat16_adaptive_beats_random={}/{} \
adaptive_null_regressions={} evictions_adaptive={} scan_cycles_per_eviction_adaptive={} \
duel_cycles_per_descent_adaptive={} adaptive_duels={} adaptive_duels_lru={} adaptive_duels_foliation={} \
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
        loc.hit_bp,
        opt.hit_bp,
        gap_closed_bp,
        NULL_SEEDS,
        null_min,
        null_max,
        digests.len(),
        null_beaten,
        NULL_SEEDS,
        chat.len(),
        chat_fo.descents,
        chat_fo.hit_bp,
        chat_lru.hit_bp,
        chat_loc.hit_bp,
        chat_opt.hit_bp,
        chat_beaten,
        NULL_SEEDS,
        csv(&SWEEP_POOLS.map(|p| p as u64)),
        sweep_fields,
        scan_fo,
        scan_lru,
        adaptive_beaten[0],
        NULL_SEEDS,
        adaptive_beaten[1],
        NULL_SEEDS,
        adaptive_beaten[2],
        NULL_SEEDS,
        adaptive_regressions,
        ad.evictions,
        scan_ad,
        duel_ad,
        ad.duels[0],
        ad.duels[1],
        ad.duels[2],
        fo.referenced_evictions
            + lru.referenced_evictions
            + rnd.referenced_evictions
            + loc.referenced_evictions
            + chat_referenced,
        fo.collapse_violations
            + lru.collapse_violations
            + loc.collapse_violations
            + chat_violations,
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
            // LRU, not the foliation ranking: that ranking beats LRU only at a
            // capacity cliff on the synthetic boot trace and ties or loses
            // elsewhere. The boot proof selects every policy explicitly and
            // does not read this default.
            Policy::Lru,
        ));
    }
    f(guard.as_mut().expect("foliation initialised above"))
}

/// Release every sequence `owner` holds in the global cache, for the task-exit
/// path. Does not build the cache for a task that never used it.
pub fn release_task(owner: u64) -> usize {
    GLOBAL.lock().as_mut().map_or(0, |f| f.release_owner(owner))
}

/// Opt the ABI cache into another eviction policy: 0 LRU (the default),
/// 1 foliation, 2 locality, 3 adaptive. Any other code is refused, which
/// covers the Belady oracle and the random null. Nothing in the kernel calls
/// this; a caller has to ask.
pub fn set_global_policy(code: u64) -> bool {
    let policy = match code {
        0 => Policy::Lru,
        1 => Policy::Foliation,
        2 => Policy::Locality,
        3 => Policy::Adaptive,
        _ => return false,
    };
    with_global(|f| f.set_policy(policy))
}

/// Proof tag of a policy.
fn policy_tag(p: Policy) -> &'static str {
    match p {
        Policy::Foliation => "foliation",
        Policy::Lru => "lru",
        Policy::Random => "random",
        Policy::Belady => "belady",
        Policy::Locality => "locality",
        Policy::Adaptive => "adaptive",
    }
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
            "policy={} ranking={} pool_blocks={} resident={} descents={} shared={} admits={} \
evictions={} frames_backed={} frames_freed={} leaf_gc={} children_full={} \
refused_budget={} refused_exhaustion={} refused_referenced_free={} scan_cycles={} \
duels={} duels_lru={} duels_foliation={}",
            policy_tag(f.policy()),
            policy_tag(f.ranking()),
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
            s.scan_cycles,
            s.duels,
            s.duels_lru,
            s.duels_foliation,
        )
    })
}

/// Per-sequence counters for `SYS_KV_SEQ_STATS`, for the task that owns `id`.
pub fn seq_stats_line(id: usize, owner: u64) -> Option<String> {
    with_global(|f| {
        let (blocks, shared, admitted) = f.seq_counts(id, owner)?;
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
        let a = fol.seq_create(8, KERNEL).unwrap_or(usize::MAX);
        let b = fol.seq_create(8, KERNEL).unwrap_or(usize::MAX);
        test_assert!(a != usize::MAX && b != usize::MAX);
        for &t in &toks {
            let _ = fol.seq_append(a, KERNEL, t);
        }
        let resident_after_a = fol.resident();
        for &t in &toks {
            let _ = fol.seq_append(b, KERNEL, t);
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
        let a = fol.seq_create(8, KERNEL).unwrap_or(usize::MAX);
        let b = fol.seq_create(8, KERNEL).unwrap_or(usize::MAX);
        test_assert!(a != usize::MAX && b != usize::MAX);
        for &t in &toks {
            let _ = fol.seq_append(a, KERNEL, t);
        }
        for &t in &toks {
            let _ = fol.seq_append(b, KERNEL, t);
        }
        let leaf = fol.seq_leaf(b, 0).unwrap_or(NONE);
        test_assert!(leaf != NONE);
        test_assert!(fol.leaf_refcount(leaf) == 2, "two holders");
        let frame_before = fol.seq_frame(b, 0);
        let _ = fol.seq_release(a, KERNEL);
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
        let live = fol.seq_create(4, KERNEL).unwrap_or(usize::MAX);
        test_assert!(live != usize::MAX);
        for &t in &prefix_tokens(13_000, 3) {
            let _ = fol.seq_append(live, KERNEL, t);
        }
        let held: Vec<u16> = (0..3).filter_map(|i| fol.seq_leaf(live, i)).collect();
        test_assert_eq!(held.len(), 3);

        // Drive churn through the remaining 3 plaques.
        for k in 0..6u32 {
            if let Ok(id) = fol.seq_create(3, KERNEL) {
                for &t in &prefix_tokens(14_000 + k * 1000, 3) {
                    let _ = fol.seq_append(id, KERNEL, t);
                }
                let _ = fol.seq_release(id, KERNEL);
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
        let _ = fol.seq_release(live, KERNEL);
        fol.teardown();
        TestResult::Pass
    }

    /// Block tables must still resolve after the plaque pool is fragmented by
    /// interleaved release and eviction.
    fn test_block_table_after_fragmentation() -> TestResult {
        let mut fol = Foliation::new(8, 64, 8, Policy::Foliation);
        let mut ids = Vec::new();
        for k in 0..3u32 {
            if let Ok(id) = fol.seq_create(2, KERNEL) {
                for &t in &prefix_tokens(15_000 + k * 1000, 2) {
                    let _ = fol.seq_append(id, KERNEL, t);
                }
                ids.push(id);
            }
        }
        test_assert_eq!(ids.len(), 3);
        // Free the middle sequence, then churn so its slots are recycled.
        let _ = fol.seq_release(ids[1], KERNEL);
        for k in 0..4u32 {
            if let Ok(id) = fol.seq_create(2, KERNEL) {
                for &t in &prefix_tokens(16_000 + k * 1000, 2) {
                    let _ = fol.seq_append(id, KERNEL, t);
                }
                let _ = fol.seq_release(id, KERNEL);
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
        let _ = fol.seq_release(ids[0], KERNEL);
        let _ = fol.seq_release(ids[2], KERNEL);
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
        let id = fol
            .seq_create(MAX_SEQ_BLOCKS as u16, KERNEL)
            .unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        let mut first_refusal = usize::MAX;
        for j in 0..(6 * BLOCK_TOKENS) {
            match fol.seq_append(id, KERNEL, 4000 + j as u32) {
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
        test_assert_eq!(fol.seq_counts(id, KERNEL).map(|c| c.0), Some(3u16));
        let _ = fol.seq_release(id, KERNEL);
        fol.teardown();

        // Fan-out: saturate the root leaf's children with live sequences at
        // the shipped ABI pool and arena, where neither is the binding
        // constraint, then seal one more distinct block off the root. The
        // ceiling binds live continuations only — a released one is reclaimed
        // (`dead_children_do_not_saturate_fanout`) — so the saturating
        // sequences stay open, which takes one slot more than the ABI table.
        let mut fol = Foliation::new(
            ABI_POOL_BLOCKS,
            ABI_LEAF_ARENA,
            ABI_MAX_SEQS + 1,
            Policy::Foliation,
        );
        let mut live = Vec::new();
        for round in 0..MAX_CHILDREN as u32 {
            let sid = fol.seq_create(1, KERNEL).unwrap_or(usize::MAX);
            test_assert!(sid != usize::MAX);
            for j in 0..BLOCK_TOKENS {
                let _ = fol.seq_append(sid, KERNEL, 60_000 + round * 100 + j as u32);
            }
            live.push(sid);
        }
        test_assert_eq!(fol.stats().children_full, 0);
        test_assert_eq!(fol.stats().descents, MAX_CHILDREN as u64);
        let sid = fol.seq_create(1, KERNEL).unwrap_or(usize::MAX);
        test_assert!(sid != usize::MAX);
        let mut refusals = 0u64;
        for j in 0..(2 * BLOCK_TOKENS) {
            if let Err(e) = fol.seq_append(sid, KERNEL, 90_000 + j as u32) {
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
        test_assert_eq!(fol.seq_counts(sid, KERNEL).map(|c| c.0), Some(0u16));
        let _ = fol.seq_release(sid, KERNEL);
        for sid in live {
            let _ = fol.seq_release(sid, KERNEL);
        }
        fol.teardown();
        TestResult::Pass
    }

    /// A refused admission must not consume one of the prefix's child slots.
    /// Otherwise repeated refusals ratchet the fan-out to `MAX_CHILDREN` and a
    /// transient capacity refusal becomes a permanent one for that prefix.
    fn test_refused_admission_leaves_no_leaf() -> TestResult {
        let mut fol = Foliation::new(3, 64, 2, Policy::Foliation);
        let id = fol
            .seq_create(MAX_SEQ_BLOCKS as u16, KERNEL)
            .unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        // Three blocks fill the pool and stay referenced; every seal after that
        // is refused, each with a distinct key, so each would invent a leaf.
        for j in 0..(3 * BLOCK_TOKENS) {
            test_assert!(fol.seq_append(id, KERNEL, 7100 + j as u32).is_ok());
        }
        let held = fol.seq_leaf(id, 2).unwrap_or(NONE);
        test_assert!(held != NONE);
        for j in 0..(20 * BLOCK_TOKENS) {
            let r = fol.seq_append(id, KERNEL, 7500 + j as u32);
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
        test_assert_eq!(fol.seq_counts(id, KERNEL).map(|c| c.0), Some(3u16));
        test_assert!(
            fol.leaf_resident(held),
            "a refused seal disturbed residency"
        );
        test_assert_eq!(fol.collapse_violations(), 0);
        let _ = fol.seq_release(id, KERNEL);
        fol.teardown();
        TestResult::Pass
    }

    /// Residency and backing must agree for every block, at every point.
    ///
    /// `leaf_resident` reports on `slot`, `seq_frame` reports on the plaque's
    /// frame, and a caller cannot tell `seq_frame`'s `None` from "no such
    /// block". Any state where one says resident and the other says nothing is
    /// a silent lookup miss on an append that returned `Ok`.
    ///
    /// The two ways to reach that state are an admission that could not obtain
    /// a frame, and a teardown that returns frames without collapsing the
    /// leaves that point at them. The first is asserted here through
    /// `frames_backed == admits`, since an unbacked admission is exactly an
    /// admission with no frame behind it; it cannot be provoked from inside the
    /// kernel because that needs the physical allocator to fail on demand. The
    /// second is driven directly.
    fn test_residency_implies_backing() -> TestResult {
        let mut fol = Foliation::new(6, 64, 4, Policy::Foliation);
        let id = fol.seq_create(4, KERNEL).unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        for &t in &prefix_tokens(17_000, 3) {
            let _ = fol.seq_append(id, KERNEL, t);
        }
        // Churn so plaques are evicted and re-admitted under the same leaves.
        for k in 0..4u32 {
            if let Ok(other) = fol.seq_create(2, KERNEL) {
                for &t in &prefix_tokens(18_000 + k * 1000, 2) {
                    let _ = fol.seq_append(other, KERNEL, t);
                }
                let _ = fol.seq_release(other, KERNEL);
            }
        }
        let s = fol.stats();
        test_assert!(
            s.frames_backed == s.admits,
            "an admission was published with no frame behind it"
        );
        let held: Vec<u16> = (0..3).filter_map(|i| fol.seq_leaf(id, i)).collect();
        test_assert_eq!(held.len(), 3);
        for (i, leaf) in held.iter().enumerate() {
            test_assert!(
                fol.leaf_resident(*leaf) == fol.seq_frame(id, i).is_some(),
                "residency and the block table disagree about a block"
            );
        }

        fol.teardown();
        test_assert_eq!(fol.resident(), 0);
        for (i, leaf) in held.iter().enumerate() {
            test_assert!(
                !fol.leaf_resident(*leaf),
                "teardown returned the frame but left the leaf resident"
            );
            test_assert!(
                fol.seq_frame(id, i).is_none(),
                "a torn-down block still resolves to a frame"
            );
        }
        let s = fol.stats();
        test_assert!(
            s.frames_freed == s.frames_backed,
            "teardown did not return every frame"
        );
        let _ = fol.seq_release(id, KERNEL);
        TestResult::Pass
    }

    /// Policy comparison on one fixed trace. Asserts only what must hold
    /// structurally — the trace is identical across policies and the offline
    /// optimum dominates — never that the foliation policy wins.
    fn test_policy_vs_lru_on_fixed_trace() -> TestResult {
        let trace = build_trace();
        let keys = trace_keys(&trace);
        let fo = replay(Policy::Foliation, &trace, &keys, null_seed(0));
        let lru = replay(Policy::Lru, &trace, &keys, null_seed(0));
        let opt = replay(Policy::Belady, &trace, &keys, null_seed(0));
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

    /// Two different blocks whose keys collide must not share a plaque. The
    /// pair is a real 64-bit `fold_key` collision off the root, pinned at
    /// compile time by the `const _` assertion beside `fold_key`.
    fn test_colliding_blocks_do_not_share() -> TestResult {
        let mut fol = Foliation::new(16, 64, 4, Policy::Foliation);
        let a = fol.seq_create(1, KERNEL).unwrap_or(usize::MAX);
        let b = fol.seq_create(1, KERNEL).unwrap_or(usize::MAX);
        test_assert!(a != usize::MAX && b != usize::MAX);
        for &t in &COLLIDE_A {
            test_assert!(fol.seq_append(a, KERNEL, t).is_ok());
        }
        for &t in &COLLIDE_B {
            test_assert!(fol.seq_append(b, KERNEL, t).is_ok());
        }
        test_assert!(fol.seq_leaf(a, 0).is_some() && fol.seq_leaf(b, 0).is_some());
        test_assert!(
            fol.seq_leaf(a, 0) != fol.seq_leaf(b, 0),
            "different tokens shared a leaf on a key collision"
        );
        test_assert!(
            fol.seq_frame(a, 0) != fol.seq_frame(b, 0),
            "different tokens shared a frame on a key collision"
        );
        test_assert_eq!(fol.stats().shared, 0);
        let _ = fol.seq_release(a, KERNEL);
        let _ = fol.seq_release(b, KERNEL);
        fol.teardown();
        TestResult::Pass
    }

    /// The safety counters must be able to fail. Tearing down under a live
    /// sequence collapses every plaque it holds, so each held block must show
    /// up once in `referenced_evictions` and once in `collapse_violations`.
    fn test_teardown_under_live_seq_is_counted() -> TestResult {
        let mut fol = Foliation::new(8, 64, 2, Policy::Foliation);
        let id = fol.seq_create(4, KERNEL).unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        for &t in &prefix_tokens(19_000, 3) {
            test_assert!(fol.seq_append(id, KERNEL, t).is_ok());
        }
        test_assert_eq!(fol.stats().referenced_evictions, 0);
        test_assert_eq!(fol.collapse_violations(), 0);
        fol.teardown();
        test_assert_eq!(fol.stats().referenced_evictions, 3);
        test_assert_eq!(fol.collapse_violations(), 3);
        let _ = fol.seq_release(id, KERNEL);
        TestResult::Pass
    }

    /// A task may not append to, release, or read the counters of a sequence
    /// another task opened. The refusal must look exactly like a missing
    /// sequence, so it cannot be used to learn whether a prefix was written.
    fn test_foreign_release_refused() -> TestResult {
        const OWNER: u64 = 7;
        const FOREIGN: u64 = 8;
        let mut fol = Foliation::new(8, 64, 2, Policy::Foliation);
        let id = fol.seq_create(4, OWNER).unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        for &t in &prefix_tokens(21_000, 2) {
            test_assert!(fol.seq_append(id, OWNER, t).is_ok());
        }
        test_assert!(
            fol.seq_release(id, FOREIGN) == Err(FoliationError::NoSuchSeq),
            "a foreign task released another task's sequence"
        );
        test_assert!(
            fol.seq_append(id, FOREIGN, 1) == Err(FoliationError::NoSuchSeq),
            "a foreign task appended to another task's sequence"
        );
        test_assert!(
            fol.seq_counts(id, FOREIGN).is_none(),
            "a foreign task read another task's sequence counters"
        );
        test_assert_eq!(fol.seq_counts(id, OWNER), Some((2u16, 0u32, 2u32)));
        let leaf = fol.seq_leaf(id, 0).unwrap_or(NONE);
        test_assert!(leaf != NONE);
        test_assert_eq!(fol.leaf_refcount(leaf), 1);
        test_assert_eq!(fol.seq_release(id, OWNER), Ok(2u16));
        fol.teardown();
        TestResult::Pass
    }

    /// A prefix's `MAX_CHILDREN` slots bound its *live* continuations. A
    /// sequence's block table is a path from the root, so a child no live
    /// sequence references heads a subtree no live sequence references. One
    /// task writing and releasing two-block sequences with distinct first
    /// blocks used to fill the root's fan-out with dead continuations for the
    /// rest of the boot — the leaf GC runs only on a full arena and never
    /// takes a resident leaf — so every later prompt with a new first block
    /// was refused with ENOSPC. The dead subtree must be reclaimed whole, its
    /// frames returned, and nothing a live sequence holds touched.
    fn test_dead_children_do_not_saturate_fanout() -> TestResult {
        const DEAD: u64 = 7;
        const LIVE: u64 = 8;
        let mut fol = Foliation::new(ABI_POOL_BLOCKS, ABI_LEAF_ARENA, ABI_MAX_SEQS, Policy::Lru);
        // A live bystander, opened first so it is the least recently used child.
        let keep = fol.seq_create(2, LIVE).unwrap_or(usize::MAX);
        test_assert!(keep != usize::MAX);
        for &t in &prefix_tokens(80_000, 2) {
            test_assert!(fol.seq_append(keep, LIVE, t).is_ok());
        }
        let held = [fol.seq_frame(keep, 0), fol.seq_frame(keep, 1)];
        test_assert!(held[0].is_some() && held[1].is_some());
        for round in 0..(MAX_CHILDREN as u32 - 1) {
            let sid = fol.seq_create(2, DEAD).unwrap_or(usize::MAX);
            test_assert!(sid != usize::MAX);
            for &t in &prefix_tokens(100_000 + round * 100, 2) {
                test_assert!(fol.seq_append(sid, DEAD, t).is_ok());
            }
            test_assert_eq!(fol.seq_release(sid, DEAD), Ok(2u16));
        }
        test_assert_eq!(fol.resident(), ABI_POOL_BLOCKS);

        let sid = fol.seq_create(1, LIVE).unwrap_or(usize::MAX);
        test_assert!(sid != usize::MAX);
        for &t in &prefix_tokens(200_000, 1) {
            test_assert!(
                fol.seq_append(sid, LIVE, t).is_ok(),
                "a new first block was refused while every other continuation was dead"
            );
        }
        test_assert_eq!(fol.seq_counts(sid, LIVE).map(|c| c.0), Some(1u16));
        test_assert!(fol.seq_frame(sid, 0).is_some());
        test_assert_eq!(fol.stats().children_full, 0);
        // Exactly one dead two-block subtree went: its two plaques and leaves.
        test_assert_eq!(fol.resident(), ABI_POOL_BLOCKS - 1);
        test_assert_eq!(fol.stats().leaf_gc, 2);
        test_assert!(
            fol.seq_frame(keep, 0) == held[0] && fol.seq_frame(keep, 1) == held[1],
            "reclaiming a dead continuation moved a live one"
        );
        test_assert_eq!(fol.stats().referenced_evictions, 0);
        test_assert_eq!(fol.collapse_violations(), 0);
        let _ = fol.seq_release(sid, LIVE);
        let _ = fol.seq_release(keep, LIVE);
        fol.teardown();
        let s = fol.stats();
        test_assert!(
            s.frames_freed == s.frames_backed,
            "a reclaimed plaque's frame was not returned"
        );
        TestResult::Pass
    }

    /// Guard. The ABI clamps a budget past `u16` to the per-sequence ceiling
    /// instead of wrapping it, seals exactly on every `BLOCK_TOKENS` boundary,
    /// refuses the first token past the block table with EFBIG without
    /// absorbing it, and answers ENOENT to a double release, to every later use
    /// of the released id, and to ids outside the table.
    fn test_abi_budget_clamp_and_block_boundaries() -> TestResult {
        use crate::syscall::table::{
            dispatch, SYS_KV_SEQ_APPEND, SYS_KV_SEQ_CREATE, SYS_KV_SEQ_RELEASE, SYS_KV_SEQ_STATS,
        };
        let id = dispatch(SYS_KV_SEQ_CREATE, u64::MAX, 0, 0).code;
        test_assert!(id >= 0, "an oversized budget must be clamped, not refused");
        let id = id as u64;
        for j in 0..MAX_SEQ_BLOCKS * BLOCK_TOKENS {
            test_assert_eq!(
                dispatch(SYS_KV_SEQ_APPEND, id, 300_000 + j as u64, 0).code,
                ((j + 1) / BLOCK_TOKENS) as i64
            );
        }
        test_assert_eq!(dispatch(SYS_KV_SEQ_APPEND, id, 1, 0).code, -27);
        test_assert_eq!(dispatch(SYS_KV_SEQ_APPEND, id, 1, 0).code, -27);
        test_assert_eq!(
            dispatch(SYS_KV_SEQ_RELEASE, id, 0, 0).code,
            MAX_SEQ_BLOCKS as i64
        );
        test_assert_eq!(dispatch(SYS_KV_SEQ_RELEASE, id, 0, 0).code, -2);
        test_assert_eq!(dispatch(SYS_KV_SEQ_APPEND, id, 1, 0).code, -2);
        test_assert_eq!(dispatch(SYS_KV_SEQ_STATS, id, 0, 0).code, -2);

        let zero = dispatch(SYS_KV_SEQ_CREATE, 0, 0, 0).code;
        test_assert!(zero >= 0);
        test_assert_eq!(dispatch(SYS_KV_SEQ_APPEND, zero as u64, 1, 0).code, -27);
        test_assert_eq!(dispatch(SYS_KV_SEQ_RELEASE, zero as u64, 0, 0).code, 0);

        for bad in [ABI_MAX_SEQS as u64, u64::MAX] {
            test_assert_eq!(dispatch(SYS_KV_SEQ_APPEND, bad, 1, 0).code, -2);
            test_assert_eq!(dispatch(SYS_KV_SEQ_RELEASE, bad, 0, 0).code, -2);
            test_assert_eq!(dispatch(SYS_KV_SEQ_STATS, bad, 0, 0).code, -2);
        }
        TestResult::Pass
    }

    /// Guard. A released id is handed to the next opener; the old owner's
    /// stale copy of it must then be refused as missing, and must neither
    /// append to, release, nor read the new owner's sequence.
    fn test_stale_handle_after_reuse_refused() -> TestResult {
        const OLD: u64 = 7;
        const NEW: u64 = 8;
        let mut fol = Foliation::new(8, 64, 2, Policy::Lru);
        let id = fol.seq_create(4, OLD).unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        for &t in &prefix_tokens(22_000, 1) {
            test_assert!(fol.seq_append(id, OLD, t).is_ok());
        }
        test_assert_eq!(fol.seq_release(id, OLD), Ok(1u16));
        test_assert_eq!(fol.seq_create(4, NEW), Ok(id));
        for &t in &prefix_tokens(23_000, 1) {
            test_assert!(fol.seq_append(id, NEW, t).is_ok());
        }
        test_assert!(fol.seq_append(id, OLD, 1) == Err(FoliationError::NoSuchSeq));
        test_assert!(fol.seq_release(id, OLD) == Err(FoliationError::NoSuchSeq));
        test_assert!(fol.seq_counts(id, OLD).is_none());
        test_assert_eq!(fol.seq_counts(id, NEW), Some((1u16, 0u32, 1u32)));
        test_assert_eq!(fol.seq_release(id, NEW), Ok(1u16));
        test_assert!(fol.seq_release(id, NEW) == Err(FoliationError::NoSuchSeq));
        fol.teardown();
        TestResult::Pass
    }

    /// Guard. At the ABI geometry the sequence table refuses the opener past
    /// `ABI_MAX_SEQS` with `TooManySeqs`, and a released slot is reusable.
    fn test_seq_table_full_refused() -> TestResult {
        let mut fol = Foliation::new(ABI_POOL_BLOCKS, ABI_LEAF_ARENA, ABI_MAX_SEQS, Policy::Lru);
        for i in 0..ABI_MAX_SEQS {
            test_assert_eq!(fol.seq_create(1, KERNEL), Ok(i));
        }
        test_assert!(fol.seq_create(1, KERNEL) == Err(FoliationError::TooManySeqs));
        test_assert_eq!(fol.seq_release(5, KERNEL), Ok(0u16));
        test_assert_eq!(fol.seq_create(1, KERNEL), Ok(5));
        for i in 0..ABI_MAX_SEQS {
            test_assert_eq!(fol.seq_release(i, KERNEL), Ok(0u16));
        }
        fol.teardown();
        TestResult::Pass
    }

    /// Guard. At the ABI geometry, with every plaque held by a live sequence,
    /// one more block is refused with `Exhausted`: no frame is requested, no
    /// held block moves, and the refused token is not absorbed. Once a holder
    /// releases, the same append goes through by collapsing a free face, and
    /// teardown returns every frame.
    fn test_pool_exhaustion_at_abi_geometry_refused() -> TestResult {
        let mut fol = Foliation::new(ABI_POOL_BLOCKS, ABI_LEAF_ARENA, ABI_MAX_SEQS, Policy::Lru);
        let mut holders = Vec::new();
        for k in 0..(ABI_POOL_BLOCKS / MAX_SEQ_BLOCKS) as u32 {
            let id = fol
                .seq_create(MAX_SEQ_BLOCKS as u16, KERNEL)
                .unwrap_or(usize::MAX);
            test_assert!(id != usize::MAX);
            for &t in &prefix_tokens(400_000 + k * 1000, MAX_SEQ_BLOCKS) {
                test_assert!(fol.seq_append(id, KERNEL, t).is_ok());
            }
            holders.push(id);
        }
        test_assert_eq!(fol.resident(), ABI_POOL_BLOCKS);
        let before: Vec<Option<PhysAddr>> = (0..MAX_SEQ_BLOCKS)
            .map(|i| fol.seq_frame(holders[0], i))
            .collect();
        let backed = fol.stats().frames_backed;

        let extra = fol.seq_create(1, KERNEL).unwrap_or(usize::MAX);
        test_assert!(extra != usize::MAX);
        let toks = prefix_tokens(500_000, 1);
        for &t in &toks[..BLOCK_TOKENS - 1] {
            test_assert_eq!(fol.seq_append(extra, KERNEL, t), Ok(0u16));
        }
        let last = toks[BLOCK_TOKENS - 1];
        test_assert!(fol.seq_append(extra, KERNEL, last) == Err(FoliationError::Exhausted));
        test_assert!(fol.seq_append(extra, KERNEL, last) == Err(FoliationError::Exhausted));
        let s = fol.stats();
        test_assert_eq!(s.frames_backed, backed);
        test_assert_eq!(s.frames_failed, 0);
        test_assert_eq!(s.referenced_evictions, 0);
        test_assert_eq!(fol.collapse_violations(), 0);
        for (i, f) in before.iter().enumerate() {
            test_assert!(fol.seq_frame(holders[0], i) == *f, "a held block moved");
        }

        test_assert_eq!(fol.seq_release(holders[0], KERNEL), Ok(MAX_SEQ_BLOCKS as u16));
        test_assert_eq!(fol.seq_append(extra, KERNEL, last), Ok(1u16));
        test_assert_eq!(fol.stats().referenced_evictions, 0);
        for id in holders.into_iter().skip(1).chain([extra]) {
            test_assert!(fol.seq_release(id, KERNEL).is_ok());
        }
        fol.teardown();
        let s = fol.stats();
        test_assert!(s.frames_freed == s.frames_backed, "teardown leaked a frame");
        TestResult::Pass
    }

    /// Guard. More distinct blocks than the ABI leaf arena holds, written and
    /// released under a fan-out that never saturates, must all be accepted:
    /// the pool is smaller than the arena, so a non-resident, childless, dead
    /// leaf always exists for the GC to take once eviction has run.
    fn test_leaf_arena_recycles_at_abi_geometry() -> TestResult {
        let mut fol = Foliation::new(ABI_POOL_BLOCKS, ABI_LEAF_ARENA, ABI_MAX_SEQS, Policy::Lru);
        let seqs = 40u32;
        let groups = 8u32;
        for k in 0..seqs {
            let id = fol
                .seq_create(MAX_SEQ_BLOCKS as u16, KERNEL)
                .unwrap_or(usize::MAX);
            test_assert!(id != usize::MAX);
            // First block shared per group, the rest distinct per sequence.
            let mut toks = prefix_tokens(600_000 + (k % groups) * 10, 1);
            toks.extend(prefix_tokens(700_000 + k * 1000, MAX_SEQ_BLOCKS - 1));
            for &t in &toks {
                test_assert!(
                    fol.seq_append(id, KERNEL, t).is_ok(),
                    "an append was refused while the arena held reclaimable leaves"
                );
            }
            test_assert_eq!(fol.seq_release(id, KERNEL), Ok(MAX_SEQ_BLOCKS as u16));
        }
        let s = fol.stats();
        test_assert!(
            (groups + seqs * (MAX_SEQ_BLOCKS as u32 - 1)) as usize > ABI_LEAF_ARENA,
            "the workload must overrun the arena, or it proves nothing"
        );
        test_assert!(s.leaf_gc > 0, "the arena was never recycled");
        test_assert_eq!(s.children_full, 0);
        test_assert_eq!(s.referenced_evictions, 0);
        test_assert_eq!(fol.collapse_violations(), 0);
        fol.teardown();
        let s = fol.stats();
        test_assert!(s.frames_freed == s.frames_backed, "teardown leaked a frame");
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

    /// The random null is a distribution, not one draw: the proof must replay
    /// it under 32 seeds and report how many of them the foliation policy
    /// beats, and the seeds must actually produce different victim choices.
    fn test_random_null_spans_seeds() -> TestResult {
        let line = foliation_proof_line();
        test_assert!(
            line.contains(" random_seeds=32 "),
            "random null is not replayed over 32 seeds"
        );
        test_assert!(
            line.contains("/32 "),
            "fraction of seeds foliation beats is not reported"
        );
        let trace = build_trace();
        let keys = trace_keys(&trace);
        let first = replay(Policy::Random, &trace, &keys, null_seed(0)).resident_digest;
        test_assert!(
            (1..NULL_SEEDS).any(|i| {
                replay(Policy::Random, &trace, &keys, null_seed(i)).resident_digest != first
            }),
            "every seed chose the same victims, so the null is one draw"
        );
        TestResult::Pass
    }

    /// Value of `key` in the proof line; an `N/M` fraction reads as `N`.
    fn proof_metric(line: &str, key: &str) -> Option<u64> {
        line.split_whitespace()
            .find_map(|t| t.strip_prefix(key))
            .and_then(|v| v.split('/').next())
            .and_then(|v| v.parse().ok())
    }

    /// The proof must replay a same-budget locality-only null: depth, then
    /// recency, which is the foliation ranking with `entrants` removed.
    /// Without it a win over random and LRU cannot be credited to the H0
    /// multiplicity rather than to depth alone. The null must be a third
    /// policy, so it may not score exactly what LRU or foliation scores, and
    /// the offline optimum must dominate it like every other policy.
    fn test_locality_null_is_measured() -> TestResult {
        let line = foliation_proof_line();
        crate::serial_println!("  kvpolicy: {}", line);
        let loc = proof_metric(&line, "hit_bp_locality=");
        test_assert!(
            loc.is_some(),
            "the proof does not measure a locality-only null"
        );
        let loc = loc.unwrap_or(0);
        let lru = proof_metric(&line, "hit_bp_lru=").unwrap_or(0);
        let fol = proof_metric(&line, "hit_bp_foliation=").unwrap_or(0);
        let opt = proof_metric(&line, "hit_bp_belady=").unwrap_or(0);
        test_assert!(
            loc != lru && loc != fol,
            "the locality null scored exactly what LRU or foliation scored"
        );
        test_assert!(opt >= loc, "oracle below the locality null");
        TestResult::Pass
    }

    /// The boot trace re-reads its only reusable prefix after 31 other blocks
    /// against a 24-plaque pool, so LRU scores 0 there by construction. The
    /// proof must also replay a request shape whose reuse follows recency —
    /// multi-turn chat, each turn resending the conversation so far — or its
    /// margins describe the trace rather than the policy. On that shape LRU
    /// must reach 90% of the offline optimum, and the optimum must dominate
    /// every policy.
    fn test_proof_replays_a_recency_trace() -> TestResult {
        let line = foliation_proof_line();
        let lru = proof_metric(&line, "chat_hit_bp_lru=");
        test_assert!(
            lru.is_some(),
            "the proof replays only a trace LRU loses by construction"
        );
        let lru = lru.unwrap_or(0);
        let fol = proof_metric(&line, "chat_hit_bp_foliation=").unwrap_or(u64::MAX);
        let loc = proof_metric(&line, "chat_hit_bp_locality=").unwrap_or(u64::MAX);
        let opt = proof_metric(&line, "chat_hit_bp_belady=").unwrap_or(0);
        test_assert!(
            lru * 10 >= opt * 9,
            "LRU is below 90% of the optimum on the chat trace, so reuse there does not follow recency"
        );
        test_assert!(
            opt >= fol && opt >= loc,
            "oracle below a realizable policy on the chat trace"
        );
        test_assert!(
            proof_metric(&line, "chat_foliation_beats_random=").is_some(),
            "the random null is not replayed on the chat trace"
        );
        TestResult::Pass
    }

    /// Comma-separated list value of `key` in the proof line.
    fn proof_list(line: &str, key: &str) -> Option<Vec<u64>> {
        let v = line.split_whitespace().find_map(|t| t.strip_prefix(key))?;
        v.split(',').map(|x| x.parse().ok()).collect()
    }

    /// One pool size is one point, and which policy wins flips with it. The
    /// proof must replay every trace — the LRU-adversarial boot trace, the
    /// 4-live chat trace and the 16-live chat trace — at pool sizes on both
    /// sides of the headline geometry, under every realizable policy, with the
    /// offline optimum bounding every point. The headline figures must be the
    /// sweep's own measurement at `BENCH_POOL_BLOCKS`, not a separate run that
    /// could disagree with it.
    fn test_proof_sweeps_pool_sizes() -> TestResult {
        let line = foliation_proof_line();
        let pools = proof_list(&line, "sweep_pools=");
        test_assert!(pools.is_some(), "the proof does not sweep the pool size");
        let pools = pools.unwrap_or_default();
        test_assert!(pools.len() >= 4, "a pool sweep needs at least four sizes");
        test_assert!(
            pools.iter().any(|&p| p < BENCH_POOL_BLOCKS as u64)
                && pools.iter().any(|&p| p > BENCH_POOL_BLOCKS as u64),
            "the sweep does not bracket the headline pool size"
        );
        let at = pools.iter().position(|&p| p == BENCH_POOL_BLOCKS as u64);
        test_assert!(at.is_some(), "the sweep skips the headline pool size");
        let at = at.unwrap_or(0);
        for trace in ["boot", "chat", "chat16"] {
            let belady = proof_list(&line, &format!("sweep_{trace}_belady="));
            test_assert!(belady.is_some(), "a trace is not swept under Belady");
            let belady = belady.unwrap_or_default();
            test_assert_eq!(belady.len(), pools.len());
            test_assert!(
                belady.last() > belady.first(),
                "the optimum gains nothing from the largest pool, so the sweep did not change capacity"
            );
            for policy in ["foliation", "lru", "locality"] {
                let hits = proof_list(&line, &format!("sweep_{trace}_{policy}="));
                test_assert!(hits.is_some(), "a trace is not swept under a policy");
                let hits = hits.unwrap_or_default();
                test_assert_eq!(hits.len(), pools.len());
                test_assert!(
                    hits.iter().zip(&belady).all(|(h, b)| h <= b),
                    "a policy beats the offline optimum at some pool size"
                );
            }
        }
        for (sweep, headline) in [
            ("sweep_boot_foliation=", "hit_bp_foliation="),
            ("sweep_boot_lru=", "hit_bp_lru="),
            ("sweep_boot_locality=", "hit_bp_locality="),
            ("sweep_boot_belady=", "hit_bp_belady="),
            ("sweep_chat_foliation=", "chat_hit_bp_foliation="),
            ("sweep_chat_lru=", "chat_hit_bp_lru="),
            ("sweep_chat_locality=", "chat_hit_bp_locality="),
            ("sweep_chat_belady=", "chat_hit_bp_belady="),
        ] {
            let col = proof_list(&line, sweep).and_then(|v| v.get(at).copied());
            test_assert!(
                col.is_some() && col == proof_metric(&line, headline),
                "the headline disagrees with the sweep at the headline pool size"
            );
        }
        TestResult::Pass
    }

    /// The victim scan is the one O(pool) step on the admission path. Its cost
    /// must be measured, not read off the complexity table: every replay that
    /// evicts must report the cycles it spent choosing victims per eviction,
    /// and the ABI stats line must carry the same counter for the live cache.
    fn test_eviction_scan_is_measured() -> TestResult {
        let line = foliation_proof_line();
        for policy in ["foliation", "lru"] {
            let per = proof_metric(&line, &format!("scan_cycles_per_eviction_{policy}="));
            test_assert!(per.is_some(), "the eviction scan cost is not reported");
            test_assert!(
                per.unwrap_or(0) > 0,
                "a replay that evicted reports a free eviction scan"
            );
        }
        test_assert!(
            global_stats_line().contains(" scan_cycles="),
            "the ABI stats line does not report the eviction scan cost"
        );
        TestResult::Pass
    }

    /// The adaptive policy must be the better of the LRU and foliation
    /// rankings wherever either one wins: at every sweep point of every trace
    /// it may trail the better base policy by at most `MARGIN_BP`, and it may
    /// never beat the offline optimum.
    ///
    /// The margin is fixed here, by the test, and was fixed before the policy
    /// existed. It is a cost of learning rather than slack for a benchmark: no
    /// evidence that one ranking chose a worse victim than the other can
    /// arrive before a disputed block is requested again, so an online
    /// selector pays for the reuses in flight. For a selector starting under
    /// LRU on the boot trace that is the hot prefix's first return, 4 of 210
    /// descents or 190 bp; 250 bp is that floor rounded up to a quarter
    /// point. The policy as built starts under the foliation ranking (see
    /// `Policy::Adaptive`), which moves its learning cost onto the chat trace.
    fn test_adaptive_tracks_the_better_policy() -> TestResult {
        const MARGIN_BP: u64 = 250;
        let line = foliation_proof_line();
        for trace in ["boot", "chat", "chat16"] {
            let ad = proof_list(&line, &format!("sweep_{trace}_adaptive="));
            test_assert!(ad.is_some(), "the proof does not sweep the adaptive policy");
            let ad = ad.unwrap_or_default();
            let fo = proof_list(&line, &format!("sweep_{trace}_foliation=")).unwrap_or_default();
            let lru = proof_list(&line, &format!("sweep_{trace}_lru=")).unwrap_or_default();
            let opt = proof_list(&line, &format!("sweep_{trace}_belady=")).unwrap_or_default();
            test_assert!(
                !ad.is_empty()
                    && ad.len() == fo.len()
                    && ad.len() == lru.len()
                    && ad.len() == opt.len(),
                "the adaptive sweep does not line up with the base policies"
            );
            for i in 0..ad.len() {
                test_assert!(
                    ad[i] + MARGIN_BP >= fo[i].max(lru[i]),
                    "the adaptive policy trails the better base policy by more than the margin"
                );
                test_assert!(
                    ad[i] <= opt[i],
                    "the adaptive policy beats the offline optimum"
                );
            }
        }
        TestResult::Pass
    }

    /// Selecting between two rankings may not throw away a win either of them
    /// had over chance: on every trace, every random-null seed that the LRU or
    /// the foliation ranking beats, the adaptive policy must beat too.
    fn test_adaptive_keeps_every_win_over_random() -> TestResult {
        let line = foliation_proof_line();
        for key in [
            "adaptive_beats_random=",
            "chat_adaptive_beats_random=",
            "chat16_adaptive_beats_random=",
        ] {
            test_assert!(
                proof_metric(&line, key).is_some(),
                "the adaptive policy is not replayed against the random null"
            );
        }
        test_assert!(
            proof_metric(&line, "adaptive_null_regressions=") == Some(0),
            "the adaptive policy lost to a random seed that a base policy beat"
        );
        TestResult::Pass
    }

    /// Three resident single-block leaves in a 3-plaque adaptive cache: `H`
    /// entered twice and oldest, `N1` and `N2` entered once. Sealing a fourth
    /// block forces an eviction the two rankings dispute — the foliation
    /// ranking evicts `N1` (fewest entrants), LRU would evict `H` (oldest).
    fn disputed_cache(leaf_arena: usize) -> Foliation {
        let mut fol = Foliation::new(3, leaf_arena, 4, Policy::Adaptive);
        for base in [100u32, 100, 200, 300, 400] {
            request_block(&mut fol, base);
        }
        fol
    }

    /// Seal one block of `base..base + BLOCK_TOKENS` in its own sequence.
    fn request_block(fol: &mut Foliation, base: u32) {
        if let Ok(id) = fol.seq_create(1, KERNEL) {
            for j in 0..BLOCK_TOKENS as u32 {
                let _ = fol.seq_append(id, KERNEL, base + j);
            }
            let _ = fol.seq_release(id, KERNEL);
        }
    }

    /// A duel goes to the ranking whose victim was needed later. When the
    /// block the foliation ranking evicted comes back before the one LRU
    /// would have evicted, LRU wins the duel and the adaptive cache switches
    /// to it; when the spared block comes back first, the foliation ranking
    /// wins and the cache stays.
    fn test_adaptive_duel_goes_to_the_later_victim() -> TestResult {
        let mut fol = disputed_cache(32);
        test_assert_eq!(fol.stats().evictions, 1);
        test_assert_eq!(fol.stats().duels, 1);
        test_assert!(
            fol.ranking() == Policy::Foliation,
            "an adaptive cache with no settled duel is not under the foliation ranking"
        );
        // The evicted block returns first: the foliation ranking chose worse.
        request_block(&mut fol, 200);
        test_assert_eq!(fol.stats().duels_lru, 1);
        test_assert_eq!(fol.stats().duels_foliation, 0);
        test_assert!(
            fol.ranking() == Policy::Lru,
            "losing a duel did not move the adaptive cache to the other ranking"
        );
        test_assert_eq!(fol.collapse_violations(), 0);
        test_assert_eq!(fol.stats().referenced_evictions, 0);
        fol.teardown();

        let mut fol = disputed_cache(32);
        // The spared block returns first: the foliation ranking chose better.
        request_block(&mut fol, 100);
        test_assert_eq!(fol.stats().duels_foliation, 1);
        test_assert_eq!(fol.stats().duels_lru, 0);
        test_assert!(
            fol.ranking() == Policy::Foliation,
            "winning a duel moved the adaptive cache off its ranking"
        );
        fol.teardown();
        TestResult::Pass
    }

    /// A duel names leaves by arena slot, and a slot outlives its block: once
    /// the evicted leaf is reclaimed, the next new block can land in the same
    /// slot. That block never took part in the duel, so requesting it must
    /// settle nothing. The arena here holds exactly the root and the four
    /// disputed blocks, so the fifth block reclaims the evicted `N1`.
    fn test_adaptive_reused_slot_settles_nothing() -> TestResult {
        let mut fol = disputed_cache(5);
        let evicted = fol.duels[0].evicted;
        test_assert_eq!(fol.stats().duels, 1);
        request_block(&mut fol, 500);
        test_assert_eq!(fol.stats().leaf_gc, 1);
        test_assert!(
            fol.leaf_resident(evicted),
            "the new block did not land in the reclaimed slot, so the test proves nothing"
        );
        test_assert_eq!(fol.stats().duels_lru, 0);
        test_assert_eq!(fol.stats().duels_foliation, 0);
        fol.teardown();
        TestResult::Pass
    }

    /// The duel score saturates at `PSEL_MAX`, so however long LRU has been
    /// winning, `PSEL_MAX` verdicts against it hand the cache back to the
    /// foliation ranking (the other way takes one more, since ties go to the
    /// foliation ranking). Drives `settle_duels` directly with synthetic duels
    /// on one leaf.
    fn test_adaptive_score_saturates() -> TestResult {
        let mut fol = Foliation::new(4, 16, 2, Policy::Adaptive);
        fol.leaves[1].used = true;
        fol.leaves[1].key = 0xD0E1;
        let verdicts = |fol: &mut Foliation, n: usize, lru_right: bool| {
            for d in fol.duels.iter_mut().take(n) {
                *d = Duel {
                    evicted: 2,
                    evicted_key: 0,
                    spared: 1,
                    spared_key: 0xD0E1,
                    by_lru: lru_right,
                };
            }
            fol.settle_duels(1);
        };
        verdicts(&mut fol, 3 * PSEL_MAX as usize, true);
        test_assert_eq!(fol.stats().duels_lru, 3 * PSEL_MAX as u64);
        test_assert!(fol.ranking() == Policy::Lru);
        verdicts(&mut fol, PSEL_MAX as usize, false);
        test_assert!(
            fol.ranking() == Policy::Foliation,
            "a long winning streak kept the ranking past PSEL_MAX verdicts against it"
        );
        TestResult::Pass
    }

    /// The adaptive policy is opt-in: the ABI cache stays LRU until a caller
    /// selects another policy, only serving policies can be selected, and the
    /// stats line reports both the policy and the ranking in force.
    fn test_adaptive_is_opt_in() -> TestResult {
        test_assert!(with_global(|f| f.policy()) == Policy::Lru);
        for code in [4u64, 5, u64::MAX] {
            test_assert!(
                !set_global_policy(code),
                "an unknown policy code was accepted"
            );
        }
        test_assert!(with_global(|f| f.policy()) == Policy::Lru);
        let mut fol = Foliation::new(4, 16, 2, Policy::Lru);
        test_assert!(
            !fol.set_policy(Policy::Belady),
            "a live cache accepted the oracle"
        );
        test_assert!(
            !fol.set_policy(Policy::Random),
            "a live cache accepted the null"
        );
        test_assert!(fol.policy() == Policy::Lru);
        test_assert!(
            set_global_policy(3),
            "the adaptive policy could not be selected"
        );
        test_assert!(
            global_stats_line().starts_with("policy=adaptive ranking=foliation "),
            "the stats line does not report the adaptive policy and its ranking"
        );
        test_assert!(set_global_policy(0));
        test_assert!(global_stats_line().starts_with("policy=lru ranking=lru "));
        TestResult::Pass
    }

    /// Seal `tokens` as one block in its own sequence.
    fn request_tokens(fol: &mut Foliation, tokens: &[u32; BLOCK_TOKENS]) {
        if let Ok(id) = fol.seq_create(1, KERNEL) {
            for &t in tokens {
                let _ = fol.seq_append(id, KERNEL, t);
            }
            let _ = fol.seq_release(id, KERNEL);
        }
    }

    /// RED: a duel named its leaves by arena slot and key, and a key is a
    /// 64-bit digest, not an identity. `COLLIDE_B` has `COLLIDE_A`'s key off
    /// the root, so once the evicted `COLLIDE_A` leaf was reclaimed and
    /// `COLLIDE_B` landed in its slot, requesting `COLLIDE_B` settled the duel
    /// for LRU: a vote by a block that was never in it. A duel naming a slot
    /// is dropped when the slot takes a new block.
    fn test_adaptive_colliding_block_settles_nothing() -> TestResult {
        // As `disputed_cache(5)`, with `COLLIDE_A` as the block the foliation
        // ranking evicts and LRU would have kept.
        let mut fol = Foliation::new(3, 5, 4, Policy::Adaptive);
        request_block(&mut fol, 100);
        request_block(&mut fol, 100);
        request_tokens(&mut fol, &COLLIDE_A);
        request_block(&mut fol, 300);
        request_block(&mut fol, 400);
        test_assert_eq!(fol.stats().duels, 1);
        let evicted = fol.duels[0].evicted;
        test_assert!(
            evicted != NONE && fol.leaves[evicted as usize].tokens == COLLIDE_A,
            "the duel must be over COLLIDE_A"
        );
        request_tokens(&mut fol, &COLLIDE_B);
        test_assert!(
            fol.leaf_resident(evicted) && fol.leaves[evicted as usize].tokens == COLLIDE_B,
            "COLLIDE_B did not land in the reclaimed slot, so the test proves nothing"
        );
        test_assert_eq!(fol.stats().duels_lru, 0);
        test_assert_eq!(fol.stats().duels_foliation, 0);
        fol.teardown();
        TestResult::Pass
    }

    /// Guard. Switching a live adaptive cache's policy drops its open duels
    /// and its score, mid-block and with a sequence holding blocks: the block
    /// whose duel was open before the switch settles nothing after switching
    /// away and back, the held sequence keeps appending, and residency,
    /// references and backing hold throughout.
    fn test_policy_switch_drops_live_duels() -> TestResult {
        let mut fol = disputed_cache(32);
        test_assert_eq!(fol.stats().duels, 1);
        let id = fol.seq_create(4, KERNEL).unwrap_or(usize::MAX);
        test_assert!(id != usize::MAX);
        let toks = prefix_tokens(33_000, 2);
        for &t in &toks[..BLOCK_TOKENS + 3] {
            test_assert!(fol.seq_append(id, KERNEL, t).is_ok());
        }
        test_assert!(fol.set_policy(Policy::Lru));
        test_assert!(fol.set_policy(Policy::Adaptive));
        test_assert!(fol.duels.iter().all(|d| d.evicted == NONE));
        request_block(&mut fol, 200);
        test_assert_eq!(fol.stats().duels_lru + fol.stats().duels_foliation, 0);
        test_assert!(fol.ranking() == Policy::Foliation);
        for &t in &toks[BLOCK_TOKENS + 3..] {
            test_assert!(fol.seq_append(id, KERNEL, t).is_ok());
        }
        test_assert!(fol.set_policy(Policy::Foliation));
        test_assert_eq!(fol.seq_counts(id, KERNEL).map(|c| c.0), Some(2u16));
        test_assert!(fol.seq_frame(id, 0).is_some() && fol.seq_frame(id, 1).is_some());
        test_assert_eq!(fol.collapse_violations(), 0);
        test_assert_eq!(fol.stats().referenced_evictions, 0);
        let _ = fol.seq_release(id, KERNEL);
        fol.teardown();
        let s = fol.stats();
        test_assert!(s.frames_freed == s.frames_backed, "teardown leaked a frame");
        TestResult::Pass
    }

    /// The cache syscalls serve under LRU. The foliation ranking only wins at
    /// a capacity cliff on a synthetic trace, so it is selected explicitly by
    /// the boot proof and is not the default for real callers.
    fn test_abi_default_is_lru() -> TestResult {
        test_assert!(
            with_global(|f| f.policy()) == Policy::Lru,
            "ABI cache default is not LRU"
        );
        test_assert!(
            global_stats_line().starts_with("policy=lru "),
            "stats line misreports the ABI policy"
        );
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
            "foliation::residency_implies_backing",
            test_residency_implies_backing,
        );
        crate::testing::register_test(
            "foliation::policy_vs_lru_on_fixed_trace",
            test_policy_vs_lru_on_fixed_trace,
        );
        crate::testing::register_test(
            "foliation::colliding_blocks_do_not_share",
            test_colliding_blocks_do_not_share,
        );
        crate::testing::register_test(
            "foliation::teardown_under_live_seq_is_counted",
            test_teardown_under_live_seq_is_counted,
        );
        crate::testing::register_test(
            "foliation::foreign_release_refused",
            test_foreign_release_refused,
        );
        crate::testing::register_test("foliation::proof_line_passes", test_proof_line_passes);
        crate::testing::register_test(
            "foliation::random_null_spans_seeds",
            test_random_null_spans_seeds,
        );
        crate::testing::register_test("foliation::abi_default_is_lru", test_abi_default_is_lru);
        crate::testing::register_test(
            "foliation::locality_null_is_measured",
            test_locality_null_is_measured,
        );
        crate::testing::register_test(
            "foliation::proof_replays_a_recency_trace",
            test_proof_replays_a_recency_trace,
        );
        crate::testing::register_test(
            "foliation::dead_children_do_not_saturate_fanout",
            test_dead_children_do_not_saturate_fanout,
        );
        crate::testing::register_test(
            "foliation::abi_budget_clamp_and_block_boundaries",
            test_abi_budget_clamp_and_block_boundaries,
        );
        crate::testing::register_test(
            "foliation::stale_handle_after_reuse_refused",
            test_stale_handle_after_reuse_refused,
        );
        crate::testing::register_test(
            "foliation::seq_table_full_refused",
            test_seq_table_full_refused,
        );
        crate::testing::register_test(
            "foliation::pool_exhaustion_at_abi_geometry_refused",
            test_pool_exhaustion_at_abi_geometry_refused,
        );
        crate::testing::register_test(
            "foliation::leaf_arena_recycles_at_abi_geometry",
            test_leaf_arena_recycles_at_abi_geometry,
        );
        crate::testing::register_test(
            "foliation::proof_sweeps_pool_sizes",
            test_proof_sweeps_pool_sizes,
        );
        crate::testing::register_test(
            "foliation::eviction_scan_is_measured",
            test_eviction_scan_is_measured,
        );
        crate::testing::register_test(
            "foliation::adaptive_tracks_the_better_policy",
            test_adaptive_tracks_the_better_policy,
        );
        crate::testing::register_test(
            "foliation::adaptive_keeps_every_win_over_random",
            test_adaptive_keeps_every_win_over_random,
        );
        crate::testing::register_test(
            "foliation::adaptive_duel_goes_to_the_later_victim",
            test_adaptive_duel_goes_to_the_later_victim,
        );
        crate::testing::register_test("foliation::adaptive_is_opt_in", test_adaptive_is_opt_in);
        crate::testing::register_test(
            "foliation::adaptive_reused_slot_settles_nothing",
            test_adaptive_reused_slot_settles_nothing,
        );
        crate::testing::register_test(
            "foliation::adaptive_score_saturates",
            test_adaptive_score_saturates,
        );
        crate::testing::register_test(
            "foliation::adaptive_colliding_block_settles_nothing",
            test_adaptive_colliding_block_settles_nothing,
        );
        crate::testing::register_test(
            "foliation::policy_switch_drops_live_duels",
            test_policy_switch_drops_live_duels,
        );
    }
}
