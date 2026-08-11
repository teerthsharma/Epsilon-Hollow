// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! `sandbox` — a confined execution envelope for a guest LLM, sized from the
//! topology of the guest's own page-access trace.
//!
//! # Why the envelope is not a number in a config
//!
//! A resident-set size picked at build time is wrong for every workload except
//! the one it was measured on. The quantity that actually predicts how much
//! memory an inference guest wants is not its allocation rate and not its heap
//! high-water mark — it is how its accesses fall into *separated regions*, and
//! how wide those regions are. A guest that alternates between k distant
//! regions per phase wants k of them resident and thrashes if it gets fewer; a
//! guest sweeping one contiguous 8 GiB weight tensor wants a stripe of that
//! tensor, and sizing it by the *number* of regions it has — one — is the
//! defect this module shipped with in b14631c and the reason
//! [`WorkingSet::Clustered`] carries `pages` beside `clusters`.
//!
//! That is an H₀ question, so it is answered with H₀. The same MST cut that
//! counts the regions also partitions the trace into them, and each part's own
//! population and page span is read off that partition.
//!
//! # The embedding
//!
//! Each observed access is a point `(tick, page)`. Both axes are then rescaled
//! to `[0, 1]` over the observed range of the window, which is what makes the
//! metric dimensionless: a page index runs to millions and a tick to 64, so an
//! un-normalised Euclidean distance is a page-index distance with rounding
//! noise attached. After rescaling, the *shape* statistics are invariant under
//! any affine relabelling of either axis — the same guest observed with a
//! different page base or a different sampling cadence measures the same
//! `clusters` and `separation`, which
//! [`tests::test_summary_invariant_under_affine_page_relabel`] asserts.
//!
//! `pages` deliberately is not invariant, because it is not a shape: it is a
//! quantity of memory, and a guest striding seven times as far over seven times
//! as much of it wants seven times the stripe. It is therefore measured from
//! the raw page indices rather than from the normalised cloud — mapping a
//! `[0, 1]` span back through the observed range would only round-trip through
//! floats an answer the integers already hold exactly.
//!
//! Time is in the embedding rather than projected out because a *phase* is what
//! makes a region worth keeping resident. Two page regions touched in strict
//! alternation are one working set; the same two touched in separate phases are
//! two, and only the time axis can tell them apart.
//!
//! # The signal
//!
//! Single-linkage clustering, read off the minimum spanning tree. Cutting the
//! MST at scale ε leaves `1 + #{edges > ε}` components, so the whole H₀
//! barcode is in the sorted edge spectrum. The cut is placed at the largest
//! *ratio* between consecutive sorted edges: that ratio is the separation of
//! the split, and it is scale free. `k` clusters separated by a factor of 10
//! is a reading worth acting on; `k` clusters separated by a factor of 1.01 is
//! noise in the sampling, and [`size_envelope`] treats it as such.
//!
//! # Unmeasurable is a state, not a default
//!
//! [`summarize`] returns [`WorkingSet::Unmeasurable`] when there is no
//! measurement to report: too few samples, or a cloud with no positive MST edge
//! at all. It does not report a plausible number it did not measure. This is
//! the defect `ml_engine/stratum.rs` was fixed for — when every pairwise
//! distance overflowed there, Prim recorded zero edges and a degenerate branch
//! published `shatter=1.000 h0_death=0 loop=0`, a diverged run reported as well
//! fit with every signal finite. The dangerous shape of that failure here would
//! be worse: a cloud with no edges has a `max/min` edge ratio of `0/0`, and a
//! fabricated "n perfectly separated clusters, separation infinite" reading
//! sends the envelope straight to its ceiling. So the rule is the inverse:
//! **unmeasurable sizes to the floor.**
//!
//! # The four rules
//!
//! 1. **The cap is not negotiable.** [`Sandbox::cap`] is fixed at construction
//!    and has no setter. [`size_envelope`] applies `min(cap)` last and
//!    unconditionally, so no signal — including a malformed one with
//!    `clusters == usize::MAX` and `pages == u64::MAX` — can raise the envelope
//!    past it. Inference may narrow, never widen. Extent makes this rule matter
//!    more, not less: it multiplies page counts rather than region counts, so
//!    every step of it saturates, a wrapped demand being one that arrives under
//!    the cap and reads as the cap having held.
//! 2. **Unmeasurable sizes to the floor**, and says so through the return type.
//! 3. **Allocation failure is refusal.** [`Sandbox::start`] either obtains its
//!    whole floor or hands back every frame it did obtain and refuses to run.
//!    It never starts partially confined. `ml_engine/foliation.rs` was fixed
//!    today for the mirror-image defect: it published a resident leaf with no
//!    frame behind it and returned `Ok`.
//! 4. **Shrinking cannot evict held frames.** A re-evaluation that lowers the
//!    envelope returns only frames the guest has not pinned, stops early when
//!    the rest are pinned, and leaves the guest running.
//!
//! # Bounds
//!
//! At most [`MAX_SAMPLES`] accesses are retained, in a fixed ring; observation
//! is O(1). The summary is O(MAX_SAMPLES²) over fixed stack buffers and
//! allocates nothing. Resident frames are bounded by the construction-time cap.
//!
//! # Confinement
//!
//! [`SandboxPolicy`] is deliberately minimal and local to this module. A total
//! permission field with deny-dominance is being built separately in
//! `security/`; wiring this boundary to it is a follow-up, and depending on an
//! unlanded API would block both.

use alloc::vec::Vec;
use core::cmp::Ordering;

use aether_core::manifold::ManifoldPoint;
use x86_64::PhysAddr;

use crate::memory::topo_ram::{self, ZoneHint};

// ── Sizing constants ────────────────────────────────────────────────────────

/// Accesses retained in the trace window. Bounds both the memory held for the
/// trace and the O(n²) MST cost. Matches `stratum::STRATUM_WINDOW`.
pub const MAX_SAMPLES: usize = 64;

/// Accesses required before a summary is attempted.
///
/// The gap statistic compares consecutive MST edges, so it needs at least two
/// edges to compare and a third to have a scale below them: 4 points.
pub const MIN_SAMPLES: usize = 4;

/// Frames granted per well-separated region.
///
/// Calibration constant. There is no derivation behind 4 — it is 16 KiB per
/// region, chosen because it is the smallest span worth the bookkeeping — and
/// it is the one number in this module that a real workload would want tuned.
/// The confidence weighting below is what keeps a wrong value from being
/// catastrophic: an unseparated reading grants none of it.
pub const FRAMES_PER_REGION: usize = 4;

/// Pages of measured region extent per resident frame.
///
/// The second calibration constant, and the one that decides how much of a
/// large region the envelope promises to hold: 32 pages of region per 4 KiB
/// frame is a 1/32 stripe, because the envelope confines a guest rather than
/// promising it full residency.
///
/// Its magnitude is chosen against the trace window rather than against a
/// workload. A region the window can cover densely — at most [`MAX_SAMPLES`]
/// pages — asks for `64 / 32 = 2` frames, which is below [`FRAMES_PER_REGION`],
/// so small regions keep exactly the count-driven sizing they had and only a
/// region wider than `FRAMES_PER_REGION * PAGES_PER_GRANTED_FRAME` = 128 pages
/// moves the envelope. That crossover is the knob: lower it to make the
/// envelope track extent sooner, raise it to hold a thinner stripe of a large
/// region.
pub const PAGES_PER_GRANTED_FRAME: u64 = 32;

/// Edge-spectrum gap ratio at which a split is fully trusted.
///
/// A ratio of exactly 1 is no gap at all: the sorted MST edges are uniform and
/// the "clusters" are an artefact of where the cut happened to land. 3 means
/// the between-region hop is three times the within-region step, which no
/// uniform sampling of a single region produces. Between 1 and 3 the grant is
/// interpolated, so a marginal reading yields a marginal envelope.
pub const SEPARATION_FULL: f64 = 3.0;

// ── Observation ─────────────────────────────────────────────────────────────

/// One observed page access.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Access {
    /// Monotone observation index. Not a wall clock: kernel sizing must be
    /// reproducible, so the time axis is a counter the sandbox owns.
    pub tick: u64,
    /// Page index the guest touched.
    pub page: u64,
}

/// Why a trace could not be summarised.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unmeasurable {
    /// Fewer than [`MIN_SAMPLES`] accesses observed.
    TooFewSamples,
    /// Single-linkage recorded no positive edge: every access coincides, or the
    /// cloud's own scale is not a finite positive number. There is no length to
    /// separate clusters at, so there is no cluster count.
    NoScale,
}

/// H₀ summary of an access trace.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WorkingSet {
    /// `clusters` single-linkage components at the most persistent cut, with
    /// `separation` the ratio between the first cut edge and the last kept
    /// edge. `separation >= 1` always; 1 means no gap was found.
    ///
    /// `pages` is how much *address range* those components cover: the sum,
    /// over every component holding at least [`MIN_SAMPLES`] accesses, of that
    /// component's own page span, in pages. `clusters` says how many regions
    /// there are and `pages` says how big they are; sizing needs both, because
    /// one 8 GiB tensor and one 8 KiB scratch buffer are both one region.
    Clustered {
        clusters: usize,
        separation: f64,
        pages: u64,
    },
    /// No summary exists for this cloud. Callers must not substitute a default.
    Unmeasurable(Unmeasurable),
}

/// Summarise an access trace as H₀ clustering structure.
///
/// Only the most recent [`MAX_SAMPLES`] accesses are read; a longer slice is
/// bounded, not refused. Allocates nothing.
pub fn summarize(samples: &[Access]) -> WorkingSet {
    let n = samples.len().min(MAX_SAMPLES);
    if n < MIN_SAMPLES {
        return WorkingSet::Unmeasurable(Unmeasurable::TooFewSamples);
    }
    let s = &samples[samples.len() - n..];

    let (mut t_lo, mut t_hi) = (u64::MAX, 0u64);
    let (mut p_lo, mut p_hi) = (u64::MAX, 0u64);
    for a in s {
        t_lo = t_lo.min(a.tick);
        t_hi = t_hi.max(a.tick);
        p_lo = p_lo.min(a.page);
        p_hi = p_hi.max(a.page);
    }
    // Rescale each axis over its observed range. A degenerate axis collapses to
    // 0 rather than dividing by zero: a guest that touched one page is a cloud
    // on the time axis, not a NaN.
    let t_span = (t_hi - t_lo) as f64;
    let p_span = (p_hi - p_lo) as f64;
    let mut cloud = [ManifoldPoint::<2>::zero(); MAX_SAMPLES];
    for (i, a) in s.iter().enumerate() {
        let x = if t_span > 0.0 {
            (a.tick - t_lo) as f64 / t_span
        } else {
            0.0
        };
        let y = if p_span > 0.0 {
            (a.page - p_lo) as f64 / p_span
        } else {
            0.0
        };
        cloud[i] = ManifoldPoint::new([x, y]);
    }

    let mut tree = [MstEdge::NONE; MAX_SAMPLES];
    let t = mst_edges(&cloud[..n], &mut tree);
    let mut edges = [0.0f64; MAX_SAMPLES];
    let mut m = 0usize;
    for edge in &tree[..t] {
        if edge.len > 0.0 {
            edges[m] = edge.len;
            m += 1;
        }
    }
    let e = &mut edges[..m];
    e.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // The whole cloud's scale, and the only place a fabricated reading could
    // enter. `mst_edges` records positive finite edges only, so `m == 0` is
    // exactly stratum's "Prim recorded no edge" state: report it as unmeasured.
    if m == 0 || !e[m - 1].is_finite() || e[m - 1] <= 0.0 {
        return WorkingSet::Unmeasurable(Unmeasurable::NoScale);
    }

    // Largest ratio between consecutive sorted edges. Cutting between `e[i-1]`
    // and `e[i]` removes `m - i` edges, leaving `1 + m - i` components.
    //
    // The gap must clear `eps` in absolute terms as well as in ratio. A cloud
    // sampled uniformly has a spectrum uniform to within rounding, so every
    // ratio in it is `1 + O(ulp)` and the largest one lands wherever the last
    // bit fell. b14631c tolerated that because a ratio that close to 1 grants
    // nothing; the partition below cannot tolerate it, because a cut placed by
    // rounding noise shreds one contiguous region into fragments and charges
    // the guest for none of them.
    let eps = e[m - 1] * 1e-9;
    let mut separation = 1.0f64;
    let mut cut = m;
    for i in 1..m {
        if e[i - 1] <= eps || e[i] - e[i - 1] <= eps {
            continue;
        }
        let ratio = e[i] / e[i - 1];
        if ratio > separation {
            separation = ratio;
            cut = i;
        }
    }

    // Everything at or below the last kept edge stays connected, which is the
    // same partition the cluster count is read off: `cut == m` keeps every
    // edge and leaves one component.
    WorkingSet::Clustered {
        clusters: 1 + (m - cut),
        separation,
        pages: measured_pages(s, &tree[..t], e[cut - 1]),
    }
}

/// Pages of address range the components at `threshold` actually cover.
///
/// Components are single-linkage at the same cut the caller read its cluster
/// count off: two accesses are in one component when the MST path between them
/// uses no edge longer than `threshold`.
///
/// A component's span is measured from the raw page indices in `samples`, not
/// from the normalised cloud. The embedding divides both axes by their observed
/// range to make the *shape* dimensionless, and multiplying a `[0, 1]` span
/// back by `p_span` only feeds the page range through a float round trip that
/// the integers already answer exactly.
///
/// A component holding fewer than [`MIN_SAMPLES`] accesses contributes nothing.
/// Two or three touches are not a measurement of a region — they are the same
/// state the whole cloud is refused for below `MIN_SAMPLES` — and the range
/// they straddle is very often a hole rather than a region. Population is used
/// as that admission gate and never as a multiplier, because [`MAX_SAMPLES`]
/// truncation makes a component's sample count a biased proxy for its true
/// traffic: a region touched steadily across a long run holds fewer of the last
/// 64 accesses than one touched in a burst, and scaling extent by a count would
/// shrink the steady region's envelope for being steady. What the gate does
/// cost is a region whose share of the window falls below `MIN_SAMPLES`: it
/// contributes no extent at all and falls back to the per-region grant, which
/// is the floor-ward direction rule 2 already chose.
fn measured_pages(samples: &[Access], tree: &[MstEdge], threshold: f64) -> u64 {
    let n = samples.len();
    let mut label = [0usize; MAX_SAMPLES];
    for (i, l) in label.iter_mut().enumerate().take(n) {
        *l = i;
    }
    for edge in tree {
        if edge.len > threshold {
            continue;
        }
        let (a, b) = (label[edge.a], label[edge.b]);
        if a == b {
            continue;
        }
        let (keep, merged) = if a < b { (a, b) } else { (b, a) };
        for l in label[..n].iter_mut() {
            if *l == merged {
                *l = keep;
            }
        }
    }

    let mut pages = 0u64;
    for c in 0..n {
        if label[c] != c {
            continue;
        }
        let mut pop = 0usize;
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for (i, a) in samples.iter().enumerate() {
            if label[i] == c {
                pop += 1;
                lo = lo.min(a.page);
                hi = hi.max(a.page);
            }
        }
        if pop >= MIN_SAMPLES {
            // Saturating throughout: a page index is a u64, so one component's
            // span alone can reach u64::MAX, and a wrapped total is a large
            // region reported as a small one.
            pages = pages.saturating_add((hi - lo).saturating_add(1));
        }
    }
    pages
}

/// Map a summary to a resident frame count within `[floor, cap]`.
///
/// Two demands, and the envelope is the larger:
///
/// * **How many regions there are**, weighted by how well separated they are.
///   `separation` is confidence in the *cut*, so it weights the quantity the
///   cut produced — the count — and nothing else.
/// * **How much range those regions cover**, at one frame per
///   [`PAGES_PER_GRANTED_FRAME`] pages. This one is not weighted by
///   `separation`, and must not be: a single contiguous region has no gap in
///   its spectrum to separate anything at, so its separation is exactly 1 and a
///   weighted extent would be worth nothing. That is the b14631c defect in its
///   deepest form — the 8 GiB tensor is *one* cluster with *no* separation, and
///   every count-shaped term about it is 1.
///
/// The cap wins, always and last. `floor` above `cap` is a malformed bound and
/// yields `cap`, never something above it, so no argument ordering mistake at a
/// call site can talk the envelope past the sandbox's own maximum.
pub fn size_envelope(ws: &WorkingSet, floor: usize, cap: usize) -> usize {
    let floor = floor.min(cap);
    let want = match *ws {
        // Rule 2. Not a generous default, and not a fabricated signal.
        WorkingSet::Unmeasurable(_) => floor,
        WorkingSet::Clustered {
            clusters,
            separation,
            pages,
        } => {
            if !separation.is_finite() {
                // A caller-supplied signal that is not a number is not evidence
                // of demand; every comparison against it would be false. That
                // disqualifies the whole summary, extent included: `pages` was
                // measured off the same partition the ratio was read from.
                floor
            } else {
                let demand = clusters.saturating_mul(FRAMES_PER_REGION);
                let confidence =
                    ((separation - 1.0) / (SEPARATION_FULL - 1.0)).clamp(0.0, 1.0);
                let grant = (demand.saturating_sub(floor) as f64 * confidence) as usize;
                floor.saturating_add(grant).max(frames_for_pages(pages))
            }
        }
    };
    // Rule 1.
    want.min(cap)
}

/// Frames a measured extent of `pages` pages asks for.
///
/// Rounds up, so a region measured at all asks for at least one frame, and
/// saturates rather than truncating: `pages` is a u64 and the envelope is a
/// usize, and a demand that wrapped on the way down would arrive *under* the
/// cap and look like the cap had held.
pub fn frames_for_pages(pages: u64) -> usize {
    usize::try_from(pages.div_ceil(PAGES_PER_GRANTED_FRAME)).unwrap_or(usize::MAX)
}

/// One tree edge: its length and the two sample indices it joins.
#[derive(Clone, Copy)]
struct MstEdge {
    len: f64,
    a: usize,
    b: usize,
}

impl MstEdge {
    const NONE: Self = Self {
        len: 0.0,
        a: 0,
        b: 0,
    };
}

/// Minimum spanning tree by Prim's algorithm, O(n²).
///
/// Writes the finite edges to `out` and returns how many there were, endpoints
/// included: the caller needs the tree itself, not just its spectrum, because
/// the components the cut leaves are what carry a region's population and page
/// span. Zero-length edges between coincident points are kept here and filtered
/// out of the spectrum by the caller — the points are at the same position and
/// belong to the same component at every scale, so dropping the edge would
/// scatter them into components of one. A cloud whose points all coincide
/// therefore yields no *positive* edge, which is the caller's signal that there
/// is no scale to measure at.
///
/// ponytail: a near-copy of the Prim loop in `ml_engine/stratum.rs`. That one
/// is private, hard-typed to `ManifoldPoint<3>` and `STRATUM_WINDOW`, and
/// returns only `(max, median)` rather than the spectrum this needs. Upgrade
/// path is to lift one generic `mst_edges` into a shared geometry module and
/// have `stratum` derive its two statistics from it; that is a two-file change
/// and this module owns neither file today.
fn mst_edges(pts: &[ManifoldPoint<2>], out: &mut [MstEdge; MAX_SAMPLES]) -> usize {
    let n = pts.len();
    if n < 2 {
        return 0;
    }
    let mut included = [false; MAX_SAMPLES];
    let mut key = [f64::INFINITY; MAX_SAMPLES];
    let mut parent = [usize::MAX; MAX_SAMPLES];
    let mut count = 0usize;
    key[0] = 0.0;
    for _ in 0..n {
        let mut best = usize::MAX;
        let mut best_key = f64::INFINITY;
        for i in 0..n {
            if !included[i] && key[i] < best_key {
                best_key = key[i];
                best = i;
            }
        }
        if best == usize::MAX {
            break;
        }
        included[best] = true;
        // The root has no parent, and a point reached by no finite distance —
        // every candidate distance was NaN — has none either. Neither is an
        // edge, and neither joins anything to anything.
        if best_key.is_finite() && parent[best] != usize::MAX {
            out[count] = MstEdge {
                len: best_key,
                a: best,
                b: parent[best],
            };
            count += 1;
        }
        for i in 0..n {
            if !included[i] {
                let d = pts[best].distance(&pts[i]);
                if d < key[i] {
                    key[i] = d;
                    parent[i] = best;
                }
            }
        }
    }
    count
}

// ── Frame supply ────────────────────────────────────────────────────────────

/// Where a sandbox's frames come from.
///
/// One frame per call, matching how `ml_engine/foliation.rs` backs a plaque.
/// The trait exists so an assertion can drive the exhaustion path with an
/// explicit budget: `topo_ram::alloc_frames` cannot be made to fail on demand
/// from inside the kernel, which is exactly why foliation's unbacked-admission
/// defect survived as long as it did.
pub trait FrameSource {
    /// One frame, or `None` when the source is exhausted.
    fn acquire(&mut self) -> Option<PhysAddr>;
    /// Return a frame previously handed out by `acquire`.
    fn release(&mut self, frame: PhysAddr);
}

/// The real supply: topological RAM, high zone.
///
/// `topo_ram::alloc_frames` falls back to the low zone on its own, and returns
/// `None` both on exhaustion and before `topo_ram::init()`. Either way the
/// sandbox treats it as refusal.
pub struct TopoRamFrames;

impl FrameSource for TopoRamFrames {
    fn acquire(&mut self) -> Option<PhysAddr> {
        topo_ram::alloc_frames(1, ZoneHint::High, None)
    }

    fn release(&mut self, frame: PhysAddr) {
        topo_ram::free_frames(frame, 1);
    }
}

// ── Confinement ─────────────────────────────────────────────────────────────

/// Minimal permission boundary for a guest.
///
/// Both fields are deny-dominant: they can only remove reach, never grant it.
/// `max_page` bounds what the guest may touch, and an access outside it is
/// refused and never enters the trace — so a guest cannot inflate its own
/// measured working set by touching memory it is not allowed to touch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SandboxPolicy {
    /// Highest page index the guest may touch.
    pub max_page: u64,
    /// When false the envelope stays at the floor whatever the topology says.
    pub may_grow: bool,
}

/// Why a sandbox operation was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SandboxError {
    /// A cap of zero confines nothing.
    BadBounds,
    /// `start` on a running sandbox.
    AlreadyRunning,
    /// The frame source could not supply the floor. The sandbox did not start
    /// and holds no frames.
    Refused,
}

struct Resident {
    addr: PhysAddr,
    /// The guest is using this frame; a shrink must not take it.
    pinned: bool,
}

/// A guest workload confined to a topologically sized memory envelope.
pub struct Sandbox {
    policy: SandboxPolicy,
    floor: usize,
    cap: usize,
    frames: Vec<Resident>,
    ring: [Access; MAX_SAMPLES],
    pos: usize,
    len: usize,
    tick: u64,
    running: bool,
    denied: u64,
}

impl Sandbox {
    /// Build a sandbox with a hard maximum of `cap` frames.
    ///
    /// `floor` is clamped into `[1, cap]`. There is no setter for either bound:
    /// the topological signal chooses inside them and nothing moves them.
    pub fn new(policy: SandboxPolicy, floor: usize, cap: usize) -> Result<Self, SandboxError> {
        if cap == 0 {
            return Err(SandboxError::BadBounds);
        }
        Ok(Self {
            policy,
            floor: floor.clamp(1, cap),
            cap,
            frames: Vec::new(),
            ring: [Access { tick: 0, page: 0 }; MAX_SAMPLES],
            pos: 0,
            len: 0,
            tick: 0,
            running: false,
            denied: 0,
        })
    }

    /// The hard maximum. Fixed at construction.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The minimum the guest is ever confined to while running.
    pub fn floor(&self) -> usize {
        self.floor
    }

    /// Frames currently in the envelope.
    pub fn resident(&self) -> usize {
        self.frames.len()
    }

    /// Frames the guest has pinned.
    pub fn held(&self) -> usize {
        self.frames.iter().filter(|f| f.pinned).count()
    }

    /// True once `start` has obtained the floor.
    pub fn running(&self) -> bool {
        self.running
    }

    /// Accesses refused by the policy.
    pub fn denied(&self) -> u64 {
        self.denied
    }

    /// Addresses currently in the envelope, in acquisition order.
    pub fn frame_addrs(&self) -> Vec<PhysAddr> {
        self.frames.iter().map(|f| f.addr).collect()
    }

    /// Record one page access. Returns false when the policy refuses it, in
    /// which case nothing enters the trace.
    pub fn observe(&mut self, page: u64) -> bool {
        if !self.running || page > self.policy.max_page {
            self.denied += 1;
            return false;
        }
        self.tick += 1;
        self.ring[self.pos] = Access {
            tick: self.tick,
            page,
        };
        self.pos = (self.pos + 1) % MAX_SAMPLES;
        if self.len < MAX_SAMPLES {
            self.len += 1;
        }
        true
    }

    /// Mark a frame as in use by the guest, so a shrink cannot take it.
    pub fn pin(&mut self, addr: PhysAddr) -> bool {
        for f in self.frames.iter_mut() {
            if f.addr == addr {
                f.pinned = true;
                return true;
            }
        }
        false
    }

    /// Release a frame back to the shrinkable set.
    pub fn unpin(&mut self, addr: PhysAddr) -> bool {
        for f in self.frames.iter_mut() {
            if f.addr == addr {
                f.pinned = false;
                return true;
            }
        }
        false
    }

    /// H₀ summary of the current trace window.
    pub fn summary(&self) -> WorkingSet {
        let mut out = [Access { tick: 0, page: 0 }; MAX_SAMPLES];
        let start = if self.len < MAX_SAMPLES { 0 } else { self.pos };
        for i in 0..self.len {
            out[i] = self.ring[(start + i) % MAX_SAMPLES];
        }
        summarize(&out[..self.len])
    }

    /// Obtain the floor and start the guest.
    ///
    /// Rule 3: either the whole floor is obtained or every frame taken along
    /// the way is handed back and the sandbox does not run. It never starts
    /// partially confined, because a guest running below its floor is a guest
    /// whose confinement is a number nobody chose.
    ///
    /// The envelope starts at the floor by measurement, not by fiat: with no
    /// accesses observed the summary is `TooFewSamples`, and unmeasurable sizes
    /// to the floor.
    pub fn start(&mut self, src: &mut dyn FrameSource) -> Result<usize, SandboxError> {
        if self.running {
            return Err(SandboxError::AlreadyRunning);
        }
        let want = size_envelope(&self.summary(), self.floor, self.cap);
        for _ in 0..want {
            match src.acquire() {
                Some(addr) => self.frames.push(Resident {
                    addr,
                    pinned: false,
                }),
                None => {
                    for f in self.frames.drain(..) {
                        src.release(f.addr);
                    }
                    return Err(SandboxError::Refused);
                }
            }
        }
        self.running = true;
        Ok(self.frames.len())
    }

    /// Re-size the envelope from the current trace. Returns the new frame count.
    ///
    /// Growth that the frame source cannot satisfy stops where it stops: the
    /// envelope is already at or above the floor and the guest keeps running,
    /// so there is nothing to refuse. Shrinking obeys rule 4 — only unpinned
    /// frames are returned, and the envelope stops above target rather than
    /// take one the guest holds.
    pub fn reevaluate(&mut self, src: &mut dyn FrameSource) -> usize {
        if !self.running {
            return 0;
        }
        let target = if self.policy.may_grow {
            size_envelope(&self.summary(), self.floor, self.cap)
        } else {
            self.floor
        };
        while self.frames.len() < target {
            match src.acquire() {
                Some(addr) => self.frames.push(Resident {
                    addr,
                    pinned: false,
                }),
                None => break,
            }
        }
        while self.frames.len() > target {
            let Some(i) = self.frames.iter().position(|f| !f.pinned) else {
                break;
            };
            let f = self.frames.swap_remove(i);
            src.release(f.addr);
        }
        self.frames.len()
    }

    /// Stop the guest and return every frame, pinned or not.
    pub fn stop(&mut self, src: &mut dyn FrameSource) -> usize {
        let n = self.frames.len();
        for f in self.frames.drain(..) {
            src.release(f.addr);
        }
        self.running = false;
        n
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(feature = "test-mode")]
pub mod tests {
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    /// Frame source with an explicit budget, so the exhaustion path is drivable.
    struct StubFrames {
        budget: usize,
        next: u64,
        outstanding: usize,
        released: usize,
    }

    impl StubFrames {
        fn new(budget: usize) -> Self {
            Self {
                budget,
                next: 0x1000,
                outstanding: 0,
                released: 0,
            }
        }
    }

    impl FrameSource for StubFrames {
        fn acquire(&mut self) -> Option<PhysAddr> {
            if self.budget == 0 {
                return None;
            }
            self.budget -= 1;
            self.outstanding += 1;
            self.next += 1;
            Some(PhysAddr::new(self.next * 4096))
        }

        fn release(&mut self, _frame: PhysAddr) {
            self.budget += 1;
            self.released += 1;
            self.outstanding = self.outstanding.saturating_sub(1);
        }
    }

    fn policy(max_page: u64, may_grow: bool) -> SandboxPolicy {
        SandboxPolicy { max_page, may_grow }
    }

    /// `k` phase-local regions: phase j touches `per` distinct pages far from
    /// every other phase's, during `per` consecutive ticks.
    fn clustered(k: usize, per: usize) -> Vec<Access> {
        let mut v = Vec::new();
        for j in 0..k {
            for i in 0..per {
                v.push(Access {
                    tick: (j * per + i) as u64,
                    page: (j as u64) * 10_000 + i as u64,
                });
            }
        }
        v
    }

    /// One region read as a strided sweep: `n` accesses `stride` pages apart,
    /// one per tick. The shape of a weight-tensor read — a single cluster whose
    /// page range is far wider than the trace window can hold points for.
    fn swept(n: usize, stride: u64) -> Vec<Access> {
        (0..n)
            .map(|i| Access {
                tick: i as u64,
                page: i as u64 * stride,
            })
            .collect()
    }

    /// One region swept linearly: a uniform edge spectrum with no gap.
    fn sweep(n: usize) -> Vec<Access> {
        (0..n)
            .map(|i| Access {
                tick: i as u64,
                page: i as u64,
            })
            .collect()
    }

    /// Every access identical: single-linkage records no positive edge. This is
    /// the cloud that made `stratum` fabricate a well-fit verdict.
    fn coincident(n: usize) -> Vec<Access> {
        (0..n)
            .map(|_| Access { tick: 7, page: 7 })
            .collect()
    }

    /// Rule 1. The topological signal chooses inside `[floor, cap]` and nothing
    /// it can say raises the cap.
    fn test_cap_bounds_topological_demand() -> TestResult {
        let cloud = clustered(8, 8);
        let ws = summarize(&cloud);
        let WorkingSet::Clustered {
            clusters,
            separation,
            ..
        } = ws
        else {
            return TestResult::Fail("eight phase-local regions must be measurable");
        };
        test_assert_eq!(clusters, 8);
        // Bound through a `bool` rather than inline: `test_assert!` expands to
        // `if !$cond`, and a negated comparison on a partially ordered type is
        // a clippy warning at the macro's expansion site.
        let well_separated = separation > SEPARATION_FULL;
        test_assert!(
            well_separated,
            "distant regions must read as well separated"
        );

        // Unconstrained, the signal asks for 8 regions x FRAMES_PER_REGION.
        test_assert_eq!(size_envelope(&ws, 2, 64), 32);
        // Against a cap it exceeds, the answer is the cap — exactly, at three
        // adjacent caps, so a cap shifted by one is visible.
        test_assert_eq!(size_envelope(&ws, 2, 8), 8);
        test_assert_eq!(size_envelope(&ws, 2, 9), 9);
        test_assert_eq!(size_envelope(&ws, 2, 10), 10);

        // No signal can raise it, including one no measurement could produce.
        // Both demands are absurd here: extent multiplies larger numbers than
        // counts do, so it is the likelier of the two to wrap under a profile
        // with overflow checks off, and a wrapped demand arrives under the cap
        // looking like the cap held.
        let absurd = WorkingSet::Clustered {
            clusters: usize::MAX,
            separation: 1e300,
            pages: u64::MAX,
        };
        test_assert_eq!(size_envelope(&absurd, 4, 6), 6);
        // Nor can a malformed bound where the floor is above the cap.
        test_assert_eq!(size_envelope(&absurd, 100, 6), 6);
        test_assert_eq!(size_envelope(&ws, 100, 6), 6);

        // Longer traces are bounded, not refused: the window is the last 64.
        let long = clustered(16, 8);
        test_assert!(long.len() > MAX_SAMPLES);
        test_assert_eq!(summarize(&long), summarize(&long[long.len() - MAX_SAMPLES..]));

        // And the same bound holds through the live sandbox.
        let mut src = StubFrames::new(1024);
        let Ok(mut sb) = Sandbox::new(policy(1_000_000, true), 2, 6) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(sb.start(&mut src), Ok(2));
        for a in clustered(8, 8) {
            test_assert!(sb.observe(a.page), "policy must admit the fixture");
        }
        test_assert_eq!(sb.reevaluate(&mut src), 6);
        test_assert!(sb.resident() <= sb.cap(), "envelope exceeded its own cap");
        sb.stop(&mut src);
        TestResult::Pass
    }

    /// Rule 2. An unmeasurable cloud sizes to the floor, never to a default
    /// that happens to be generous, and never through a fabricated signal.
    fn test_unmeasurable_sizes_to_floor() -> TestResult {
        let few = sweep(MIN_SAMPLES - 1);
        test_assert_eq!(
            summarize(&few),
            WorkingSet::Unmeasurable(Unmeasurable::TooFewSamples)
        );
        // Zero positive MST edges — the exact state that made stratum publish
        // `shatter=1.000 h0_death=0 loop=0` for a diverged run.
        let flat = coincident(16);
        test_assert_eq!(
            summarize(&flat),
            WorkingSet::Unmeasurable(Unmeasurable::NoScale)
        );

        for ws in [summarize(&few), summarize(&flat)] {
            test_assert_eq!(size_envelope(&ws, 2, 40), 2);
            test_assert_eq!(size_envelope(&ws, 3, 40), 3);
            test_assert_eq!(size_envelope(&ws, 4, 40), 4);
        }

        // A signal that is not a number is not evidence of demand either, and
        // that disqualifies the extent it was measured alongside: 4096 pages
        // would otherwise buy 128 frames.
        let nan = WorkingSet::Clustered {
            clusters: 16,
            separation: f64::NAN,
            pages: 4096,
        };
        test_assert_eq!(size_envelope(&nan, 3, 40), 3);

        // A sandbox that has observed nothing starts at its floor, by
        // measurement rather than by a default.
        let mut src = StubFrames::new(64);
        let Ok(mut sb) = Sandbox::new(policy(1_000_000, true), 3, 40) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(sb.start(&mut src), Ok(3));
        test_assert_eq!(sb.reevaluate(&mut src), 3);
        sb.stop(&mut src);
        TestResult::Pass
    }

    /// Rule 3. Fewer frames than the floor is refusal, not partial confinement.
    fn test_start_refuses_below_floor() -> TestResult {
        let mut short = StubFrames::new(3);
        let Ok(mut sb) = Sandbox::new(policy(1_000_000, true), 4, 16) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(sb.start(&mut short), Err(SandboxError::Refused));
        test_assert_eq!(sb.resident(), 0);
        test_assert!(!sb.running(), "a refused sandbox must not be running");
        test_assert_eq!(short.outstanding, 0);
        test_assert_eq!(short.budget, 3);
        test_assert_eq!(short.released, 3);
        // A refused sandbox observes nothing, so it cannot be sized either.
        test_assert!(!sb.observe(1), "a sandbox that did not start must not run");
        test_assert_eq!(sb.reevaluate(&mut short), 0);

        // Control: exactly the floor is enough, and nothing more is taken.
        let mut exact = StubFrames::new(4);
        let Ok(mut ok) = Sandbox::new(policy(1_000_000, true), 4, 16) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(ok.start(&mut exact), Ok(4));
        test_assert!(ok.running());
        test_assert_eq!(ok.resident(), 4);
        test_assert_eq!(exact.outstanding, 4);
        test_assert_eq!(exact.budget, 0);
        ok.stop(&mut exact);
        test_assert_eq!(exact.outstanding, 0);
        TestResult::Pass
    }

    /// Rule 4. A shrink returns only frames the guest is not holding, and the
    /// guest keeps running.
    fn test_shrink_spares_held_frames() -> TestResult {
        let mut src = StubFrames::new(64);
        let Ok(mut sb) = Sandbox::new(policy(1_000_000, true), 2, 16) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(sb.start(&mut src), Ok(2));

        // Four separated regions: 4 x FRAMES_PER_REGION = the whole cap.
        for a in clustered(4, 8) {
            test_assert!(sb.observe(a.page));
        }
        test_assert_eq!(sb.reevaluate(&mut src), 16);

        let held: Vec<PhysAddr> = sb.frame_addrs().into_iter().take(6).collect();
        for addr in &held {
            test_assert!(sb.pin(*addr), "pin must find a resident frame");
        }
        test_assert_eq!(sb.held(), 6);

        // Phase change: one swept region, no separation, so the envelope wants
        // the floor of 2 — well below the 6 frames the guest is holding.
        for i in 0..MAX_SAMPLES {
            test_assert!(sb.observe(i as u64));
        }
        let after = sb.reevaluate(&mut src);
        test_assert_eq!(after, 6);
        test_assert!(sb.running(), "shrinking must leave the guest running");
        test_assert_eq!(sb.held(), 6);
        let remaining = sb.frame_addrs();
        for addr in &held {
            test_assert!(
                remaining.contains(addr),
                "shrink evicted a frame the guest still holds"
            );
        }
        test_assert_eq!(src.outstanding, 6);

        // Unpinning lets the envelope reach the floor it was already asking for.
        for addr in &held {
            test_assert!(sb.unpin(*addr));
        }
        test_assert_eq!(sb.reevaluate(&mut src), 2);
        test_assert_eq!(src.outstanding, 2);
        sb.stop(&mut src);
        test_assert_eq!(src.outstanding, 0);
        TestResult::Pass
    }

    /// The permission boundary only ever narrows the envelope. An access
    /// outside the window never enters the trace, and a policy that denies
    /// growth pins the envelope at the floor whatever the topology reads.
    fn test_policy_only_narrows() -> TestResult {
        let mut src = StubFrames::new(64);
        let Ok(mut sb) = Sandbox::new(policy(100, true), 2, 16) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(sb.start(&mut src), Ok(2));
        for a in clustered(8, 8) {
            test_assert_eq!(sb.observe(a.page), a.page <= 100);
        }
        test_assert!(sb.denied() > 0, "the window must have refused something");
        // Only the first region is inside the window, so the trace is one
        // region and the envelope stays at the floor.
        test_assert_eq!(sb.reevaluate(&mut src), 2);
        sb.stop(&mut src);

        let mut src2 = StubFrames::new(64);
        let Ok(mut pinned) = Sandbox::new(policy(1_000_000, false), 2, 16) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(pinned.start(&mut src2), Ok(2));
        for a in clustered(8, 8) {
            test_assert!(pinned.observe(a.page));
        }
        test_assert!(
            matches!(pinned.summary(), WorkingSet::Clustered { clusters: 8, .. }),
            "the signal must still read eight regions"
        );
        test_assert_eq!(pinned.reevaluate(&mut src2), 2);
        pinned.stop(&mut src2);
        TestResult::Pass
    }

    /// The summary is a statement about shape, so relabelling either axis
    /// affinely must not move it. Without the per-axis rescaling the page axis
    /// swamps the time axis and this fails.
    fn test_summary_invariant_under_affine_page_relabel() -> TestResult {
        let base = clustered(6, 8);
        let relabelled: Vec<Access> = base
            .iter()
            .map(|a| Access {
                tick: a.tick * 3 + 11,
                page: a.page * 7 + 1_000_000,
            })
            .collect();
        let (WorkingSet::Clustered { clusters: ka, separation: sa, .. }, WorkingSet::Clustered { clusters: kb, separation: sb, .. }) =
            (summarize(&base), summarize(&relabelled))
        else {
            return TestResult::Fail("both fixtures must be measurable");
        };
        test_assert_eq!(ka, 6);
        test_assert_eq!(kb, 6);
        let agrees = libm::fabs(sa - sb) < 1e-9;
        test_assert!(
            agrees,
            "separation must not depend on the page base or stride"
        );
        TestResult::Pass
    }

    /// The defect b14631c named as its own ceiling: an inference guest with one
    /// contiguous multi-gigabyte weight tensor reads as one cluster, and a
    /// count-driven envelope hands it `FRAMES_PER_REGION` frames.
    ///
    /// The envelope must scale with the range the region covers. Nothing else
    /// about the reading changes between the three tensors below — one cluster,
    /// no separation — so a count-driven envelope returns the same number for
    /// all three, and an extent-driven one returns three different ones in
    /// proportion.
    fn test_extent_sizes_one_large_region() -> TestResult {
        // 8 GiB of weights is 2^21 pages. 64 retained accesses land one every
        // 32_768 pages, spanning 63 * 32_768 + 1.
        let tensor = swept(MAX_SAMPLES, 32_768);
        let ws = summarize(&tensor);
        let WorkingSet::Clustered { clusters, .. } = ws else {
            return TestResult::Fail("a swept region must be measurable");
        };
        test_assert_eq!(clusters, 1);
        test_assert_eq!(size_envelope(&ws, 2, 1 << 20), 64_513);

        // Half the tensor, same one cluster, half the envelope.
        let half = summarize(&swept(MAX_SAMPLES, 16_384));
        test_assert!(matches!(half, WorkingSet::Clustered { clusters: 1, .. }));
        test_assert_eq!(size_envelope(&half, 2, 1 << 20), 32_257);

        // Extent is a page count, so a seven-fold page stride is a seven-fold
        // envelope. Reading the span off the normalised coordinates instead of
        // mapping it back through the observed range gives all three tensors
        // the same span of 1.
        let wide = summarize(&swept(MAX_SAMPLES, 7 * 32_768));
        test_assert!(matches!(wide, WorkingSet::Clustered { clusters: 1, .. }));
        test_assert_eq!(size_envelope(&wide, 2, 1 << 20), 451_585);

        // Rule 1 at the new arithmetic: extent is demand, and demand does not
        // outrank the cap.
        test_assert_eq!(size_envelope(&ws, 2, 256), 256);
        test_assert_eq!(size_envelope(&ws, 2, 3), 3);
        test_assert_eq!(size_envelope(&wide, 2, 256), 256);

        // Extent runs up against u64, and this profile has no overflow checks.
        // A demand that wrapped would land under any cap and read as the cap
        // holding, so the saturation is asserted where it is visible: against a
        // cap nothing can reach.
        let absurd = WorkingSet::Clustered {
            clusters: usize::MAX,
            separation: 1e300,
            pages: u64::MAX,
        };
        test_assert_eq!(size_envelope(&absurd, 4, usize::MAX), usize::MAX);
        test_assert_eq!(frames_for_pages(u64::MAX), 576_460_752_303_423_488);
        // Extent alone, with a count that asks for nothing.
        let vast = WorkingSet::Clustered {
            clusters: 1,
            separation: 1.0,
            pages: u64::MAX,
        };
        test_assert_eq!(size_envelope(&vast, 4, usize::MAX), 576_460_752_303_423_488);
        test_assert_eq!(size_envelope(&vast, 4, 9), 9);

        // Saturation at the measurement, not only at the sizing: a guest that
        // touches page 0 and page u64::MAX in one region spans u64::MAX + 1
        // pages, which is not a u64. Wrapping there reports the largest region
        // measurable as no region at all, and sends the envelope to the floor.
        let mut everything = swept(MAX_SAMPLES, u64::MAX / MAX_SAMPLES as u64);
        everything[MAX_SAMPLES - 1].page = u64::MAX;
        let all = summarize(&everything);
        test_assert!(matches!(all, WorkingSet::Clustered { clusters: 1, .. }));
        test_assert_eq!(size_envelope(&all, 2, usize::MAX), 576_460_752_303_423_488);
        test_assert_eq!(size_envelope(&all, 2, 1_000_000), 1_000_000);

        // And through the live sandbox, where the guest touches the tensor.
        let mut src = StubFrames::new(1024);
        let Ok(mut sb) = Sandbox::new(policy(1 << 22, true), 2, 128) else {
            return TestResult::Fail("bounds must be accepted");
        };
        test_assert_eq!(sb.start(&mut src), Ok(2));
        for a in swept(MAX_SAMPLES, 32_768) {
            test_assert!(sb.observe(a.page), "policy must admit the fixture");
        }
        test_assert_eq!(sb.reevaluate(&mut src), 128);
        test_assert!(sb.resident() <= sb.cap(), "envelope exceeded its own cap");
        sb.stop(&mut src);
        TestResult::Pass
    }

    /// A cluster too thin to have measured a region buys no frames for the
    /// range it happens to straddle.
    ///
    /// Three phases: 40 accesses over 39_937 pages, exactly `MIN_SAMPLES`
    /// accesses over 3_073 pages, and `MIN_SAMPLES - 1` accesses straddling
    /// 100_001 pages. The third is two or three stray touches, not a region,
    /// and its span is charged to nobody.
    fn test_thin_cluster_buys_no_extent() -> TestResult {
        let mut v = Vec::new();
        for i in 0..40u64 {
            v.push(Access {
                tick: i,
                page: i * 1024,
            });
        }
        for i in 0..MIN_SAMPLES as u64 {
            v.push(Access {
                tick: 50 + i,
                page: 500_000 + i * 1024,
            });
        }
        for i in 0..MIN_SAMPLES as u64 - 1 {
            v.push(Access {
                tick: 60 + i,
                page: 900_000 + i * 50_000,
            });
        }
        let ws = summarize(&v);
        let WorkingSet::Clustered {
            clusters,
            separation,
            pages,
        } = ws
        else {
            return TestResult::Fail("three distant phases must be measurable");
        };
        test_assert_eq!(pages, 39_937 + 3_073);
        test_assert_eq!(clusters, 3);
        let well_separated = separation > SEPARATION_FULL;
        test_assert!(well_separated, "distant phases must read as separated");

        // 39_937 + 3_073 measured pages. The thin phase's 100_001 would take
        // this to 4_375.
        test_assert_eq!(size_envelope(&ws, 2, 1 << 20), 1_345);
        test_assert_eq!(size_envelope(&ws, 2, 64), 64);
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test(
            "sandbox::cap_bounds_topological_demand",
            test_cap_bounds_topological_demand,
        );
        crate::testing::register_test(
            "sandbox::unmeasurable_sizes_to_floor",
            test_unmeasurable_sizes_to_floor,
        );
        crate::testing::register_test(
            "sandbox::start_refuses_below_floor",
            test_start_refuses_below_floor,
        );
        crate::testing::register_test(
            "sandbox::shrink_spares_held_frames",
            test_shrink_spares_held_frames,
        );
        crate::testing::register_test("sandbox::policy_only_narrows", test_policy_only_narrows);
        crate::testing::register_test(
            "sandbox::extent_sizes_one_large_region",
            test_extent_sizes_one_large_region,
        );
        crate::testing::register_test(
            "sandbox::thin_cluster_buys_no_extent",
            test_thin_cluster_buys_no_extent,
        );
        crate::testing::register_test(
            "sandbox::summary_invariant_under_affine_page_relabel",
            test_summary_invariant_under_affine_page_relabel,
        );
    }
}
