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
//! high-water mark — it is how many *separated regions* its accesses fall into.
//! A guest whose accesses form one contiguous sweep wants one resident region
//! however many pages it touches; a guest that alternates between k distant
//! regions per phase wants k, and thrashes if it gets fewer.
//!
//! That is an H₀ question, so it is answered with H₀.
//!
//! # The embedding
//!
//! Each observed access is a point `(tick, page)`. Both axes are then rescaled
//! to `[0, 1]` over the observed range of the window, which is what makes the
//! metric dimensionless: a page index runs to millions and a tick to 64, so an
//! un-normalised Euclidean distance is a page-index distance with rounding
//! noise attached. After rescaling, the summary is invariant under any affine
//! relabelling of either axis — the same guest observed with a different page
//! base or a different sampling cadence measures identically, which
//! [`tests::test_summary_invariant_under_affine_page_relabel`] asserts.
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
//!    `clusters == usize::MAX` — can raise the envelope past it. Inference may
//!    narrow, never widen.
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
    Clustered { clusters: usize, separation: f64 },
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

    let mut edges = [0.0f64; MAX_SAMPLES];
    let m = mst_edges(&cloud[..n], &mut edges);
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
    let eps = e[m - 1] * 1e-9;
    let mut separation = 1.0f64;
    let mut cut = m;
    for i in 1..m {
        if e[i - 1] <= eps {
            continue;
        }
        let ratio = e[i] / e[i - 1];
        if ratio > separation {
            separation = ratio;
            cut = i;
        }
    }
    WorkingSet::Clustered {
        clusters: 1 + (m - cut),
        separation,
    }
}

/// Map a summary to a resident frame count within `[floor, cap]`.
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
        } => {
            if !separation.is_finite() {
                // A caller-supplied signal that is not a number is not evidence
                // of demand; every comparison against it would be false.
                floor
            } else {
                let demand = clusters.saturating_mul(FRAMES_PER_REGION);
                let confidence =
                    ((separation - 1.0) / (SEPARATION_FULL - 1.0)).clamp(0.0, 1.0);
                let grant = (demand.saturating_sub(floor) as f64 * confidence) as usize;
                floor.saturating_add(grant)
            }
        }
    };
    // Rule 1.
    want.min(cap)
}

/// Minimum spanning tree edge lengths by Prim's algorithm, O(n²).
///
/// Writes the positive finite edges to `out` and returns how many there were.
/// Coincident points contribute a zero-length edge, which is dropped: they are
/// the same position and merge into the same component at every scale. A cloud
/// whose points all coincide therefore yields 0, which is the caller's signal
/// that there is no scale to measure at.
///
/// ponytail: a near-copy of the Prim loop in `ml_engine/stratum.rs`. That one
/// is private, hard-typed to `ManifoldPoint<3>` and `STRATUM_WINDOW`, and
/// returns only `(max, median)` rather than the spectrum this needs. Upgrade
/// path is to lift one generic `mst_edges` into a shared geometry module and
/// have `stratum` derive its two statistics from it; that is a two-file change
/// and this module owns neither file today.
fn mst_edges(pts: &[ManifoldPoint<2>], out: &mut [f64; MAX_SAMPLES]) -> usize {
    let n = pts.len();
    if n < 2 {
        return 0;
    }
    let mut included = [false; MAX_SAMPLES];
    let mut key = [f64::INFINITY; MAX_SAMPLES];
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
        if best_key.is_finite() && best_key > 0.0 {
            out[count] = best_key;
            count += 1;
        }
        for i in 0..n {
            if !included[i] {
                let d = pts[best].distance(&pts[i]);
                if d < key[i] {
                    key[i] = d;
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
        let absurd = WorkingSet::Clustered {
            clusters: usize::MAX,
            separation: 1e300,
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

        // A signal that is not a number is not evidence of demand either.
        let nan = WorkingSet::Clustered {
            clusters: 16,
            separation: f64::NAN,
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
        let (WorkingSet::Clustered { clusters: ka, separation: sa }, WorkingSet::Clustered { clusters: kb, separation: sb }) =
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
            "sandbox::summary_invariant_under_affine_page_relabel",
            test_summary_invariant_under_affine_page_relabel,
        );
    }
}
