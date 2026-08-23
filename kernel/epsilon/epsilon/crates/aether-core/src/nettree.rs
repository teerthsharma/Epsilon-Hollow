// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Hierarchical net-tree over a point cloud.
//!
//! A net-tree is the structure Sheehy's sparse Vietoris-Rips construction is
//! built on (*Linear-Size Approximations to the Vietoris-Rips Filtration*,
//! arXiv:1203.6786, Discrete & Computational Geometry 49(4):778-796, 2013). It
//! is a sequence of nested nets at geometrically increasing radii, each
//! satisfying two invariants:
//!
//! * **Packing** — distinct net points at level `l` are strictly more than
//!   `r_l` apart.
//! * **Covering** — every point of level `l - 1` lies within `r_l` of some
//!   level-`l` net point.
//!
//! Both are enforced by construction and asserted per level in
//! `tests/house_nettree_invariants.rs`. A greedy pass gives both: a candidate
//! is selected exactly when it is not already covered, so the unselected are
//! covered by definition and the selected are separated by definition.
//!
//! # Why this is cheap on the sphere
//!
//! The size of each net is governed by the doubling constant of the metric, in
//! Sheehy's sense — the least number of radius-`r` balls needed to cover a ball
//! of radius `2r`, taken over the finite point set. For `S^2` under the chordal
//! metric that constant is at most 25 for every radius, because the chordal
//! ball of radius `t` has area exactly `pi * t^2` (the sphere is Ahlfors
//! 2-regular in this metric) and a disjoint-cap count applies. The bound is
//! inherited by every subset, so it holds for any point configuration the
//! kernel produces, with no assumption about how the points are distributed.

extern crate alloc;

use alloc::vec::Vec;

use crate::manifold::ManifoldPoint;

/// Nested nets at geometrically increasing radii.
#[derive(Debug, Clone)]
pub struct NetTree {
    /// `levels[l]` holds indices into the original point slice.
    levels: Vec<Vec<usize>>,
    /// `radii[l]` is the separation and covering radius of level `l`.
    radii: Vec<f64>,
}

impl NetTree {
    /// Build the hierarchy with ratio `tau` between consecutive radii.
    ///
    /// `tau` is clamped to at least 1.5 so the radii grow and the construction
    /// terminates. The base radius is half the smallest positive interpoint
    /// distance, which places every distinct point in level 0 and collapses
    /// exact duplicates into a single representative.
    ///
    /// Returns an empty tree for empty input.
    pub fn build<const D: usize>(points: &[ManifoldPoint<D>], tau: f64) -> Self {
        if points.is_empty() {
            return Self { levels: Vec::new(), radii: Vec::new() };
        }
        let tau = if tau.is_finite() && tau >= 1.5 { tau } else { 2.0 };

        // Smallest positive interpoint distance; `None` when all coincide.
        let mut min_positive = f64::INFINITY;
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let d = points[i].distance(&points[j]);
                if d > 0.0 && d < min_positive {
                    min_positive = d;
                }
            }
        }
        // All points coincident, or a single point: any positive radius works.
        let mut radius = if min_positive.is_finite() { min_positive / 2.0 } else { 1.0 };

        let mut levels: Vec<Vec<usize>> = Vec::new();
        let mut radii: Vec<f64> = Vec::new();
        let mut candidates: Vec<usize> = (0..points.len()).collect();

        loop {
            let net = greedy_net(points, &candidates, radius);
            let done = net.len() == 1;
            candidates = net.clone();
            levels.push(net);
            radii.push(radius);
            if done {
                break;
            }
            radius *= tau;
        }

        Self { levels, radii }
    }

    /// Number of levels, root included.
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Indices of the level-`l` net, into the slice passed to [`Self::build`].
    pub fn net(&self, level: usize) -> &[usize] {
        &self.levels[level]
    }

    /// Separation and covering radius of level `l`.
    pub fn radius(&self, level: usize) -> f64 {
        self.radii[level]
    }
}

/// Select a maximal `radius`-separated subset of `candidates`, greedily.
///
/// A candidate is kept exactly when no already-kept point lies within `radius`
/// of it. The kept points are therefore pairwise more than `radius` apart
/// (packing), and every discarded point is within `radius` of a kept one
/// (covering). Both invariants hold by construction, not by later repair.
fn greedy_net<const D: usize>(
    points: &[ManifoldPoint<D>],
    candidates: &[usize],
    radius: f64,
) -> Vec<usize> {
    let mut net: Vec<usize> = Vec::new();
    for &c in candidates {
        let covered = net.iter().any(|&k| points[c].distance(&points[k]) <= radius);
        if !covered {
            net.push(c);
        }
    }
    net
}

// ═══════════════════════════════════════════════════════════════════════════════
// Relaxed (weighted) distance — Sheehy, arXiv:1203.6786 section 4
// ═══════════════════════════════════════════════════════════════════════════════

/// Weight of a point at scale `alpha`, given its deletion time `t` and the
/// tightness parameter `eps`.
///
/// Sheehy section 4, transcribed from the paper:
///
/// ```text
/// w_p(alpha) = 0                             if alpha <= (1 - 2 eps) t_p
///            = (alpha - (1 - 2 eps) t_p) / 2 if (1 - 2 eps) t_p < alpha < t_p
///            = eps * alpha                   if t_p <= alpha
/// ```
///
/// The weight is zero until just before the point's removal time, then rises at
/// slope `1/2`, then at slope `eps`. It is continuous at both breakpoints — at
/// `alpha = t_p` the middle branch gives `(t_p - (1 - 2 eps) t_p)/2 = eps t_p`,
/// which is the third branch's value — and it is `1/2`-Lipschitz, since
/// `eps <= 1/3 < 1/2`. Both facts are asserted in
/// `tests/house_relaxed_distance.rs`; the Lipschitz constant is what makes the
/// paper's Lemma 4.1 go through.
///
/// `eps` is clamped to `(0, 1/3]`, the range the paper assumes throughout.
pub fn weight(alpha: f64, deletion_time: f64, eps: f64) -> f64 {
    let eps = if eps.is_finite() && eps > 0.0 && eps <= 1.0 / 3.0 { eps } else { 1.0 / 3.0 };
    let t = if deletion_time.is_finite() && deletion_time > 0.0 { deletion_time } else { 0.0 };
    let knee = (1.0 - 2.0 * eps) * t;

    if alpha <= knee {
        0.0
    } else if alpha < t {
        0.5 * (alpha - knee)
    } else {
        eps * alpha
    }
}

/// Relaxed distance at scale `alpha`:
/// `d_alpha(p, q) = d(p, q) + w_p(alpha) + w_q(alpha)`.
///
/// This is deliberately **not** a metric — the paper relaxes the input metric so
/// that a ball may be covered by nearby balls, which is what permits points to
/// be deleted without changing the topology. Two properties survive the
/// relaxation and both are tested:
///
/// * `d_alpha` is monotonically non-decreasing in `alpha`, and `d_0 = d`, so
///   `d_alpha >= d` always.
/// * **Lemma 4.1**: if `d_alpha(p,q) <= alpha <= beta` then
///   `d_beta(p,q) <= beta`.
///
/// The deletion times need only be non-negative; the paper assumes nothing else
/// about them in this section. Their specific choice comes from the net-tree
/// and affects the *size* guarantee, not the correctness of this layer.
pub fn relaxed_distance(d: f64, alpha: f64, t_p: f64, t_q: f64, eps: f64) -> f64 {
    d + weight(alpha, t_p, eps) + weight(alpha, t_q, eps)
}

impl NetTree {
    /// Deletion time of each input point, per Sheehy section 6.
    ///
    /// Verbatim from arXiv:1203.6786: "let `v_p` denote the least ancestor among
    /// the nodes in T represented by p. For each `p in P` the deletion time
    /// `t_p` is defined as `t_p := (1 / (eps(1 - 2 eps))) rad(par(v_p))`. This
    /// is just the radius of the parent of `v_p` with a small scaling factor
    /// included for technical reasons."
    ///
    /// Two details matter and both were got wrong here before the paper was
    /// read. It is the radius of the **parent** of the last node representing
    /// `p`, one level coarser than `p`'s own; and it carries the factor
    /// `1 / (eps(1 - 2 eps))`, which is 9 at `eps = 1/3` and 22.2 at
    /// `eps = 0.05`.
    ///
    /// That factor is the whole reason the construction tightens as `eps`
    /// shrinks: smaller `eps` gives **later** deletion, so points are retained
    /// longer and the approximation improves. A deletion rule with no `eps`
    /// dependence cannot do that, and iteration 20 measured the consequence —
    /// bottleneck error superlinear in `eps`, exceeding the whole feature scale
    /// at `eps = 1/3`.
    ///
    /// Points that survive to the root have no parent; they take the coarsest
    /// radius, scaled the same way.
    pub fn deletion_times(&self, point_count: usize, eps: f64) -> Vec<f64> {
        let eps = if eps.is_finite() && eps > 0.0 && eps <= 1.0 / 3.0 { eps } else { 1.0 / 3.0 };
        let scale = 1.0 / (eps * (1.0 - 2.0 * eps));

        // Last level at which each point is still a net representative.
        let mut last_level = alloc::vec![0usize; point_count];
        for level in 0..self.levels() {
            for &idx in &self.levels[level] {
                if idx < point_count {
                    last_level[idx] = level;
                }
            }
        }

        let top = self.radii.len().saturating_sub(1);
        last_level
            .into_iter()
            .map(|l| {
                // rad(par(v_p)): one level coarser than the node representing p.
                let parent = (l + 1).min(top);
                self.radii.get(parent).copied().unwrap_or(0.0) * scale
            })
            .collect()
    }
}

/// The filtration value of a pair under the relaxed distance: the smallest
/// `alpha` at which `d_alpha(p, q) <= alpha`.
///
/// The relaxed distance moves with `alpha`, so it is not a distance matrix and
/// cannot be handed to a Rips builder directly. Sheehy's Lemma 4.1 —
/// `d_alpha(p,q) <= alpha <= beta` implies `d_beta(p,q) <= beta` — says the set
/// of admitting `alpha` is an upward-closed ray, so its infimum is a
/// well-defined entry time and the resulting complex is a genuine filtration.
///
/// Solved by bisection on the monotone predicate `d_alpha(p,q) <= alpha`, which
/// Lemma 4.1 guarantees flips exactly once. The weight is piecewise linear with
/// slopes `0`, `1/2` and `eps`, all below 1, so `alpha - d_alpha` is strictly
/// increasing and a bracket always exists.
///
/// Properties asserted in `tests/house_relaxed_entry_time.rs`:
///
/// * the pair is admitted at its entry time and not strictly below it;
/// * the entry time is never earlier than the true distance, since
///   `d_alpha >= d`;
/// * with deletion times far beyond the scale both weights vanish and the entry
///   time **is** the true distance — the relaxed filtration degenerates to the
///   exact Rips filtration, which is the reduction the construction must
///   satisfy;
/// * it is monotone in the true distance, without which the complex is not a
///   filtration;
/// * a shorter deletion time never makes a pair enter earlier.
pub fn relaxed_entry_time(d: f64, t_p: f64, t_q: f64, eps: f64) -> f64 {
    if !d.is_finite() || d < 0.0 {
        return f64::INFINITY;
    }
    let admits = |a: f64| relaxed_distance(d, a, t_p, t_q, eps) <= a;

    // `d` itself is the earliest conceivable entry time, since `d_alpha >= d`.
    if admits(d) {
        return d;
    }
    // Grow until the predicate flips. Weight slopes are at most 1/2 < 1, so
    // `alpha - d_alpha(alpha)` increases without bound and this terminates.
    let mut hi = if d > 0.0 { d * 2.0 } else { 1.0 };
    let mut guard = 0;
    while !admits(hi) {
        hi *= 2.0;
        guard += 1;
        if guard > 200 {
            return f64::INFINITY;
        }
    }
    let mut lo = d;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if admits(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

// ═══════════════════════════════════════════════════════════════════════════════
// Range queries — Sheehy section 10's neighbour search
// ═══════════════════════════════════════════════════════════════════════════════

impl NetTree {
    /// For each level, which points each net point covers at that level.
    ///
    /// The covering invariant says every point of level `l - 1` lies within
    /// `r_l` of some level-`l` net point. Recording *which* one turns the
    /// hierarchy into a search structure: a ball of radius `q` around a query
    /// can only contain points covered by level-`l` net points within
    /// `q + r_l` of it, by the triangle inequality.
    ///
    /// Returned as `cover[l][i]`, the points covered by `net(l)[i]`.
    pub fn covering<const D: usize>(&self, points: &[ManifoldPoint<D>]) -> Vec<Vec<Vec<usize>>> {
        let mut out = Vec::with_capacity(self.levels());
        for level in 0..self.levels() {
            let net = &self.levels[level];
            let radius = self.radii[level];
            let source: Vec<usize> = if level == 0 {
                (0..points.len()).collect()
            } else {
                self.levels[level - 1].clone()
            };
            let mut buckets = alloc::vec![Vec::new(); net.len()];
            for &p in &source {
                // Nearest net point; the covering invariant guarantees one
                // within `radius`, and ties are broken by index for
                // determinism.
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for (slot, &c) in net.iter().enumerate() {
                    let d = points[p].distance(&points[c]);
                    if d < best_d {
                        best_d = d;
                        best = slot;
                    }
                }
                let _ = radius;
                buckets[best].push(p);
            }
            out.push(buckets);
        }
        out
    }

    /// Every point within `radius` of `points[query]`, found by descending the
    /// hierarchy instead of scanning all points.
    ///
    /// At level `l` a candidate can only lie under a net point within
    /// `radius + r_l` of the query, since the covering invariant places it
    /// within `r_l` of that net point and the triangle inequality does the
    /// rest. Branches outside that bound are discarded whole.
    ///
    /// This is the search Sheehy's section 10 uses to reach `O(n log n)`
    /// construction. It returns exactly what a linear scan returns — asserted
    /// edge-for-edge against the quadratic enumeration in
    /// `tests/house_range_query.rs` — so it is a speedup, never a different
    /// answer.
    pub fn range_query<const D: usize>(
        &self,
        points: &[ManifoldPoint<D>],
        cover: &[Vec<Vec<usize>>],
        query: usize,
        radius: f64,
    ) -> Vec<usize> {
        if self.levels() == 0 {
            return Vec::new();
        }
        let top = self.levels() - 1;
        // Start from the root net, keep the slots whose subtree can reach.
        let mut frontier: Vec<usize> = (0..self.levels[top].len()).collect();
        let mut level = top;

        // Cumulative reach: a point at the bottom sits within `r_0` of its
        // level-0 net point, which sits within `r_1` of its level-1 ancestor,
        // and so on. Its distance to a level-`l` ancestor is therefore bounded
        // by the SUM of radii up to `l`, not by `r_l` alone. Pruning on the
        // single radius discards reachable branches — the oracle caught exactly
        // that on the first run.
        let mut reach = alloc::vec![0.0f64; self.radii.len()];
        let mut acc = 0.0;
        for (i, &r) in self.radii.iter().enumerate() {
            acc += r;
            reach[i] = acc;
        }

        loop {
            let r = reach[level];
            let keep: Vec<usize> = frontier
                .into_iter()
                .filter(|&slot| {
                    let c = self.levels[level][slot];
                    points[query].distance(&points[c]) <= radius + r
                })
                .collect();

            if level == 0 {
                let mut out = Vec::new();
                for slot in keep {
                    for &p in &cover[0][slot] {
                        if p != query && points[query].distance(&points[p]) <= radius {
                            out.push(p);
                        }
                    }
                }
                out.sort_unstable();
                return out;
            }

            // Descend: the points covered here are level-(l-1) net points.
            // The slot lookup is a precomputed map, not a linear search — an
            // earlier version called `position()` inside this loop, which made
            // the query 100x SLOWER than a plain scan.
            let mut slot_of = alloc::collections::BTreeMap::new();
            for (pos, &x) in self.levels[level - 1].iter().enumerate() {
                slot_of.insert(x, pos);
            }
            let mut next = Vec::new();
            for slot in keep {
                for p in &cover[level][slot] {
                    if let Some(&pos) = slot_of.get(p) {
                        next.push(pos);
                    }
                }
            }
            next.sort_unstable();
            next.dedup();
            frontier = next;
            level -= 1;
        }
    }
}
