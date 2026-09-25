//! Sparse attention with topology-derived key selection — CPU reference.
//!
//! This is a reference implementation, not a fast kernel. It exists so the
//! correctness contracts in `tests/attention_contracts.rs` have something to
//! bind to, and so the same-budget ablation can be run at all. There is no GPU
//! path in this workspace; a fast kernel needs one, and claiming a speedup
//! without it would be measuring nothing.
//!
//! What it does provide:
//!
//!   - `dense_attention`: ordinary scaled dot-product attention, the reference
//!     every sparse path is checked against.
//!   - `sparse_attention`: the same computation restricted to a boolean mask,
//!     with an explicit guard for rows that select nothing.
//!   - `Selector`: the key-selection rules, including the topological one and the
//!     three baselines an honest evaluation needs.
//!   - `attention_mass_recovered`: the diagnostic that says whether topology did
//!     the work or sparsity did.
//!
//! No backward pass. When one is added, a `gradcheck` against the dense path on
//! the same mask is the first test that must accompany it: a forward-correct
//! kernel with a wrong backward trains to a plausible worse optimum, which loss
//! curves alone will not reveal.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use libm::{exp, nextafter, sqrt};

/// How a row picks the keys it may attend to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Selector {
    /// Every key. The control: sparse must equal dense here.
    Dense,

    /// A sliding window of the `window` most recent keys (including self).
    ///
    /// The locality-only baseline. If a topological selector cannot beat this, it
    /// is rediscovering adjacency, not topology.
    Local { window: usize },

    /// `budget` keys drawn uniformly without replacement from the legal set.
    ///
    /// The sparsity baseline. Any gain a selector shows over this is the gain
    /// attributable to the selection rule rather than to sparsity itself.
    Random { budget: usize, seed: u64 },

    /// The `budget` keys with the genuinely largest attention weights, computed
    /// densely.
    ///
    /// Unimplementable in production — it needs the scores it is meant to avoid
    /// computing — but decisive as a diagnostic upper bound.
    ///
    /// The set is certified against float rounding; a row whose boundary the
    /// rounding bound cannot decide is widened past `budget` and reported (see
    /// [`select_mask_with_report`]).
    OracleTopK { budget: usize },

    /// The `budget` keys nearest the query in the shared embedding space, keeping
    /// only those within `radius_scale` times the row's median query-key
    /// distance.
    ///
    /// This is the mechanism under test: the claim is that geometric proximity in
    /// key space proxies attention mass.
    ///
    /// `radius_scale` is **relative**, not an absolute length. An absolute radius
    /// makes the selector silently dataset-dependent — the same failure the
    /// persistence scale-equivariance test exists to catch — and in practice it
    /// under-spends the budget: with an absolute radius of 0.6 against a median
    /// query-key distance of 2.4, the selector picked 1.0 keys per row while its
    /// same-budget baselines picked 5.5, and lost the ablation on budget rather
    /// than on mechanism. Use `f64::INFINITY` for pure budget-nearest.
    Topological { budget: usize, radius_scale: f64 },

    /// Route through a topological clustering of key *directions*, then rank the
    /// resulting candidate set by the exact dot product.
    ///
    /// The fix the ablation demanded. `Topological` conflated two jobs: finding
    /// candidates and scoring them. Euclidean proximity does both badly once key
    /// norms vary, because `‖q − k‖² = ‖q‖² + ‖k‖² − 2·q·k` — a large-norm key is
    /// far away *and* high-scoring. This variant splits them:
    ///
    ///   1. Single-linkage (H0) clustering of the unit-normalised keys builds the
    ///      candidate set. Normalising first makes the clustering invariant to
    ///      per-key rescaling, which is exactly what the old rule could not be.
    ///   2. The exact dot product ranks within the candidates, restoring the norm
    ///      sensitivity the geometry deliberately discarded.
    ///
    /// Cost: `clusters` centroid dot products to route, plus one per candidate to
    /// rank — not `seq`. That is what makes it a sparse method rather than a
    /// re-described dense one.
    TopologicalRouted { budget: usize, clusters: usize },

    /// Route when [`routing_plan`] says routing will pay, and fall back to a
    /// sliding window of the same budget when it will not.
    ///
    /// `TopologicalRouted` is a real sparsity win only when the keys have H0
    /// structure. On a cloud with no density gaps, single-linkage chains — 64
    /// uniform keys give components of size `[61, 1, 1, 1]` — so routing to the
    /// top cluster is routing to everything, at 0.999x the dense dot-product
    /// count. That is not a clustering bug; it is H0 correctly reporting that
    /// there is nothing to route on.
    ///
    /// A condition nobody checks at runtime is an assumption. This variant checks
    /// it. Checking is not free: the check aligns every query with every cluster
    /// centroid, `cluster_count` dot products per row. Routing reuses that pass;
    /// the dense fallback cannot, so it costs dense plus `cluster_count` per row.
    /// The bound is therefore dense plus the plan, not dense
    /// (`the_adaptive_selector_costs_at_most_dense_plus_its_plan`).
    Adaptive { budget: usize, clusters: usize },
}

/// What routing will cost on a given query and key tensor, decided before any
/// attention runs.
///
/// The H0 clustering is computed once per key tensor and amortised over every
/// query, head, and layer that reuses it, so this is cheap relative to attention
/// itself. The cost is not a property of the keys alone: each row pays for the
/// clusters its query aligns with, so the plan needs the queries to report the
/// cost the selector will actually incur.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingPlan {
    /// Dot products per row the router will perform, as a fraction of dense.
    pub cost_ratio: f64,

    /// Share of keys in the largest H0 component. Near 1 means chaining: one
    /// giant component and singletons, so the candidate set is everything.
    pub largest_cluster_share: f64,

    /// Ratio of the first merge height above the cut to the last one below it,
    /// derived from the H0 barcode alone.
    ///
    /// A clear separation between clusters shows up as a large ratio; chaining
    /// shows up as a ratio near 1. This is the signal a runtime could cache and
    /// reuse without redoing the clustering.
    pub gap_ratio: f64,

    /// The `cost_ratio` below which routing is judged worthwhile.
    pub threshold: f64,

    /// Whether to route. `cost_ratio < threshold`.
    pub worth_routing: bool,
}

/// Cost ratio below which routing is judged to pay for itself.
///
/// Routing has to beat dense by enough to be worth the clustering and the extra
/// control flow; 0.6 is the measured point where the discount is unambiguous
/// (4 real key clusters give 0.449, uniform keys give 0.999). Nothing depends on
/// the exact value — `the_plan_declines_to_route_exactly_when_routing_would_not_pay`
/// asserts the decision against whatever this is.
pub const ROUTING_COST_THRESHOLD: f64 = 0.6;

/// Decide whether topological routing will pay on these queries and keys.
pub fn routing_plan(
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    clusters: usize,
    budget: usize,
    causal: bool,
) -> RoutingPlan {
    let (assignment, merge_heights) = single_linkage_clusters(k, seq, head_dim, clusters, true);
    let cluster_count = assignment.iter().copied().max().map_or(0, |m| m + 1);

    let mut sizes = vec![0usize; cluster_count.max(1)];
    for &label in &assignment {
        sizes[label] += 1;
    }
    let largest = sizes.iter().copied().max().unwrap_or(0);

    let (routed_clusters, rows) = route_rows(q, k, seq, head_dim, clusters, budget, causal);
    let cost_ratio = routed_cost_ratio(routed_clusters * seq, &rows, seq, causal);

    // The cut sits between merge `seq - cluster_count - 1` and `seq - cluster_count`:
    // that many merges are taken, the rest are not. The ratio across it is how
    // sharply the data separates.
    let gap_ratio = {
        let taken = seq.saturating_sub(cluster_count);
        match (
            taken.checked_sub(1).and_then(|i| merge_heights.get(i)),
            merge_heights.get(taken),
        ) {
            (Some(&below), Some(&above)) if below > 0.0 => above / below,
            // No merge on one side of the cut, or a degenerate zero-height merge:
            // report 1.0, which reads as "no separation" and never triggers routing
            // on its own.
            _ => 1.0,
        }
    };

    RoutingPlan {
        cost_ratio,
        largest_cluster_share: largest as f64 / seq.max(1) as f64,
        gap_ratio,
        threshold: ROUTING_COST_THRESHOLD,
        worth_routing: cost_ratio < ROUTING_COST_THRESHOLD,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Attention
// ═══════════════════════════════════════════════════════════════════════════════

/// Scaled dot-product attention over a single head.
///
/// `q`, `k`, `v` are row-major `[seq, head_dim]`. Returns `[seq, head_dim]`.
///
/// Equivalent to `sparse_attention` with an all-true mask, and asserted to be so.
pub fn dense_attention(q: &[f64], k: &[f64], v: &[f64], seq: usize, head_dim: usize) -> Vec<f64> {
    sparse_attention(q, k, v, seq, head_dim, &vec![true; seq * seq])
}

/// Scaled dot-product attention restricted to `mask`, a row-major `[seq, seq]`
/// boolean where `mask[i * seq + j]` permits query `i` to attend to key `j`.
///
/// A row that selects no keys returns zeros rather than NaN. That case is
/// reachable: a topological selector on an isolated token — a point that is its
/// own connected component — selects nothing. Softmax over an empty set is
/// undefined, and the naive form (exp of all `-inf`) yields `0/0`.
pub fn sparse_attention(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    mask: &[bool],
) -> Vec<f64> {
    assert_eq!(q.len(), seq * head_dim, "q must be [seq, head_dim]");
    assert_eq!(k.len(), seq * head_dim, "k must be [seq, head_dim]");
    assert_eq!(v.len(), seq * head_dim, "v must be [seq, head_dim]");
    assert_eq!(mask.len(), seq * seq, "mask must be [seq, seq]");

    let scale = 1.0 / sqrt(head_dim as f64);
    let mut out = vec![0.0f64; seq * head_dim];
    let mut logits = vec![0.0f64; seq];

    for i in 0..seq {
        // Scores for the permitted keys, and their running maximum. Subtracting
        // the max before exp is what keeps large logits from overflowing to +inf.
        let mut max_logit = f64::NEG_INFINITY;
        let mut any = false;
        for j in 0..seq {
            if !mask[i * seq + j] {
                continue;
            }
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            let logit = dot * scale;
            logits[j] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
            any = true;
        }

        // The guard. Zeros, not NaN, and documented as the defined value.
        if !any {
            continue;
        }

        let mut denominator = 0.0;
        for j in 0..seq {
            if mask[i * seq + j] {
                denominator += exp(logits[j] - max_logit);
            }
        }

        for j in 0..seq {
            if !mask[i * seq + j] {
                continue;
            }
            let weight = exp(logits[j] - max_logit) / denominator;
            for d in 0..head_dim {
                out[i * head_dim + d] += weight * v[j * head_dim + d];
            }
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// Selection
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the `[seq, seq]` boolean mask a selector produces.
///
/// `causal` restricts every row `i` to keys `j <= i`. Selectors apply their rule
/// within the causal set, never outside it, so no budget is spent on keys that
/// would be masked away afterwards.
///
/// Every row selects at least one key: a row's own position is always legal under
/// causal masking, so the budgeted selectors always have something to fall back
/// on. The all-masked guard in `sparse_attention` therefore covers hand-built
/// masks, not selector output.
pub fn select_mask(
    selector: Selector,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<bool> {
    select_mask_with_report(selector, q, k, v, seq, head_dim, causal).0
}

/// What `select_mask_with_report` did beyond building the mask.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SelectionReport {
    /// Every row whose top-k boundary the rounding enclosures did not decide,
    /// with the refusal that names the pair. Its length is the count of widened
    /// rows.
    pub widened_rows: Vec<(usize, BoundaryRefusal)>,
    /// Every row that fell back to dense because a score or radius was not
    /// finite, with the key that carried it. Its length is the count of
    /// densified rows.
    pub dense_rows: Vec<(usize, usize)>,
    /// Keys selected past the budget by widening or densifying, summed over rows.
    pub extra_keys: usize,
    /// `head_dim`-length inner products the path actually ran performed, summed
    /// over rows, counted on the same terms as [`selection_dot_cost`]: each score
    /// or distance evaluated, each centroid alignment, and for `Dense` the full
    /// legal set it commits the kernel to. Dividing by `seq` gives the executed
    /// per-row cost, which `selection_dot_cost` must equal.
    pub dot_products: usize,
}

/// [`select_mask`], plus the report of which score-ranked rows were refused
/// certification and widened.
///
/// `OracleTopK` and step 3 of `TopologicalRouted` rank by float dot product.
/// Each score is enclosed by its a-priori rounding bound, and the top-`budget`
/// set is kept only when [`certified_top_k`] proves no rounding could have moved
/// a key across the boundary.
///
/// On a refusal the row is **widened**, not densified: it takes every key whose
/// enclosure reaches the lowest selected lower end. Every key left out lies
/// strictly below `budget` selected keys in exact arithmetic, so the widened set
/// provably contains the exact top-`budget`. Widening was chosen over a dense
/// fallback because it keeps that guarantee at the cost of the few keys inside
/// the ambiguous band rather than the whole row, and needs no extra dot products:
/// every score it consults was already computed. The growth is not silent — it is
/// counted here and exceeds the budget only on refused rows.
///
/// A score or radius that is not finite has no enclosure — an overflowed dot
/// product can read NaN or -inf while the exact score is finite and largest — so
/// no floor can be derived from it and widening cannot bound it. Such a row falls
/// back to dense over its legal keys and is reported in `dense_rows`.
pub fn select_mask_with_report(
    selector: Selector,
    q: &[f64],
    k: &[f64],
    _v: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> (Vec<bool>, SelectionReport) {
    // Routing structure is built once over all keys, not per row: a key's cluster
    // does not depend on which query is looking at it. This is what keeps the
    // clustering cost amortised across the sequence rather than paid per row.
    // `resolve` also runs the alignment pass, once, and it is charged here.
    let Resolved {
        selector,
        mut routed,
        paid,
    } = resolve(selector, q, k, seq, head_dim, causal);
    let mut mask = vec![false; seq * seq];
    let mut report = SelectionReport {
        dot_products: paid,
        ..SelectionReport::default()
    };

    for i in 0..seq {
        let legal_end = if causal { i + 1 } else { seq };
        let legal: Vec<usize> = (0..legal_end).collect();

        let chosen: Vec<usize> = match selector {
            Selector::Dense => {
                report.dot_products += legal.len();
                legal.clone()
            }

            Selector::Local { window } => {
                let start = legal_end.saturating_sub(window.max(1));
                (start..legal_end).collect()
            }

            Selector::Random { budget, seed } => {
                // Deterministic per (seed, row): the same call must always produce
                // the same mask, or every A/B comparison is measuring noise.
                let mut pool = legal.clone();
                let mut state = splitmix(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let take = budget.min(pool.len());
                for slot in 0..take {
                    state = splitmix(state);
                    let pick = slot + (state as usize) % (pool.len() - slot);
                    pool.swap(slot, pick);
                }
                pool.truncate(take);
                pool
            }

            Selector::OracleTopK { budget } => {
                let scored: Vec<(usize, f64, f64)> = legal
                    .iter()
                    .map(|&j| enclosed_dot(q, k, i, j, head_dim))
                    .collect();
                report.dot_products += scored.len();
                certified_or_widened(&scored, budget, i, legal_end, &mut report)
            }

            Selector::Topological {
                budget,
                radius_scale,
            } => {
                // Geometric proximity in key space, NOT the dot product: this rule
                // must not peek at the scores it claims to predict, or the
                // comparison against OracleTopK is circular.
                let mut scored: Vec<(usize, f64)> = legal
                    .iter()
                    .map(|&j| (j, key_distance(q, k, i, j, head_dim)))
                    .collect();
                report.dot_products += scored.len();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

                // The radius is a multiple of this row's median query-key
                // distance, which makes the whole rule scale-equivariant: scaling
                // q and k by c scales every distance and the median alike, so the
                // selected set is unchanged.
                let cutoff = if radius_scale.is_finite() {
                    radius_scale * scored[scored.len() / 2].1
                } else {
                    f64::INFINITY
                };

                let mut chosen: Vec<usize> = scored
                    .iter()
                    .filter(|&&(_, distance)| distance <= cutoff)
                    .take(budget)
                    .map(|&(j, _)| j)
                    .collect();

                // An isolated token has no neighbour inside the radius. Falling
                // back to self keeps the row well-defined without silently
                // widening the radius, which would make the parameter meaningless.
                if chosen.is_empty() {
                    chosen.push(if causal { i } else { i.min(seq - 1) });
                }
                chosen
            }

            // `resolve` collapses Adaptive before the loop, so it cannot appear
            // here. Panicking rather than silently picking a branch keeps the
            // "cost reported equals path taken" guarantee honest.
            Selector::Adaptive { .. } => unreachable!("Adaptive is resolved before selection"),

            Selector::TopologicalRouted { budget, .. } => {
                // Steps 1 and 2, already run by `resolve`: the legal members of
                // the best-aligned clusters.
                let candidates =
                    core::mem::take(&mut routed.as_mut().expect("resolve routes this selector")[i]);

                // Step 3: the exact dot product decides within the candidate set.
                // This is what the nearest-neighbour rule was missing: geometry
                // finds where to look, the score decides what to take. The
                // decision is certified against rounding, as for `OracleTopK`.
                let scored: Vec<(usize, f64, f64)> = candidates
                    .into_iter()
                    .map(|j| enclosed_dot(q, k, i, j, head_dim))
                    .collect();
                report.dot_products += scored.len();
                let mut chosen = certified_or_widened(&scored, budget, i, legal_end, &mut report);
                if chosen.is_empty() {
                    chosen.push(if causal { i } else { i.min(seq - 1) });
                }
                chosen
            }
        };

        for j in chosen {
            mask[i * seq + j] = true;
        }
    }
    (mask, report)
}

/// The H0 routing structure over one key tensor: cluster labels and the
/// unit-direction centroid of each cluster.
///
/// Shared by the selector and by [`selection_dot_cost`], so the cost model walks
/// exactly the clusters the selector walks, in the same order.
struct Routing {
    assignment: Vec<usize>,
    centroids: Vec<f64>,
    cluster_count: usize,
}

impl Routing {
    fn build(k: &[f64], seq: usize, head_dim: usize, clusters: usize) -> Self {
        let (assignment, _) = single_linkage_clusters(k, seq, head_dim, clusters, true);
        let cluster_count = assignment.iter().copied().max().map_or(0, |m| m + 1);

        // Centroids of the unit-normalised keys: the direction each cluster
        // represents. Routing compares queries against these, so the router
        // does `cluster_count` dot products instead of `seq`.
        let mut centroids = vec![0.0f64; cluster_count * head_dim];
        let mut members = vec![0usize; cluster_count];
        for (t, &label) in assignment.iter().enumerate() {
            let norm = sqrt(
                (0..head_dim)
                    .map(|d| k[t * head_dim + d] * k[t * head_dim + d])
                    .sum::<f64>(),
            );
            let inverse = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for d in 0..head_dim {
                centroids[label * head_dim + d] += k[t * head_dim + d] * inverse;
            }
            members[label] += 1;
        }
        for label in 0..cluster_count {
            let count = members[label].max(1) as f64;
            for d in 0..head_dim {
                centroids[label * head_dim + d] /= count;
            }
        }
        Self {
            assignment,
            centroids,
            cluster_count,
        }
    }

    /// The keys below `legal_end` that query `i` is routed to.
    ///
    /// Step 1 ranks clusters by query-to-centroid alignment — the only place the
    /// query meets the topology, at `cluster_count` dot products. Step 2 pulls the
    /// legal members of the best clusters, whole, until there are at least
    /// `budget` candidates to choose between.
    fn candidates(
        &self,
        q: &[f64],
        i: usize,
        head_dim: usize,
        legal_end: usize,
        budget: usize,
    ) -> Vec<usize> {
        let mut ranked: Vec<(usize, f64)> = (0..self.cluster_count)
            .map(|label| {
                let score: f64 = (0..head_dim)
                    .map(|d| q[i * head_dim + d] * self.centroids[label * head_dim + d])
                    .sum();
                (label, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut candidates: Vec<usize> = Vec::new();
        for (label, _) in ranked {
            if candidates.len() >= budget {
                break;
            }
            candidates.extend((0..legal_end).filter(|&j| self.assignment[j] == label));
        }
        candidates
    }
}

/// A top-k boundary the rounding enclosures do not decide.
///
/// Names the selected key whose enclosure reaches lowest and the unselected key
/// whose enclosure reaches highest. The boundary is decided only when
/// `gap > needed_margin`; a refusal says nothing about which key is really
/// larger, only that the float arithmetic cannot tell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryRefusal {
    /// The selected key with the lowest enclosure lower end.
    pub inside: usize,
    /// The unselected key with the highest enclosure upper end.
    pub outside: usize,
    /// Float score of `inside` minus float score of `outside`.
    pub gap: f64,
    /// The gap that would certify the pair: the sum of their two radii.
    pub needed_margin: f64,
}

/// The top-`budget` ids of `scored`, proven independent of rounding, or the
/// refusal that blocks the proof.
///
/// `scored` holds `(id, score, radius)` with `|score - exact| <= radius`. The set
/// `T` of the `budget` highest scores (ties by id) is certified when
///
/// ```text
/// min_{i in T} (score_i - radius_i)  >  max_{j not in T} (score_j + radius_j)
/// ```
///
/// which makes `T` the top-`budget` set of every score vector inside the
/// enclosures, the exact one included. Comparing only the rank-`budget` and
/// rank-`budget + 1` enclosures is not this rule and is unsound once radii vary:
/// a low-ranked key with a wide radius can reach past both
/// (`the_rule_is_not_the_rank_k_versus_rank_k_plus_one_pair`).
///
/// Any score or radius that is NaN or infinite refuses first, before any
/// ordering: `total_cmp` sorts `+NaN` above every number and `-NaN` below, so an
/// extreme-only comparison steps over it, and an overflowed score has no
/// enclosure to compare.
///
/// Returns ids in descending score order. A `budget` of zero, or one covering
/// every key, is certified trivially: there is no boundary to decide.
pub fn certified_top_k(
    scored: &[(usize, f64, f64)],
    budget: usize,
) -> Result<Vec<usize>, TopKRefusal> {
    if let Some(&(key, _, _)) = scored
        .iter()
        .find(|&&(_, score, radius)| !score.is_finite() || !radius.is_finite())
    {
        return Err(TopKRefusal::NonFinite { key });
    }
    let mut ranked = scored.to_vec();
    // Descending by score, ties broken by id so the result is deterministic
    // rather than sort-stability-dependent.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if budget == 0 || budget >= ranked.len() {
        ranked.truncate(budget);
        return Ok(ranked.into_iter().map(|(j, _, _)| j).collect());
    }

    let (inside, outside) = ranked.split_at(budget);
    let lowest = inside
        .iter()
        .copied()
        .min_by(|a, b| lower_end(a).total_cmp(&lower_end(b)))
        .expect("budget > 0");
    let highest = outside
        .iter()
        .copied()
        .max_by(|a, b| upper_end(a).total_cmp(&upper_end(b)))
        .expect("budget < len");

    if lower_end(&lowest) > upper_end(&highest) {
        Ok(inside.iter().map(|&(j, _, _)| j).collect())
    } else {
        Err(TopKRefusal::Boundary(BoundaryRefusal {
            inside: lowest.0,
            outside: highest.0,
            gap: lowest.1 - highest.1,
            needed_margin: lowest.2 + highest.2,
        }))
    }
}

/// Why [`certified_top_k`] declined to certify.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopKRefusal {
    /// The enclosures of a selected and an unselected key overlap.
    Boundary(BoundaryRefusal),
    /// `key`'s score or radius is NaN or infinite, so it has no enclosure.
    NonFinite { key: usize },
}

/// `score - radius`, rounded to nearest.
///
/// Directed rounding is not needed at either end. Round-to-nearest is monotone,
/// so for finite inputs `fl(a) > fl(b)` implies `a > b` exactly: a certificate
/// `fl(lo_in) > fl(hi_out)` proves `lo_in > hi_out`, and a widening exclusion
/// `fl(hi_j) < fl(lo_in)` proves `hi_j < lo_in`.
fn lower_end(&(_, score, radius): &(usize, f64, f64)) -> f64 {
    score - radius
}

/// `score + radius`, rounded to nearest; see [`lower_end`].
fn upper_end(&(_, score, radius): &(usize, f64, f64)) -> f64 {
    score + radius
}

/// The certified top-`budget` of `scored`, or, on a refusal, a superset of the
/// exact top-`budget`, recorded in `report`.
///
/// On an overlapping boundary the row widens to every key whose enclosure reaches
/// the lowest selected lower end: any key left out has an upper end strictly
/// below the lower end of all `budget` selected keys, so it is outside the exact
/// top-`budget`. On a non-finite score or radius there is no floor to widen to,
/// so the row falls back to every legal key, `0..legal_end`.
fn certified_or_widened(
    scored: &[(usize, f64, f64)],
    budget: usize,
    row: usize,
    legal_end: usize,
    report: &mut SelectionReport,
) -> Vec<usize> {
    let chosen: Vec<usize> = match certified_top_k(scored, budget) {
        Ok(chosen) => return chosen,
        Err(TopKRefusal::NonFinite { key }) => {
            report.dense_rows.push((row, key));
            (0..legal_end).collect()
        }
        Err(TopKRefusal::Boundary(refusal)) => {
            let floor = scored
                .iter()
                .find(|entry| entry.0 == refusal.inside)
                .map(lower_end)
                .expect("the refused key is one of the scored keys");
            report.widened_rows.push((row, refusal));
            scored
                .iter()
                .filter(|entry| upper_end(entry) >= floor)
                .map(|&(j, _, _)| j)
                .collect()
        }
    };
    report.extra_keys += chosen.len().saturating_sub(budget);
    chosen
}

/// Single-linkage clustering of `points` (row-major `[count, dim]`) cut so that
/// `clusters` components remain.
///
/// Returns `(assignment, merge_heights)`. `assignment[i]` is the component label
/// of point `i`, canonicalised so labels appear in ascending order of first
/// occurrence — otherwise the labels would depend on union-find internals and the
/// result would not be comparable across runs. `merge_heights` is every merge
/// distance in ascending order.
///
/// Single-linkage merge heights are exactly the finite H0 deaths of the
/// Vietoris-Rips persistence of the same cloud; `single_linkage_merge_heights_
/// equal_the_h0_persistence_deaths` asserts that against the persistence engine
/// rather than trusting two implementations of one theorem.
///
/// With `normalize`, points are projected to the unit sphere first, which makes
/// the clustering invariant to per-point rescaling. That is the property the
/// routed selector needs and the nearest-neighbour selector lacks.
///
/// ponytail: O(n^2) edge enumeration plus a sort. This is not the binding term —
/// `sparse_attention` takes a dense `[seq, seq]` mask, so the kernel around this
/// call is already Theta(seq^2) in both time and memory, and a neighbour graph
/// here would save nothing. Trigger: revisit when the mask stops being
/// materialised dense (a block or index list instead of `[seq, seq]` bools), at
/// which point the quadratic edge scan becomes the ceiling and a k-NN graph over
/// `prepared` is the upgrade.
pub fn single_linkage_clusters(
    points: &[f64],
    count: usize,
    dim: usize,
    clusters: usize,
    normalize: bool,
) -> (Vec<usize>, Vec<f64>) {
    assert_eq!(points.len(), count * dim, "points must be [count, dim]");

    let prepared: Vec<f64> = if normalize {
        let mut out = vec![0.0; count * dim];
        for t in 0..count {
            let norm = sqrt(
                (0..dim)
                    .map(|d| points[t * dim + d] * points[t * dim + d])
                    .sum::<f64>(),
            );
            // A zero vector has no direction; leave it at the origin rather than
            // inventing one, and let it cluster with whatever is nearest.
            let inverse = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for d in 0..dim {
                out[t * dim + d] = points[t * dim + d] * inverse;
            }
        }
        out
    } else {
        points.to_vec()
    };

    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(count * count / 2);
    for i in 0..count {
        for j in (i + 1)..count {
            let mut sum = 0.0;
            for d in 0..dim {
                let delta = prepared[i * dim + d] - prepared[j * dim + d];
                sum += delta * delta;
            }
            edges.push((sqrt(sum), i, j));
        }
    }
    // Ties broken by index so the merge order, and therefore the cut, is
    // deterministic rather than sort-stability-dependent.
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut parent: Vec<usize> = (0..count).collect();
    let mut merge_heights = Vec::with_capacity(count.saturating_sub(1));
    let mut components = count;
    let target = clusters.clamp(1, count.max(1));

    for (height, i, j) in edges {
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
        if ri == rj {
            continue;
        }
        // Every merge is recorded, including those past the cut: the heights are
        // the full H0 barcode, and the cut is applied separately below.
        merge_heights.push(height);
        if components > target {
            parent[ri] = rj;
            components -= 1;
        }
    }

    // Canonical labels: first occurrence order.
    let mut label_of = vec![usize::MAX; count];
    let mut assignment = vec![0usize; count];
    let mut next = 0usize;
    for (t, slot) in assignment.iter_mut().enumerate() {
        let root = find(&mut parent, t);
        if label_of[root] == usize::MAX {
            label_of[root] = next;
            next += 1;
        }
        *slot = label_of[root];
    }
    (assignment, merge_heights)
}

fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    // Path compression, iterative: recursion here would blow the kernel stack on
    // a long chain, and single-linkage produces exactly those.
    let mut current = x;
    while parent[current] != root {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

fn dot(q: &[f64], k: &[f64], i: usize, j: usize, head_dim: usize) -> f64 {
    let mut sum = 0.0;
    for d in 0..head_dim {
        sum += q[i * head_dim + d] * k[j * head_dim + d];
    }
    sum
}

/// `(j, fl(q_i · k_j), radius)` with `|fl(q_i · k_j) - q_i · k_j| <= radius`.
///
/// The score is `dot`'s, bit for bit. The radius is Higham's a-priori bound for a
/// float inner product of length `n = head_dim` (ASNA 2nd ed., Theorem 3.1):
///
/// ```text
/// |fl(q·k) - q·k| <= gamma_n · Σ|q_d k_d|,   gamma_n = n·u / (1 - n·u),   u = 2^-53
/// ```
///
/// with the absolute value inside the sum; `gamma_n · |q·k|` is not this bound and
/// is far below it under cancellation, which is exactly where it matters. Two
/// terms make the computed radius a bound rather than an estimate: the float
/// `Σ|q_d k_d|` can itself round low by a relative `gamma_n`, and the remaining
/// four roundings cost `4u`, so the radius is scaled by `1 + 2·gamma_n + 4u` and
/// rounded up; and a product that lands subnormal carries absolute rather than
/// relative error, so `n` times the smallest subnormal is added unconditionally.
///
/// The absolute sum rides the same pass over `head_dim` as the score, so the
/// enclosure adds no dot products to the selection cost.
fn enclosed_dot(q: &[f64], k: &[f64], i: usize, j: usize, head_dim: usize) -> (usize, f64, f64) {
    let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
    for d in 0..head_dim {
        let product = q[i * head_dim + d] * k[j * head_dim + d];
        sum += product;
        magnitude += product.abs();
    }
    let n = head_dim as f64;
    let u = f64::EPSILON / 2.0;
    let gamma = nextafter(n * u / (1.0 - n * u), f64::INFINITY);
    let underflow = n * f64::from_bits(1);
    let radius = gamma * magnitude * (1.0 + 2.0 * gamma + 4.0 * u) + underflow;
    (j, sum, nextafter(radius, f64::INFINITY))
}

/// Euclidean distance between query `i` and key `j` in the shared embedding
/// space. The topological selector's only view of the data.
fn key_distance(q: &[f64], k: &[f64], i: usize, j: usize, head_dim: usize) -> f64 {
    let mut sum = 0.0;
    for d in 0..head_dim {
        let delta = q[i * head_dim + d] - k[j * head_dim + d];
        sum += delta * delta;
    }
    sqrt(sum)
}

fn splitmix(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ═══════════════════════════════════════════════════════════════════════════════
// The ablation
// ═══════════════════════════════════════════════════════════════════════════════

/// A selector collapsed to the concrete rule it will run, with the routing pass
/// already performed to decide it.
struct Resolved {
    selector: Selector,
    /// Per-row routed candidates, when the rule is `TopologicalRouted`.
    routed: Option<Vec<Vec<usize>>>,
    /// Query-to-centroid dot products spent while resolving, summed over rows.
    paid: usize,
}

/// Collapse [`Selector::Adaptive`] to the concrete rule it will actually run.
///
/// Everything downstream — mask construction, cost accounting, the ablation — goes
/// through this, so the adaptive selector cannot report one path's cost while
/// running another's. The alignment pass is performed here once and handed on:
/// the routed path reuses it rather than repeating it, and a fallback still pays
/// for it, because the decision could not be made without it.
fn resolve(
    selector: Selector,
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> Resolved {
    let routed = |budget, clusters| {
        let (cluster_count, rows) = route_rows(q, k, seq, head_dim, clusters, budget, causal);
        (cluster_count * seq, rows)
    };
    match selector {
        Selector::TopologicalRouted { budget, clusters } => {
            let (paid, rows) = routed(budget, clusters);
            Resolved {
                selector,
                routed: Some(rows),
                paid,
            }
        }
        Selector::Adaptive { budget, clusters } => {
            let (paid, rows) = routed(budget, clusters);
            if routed_cost_ratio(paid, &rows, seq, causal) < ROUTING_COST_THRESHOLD {
                Resolved {
                    selector: Selector::TopologicalRouted { budget, clusters },
                    routed: Some(rows),
                    paid,
                }
            } else {
                // Dense, not a cheap window.
                //
                // Measured: on unstructured keys a budget-6 sliding window lands at
                // placement +0.014 — indistinguishable from picking keys at random.
                // There is no cheap-and-good option when the keys have no structure,
                // because finding the top-k without computing the scores is exactly
                // what the structure was supposed to make possible.
                //
                // So the fallback preserves correctness rather than cost. Its
                // quality is dense's; its cost is dense plus the alignment pass
                // spent deciding, which is `cluster_count` dot products per row
                // above dense. A caller that would rather trade quality for cost
                // should ask for `Local` explicitly.
                Resolved {
                    selector: Selector::Dense,
                    routed: None,
                    paid,
                }
            }
        }
        other => Resolved {
            selector: other,
            routed: None,
            paid: 0,
        },
    }
}

/// Every row's routed candidates, from one alignment pass over the H0 clusters.
/// Returns `(cluster_count, candidates per row)`.
fn route_rows(
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    clusters: usize,
    budget: usize,
    causal: bool,
) -> (usize, Vec<Vec<usize>>) {
    let routing = Routing::build(k, seq, head_dim, clusters);
    let rows = (0..seq)
        .map(|i| {
            let legal_end = if causal { i + 1 } else { seq };
            routing.candidates(q, i, head_dim, legal_end, budget)
        })
        .collect();
    (routing.cluster_count, rows)
}

/// The routed path's dot products per row as a fraction of dense: the alignment
/// pass (`paid`) plus one score per candidate.
fn routed_cost_ratio(paid: usize, rows: &[Vec<usize>], seq: usize, causal: bool) -> f64 {
    let total = paid + rows.iter().map(Vec::len).sum::<usize>();
    total as f64 / seq as f64 / dense_dot_cost(seq, causal)
}

/// Mean number of `head_dim`-length dot products a selector performs per row.
///
/// Quality without cost is not a result. `attention_mass_recovered` says how good
/// a selector's choice is; this says what the choice cost to make. A selector that
/// recovers 90% of attention mass while examining every key is dense attention
/// with extra steps, and only this function reveals that.
///
/// Dense attention costs `i + 1` dot products at row `i` under causal masking, so
/// compare against `dense_dot_cost`.
///
/// The number is the cost of the path [`select_mask`] runs on the same `q` and
/// `k`, not an estimate of it: `TopologicalRouted` visits clusters in each
/// query's alignment order, so its cost depends on the queries, and the count is
/// taken from that same walk. [`SelectionReport::dot_products`] measures the run
/// independently, and `the_reported_cost_is_the_cost_of_the_path_actually_run`
/// asserts the two are equal.
///
/// Selectors that inspect no scores (`Local`, `Random`) cost nothing to evaluate
/// and report 0; the number is about search, not about the attention arithmetic
/// that follows. `OracleTopK` costs the full dense count by construction — that is
/// precisely why it is a diagnostic and not an implementation.
pub fn selection_dot_cost(
    selector: Selector,
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> f64 {
    let resolved = resolve(selector, q, k, seq, head_dim, causal);
    let path: usize = match resolved.selector {
        Selector::Adaptive { .. } => unreachable!("Adaptive is resolved before costing"),
        Selector::Local { .. } | Selector::Random { .. } => 0,
        Selector::Dense | Selector::OracleTopK { .. } | Selector::Topological { .. } => {
            (0..seq).map(|i| if causal { i + 1 } else { seq }).sum()
        }
        // A row pays for the clusters its query aligns with, so the cost is per
        // query: the candidates of the selector's own ranking, not a proxy for it.
        Selector::TopologicalRouted { .. } => resolved
            .routed
            .as_ref()
            .expect("resolve routes this selector")
            .iter()
            .map(Vec::len)
            .sum(),
    };
    (resolved.paid + path) as f64 / seq as f64
}

/// Mean dot products per row that dense attention performs. The denominator for
/// any sparsity claim.
pub fn dense_dot_cost(seq: usize, causal: bool) -> f64 {
    let total: usize = (0..seq).map(|i| if causal { i + 1 } else { seq }).sum();
    total as f64 / seq as f64
}

/// Fraction of true dense attention mass that a selector's chosen keys carry,
/// averaged over rows.
///
/// For each row, computes the full softmax over the legal keys and sums the
/// weights of the keys the selector picked. `Selector::OracleTopK` maximises this
/// by construction, so it is the upper bound at a given budget; `Random` is the
/// floor. Where a selector lands between them is the measurement that decides
/// whether the mechanism contributes anything:
///
///   - at random → the mechanism contributes nothing, and any downstream gain is
///     sparsity acting as a regulariser
///   - at oracle → the selector's summary is a good proxy for attention mass,
///     which is the actual scientific claim
///   - between → report the fraction, not the headline
pub fn attention_mass_recovered(
    selector: Selector,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> f64 {
    let mask = select_mask(selector, q, k, v, seq, head_dim, causal);
    let scale = 1.0 / sqrt(head_dim as f64);
    let mut total = 0.0;

    for i in 0..seq {
        let legal_end = if causal { i + 1 } else { seq };

        let mut max_logit = f64::NEG_INFINITY;
        let mut logits = vec![0.0f64; legal_end];
        for (j, logit) in logits.iter_mut().enumerate() {
            *logit = dot(q, k, i, j, head_dim) * scale;
            if *logit > max_logit {
                max_logit = *logit;
            }
        }

        let denominator: f64 = logits.iter().map(|&l| exp(l - max_logit)).sum();
        let recovered: f64 = (0..legal_end)
            .filter(|&j| mask[i * seq + j])
            .map(|j| exp(logits[j] - max_logit) / denominator)
            .sum();
        total += recovered;
    }
    total / seq as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `r = M · 2^e` with `M` the integer significand, for a positive normal `r`.
    fn significand_and_exponent(r: f64) -> (u128, i32) {
        assert!(r > 0.0 && r.is_normal(), "{r} is not a positive normal");
        let bits = r.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32 - 1075;
        (
            ((bits & ((1u64 << 52) - 1)) | (1u64 << 52)) as u128,
            exponent,
        )
    }

    #[test]
    fn the_radius_dominates_highams_bound_on_the_exact_magnitude() {
        // q = 1 x 8, k = [1, u, u, u, u, u, u, u], u = 2^-53. Every `1 + u` ties
        // to even and drops the u, so the float magnitude is 1 while the exact
        // magnitude is S = 1 + 7u: the float sum under-reads it by 7u. The radius
        // must still cover gamma_8 * S, which in exact rationals is
        //
        //   gamma_8 * S = 8 / (2^53 - 8) * (2^53 + 7) / 2^53.
        //
        // With r = M * 2^e, `r >= gamma_8 * S` is, after dividing by 2^53,
        //   M * (2^53 - 8) >= 8 * (2^53 + 7) * 2^(-e - 53),
        // both sides below 2^107 and compared in u128 with no rounding.
        let u = f64::EPSILON / 2.0;
        let q = [1.0; 8];
        let mut k = [u; 8];
        k[0] = 1.0;
        let (_, score, radius) = enclosed_dot(&q, &k, 0, 0, 8);
        assert_eq!(score, 1.0, "premise: the float sum drops all seven u");

        let (m, e) = significand_and_exponent(radius);
        let shift = u32::try_from(-e - 53).expect("radius is far below 1");
        let lhs = m * ((1u128 << 53) - 8);
        let rhs = (8 * ((1u128 << 53) + 7)) << shift;
        assert!(
            lhs >= rhs,
            "radius {radius:e} is below gamma_8 times the exact magnitude"
        );
    }

    #[test]
    fn the_enclosure_contains_the_exact_score_when_products_underflow() {
        // Each product is (1 - 2^-10) * 2^-1075, below half the smallest
        // subnormal, so it rounds to 0 and the float score is 0. The relative
        // error model does not hold there: the magnitude is 0 too, so gamma
        // contributes nothing, and only the underflow term can cover the exact
        // score 3 * (1 - 2^-10) * 2^-1075 = 3069 * 2^-1085.
        //
        // With r = R * 2^-1074, `r >= 3069 * 2^-1085` is `R * 2^11 >= 3069`.
        let q = [2f64.powi(-600); 3];
        let k = [(1.0 - 2f64.powi(-10)) * 2f64.powi(-475); 3];
        let (_, score, radius) = enclosed_dot(&q, &k, 0, 0, 3);
        assert_eq!(score, 0.0, "premise: every product underflows to zero");

        let in_subnormals = radius * 2f64.powi(1000) * 2f64.powi(74);
        assert!(
            in_subnormals * 2048.0 >= 3069.0,
            "radius of {in_subnormals} subnormals misses an exact score of 3069/2048"
        );
    }
}
