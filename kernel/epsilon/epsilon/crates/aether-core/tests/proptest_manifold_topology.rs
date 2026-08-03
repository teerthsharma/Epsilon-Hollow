// AETHER Core — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Invariant tests for `SparseAttentionGraph`'s topological summary.
//!
//! # Why this file exists
//!
//! `SparseAttentionGraph::compute_betti_0` and `estimate_betti_1` produce a
//! plausible number for every input. A wrong Betti number does not panic, does
//! not produce a NaN, and does not fail a smoke test — it produces a slightly
//! different integer, downstream code accepts it, and nothing surfaces. The
//! only way to catch that class of defect is to assert the properties a correct
//! implementation must satisfy and a broken one cannot.
//!
//! `ml_engine/stratum.rs` in Seal OS already declines to use these functions,
//! citing the 64-bit adjacency bitmask and the bounded DFS stack. That comment
//! is a workaround at one call site. These tests establish what is actually
//! true, for every caller.
//!
//! # The reference
//!
//! Each property is checked against an independent, deliberately slow and
//! obviously-correct union-find over the raw predicate
//! `p_i.distance(p_j) < epsilon`. That predicate — strict `<`, see
//! `ManifoldPoint::is_neighbor` — is the definition of the Vietoris–Rips
//! 1-skeleton at scale `epsilon`, so the reference is the specification.
//!
//! # Derived constants
//!
//! The `√3·r` Rips death time for a circle does not apply here: this structure
//! stores only a 1-skeleton and fills no 2-simplices, so cycles are never
//! killed by triangles. The constant is re-derived for this filtration in
//! [`circle_cycle_rank_is_one`].

use aether_core::manifold::{ManifoldPoint, SparseAttentionGraph};
use proptest::prelude::*;

// ── Independent reference implementation ────────────────────────────────────

/// Connected components of the ε-neighbourhood graph, by union-find over every
/// pair. O(n²) and intentionally so: this is the specification, not the
/// optimisation.
fn reference_betti_0<const D: usize>(pts: &[ManifoldPoint<D>], eps: f64) -> u32 {
    let n = pts.len();
    if n == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if pts[i].distance(&pts[j]) < eps {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count() as u32
}

/// Edge count of the ε-neighbourhood graph, counted once per unordered pair.
fn reference_edges<const D: usize>(pts: &[ManifoldPoint<D>], eps: f64) -> usize {
    let n = pts.len();
    let mut e = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if pts[i].distance(&pts[j]) < eps {
                e += 1;
            }
        }
    }
    e
}

/// Cycle rank of the 1-skeleton: `β₁ = E − V + β₀`. Exact for a graph — this is
/// the Euler characteristic identity, not an approximation, because a graph has
/// no 2-cells to kill cycles.
fn reference_cycle_rank<const D: usize>(pts: &[ManifoldPoint<D>], eps: f64) -> u32 {
    let e = reference_edges(pts, eps) as i64;
    let v = pts.len() as i64;
    let b0 = reference_betti_0(pts, eps) as i64;
    (e - v + b0).max(0) as u32
}

fn build<const D: usize>(pts: &[ManifoldPoint<D>], eps: f64) -> SparseAttentionGraph<D> {
    let mut g = SparseAttentionGraph::<D>::new(eps);
    for p in pts {
        g.add_point(*p);
    }
    g
}

// ── Strategies ──────────────────────────────────────────────────────────────

/// Point clouds spanning the 64-point adjacency-bitmask boundary. The range is
/// chosen deliberately: `MAX_POINTS` is 256 but adjacency is a single `u64`, so
/// any defect tied to that mismatch lives at n > 64 and is invisible below it.
fn cloud_2d(max_n: usize) -> impl Strategy<Value = Vec<ManifoldPoint<2>>> {
    prop::collection::vec(
        (-10.0f64..10.0, -10.0f64..10.0).prop_map(|(x, y)| ManifoldPoint::<2>::new([x, y])),
        2..max_n,
    )
}

// ── 1. Adjacency fidelity ───────────────────────────────────────────────────
// The sharpest available test: the stored adjacency must equal the predicate it
// claims to store. Everything downstream is a function of this, so a failure
// here explains every other failure in the file.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn adjacency_matches_distance_predicate(pts in cloud_2d(120), eps in 0.5f64..8.0) {
        let g = build(&pts, eps);
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                if i == j { continue; }
                let truth = pts[i].distance(&pts[j]) < eps;
                prop_assert_eq!(
                    g.are_neighbors(i, j), truth,
                    "are_neighbors({}, {}) disagrees with distance predicate at n={}",
                    i, j, pts.len()
                );
            }
        }
    }

    /// Adjacency of an undirected graph is symmetric. A stored asymmetry means
    /// the reverse edge was written to the wrong slot.
    #[test]
    fn adjacency_is_symmetric(pts in cloud_2d(120), eps in 0.5f64..8.0) {
        let g = build(&pts, eps);
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                prop_assert_eq!(
                    g.are_neighbors(i, j), g.are_neighbors(j, i),
                    "adjacency asymmetric at ({}, {}) with n={}", i, j, pts.len()
                );
            }
        }
    }
}

// ── 2. β₀ parity against the reference ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn betti_0_matches_reference(pts in cloud_2d(120), eps in 0.5f64..8.0) {
        let g = build(&pts, eps);
        prop_assert_eq!(
            g.compute_betti_0(), reference_betti_0(&pts, eps),
            "β₀ disagrees with union-find reference at n={}", pts.len()
        );
    }

    #[test]
    fn cycle_rank_matches_reference(pts in cloud_2d(120), eps in 0.5f64..8.0) {
        let g = build(&pts, eps);
        prop_assert_eq!(
            g.estimate_betti_1(), reference_cycle_rank(&pts, eps),
            "cycle rank disagrees with E−V+β₀ reference at n={}", pts.len()
        );
    }
}

// ── 3. Permutation invariance ───────────────────────────────────────────────
// A Betti number is a function of the point *set*. Insertion order is not part
// of the input.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn betti_0_is_permutation_invariant(
        pts in cloud_2d(120),
        eps in 0.5f64..8.0,
        rot in 1usize..64,
    ) {
        let mut shuffled = pts.clone();
        shuffled.rotate_left(rot % pts.len().max(1));

        prop_assert_eq!(
            build(&pts, eps).compute_betti_0(),
            build(&shuffled, eps).compute_betti_0(),
            "β₀ changed under reordering at n={}", pts.len()
        );
    }
}

// ── 4. Scale equivariance ───────────────────────────────────────────────────
// Scaling every coordinate and ε by the same c > 0 rescales every distance by
// exactly c, so the predicate `d < ε` is unchanged pointwise and the graph is
// identical. A failure here means an absolute threshold is hiding somewhere.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn betti_numbers_are_scale_equivariant(
        pts in cloud_2d(80),
        eps in 0.5f64..8.0,
        c in 0.25f64..4.0,
    ) {
        let scaled: Vec<_> = pts.iter()
            .map(|p| ManifoldPoint::<2>::new([p.coords[0] * c, p.coords[1] * c]))
            .collect();

        let base = build(&pts, eps);
        let up = build(&scaled, eps * c);

        prop_assert_eq!(base.compute_betti_0(), up.compute_betti_0(), "β₀ not scale equivariant");
        prop_assert_eq!(base.estimate_betti_1(), up.estimate_betti_1(), "β₁ not scale equivariant");
    }

    /// Translation preserves every pairwise distance exactly.
    #[test]
    fn betti_numbers_are_translation_invariant(
        pts in cloud_2d(80),
        eps in 0.5f64..8.0,
        dx in -50.0f64..50.0,
        dy in -50.0f64..50.0,
    ) {
        let moved: Vec<_> = pts.iter()
            .map(|p| ManifoldPoint::<2>::new([p.coords[0] + dx, p.coords[1] + dy]))
            .collect();

        let base = build(&pts, eps);
        let shifted = build(&moved, eps);

        prop_assert_eq!(base.compute_betti_0(), shifted.compute_betti_0(), "β₀ not translation invariant");
        prop_assert_eq!(base.estimate_betti_1(), shifted.estimate_betti_1(), "β₁ not translation invariant");
    }
}

// ── 5. Known-manifold ground truth ──────────────────────────────────────────

/// `k` tight clusters placed far enough apart that no cross-cluster pair is
/// within ε. β₀ must be exactly `k`.
#[test]
fn separated_clusters_give_exact_component_count() {
    for k in 2..=6usize {
        for per in [3usize, 10, 20] {
            let eps = 1.0;
            let mut pts = Vec::new();
            for c in 0..k {
                // Cluster centres 100 apart; members within 0.1 of the centre.
                let cx = c as f64 * 100.0;
                for m in 0..per {
                    let t = m as f64 / per as f64 * core::f64::consts::TAU;
                    pts.push(ManifoldPoint::<2>::new([cx + 0.1 * t.cos(), 0.1 * t.sin()]));
                }
            }
            let g = build(&pts, eps);
            assert_eq!(
                g.compute_betti_0(),
                k as u32,
                "expected {k} components for {k} clusters of {per} points (n={})",
                pts.len()
            );
        }
    }
}

/// `n` points evenly spaced on a circle of radius `r`.
///
/// Consecutive spacing is `2r·sin(π/n)`; next-nearest spacing is `2r·sin(2π/n)`.
/// Choosing ε strictly between them makes the graph exactly an `n`-cycle, so
/// `V = n`, `E = n`, `β₀ = 1` and the cycle rank is `E − V + β₀ = 1`.
///
/// This is the constant derived for *this* filtration. The familiar `√3·r`
/// death time is a Vietoris–Rips result that depends on 2-simplices filling in
/// and killing the cycle; this structure never fills one, so `√3·r` would be
/// the wrong assertion here.
#[test]
fn circle_cycle_rank_is_one() {
    let r = 5.0f64;
    for n in [8usize, 16, 32, 60, 100] {
        let pts: Vec<_> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64 * core::f64::consts::TAU;
                ManifoldPoint::<2>::new([r * t.cos(), r * t.sin()])
            })
            .collect();

        let near = 2.0 * r * (core::f64::consts::PI / n as f64).sin();
        let next = 2.0 * r * (2.0 * core::f64::consts::PI / n as f64).sin();
        let eps = (near + next) / 2.0;

        // The reference confirms the geometry is what the derivation says it is,
        // independently of the structure under test.
        assert_eq!(
            reference_betti_0(&pts, eps),
            1,
            "circle should be connected at n={n}"
        );
        assert_eq!(
            reference_cycle_rank(&pts, eps),
            1,
            "circle should have one cycle at n={n}"
        );

        let g = build(&pts, eps);
        assert_eq!(
            g.compute_betti_0(),
            1,
            "β₀ should be 1 on a circle at n={n}"
        );
        assert_eq!(
            g.estimate_betti_1(),
            1,
            "cycle rank should be 1 on a circle at n={n}"
        );
    }
}

/// Negative control. A single tight blob has one component and no cycle
/// structure worth reporting — a pipeline that finds topology in a blob finds
/// it everywhere, which is worse than finding nothing.
#[test]
fn tight_blob_is_one_component() {
    for n in [10usize, 50, 70, 120] {
        let pts: Vec<_> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.6180339887;
                ManifoldPoint::<2>::new([0.01 * (t * 7.0).sin(), 0.01 * (t * 11.0).cos()])
            })
            .collect();
        let g = build(&pts, 1.0);
        assert_eq!(
            g.compute_betti_0(),
            1,
            "a tight blob is one component at n={n}"
        );
        assert_eq!(
            g.compute_betti_0(),
            reference_betti_0(&pts, 1.0),
            "blob β₀ disagrees with reference at n={n}"
        );
    }
}

// ── 6. The boundary, localised ──────────────────────────────────────────────

/// Two points at index 0 and index 64+ that are geometrically far apart must
/// not be reported as neighbours. This isolates the bitmask-aliasing question
/// from every other moving part, so a failure here is unambiguous.
#[test]
fn far_apart_points_across_the_64_boundary_are_not_neighbours() {
    let mut pts: Vec<ManifoldPoint<2>> = Vec::new();
    // 64 points in a tight cluster at the origin.
    for i in 0..64 {
        pts.push(ManifoldPoint::new([0.001 * i as f64, 0.0]));
    }
    // One point very far away — adjacent to nothing.
    pts.push(ManifoldPoint::new([1000.0, 1000.0]));

    let g = build(&pts, 1.0);
    let far = pts.len() - 1;

    for i in 0..far {
        assert!(
            !g.are_neighbors(i, far),
            "point {i} reported adjacent to the far point at index {far}"
        );
        assert!(
            !g.are_neighbors(far, i),
            "far point {far} reported adjacent to point {i}"
        );
    }
    assert_eq!(
        g.compute_betti_0(),
        2,
        "one cluster plus one isolated point is 2 components"
    );
}
