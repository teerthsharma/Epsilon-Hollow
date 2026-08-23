//! `compute_betti_1` is named for a homology rank and is not one.
//!
//! Two properties hold for the first Betti number of a point cloud and for
//! nothing that slides a window along a sequence:
//!
//! * it does not grow when the same shape is sampled more densely;
//! * it does not depend on the order the samples arrive in, because a point
//!   cloud has no intrinsic order.
//!
//! The crate already asserts the second for beta_0, in
//! `proptest_manifold_topology.rs::betti_0_is_permutation_invariant`. This file
//! asserts both for whatever is called beta_1.
//!
//! The third test is the mathematical statement the repair rests on, checked
//! against the crate's own persistence machinery rather than against the
//! function under repair.

use aether_core::persistence::{persistent_homology_from_distances, PersistenceConfig};
use aether_core::topology::{
    betti_1, compute_shape, oscillation_count, verify_shape, VerifyResult,
};

/// `[0, 60, 120, 0]` repeated. Every repetition samples the same three-value
/// pattern again; the underlying shape does not change.
fn repeated(reps: usize) -> Vec<u8> {
    [0u8, 60, 120, 0]
        .iter()
        .cycle()
        .take(4 * reps)
        .copied()
        .collect()
}

#[test]
fn a_homology_rank_does_not_grow_with_sample_count() {
    let base = betti_1(&repeated(1));
    let mut observed = Vec::new();
    for reps in [1usize, 2, 4, 8, 16, 32] {
        let v = repeated(reps);
        observed.push((v.len(), betti_1(&v)));
    }
    println!("betti_1 against sample count: {observed:?}");
    for (len, b) in observed {
        assert_eq!(
            b, base,
            "beta_1 changed to {b} at length {len} while the sampled shape stayed \
             the same; a homology rank does not scale with sample count"
        );
    }
}

#[test]
fn a_homology_rank_is_permutation_invariant() {
    // A point cloud has no order. Reversal and a fixed deterministic shuffle
    // are both relabelings, so both are provable no-ops against homology.
    let cases: [Vec<u8>; 3] = [
        repeated(4),
        vec![7, 200, 13, 200, 7, 99, 4, 250, 4, 99],
        vec![0, 1, 2, 3, 250, 251, 252, 128, 129, 130, 5, 5, 5],
    ];
    for data in cases {
        let forward = betti_1(&data);

        let mut reversed = data.clone();
        reversed.reverse();
        assert_eq!(
            betti_1(&reversed),
            forward,
            "beta_1 changed under reversal of {data:?}"
        );

        // A deterministic stride permutation: same multiset, different order.
        let mut strided: Vec<u8> = data.iter().step_by(3).copied().collect();
        strided.extend(
            data.iter()
                .enumerate()
                .filter(|(i, _)| i % 3 != 0)
                .map(|(_, &b)| b),
        );
        assert_eq!(strided.len(), data.len());
        assert_eq!(
            betti_1(&strided),
            forward,
            "beta_1 changed under a stride permutation of {data:?}"
        );
    }
}

/// The claim the repair rests on: a point cloud **on a line** has no 1-cycles,
/// at any scale.
///
/// Proof. Order the distinct values `x_1 < ... < x_n`. In the Rips graph at
/// scale `t`, the neighbours of `x_1` are exactly the values in
/// `(x_1, x_1 + t]`, and any two of those differ by at most `t`, so they are
/// pairwise adjacent. Hence `N[x_1]` is a clique, `x_1` is a simplicial vertex,
/// its closed star is a full simplex, and deleting it is an elementary collapse.
/// Induct. Every component collapses to a point, so `H_1 = 0`.
///
/// This test does not take that on trust. It runs the crate's own persistent
/// homology over the byte values and requires zero H1 bars — an independent
/// check, since a disagreement would indict either the proof above or
/// `persistence.rs`, and either is worth knowing.
#[test]
fn one_dimensional_data_has_no_one_cycles_at_any_scale() {
    let corpora: [Vec<u8>; 5] = [
        repeated(6),
        vec![7, 200, 13, 200, 7, 99, 4, 250, 4, 99],
        vec![0, 1, 2, 3, 250, 251, 252, 128, 129, 130],
        (0u8..40).map(|i| i.wrapping_mul(7)).collect(),
        vec![42; 16],
    ];
    let mut total_h1 = 0usize;
    for data in &corpora {
        // Distinct values only: duplicates are distance-zero copies and add
        // nothing but n^2 work.
        let mut vals: Vec<f64> = data.iter().map(|&b| f64::from(b)).collect();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        let n = vals.len();
        let mut dist = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                dist[i * n + j] = (vals[i] - vals[j]).abs();
            }
        }
        let cfg = PersistenceConfig {
            max_radius: 300.0,
            ..Default::default()
        };
        let diagram = persistent_homology_from_distances(&dist, n, cfg)
            .expect("one-dimensional distances are a valid metric");
        for radius in [0.5, 1.0, 3.0, 8.0, 20.0, 64.0, 200.0] {
            let b = diagram.betti_at(radius);
            total_h1 += b.beta_1 as usize;
            // Tie the measurement to the function it is supposed to justify.
            // Without this line the test verifies the MATHEMATICS and says
            // nothing about `betti_1`: it never mentions it, so the collapse
            // argument would be decoration and any constant would pass.
            assert_eq!(
                betti_1(data),
                b.beta_1,
                "betti_1 returned {} where persistent homology measured {} at                  radius {radius} on {data:?}",
                betti_1(data),
                b.beta_1
            );
            assert_eq!(
                b.beta_1, 0,
                "persistent homology found {} one-cycles at radius {radius} in \
                 one-dimensional data {data:?}, which the collapse argument says \
                 is impossible",
                b.beta_1
            );
        }
    }
    println!("checked 5 corpora at 7 radii each; total H1 bars found: {total_h1}");
}

/// The statistic itself is kept, under its own name, and still does what it did.
/// Renaming it is the repair; deleting it would throw away a usable signal.
#[test]
fn the_oscillation_statistic_still_counts_windows() {
    // [0,60,120,0] repeated: every 4-window starting on a 0 closes, and the
    // count rises with length. That is correct behaviour for a window count and
    // disqualifying for a Betti number.
    let a = oscillation_count(&repeated(1));
    let b = oscillation_count(&repeated(8));
    assert!(
        b > a,
        "the oscillation statistic should grow with sample count ({a} then {b}); \
         if it no longer does, this test is guarding the wrong function"
    );
    assert_eq!(oscillation_count(&[]), 0);
    assert_eq!(oscillation_count(&[1, 2, 3]), 0, "needs a full 4-window");
}

/// The `MAX_OSCILLATION` branch of [`verify_shape`] must be reachable.
///
/// `topology.rs` justifies keeping the statistic in that branch by arguing that
/// substituting the true beta_1 would make the branch unreachable. That argument
/// is worthless unless the branch is reachable *now*, and before this test no
/// assertion anywhere in the repository exercised `ExcessiveLoops`. This is the
/// required-misfire control on it.
///
/// The window is narrow and worth stating. Rejection needs
/// `density = betti_0 / len` in `[0.1, 0.6]` and `oscillation > 10`
/// simultaneously. A period-3 pattern of three well-separated values gives
/// `betti_0 = 3` and `oscillation = len - 3`, so `len` must satisfy
/// `3 / len >= 0.1` and `len - 3 > 10`, that is `14 <= len <= 30`.
#[test]
fn verify_shape_rejects_excessive_oscillation() {
    let data: Vec<u8> = [0u8, 96, 192].iter().cycle().take(21).copied().collect();
    let shape = compute_shape(&data);
    println!(
        "len 21: betti_0 {} density {:.4} oscillation {}",
        shape.betti_0, shape.density, shape.oscillation
    );
    assert!(
        matches!(verify_shape(&data), VerifyResult::ExcessiveLoops { .. }),
        "expected ExcessiveLoops, got {:?} for shape {shape:?}",
        verify_shape(&data)
    );
}

/// An exact closed form rather than an inequality, which is what this loop's
/// own rule R2 asks for wherever one exists.
///
/// For a period-3 repetition of three values that are pairwise further apart
/// than the tolerance, **every** 4-window qualifies: `w[0]` and `w[3]` are the
/// same value, and `w[1]` differs from it. There are `len - 3` windows, so
///
/// ```text
///     oscillation_count = len - 3        for len >= 3
/// ```
///
/// This pins the value at every length. The weaker "it grows with length" test
/// above cannot distinguish `len - 3` from `len / 4`.
#[test]
fn oscillation_count_of_a_period_three_pattern_is_exactly_len_minus_three() {
    for len in 4usize..=64 {
        let data: Vec<u8> = [0u8, 96, 192].iter().cycle().take(len).copied().collect();
        assert_eq!(
            oscillation_count(&data) as usize,
            len - 3,
            "period-3 pattern of length {len} should give exactly {} windows",
            len - 3
        );
    }
}
