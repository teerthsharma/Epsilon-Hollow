//! RED tests for `topology::compute_betti_0`.
//!
//! The module doc says bytes are "a 1D point cloud on R" and beta_0 is its
//! component count at scale epsilon. For a 1-D Rips complex on a value multiset
//! that is (number of gaps exceeding epsilon) + 1, computed on the SORTED
//! values — which makes it permutation invariant, as a homology invariant must
//! be. These tests pin that definition.
use aether_core::compute_betti_0;

/// Independent reference: sort values, union consecutive ones within threshold,
/// count components. Deliberately written a different way from the source.
fn reference_betti_0(data: &[u8], threshold: i16) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let mut v: Vec<i16> = data.iter().map(|&b| b as i16).collect();
    v.sort_unstable();
    let mut components = 1u32;
    for w in v.windows(2) {
        if (w[1] - w[0]).abs() > threshold {
            components += 1;
        }
    }
    components
}

const T: i16 = 15; // CLUSTER_THRESHOLD

#[test]
fn uniform_data_is_one_component() {
    assert_eq!(
        compute_betti_0(&[0x90u8; 64]),
        1,
        "64 identical bytes are one cluster"
    );
}

#[test]
fn beta_0_is_monotone_under_adding_a_duplicate_point() {
    // beta_0 of {5} is 1. Adding a second point at the SAME value cannot
    // increase the component count, and certainly cannot decrease it below 1.
    let one = compute_betti_0(&[5u8]);
    let two = compute_betti_0(&[5u8, 5u8]);
    assert_eq!(one, 1, "a single point is one component");
    assert_eq!(
        two, 1,
        "two coincident points are still one component, got {two}"
    );
}

#[test]
fn separated_values_give_one_component_each() {
    assert_eq!(
        compute_betti_0(&[0u8, 100, 200]),
        3,
        "three well-separated values"
    );
}

#[test]
fn matches_independent_reference_on_many_inputs() {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for trial in 0..300 {
        let n = 1 + (next() % 40) as usize;
        let data: Vec<u8> = (0..n).map(|_| (next() % 256) as u8).collect();
        let got = compute_betti_0(&data);
        let want = reference_betti_0(&data, T);
        assert_eq!(
            got, want,
            "trial {trial}: data={data:?} got={got} want={want}"
        );
    }
}

#[test]
fn beta_0_is_permutation_invariant() {
    // A homology invariant of a point CLOUD cannot depend on input order.
    let mut s: u64 = 0xDEADBEEFCAFEBABE;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for trial in 0..200 {
        let n = 2 + (next() % 30) as usize;
        let data: Vec<u8> = (0..n).map(|_| (next() % 256) as u8).collect();
        let mut shuffled = data.clone();
        for i in (1..shuffled.len()).rev() {
            let j = (next() as usize) % (i + 1);
            shuffled.swap(i, j);
        }
        assert_eq!(
            compute_betti_0(&data),
            compute_betti_0(&shuffled),
            "trial {trial}: beta_0 changed under permutation. data={data:?} shuffled={shuffled:?}"
        );
    }
}

#[test]
fn empty_input_is_zero_components() {
    assert_eq!(compute_betti_0(&[]), 0);
}
