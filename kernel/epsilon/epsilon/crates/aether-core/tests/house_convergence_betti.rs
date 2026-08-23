//! `ResidualAnalyzer::compute_betti` is named after homology but computes a
//! sign-change count and an oscillation count over the residual sequence, in
//! order. These tests pin what it ACTUALLY computes, and demonstrate the two
//! properties it does not have.
use aether_core::ml::convergence::ResidualAnalyzer;

fn analyzer(vals: &[f64]) -> ResidualAnalyzer<1> {
    let mut a = ResidualAnalyzer::<1>::new(1e-9);
    a.set_residuals(vals);
    a
}

#[test]
fn it_is_not_permutation_invariant() {
    // A homology invariant of a point set cannot depend on input order.
    // This quantity does, because it walks the sequence.
    let ordered = analyzer(&[1.0, 1.0, 1.0, -1.0, -1.0, -1.0]).compute_betti();
    let shuffled = analyzer(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).compute_betti();
    println!(
        "same multiset, two orders: b0 {} vs {}",
        ordered.beta_0, shuffled.beta_0
    );
    assert_ne!(
        ordered.beta_0, shuffled.beta_0,
        "if these agreed the quantity might be order-free; it is not, and the \
         doc must not call it a Betti number"
    );
}

#[test]
fn it_is_a_sign_change_count_with_a_halving() {
    // Pinned behaviour: beta_0 = ceil((sign_changes + 1 + 1) / 2).
    for (vals, want) in [
        (vec![1.0, 1.0, 1.0], 1u32),     // 0 changes -> ceil(2/2)=1
        (vec![1.0, -1.0], 2),            // 1 change  -> ceil(3/2)=2
        (vec![1.0, -1.0, 1.0], 2),       // 2 changes -> ceil(4/2)=2
        (vec![1.0, -1.0, 1.0, -1.0], 3), // 3 changes -> ceil(5/2)=3
    ] {
        let got = analyzer(&vals).compute_betti().beta_0;
        println!("{vals:?} -> beta_0 {got} (want {want})");
        assert_eq!(
            got, want,
            "pinned sign-change behaviour changed for {vals:?}"
        );
    }
}

#[test]
fn empty_input_returns_perfect_convergence_not_an_empty_space() {
    // beta_0 of the EMPTY space is 0 — there are no components. This returns
    // BettiNumbers::default(), which is (1, 0), documented elsewhere as
    // "perfect convergence: single component, no loops".
    //
    // That is a defensible convergence-detector choice: no residuals means
    // nothing left to converge. It is NOT beta_0. Pinned here so the choice is
    // explicit rather than mistaken for topology.
    let b = ResidualAnalyzer::<1>::new(1e-9).compute_betti();
    assert_eq!(
        (b.beta_0, b.beta_1),
        (1, 0),
        "empty input returns the perfect-convergence default, not the empty space"
    );
}

#[test]
fn it_is_not_scale_invariant_in_the_way_a_betti_number_would_be() {
    // Scaling every residual by a positive constant preserves all signs, so
    // this quantity is unchanged — that much matches. But scaling by a NEGATIVE
    // constant also preserves the sign-change PATTERN while inverting every
    // sign, and a genuine beta_0 of the residual set would be unaffected too.
    // The point of this test is the recorded fact that sign is the only input:
    // magnitudes are discarded entirely.
    let small = analyzer(&[1e-12, -1e-12, 1e-12]).compute_betti();
    let large = analyzer(&[1e9, -1e9, 1e9]).compute_betti();
    println!(
        "magnitudes 1e-12 vs 1e9 -> b0 {} vs {}",
        small.beta_0, large.beta_0
    );
    assert_eq!(
        small.beta_0, large.beta_0,
        "only signs are read; if this ever differs the implementation changed"
    );
}
