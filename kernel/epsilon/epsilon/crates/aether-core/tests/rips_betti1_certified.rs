//! `SparseAttentionGraph::rips_betti_1` decides every edge `d < ε` on a
//! rounded distance. When a rounded distance lands on `ε`, the float edge set
//! is not the exact one and the returned β₁ is not the β₁ of the input.
//!
//! The unit square at `ε = fl(√2)`: the diagonals are exactly `√2`, which is
//! below `fl(√2) = 1.4142135623730951`, so in exact arithmetic both diagonals
//! are edges, the four triangles fill, and Rips β₁ = 0. The rounded diagonal
//! is `fl(√2) = ε`, fails `d < ε`, and the float complex is a bare 4-cycle
//! with β₁ = 1. The answer was chosen by the rounding of one `sqrt`.
use aether_core::manifold::{ManifoldPoint, SparseAttentionGraph};
use aether_core::persistence::PersistenceError;

fn square(eps: f64) -> SparseAttentionGraph<2> {
    let mut g = SparseAttentionGraph::new(eps);
    for p in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] {
        g.add_point(ManifoldPoint::new(p));
    }
    g
}

#[test]
fn a_diagonal_rounded_onto_epsilon_is_refused_not_counted() {
    let got = square(2.0f64.sqrt()).rips_betti_1();
    assert!(
        got.is_err(),
        "exact Rips β₁ of the unit square at ε = fl(√2) is 0; rounding answered {got:?}"
    );
    // The refusal names the boundary: the first diagonal, points 0 and 2.
    assert!(
        matches!(got, Err(PersistenceError::UndecidedEdge { i: 0, j: 2, .. })),
        "{got:?}"
    );
}

#[test]
fn edges_far_from_epsilon_still_count() {
    // Diagonal 1.414 is outside ε = 1.2 and inside ε = 1.5, by far more than
    // any rounding of a 2-D distance: both answers are decided.
    assert_eq!(square(1.2).rips_betti_1(), Ok(1));
    assert_eq!(square(1.5).rips_betti_1(), Ok(0));
}
