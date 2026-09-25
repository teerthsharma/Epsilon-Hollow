// Epsilon — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Every beta_0 this crate decides is certified or refused.
//!
//! `SparseGraph::compute_betti_0` is union-find at one strict threshold, so a
//! pair of points one ulp either side of epsilon gives 1 or 2 with nothing in
//! the input to choose between them. The certified count answers only when no
//! minimum-spanning-tree edge lies within a factor `sqrt(BETA0_RATIO)` of
//! epsilon, and otherwise names the ambiguous pair.

use epsilon::manifold::Beta0;
use epsilon::{EpsilonPoint, SparseGraph};

fn pair(x: f64) -> SparseGraph<3> {
    let mut g = SparseGraph::new(1.0);
    g.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
    g.add_point(EpsilonPoint::new([x, 0.0, 0.0]));
    g
}

#[test]
fn graph_beta0_one_ulp_from_epsilon_is_refused() {
    let below = pair(0.999_999_999_999_999_9);
    let at = pair(1.0);
    // The uncertified count flips across one ulp.
    assert_eq!(below.compute_betti_0(), 1);
    assert_eq!(at.compute_betti_0(), 2);

    assert_eq!(
        below.certified_betti_0(),
        Beta0::Refused {
            i: 0,
            j: 1,
            height: 0.999_999_999_999_999_9
        }
    );
    assert_eq!(
        at.certified_betti_0(),
        Beta0::Refused {
            i: 0,
            j: 1,
            height: 1.0
        }
    );
}

#[test]
fn graph_beta0_far_from_epsilon_is_certified() {
    assert!(matches!(
        pair(0.01).certified_betti_0(),
        Beta0::Certified { value: 1, .. }
    ));
    assert!(matches!(
        pair(10.0).certified_betti_0(),
        Beta0::Certified { value: 2, .. }
    ));
}
