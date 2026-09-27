//! Seal OS builds four 8-cell `SphericalVoronoiIndex`es, in the scheduler,
//! the compositor, the firewall and the router, and each treats the slot ->
//! point-on-S² map as injective: slot `k` is cell `k`. Their shared table must
//! therefore put eight distinct, well-separated points on the sphere.
//!
//! The `{0, pi/2, pi} x {0, pi/2, pi}` lattice they used does not. In
//! colatitude, `theta = 0` is the north pole for every `phi`, so slots 0, 3 and
//! 6 are one point, and `theta = pi` is the south pole, so slots 2 and 5 are one
//! point up to `sin(fl(pi)) = 1.2e-16`: which of them a query reaches is decided
//! by that rounding residue, not by the query.
use aether_core::tss::{verify_separation, SphericalVoronoiIndex, CUBE_CENTROIDS};

#[test]
fn every_slot_is_the_nearest_cell_to_its_own_centroid() {
    let idx = SphericalVoronoiIndex::<8>::new(CUBE_CENTROIDS);
    let located: Vec<usize> = CUBE_CENTROIDS.iter().map(|&c| idx.locate(c)).collect();
    let mut distinct = located.clone();
    distinct.sort_unstable();
    distinct.dedup();
    // from teerthsharma/branchcut branchcut/partition.py:197: a map meant to be injective onto m values from n misassigns at least n - m
    let aliased = located.len() - distinct.len();
    assert_eq!(
        aliased, 0,
        "{aliased} slots unreachable: locate(centroid[k]) = {located:?}"
    );
    assert_eq!(located, (0..8).collect::<Vec<_>>());
}

#[test]
fn slots_are_separated_far_beyond_rounding() {
    // A pair separated by less than any rounding radius is a collision whose
    // nearest-cell answer rounding decides. 1.2 rad is below acos(1/3), the
    // separation of adjacent cube vertices, by 0.03 rad.
    assert!(verify_separation(&CUBE_CENTROIDS, 1.2));
}
