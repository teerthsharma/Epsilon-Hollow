//! `aether_core::tss` measures great-circle distance on the same colatitude
//! `(theta, phi)` pairs `aether_verified::aether_tss` does. Its inline copy of
//! the distance kept the latitude formula after the verified kernel moved to
//! colatitude, so `locate` and `verify_separation` disagreed with the kernel
//! the seal-os boot gate calls.
use aether_core::tss::{compute_theta_min, verify_separation, SphericalVoronoiIndex};
use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// Cube vertices `(+-1, +-1, +-1)/sqrt(3)` in colatitude, the seal-os boot set.
fn cube_vertices() -> [(f64, f64); 8] {
    let north = (1.0f64 / 3.0f64.sqrt()).acos();
    let south = PI - north;
    let mut c = [(0.0, 0.0); 8];
    for (i, slot) in c.iter_mut().enumerate() {
        let theta = if i < 4 { north } else { south };
        *slot = (theta, FRAC_PI_4 * (2 * (i % 4) + 1) as f64);
    }
    c
}

#[test]
fn cube_vertex_centroids_are_eight_distinct_cells() {
    let c = cube_vertices();
    let idx = SphericalVoronoiIndex::<8>::new(c);
    let located: Vec<usize> = c.iter().map(|&q| idx.locate(q)).collect();
    println!("locate(centroid[i]) = {located:?}");
    assert_eq!(located, (0..8).collect::<Vec<_>>());
    assert!(
        verify_separation(&c, compute_theta_min(0.5)),
        "cube vertices are acos(1/3) apart, far above theta_min"
    );
}

#[test]
fn equator_query_lands_in_its_own_cell() {
    // The fixture `kernel/seal-os/src/fs/voronoi_cap.rs` builds. Under the
    // latitude formula every equatorial centroid sits acos(sin qt) from any
    // query, so all distances tie and `locate` returns cell 0.
    let mut c = [(0.0, 0.0); 8];
    for (i, slot) in c.iter_mut().enumerate() {
        *slot = (FRAC_PI_2, FRAC_PI_4 * i as f64);
    }
    let idx = SphericalVoronoiIndex::<8>::new(c);
    assert_eq!(idx.locate((FRAC_PI_2, FRAC_PI_2)), 2);
}
