//! RED test for the imposed-beta_2 defect at src/manifold.rs:222.
//!
//! `compute_betti_2_euler` computes `b2 = 2 - b0 + b1`, whose doc comment states
//! it assumes the space is homeomorphic to S2 and therefore imposes chi = 2. On a
//! `SparseGraph`, which is a 1-skeleton, H2 is identically 0. These tests feed it
//! clouds that are NOT spheres and assert the correct answer, beta_2 = 0.
use epsilon::manifold::{EpsilonPoint, SparseGraph};

/// A filled planar disc embedded in R^3. Contractible: beta_0 = 1, beta_2 = 0.
fn disc(k: i32, step: f64) -> Vec<EpsilonPoint<3>> {
    let mut v = Vec::new();
    for i in -k..=k {
        for j in -k..=k {
            let (x, y) = (i as f64 * step, j as f64 * step);
            if x * x + y * y <= (k as f64 * step) * (k as f64 * step) {
                v.push(EpsilonPoint::new([x, y, 0.0]));
            }
        }
    }
    v
}

/// A straight line segment. Contractible: beta_0 = 1, beta_1 = 0, beta_2 = 0.
fn segment(n: usize, step: f64) -> Vec<EpsilonPoint<3>> {
    (0..n).map(|i| EpsilonPoint::new([i as f64 * step, 0.0, 0.0])).collect()
}

#[test]
fn disc_is_not_a_sphere_so_betti_2_must_be_zero() {
    let pts = disc(4, 1.0);
    let mut g = SparseGraph::<3>::new(1.5);
    for p in &pts { g.add_point(*p); }
    let (b0, b1, defect) = g.full_shape();
    let b2 = g.betti_2();
    println!("DISC   n={} -> b0={} b1={} euler_defect={} betti_2={}", pts.len(), b0, b1, defect, b2);
    assert_eq!(b0, 1, "disc is connected");
    assert_eq!(b2, 0, "a filled planar disc has H2 = 0; got beta_2 = {b2}");
    assert_eq!(defect, b1 + 1, "euler defect is 1 + b1 on connected input");
}

#[test]
fn segment_is_not_a_sphere_so_betti_2_must_be_zero() {
    let pts = segment(10, 1.0);
    let mut g = SparseGraph::<3>::new(1.5);
    for p in &pts { g.add_point(*p); }
    let (b0, b1, defect) = g.full_shape();
    let b2 = g.betti_2();
    println!("SEGMENT n={} -> b0={} b1={} euler_defect={} betti_2={}", pts.len(), b0, b1, defect, b2);
    assert_eq!(b0, 1, "segment is connected");
    assert_eq!(b1, 0, "a path graph has cycle rank 0");
    assert_eq!(b2, 0, "a line segment has H2 = 0; got beta_2 = {b2}");
    assert_eq!(defect, 1, "euler defect of a path graph is 2 - 1 + 0 = 1, and it is NOT beta_2");
}

/// The mutant this test kills: "assume chi = 2 and solve for beta_2".
/// Any implementation that computes beta_2 from an imposed Euler characteristic
/// fails these two cases, because it cannot represent a non-sphere.
#[test]
fn betti_2_of_a_one_skeleton_is_identically_zero() {
    // H_2 of any 1-dimensional simplicial complex is 0, for every input.
    for eps in [0.5f64, 1.5, 3.0, 100.0] {
        let mut g = SparseGraph::<3>::new(eps);
        for p in &disc(3, 1.0) { g.add_point(*p); }
        let b2 = g.betti_2();
        let (_, _, defect) = g.full_shape();
        println!("eps={:6} -> betti_2={} euler_defect={}", eps, b2, defect);
        assert_eq!(b2, 0, "SparseGraph is a 1-skeleton; H2 is identically 0, but eps={eps} gave beta_2={b2}");
    }
}
