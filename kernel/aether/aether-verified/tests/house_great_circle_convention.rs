//! D10 — `great_circle_distance` uses the LATITUDE formula while the index that
//! consumes the same (theta, phi) pairs uses COLATITUDE.
//!
//! `aether-core`'s `spherical_to_unit_vector(theta, phi)` is
//! `[sin t cos p, sin t sin p, cos t]`, the colatitude convention. The cosine of
//! the central angle between two such points is
//!
//!     sin t1 sin t2 cos(p1 - p2)  +  cos t1 cos t2
//!
//! The implemented formula puts `cos(p1 - p2)` on the OTHER term.
use aether_verified::aether_tss::great_circle_distance;

const PI: f64 = core::f64::consts::PI;

/// Ground truth under the colatitude convention, derived from the dot product
/// of the two unit vectors the index actually builds.
fn colatitude_truth(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let c = t1.sin() * t2.sin() * (p1 - p2).cos() + t1.cos() * t2.cos();
    c.clamp(-1.0, 1.0).acos()
}

#[test]
fn two_points_a_quarter_turn_apart_on_the_equator() {
    // THE WITNESS. Colatitude pi/2 is the equator. Longitudes 0 and pi/2 are
    // 90 degrees apart, so the great-circle distance is exactly pi/2.
    let (t1, p1, t2, p2) = (PI / 2.0, 0.0, PI / 2.0, PI / 2.0);
    let got = great_circle_distance(t1, p1, t2, p2);
    let want = colatitude_truth(t1, p1, t2, p2);
    println!("equator, 90 deg apart: got={got:.6} want={want:.6}");
    assert!(
        (want - PI / 2.0).abs() < 1e-12,
        "ground truth itself must be pi/2"
    );
    assert!(
        (got - want).abs() < 1e-9,
        "great_circle_distance returned {got} for two points {want} apart; \
         the cos(delta phi) factor is on the wrong term"
    );
}

#[test]
fn north_pole_to_equator_is_a_quarter_turn() {
    // Colatitude 0 is the north pole; colatitude pi/2 is the equator.
    let (t1, p1, t2, p2) = (0.0, 0.0, PI / 2.0, 0.0);
    let got = great_circle_distance(t1, p1, t2, p2);
    let want = colatitude_truth(t1, p1, t2, p2);
    println!("pole to equator: got={got:.6} want={want:.6}");
    assert!((want - PI / 2.0).abs() < 1e-12);
    assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
}

#[test]
fn agrees_with_the_index_convention_on_random_pairs() {
    let mut s: u64 = 0x5EED_1234_ABCD_9876;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut worst = 0.0f64;
    for _ in 0..2000 {
        let (t1, t2) = (PI * u(), PI * u());
        let (p1, p2) = (2.0 * PI * u(), 2.0 * PI * u());
        let d = (great_circle_distance(t1, p1, t2, p2) - colatitude_truth(t1, p1, t2, p2)).abs();
        worst = worst.max(d);
    }
    println!("worst disagreement over 2000 random pairs: {worst:.6}");
    assert!(
        worst < 1e-9,
        "disagrees with the index convention by up to {worst} radians"
    );
}

#[test]
fn small_separations_survive_the_regime_a_separation_check_operates_in() {
    // `verify_separation` compares against `theta_min - 1e-6`, so accuracy at
    // small separations is what actually matters. The acos-of-dot-product form
    // returned 9.998224e-7 for a true 1e-6 and exactly 0.0 for a true 1e-8,
    // reporting distinct points as coincident. Haversine holds both.
    let theta = 1.0f64;
    for &sep in &[1e-2f64, 1e-4, 1e-6, 1e-8] {
        let d = great_circle_distance(theta, 0.0, theta + sep, 0.0);
        let rel = (d - sep).abs() / sep;
        println!("true {sep:.0e} -> {d:.6e}  relative error {rel:.2e}");
        assert!(
            rel < 1e-6,
            "separation {sep:.0e} came back as {d:.6e}, rel err {rel:.2e}"
        );
        assert!(
            d > 0.0,
            "two distinct points {sep:.0e} apart reported as coincident"
        );
    }
}

#[test]
fn d_p_p_is_exactly_zero() {
    let mut s: u64 = 0x7777_3333_1111_9999;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut worst = 0.0f64;
    for _ in 0..5000 {
        let (t, p) = (PI * u(), 2.0 * PI * u());
        worst = worst.max(great_circle_distance(t, p, t, p).abs());
    }
    println!("worst d(p,p) over 5000 points: {worst:.3e}");
    assert!(
        worst < 1e-15,
        "d(p,p) reached {worst:.3e}; the acos form gave up to 2.1e-8"
    );
}

/// Copy of the eight centroids `tss_boot_centroids` in
/// `kernel/seal-os/src/lib.rs` returns; that function is the source this copy
/// must match. seal-os sits outside the workspace and its `#[cfg(test)]` tests
/// never run, so the T1 boot gate is checked here on the host instead.
fn seal_os_boot_centroids() -> [(f64, f64); 8] {
    let north = 0.955_316_618_124_509_2; // acos(1 / sqrt(3)), a colatitude
    let south = core::f64::consts::PI - north;
    let step = core::f64::consts::FRAC_PI_4;
    [
        (north, step),
        (north, step * 3.0),
        (north, step * 5.0),
        (north, step * 7.0),
        (south, step),
        (south, step * 3.0),
        (south, step * 5.0),
        (south, step * 7.0),
    ]
}

#[test]
fn boot_centroids_separate_under_colatitude() {
    use aether_verified::aether_tss::{
        theta_min_from_epsilon, verify_packing_bound, verify_separation,
    };
    // The exact T1 predicate `verify_topology_theorems` evaluates at boot.
    let c = seal_os_boot_centroids();
    let theta = theta_min_from_epsilon(0.5);
    let mut min_sep = f64::INFINITY;
    for i in 0..c.len() {
        for j in (i + 1)..c.len() {
            min_sep = min_sep.min(great_circle_distance(c[i].0, c[i].1, c[j].0, c[j].1));
        }
    }
    println!("boot centroids: theta_min={theta:.5} min_sep={min_sep:.6}");
    assert!(
        verify_packing_bound(c.len(), theta) && verify_separation(&c, theta),
        "T1 fails on the boot centroids: min separation {min_sep} < theta_min {theta}"
    );
    // The eight points are meant to be the cube vertices (+-1, +-1, +-1)/sqrt(3),
    // whose nearest neighbours are one cube edge apart: acos(1/3).
    let edge = (1.0f64 / 3.0).acos();
    assert!(
        (min_sep - edge).abs() < 1e-12,
        "min separation {min_sep} is not the cube edge angle {edge}"
    );
}
