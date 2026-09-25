//! `SphericalGridHashIndex::locate` returns the nearest centroid, not the
//! nearest one inside the 3x3 cell neighbourhood of the query.
//!
//! The neighbourhood answer is accepted only when its distance is at most the
//! distance from the query to the boundary of the searched region; otherwise
//! the index scans every centroid. `o1_hit_rate` counts only the certified
//! answers.
use aether_core::tss::SphericalGridHashIndex;
use core::f64::consts::PI;

fn angle(a: (f64, f64), b: (f64, f64)) -> f64 {
    let u = [a.0.sin() * a.1.cos(), a.0.sin() * a.1.sin(), a.0.cos()];
    let v = [b.0.sin() * b.1.cos(), b.0.sin() * b.1.sin(), b.0.cos()];
    (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
        .clamp(-1.0, 1.0)
        .acos()
}

fn brute_nearest(centroids: &[(f64, f64)], q: (f64, f64)) -> usize {
    let mut best = 0;
    for i in 1..centroids.len() {
        if angle(q, centroids[i]) < angle(q, centroids[best]) {
            best = i;
        }
    }
    best
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn nearest_centroid_across_the_pole_is_found() {
    // Centroid 0 sits 0.1995 rad away across the north pole, in cell (0, 3)
    // under [0, 2pi) columns and (0, 7) under the old [-pi, pi] clamp; both lie
    // outside the 3x3 block around the query. Centroid 1 is 0.9 rad away.
    let c = [(0.1, 3.0), (1.0, 0.0)];
    let mut grid = SphericalGridHashIndex::<2>::new(c);
    let q = (0.1, 0.0);
    assert!(angle(q, c[0]) < angle(q, c[1]));
    assert_eq!(grid.locate(q), 0);
}

fn check_random<const K: usize>(seed: u64, queries: usize) -> f64 {
    let mut rng = Lcg(seed);
    let mut c = [(0.0, 0.0); K];
    for slot in c.iter_mut() {
        *slot = ((rng.unit() * 2.0 - 1.0).acos(), rng.unit() * 2.0 * PI);
    }
    let mut grid = SphericalGridHashIndex::<K>::new(c);
    let mut wrong = 0usize;
    for _ in 0..queries {
        // phi spans [-2pi, 4pi) so the index sees unnormalised longitudes.
        let q = (
            (rng.unit() * 2.0 - 1.0).acos(),
            rng.unit() * 6.0 * PI - 2.0 * PI,
        );
        if grid.locate(q) != brute_nearest(&c, q) {
            wrong += 1;
        }
    }
    println!(
        "K={K} seed={seed}: {wrong}/{queries} wrong, o1_hit_rate={:.4}",
        grid.o1_hit_rate()
    );
    assert_eq!(wrong, 0, "K={K}: locate disagreed with brute force");
    grid.o1_hit_rate()
}

#[test]
fn locate_matches_brute_force_on_random_queries() {
    check_random::<2>(1, 5000);
    check_random::<8>(2, 5000);
    check_random::<20>(3, 5000);
    check_random::<64>(4, 5000);
    check_random::<200>(5, 5000);
}

#[test]
fn hit_rate_counts_only_certified_answers() {
    // Both centroids sit near the south pole. A query on the equator finds
    // them in the searched rows, but 1.47 rad away while the top of the
    // searched band is 0.785 rad away, so the answer needs a full scan.
    let c = [(PI - 0.1, 0.0), (PI - 0.1, PI)];
    let mut grid = SphericalGridHashIndex::<2>::new(c);
    assert_eq!(grid.locate((PI / 2.0, 0.0)), 0);
    assert_eq!(
        grid.o1_hit_rate(),
        0.0,
        "an uncertified answer is not a hit"
    );

    // A query on top of a centroid is certified from its own cell.
    assert_eq!(grid.locate(c[1]), 1);
    assert_eq!(grid.o1_hit_rate(), 0.5);
    assert_eq!(grid.stats().total_lookups, 2);
}
