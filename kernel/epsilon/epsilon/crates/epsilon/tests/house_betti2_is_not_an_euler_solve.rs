//! `b2 = 2 - b0 + b1` cannot be a second Betti number, argued on the inputs the
//! expression actually receives.
//!
//! # A retracted argument, kept because the way it failed is the point
//!
//! The first version of this file argued: a disc and a sphere have the same
//! `beta_0 = 1` and `beta_1 = 0` and different `beta_2`, so a function of
//! `(beta_0, beta_1)` alone cannot be `beta_2`. The logic is valid and the
//! measurement was real, and the argument is still **wrong**, because those
//! `(beta_0, beta_1)` are Vietoris-Rips invariants of the point cloud and
//! `euler_defect` never sees them. It reads `SparseGraph::compute_betti_0` and
//! `SparseGraph::estimate_betti_1`, the invariants of the 1-skeleton, and on
//! that domain the disc is `(1, 511)`, not `(1, 0)`. The two samples do not
//! collide on the inputs the code consumes, so the counterexample was built
//! against a function that does not exist.
//!
//! That is this repository's named characteristic defect - a hypothesis that is
//! not load-bearing - committed inside the test written to expose it. Recording
//! it costs one paragraph and is cheaper than the next reader re-deriving it.
//!
//! # The argument that works, on the right domain
//!
//! For a **connected** graph, `beta_0 = 1`, so
//!
//! ```text
//!     euler_defect = 2 - 1 + beta_1 = 1 + beta_1 >= 1
//! ```
//!
//! It is bounded strictly below by 1 and can never be 0. The true `beta_2` of a
//! connected contractible sample **is** 0. So the expression disagrees with
//! `beta_2` on every connected contractible input, and no choice of constant
//! rescues it: the quantity has the wrong range, not merely the wrong `chi`.
//! No collision is needed and no second sample is needed.

use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{persistent_homology, PersistenceConfig};
use epsilon::manifold::{EpsilonPoint, SparseGraph};

/// Points on the unit sphere, spread by the Fibonacci lattice.
fn sphere(n: usize) -> Vec<[f64; 3]> {
    let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64) / ((n.max(2) - 1) as f64);
            let r = (1.0 - y * y).max(0.0).sqrt();
            let t = ga * (i as f64);
            [t.cos() * r, y, t.sin() * r]
        })
        .collect()
}

/// A flat disc sampled on the same lattice in the plane. Contractible: every
/// Betti number above zero vanishes.
fn disc(n: usize) -> Vec<[f64; 3]> {
    let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n)
        .map(|i| {
            let r = ((i as f64 + 0.5) / (n as f64)).sqrt();
            let t = ga * (i as f64);
            [r * t.cos(), r * t.sin(), 0.0]
        })
        .collect()
}

fn torus(n: usize, big: f64, small: f64) -> Vec<[f64; 3]> {
    let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n)
        .map(|i| {
            let u = ga * (i as f64);
            let v = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
            let rr = big + small * v.cos();
            [rr * u.cos(), rr * u.sin(), small * v.sin()]
        })
        .collect()
}

fn graph_of(pts: &[[f64; 3]], eps: f64) -> SparseGraph<3> {
    let mut g = SparseGraph::<3>::new(eps);
    for c in pts {
        g.add_point(EpsilonPoint::new(*c));
    }
    g
}

fn measure(pts: &[[f64; 3]], radius: f64) -> (u32, u32, u32) {
    let mp: Vec<ManifoldPoint<3>> = pts.iter().map(|c| ManifoldPoint::new(*c)).collect();
    let diagram = persistent_homology(&mp, PersistenceConfig::h2_default())
        .expect("h2_default admits 48 points");
    let b = diagram.betti_at(radius);
    (b.beta_0, b.beta_1, b.beta_2)
}

/// **Lemma M.** On a connected graph the Euler defect is at least 1, so it
/// cannot represent a second Betti number that is 0.
///
/// Checked across several samples and epsilons rather than argued from the
/// formula, so that a change to `euler_defect` breaking the bound is caught
/// here rather than reasoned away.
#[test]
fn lemma_m_euler_defect_is_at_least_one_on_every_connected_graph() {
    let mut checked = 0usize;
    for pts in [disc(48), sphere(48), torus(48, 1.0, 0.35)] {
        for &eps in &[0.5f64, 0.7, 0.9, 1.2, 2.5] {
            let g = graph_of(&pts, eps);
            let (b0, b1, defect) = g.full_shape();
            if b0 != 1 {
                continue; // not connected at this epsilon
            }
            checked += 1;
            assert_eq!(
                defect,
                b1 + 1,
                "connected graph at eps={eps} should give defect = 1 + beta_1"
            );
            assert!(defect >= 1, "defect {defect} fell below 1 at eps={eps}");
        }
    }
    println!("Lemma M checked on {checked} connected graphs");
    assert!(
        checked >= 8,
        "only {checked} connected configurations reached"
    );
}

/// The plan's gate, run on the domain the expression reads: a cloud that is not
/// a sphere must be able to come back with `beta_2 = 0`, and the Euler defect
/// provably cannot.
#[test]
fn the_euler_defect_cannot_report_the_true_beta_2_of_a_contractible_sample() {
    let d = disc(48);

    // Truth, from an exact boundary-matrix reduction over Z2.
    let (rb0, rb1, rb2) = measure(&d, 0.9);
    assert_eq!(rb0, 1, "the disc sample is connected at radius 0.9");
    assert_eq!(
        rb2, 0,
        "a flat disc is contractible; measured ({rb0},{rb1},{rb2})"
    );

    // What the expression under repair actually consumes, on the same points.
    let g = graph_of(&d, 0.9);
    let (gb0, gb1, defect) = g.full_shape();
    println!(
        "48 disc points at 0.9 -- Rips (b0,b1,b2) = ({rb0},{rb1},{rb2}); \
         graph (b0,b1) = ({gb0},{gb1}); euler_defect = {defect}; betti_2() = {}",
        g.betti_2()
    );

    assert_eq!(gb0, 1, "the graph is connected too");
    assert!(
        defect > rb2,
        "the Euler defect {defect} should exceed the true beta_2 {rb2}"
    );
    assert!(
        defect >= 512,
        "the reported defect on this fixture is {defect}; the point of the number \
         is that it is not a small error"
    );

    // And the replacement returns the right value. This assertion alone is
    // vacuous - `betti_2` returns a constant - so the two above carry the
    // argument and this one only pins the direction of the repair.
    assert_eq!(
        g.betti_2(),
        rb2,
        "betti_2 should agree with the exact reduction"
    );
}

/// Point-cloud invariants for the two samples, recorded because the retracted
/// argument above used them and a reader deserves to see the numbers that made
/// it look sound.
///
/// This is an observation, **not** the argument: `euler_defect` never reads
/// these. The Rips beta_1 of the disc is 0 while the graph's is 511, and that
/// gap is exactly why the domain mistake was easy to make.
#[test]
fn point_cloud_invariants_are_not_what_the_expression_reads() {
    let (sb0, sb1, sb2) = measure(&sphere(48), 0.9);
    let (db0, db1, db2) = measure(&disc(48), 0.9);
    println!("sphere(48) Rips at r=0.9 -> ({sb0},{sb1},{sb2})");
    println!("disc(48)   Rips at r=0.9 -> ({db0},{db1},{db2})");

    let (ggb0, ggb1, _) = graph_of(&disc(48), 0.9).full_shape();
    println!("disc(48)   graph at eps=0.9 -> (b0,b1) = ({ggb0},{ggb1})");

    assert_eq!((sb0, sb1), (db0, db1), "the Rips invariants do agree");
    assert_ne!(sb2, db2, "and the Rips beta_2 differs");
    assert_ne!(
        (ggb0, ggb1),
        (db0, db1),
        "but the graph invariants the expression reads are different numbers, \
         which is what made the earlier argument unsound"
    );
}

/// Reported, not asserted. 48 points is a thin sample of a genus-1 surface, so
/// whether the Rips complex resolves both 1-cycles is a question about the
/// sample rather than about the repair.
#[test]
fn torus_sample_is_reported_at_several_radii() {
    let t = torus(48, 1.0, 0.35);
    println!("torus(48, R=1.0, r=0.35)");
    println!("{:>8} {:>6} {:>6} {:>6}", "radius", "b0", "b1", "b2");
    for &r in &[0.35, 0.5, 0.7, 0.9, 1.2] {
        let (b0, b1, b2) = measure(&t, r);
        println!("{r:>8.2} {b0:>6} {b1:>6} {b2:>6}");
    }
    println!("A true torus is (1, 2, 1); this sample does not resolve it.");
}

/// `estimate_betti_1` is the **exact** first Betti number of the graph, not an
/// approximation, and the graph's first Betti number is not the point cloud's.
#[test]
fn the_graph_betti_1_is_exact_for_the_graph_and_wrong_for_the_space() {
    let d = disc(48);
    let g = graph_of(&d, 0.9);

    // Count edges independently of the implementation under test.
    let mut edges = 0i64;
    for i in 0..d.len() {
        for j in (i + 1)..d.len() {
            let dx = d[i][0] - d[j][0];
            let dy = d[i][1] - d[j][1];
            let dz = d[i][2] - d[j][2];
            if (dx * dx + dy * dy + dz * dz).sqrt() <= 0.9 {
                edges += 1;
            }
        }
    }
    let (gb0, gb1, _) = g.full_shape();
    assert_eq!(
        i64::from(gb1),
        edges - d.len() as i64 + i64::from(gb0),
        "beta_1 = E - V + beta_0 is exact for a 1-complex: E={edges} V={} beta_0={gb0}",
        d.len()
    );

    let (_, rips_b1, _) = measure(&d, 0.9);
    println!("48 disc points at 0.9: graph beta_1 = {gb1} (E={edges}), Rips beta_1 = {rips_b1}");
    assert_eq!(rips_b1, 0, "a disc is contractible");
    assert!(
        gb1 > 100,
        "the graph should carry hundreds of unfilled cycles here, got {gb1}"
    );
}
