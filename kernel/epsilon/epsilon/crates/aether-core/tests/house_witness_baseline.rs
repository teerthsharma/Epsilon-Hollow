//! Iteration 6 — the incumbent baseline: uncertified Witness vs exact Rips oracle.
use aether_core::diagram::bottleneck_distance;
use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{
    persistent_homology, ComplexKind, PersistenceConfig, PersistenceDiagram,
};
use std::time::Instant;

fn cf(dim: usize, mp: usize, k: ComplexKind) -> PersistenceConfig {
    PersistenceConfig {
        max_homology_dim: dim,
        max_points: mp,
        max_simplices: 4_000_000,
        max_radius: f64::INFINITY,
        complex_kind: k,
    }
}
fn circle(n: usize, r: f64) -> Vec<ManifoldPoint<2>> {
    (0..n)
        .map(|i| {
            let t = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
            ManifoldPoint::new([r * t.cos(), r * t.sin()])
        })
        .collect()
}
fn h1_pairs(d: &PersistenceDiagram) -> usize {
    d.pairs.iter().filter(|p| p.dimension == 1).count()
}
fn longest_h1(d: &PersistenceDiagram) -> (f64, f64) {
    d.pairs
        .iter()
        .filter(|p| p.dimension == 1)
        .filter_map(|p| p.death.map(|dd| (dd - p.birth, dd)))
        .fold((0.0, f64::NAN), |a, b| if b.0 > a.0 { b } else { a })
}

#[test]
fn witness_vs_exact_oracle() {
    println!("=== step 1: how big do H1 diagrams get? (oracle cost driver) ===");
    for &n in &[16usize, 24, 32, 40] {
        let t = Instant::now();
        let e =
            persistent_homology(&circle(n, 1.0), cf(1, 128, ComplexKind::VietorisRips)).unwrap();
        println!(
            "n={:3}  total pairs={:6}  H1 pairs={:6}  build={:.3}s",
            n,
            e.pairs.len(),
            h1_pairs(&e),
            t.elapsed().as_secs_f64()
        );
    }

    println!();
    println!("=== step 2: Witness vs exact oracle, circle n=32 r=1.0, H1 ===");
    let n = 32usize;
    let pts = circle(n, 1.0);
    let t = Instant::now();
    let exact = persistent_homology(&pts, cf(1, 128, ComplexKind::VietorisRips)).unwrap();
    let t_exact = t.elapsed().as_secs_f64();
    let (ep, ed) = longest_h1(&exact);
    // exact expected death: 2*sin(pi*ceil(n/3)/n)
    let k = (n as f64 / 3.0).ceil();
    let expect = 2.0 * (core::f64::consts::PI * k / (n as f64)).sin();
    println!("EXACT  H1pairs={:5}  longest(pers={:.5}, death={:.5})  closed-form death={:.5}  time={:.4}s",
        h1_pairs(&exact), ep, ed, expect, t_exact);
    println!();
    println!(
        "{:>9} {:>8} {:>10} {:>10} {:>12} {:>9} {:>8}",
        "landmark", "H1pairs", "pers", "death", "bottleneck", "time_s", "speedup"
    );
    for &l in &[8usize, 12, 16, 20, 24, 28, 32] {
        let t = Instant::now();
        match persistent_homology(&pts, cf(1, 128, ComplexKind::Witness { max_landmarks: l })) {
            Ok(w) => {
                let el = t.elapsed().as_secs_f64();
                let (wp, wd) = longest_h1(&w);
                let b = bottleneck_distance(&exact, &w, 1);
                println!(
                    "{:>9} {:>8} {:>10.5} {:>10.5} {:>12.5} {:>9.4} {:>7.2}x",
                    l,
                    h1_pairs(&w),
                    wp,
                    wd,
                    b,
                    el,
                    t_exact / el.max(1e-9)
                );
            }
            Err(e) => println!("{:>9}  ERROR {:?}", l, e),
        }
    }
}
