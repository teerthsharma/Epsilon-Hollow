//! Iteration 5 — exact-oracle ground truth. Observes and prints; the gate is read
//! off the printed values. persistence.rs and diagram.rs are untouched.
use aether_core::manifold::ManifoldPoint;
use aether_core::persistence::{persistent_homology, ComplexKind, PersistenceConfig};

fn cfg(dim: usize, max_points: usize) -> PersistenceConfig {
    PersistenceConfig {
        max_homology_dim: dim,
        max_points,
        max_simplices: 4_000_000,
        max_radius: f64::INFINITY,
        complex_kind: ComplexKind::VietorisRips,
    }
}

/// Longest finite bar in a dimension, as (birth, death, persistence).
fn longest(
    d: &aether_core::persistence::PersistenceDiagram,
    dim: usize,
) -> Option<(f64, f64, f64)> {
    d.pairs
        .iter()
        .filter(|p| p.dimension == dim)
        .filter_map(|p| p.death.map(|dd| (p.birth, dd, dd - p.birth)))
        .fold(None, |acc: Option<(f64, f64, f64)>, b| match acc {
            Some(a) if a.2 >= b.2 => Some(a),
            _ => Some(b),
        })
}

fn count_long(d: &aether_core::persistence::PersistenceDiagram, dim: usize, thresh: f64) -> usize {
    d.pairs
        .iter()
        .filter(|p| p.dimension == dim)
        .filter(|p| p.death.map(|dd| dd - p.birth > thresh).unwrap_or(true))
        .count()
}

#[test]
fn oracle_ground_truth() {
    // ---- 1. CIRCLE: exactly one long H1 bar, death should approach sqrt(3)*r ----
    println!("=== CIRCLE, radius r, n points: H1 death vs sqrt(3)*r ===");
    let sqrt3 = 3.0f64.sqrt();
    for &(n, r) in &[(24usize, 1.0f64), (40, 1.0), (60, 1.0), (40, 2.5)] {
        let pts: Vec<ManifoldPoint<2>> = (0..n)
            .map(|i| {
                let t = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
                ManifoldPoint::new([r * t.cos(), r * t.sin()])
            })
            .collect();
        let d = persistent_homology(&pts, cfg(1, 128)).unwrap();
        let l = longest(&d, 1);
        match l {
            Some((b, dd, p)) => println!(
                "n={:3} r={:.1}  H1 longest: birth={:.5} death={:.5} pers={:.5} | sqrt(3)*r={:.5} | ratio={:.5} | #long H1={}",
                n, r, b, dd, p, sqrt3 * r, dd / (sqrt3 * r), count_long(&d, 1, 0.25 * r)),
            None => println!("n={:3} r={:.1}  NO FINITE H1 BAR", n, r),
        }
    }

    // ---- 2. SPHERE: one H2 bar ----
    println!();
    println!("=== SPHERE S2 (Fibonacci sample), H2 ===");
    for &n in &[30usize, 42] {
        let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt()); // golden angle
        let pts: Vec<ManifoldPoint<3>> = (0..n)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f64) / ((n - 1) as f64);
                let rad = (1.0 - y * y).max(0.0).sqrt();
                let th = ga * (i as f64);
                ManifoldPoint::new([th.cos() * rad, y, th.sin() * rad])
            })
            .collect();
        let d = persistent_homology(&pts, cfg(2, 48)).unwrap();
        println!(
            "n={:3}  #H2 bars={}  longest H2={:?}  #long H2(>0.15)={}  #long H1(>0.30)={}",
            n,
            d.pairs.iter().filter(|p| p.dimension == 2).count(),
            longest(&d, 2).map(|(b, dd, p)| (
                format!("{:.4}", b),
                format!("{:.4}", dd),
                format!("{:.4}", p)
            )),
            count_long(&d, 2, 0.15),
            count_long(&d, 1, 0.30)
        );
    }

    // ---- 3. GAUSSIAN BLOB: negative control, no long bars above H0 ----
    println!();
    println!("=== GAUSSIAN BLOB (negative control): expect NO long H1 ===");
    let mut s: u64 = 0x2545F4914F6CDD1D;
    let mut nx = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        (u - 0.5) * 2.0
    };
    for &n in &[40usize, 80] {
        let pts: Vec<ManifoldPoint<2>> = (0..n)
            .map(|_| {
                let (a, b) = (nx(), nx());
                let (c, d2) = (nx(), nx());
                ManifoldPoint::new([(a + b + c) / 3.0, (d2 + nx() + nx()) / 3.0])
            })
            .collect();
        let d = persistent_homology(&pts, cfg(1, 128)).unwrap();
        let sp = pts
            .iter()
            .flat_map(|p| (0..2).map(move |k| p.coords[k]))
            .fold(f64::MIN, f64::max);
        println!(
            "n={:3} spread~{:.3}  longest H1={:?}  #H1 bars={}",
            n,
            sp,
            longest(&d, 1).map(|(_, _, p)| format!("{:.5}", p)),
            d.pairs.iter().filter(|p| p.dimension == 1).count()
        );
    }
}
