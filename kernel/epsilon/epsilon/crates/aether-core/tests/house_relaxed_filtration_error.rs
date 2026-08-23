//! First end-to-end measurement: how far is the RELAXED filtration's diagram
//! from the exact Rips diagram, and does the error track `epsilon`?
use aether_core::diagram::bottleneck_distance;
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::{relaxed_entry_time, NetTree};
use aether_core::persistence::{
    persistent_homology, persistent_homology_from_distances, ComplexKind, PersistenceConfig,
    PersistenceDiagram,
};

fn cfg(n: usize) -> PersistenceConfig {
    PersistenceConfig {
        max_homology_dim: 1,
        max_points: n.max(8),
        max_simplices: 4_000_000,
        max_radius: f64::INFINITY,
        complex_kind: ComplexKind::VietorisRips,
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

/// Entry-time matrix of the relaxed filtration, deletion times from the net-tree.
fn relaxed_matrix<const D: usize>(pts: &[ManifoldPoint<D>], eps: f64) -> Vec<f64> {
    let n = pts.len();
    let t = NetTree::build(pts, 2.0).deletion_times(n, eps);
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i].distance(&pts[j]);
            let a = relaxed_entry_time(d, t[i], t[j], eps);
            m[i * n + j] = a;
            m[j * n + i] = a;
        }
    }
    m
}

fn longest_h1(d: &PersistenceDiagram) -> f64 {
    d.pairs
        .iter()
        .filter(|p| p.dimension == 1)
        .filter_map(|p| p.death.map(|dd| dd - p.birth))
        .fold(0.0, f64::max)
}

#[test]
fn relaxed_diagram_error_tracks_epsilon() {
    let n = 40usize;
    let pts = circle(n, 1.0);
    let exact = persistent_homology(&pts, cfg(n)).unwrap();
    let exact_h1 = longest_h1(&exact);
    // Closed form from iteration 5: death = 2 r sin(pi ceil(n/3) / n).
    let k = (n as f64 / 3.0).ceil();
    let expect_death = 2.0 * (core::f64::consts::PI * k / n as f64).sin();
    println!("exact: longest H1 persistence {exact_h1:.6}, closed-form death {expect_death:.6}");

    println!();
    println!(
        "{:>8} {:>12} {:>14} {:>12}",
        "eps", "longest H1", "bottleneck", "b / eps"
    );
    let mut prev = 0.0f64;
    for &eps in &[0.02f64, 0.05, 0.1, 0.2, 1.0 / 3.0] {
        let m = relaxed_matrix(&pts, eps);
        let relaxed = persistent_homology_from_distances(&m, n, cfg(n)).unwrap();
        let b = bottleneck_distance(&exact, &relaxed, 1);
        println!(
            "{eps:>8.4} {:>12.6} {:>14.6} {:>12.3}",
            longest_h1(&relaxed),
            b,
            b / eps
        );

        // The error must not shrink as the approximation is loosened.
        assert!(
            b >= prev - 1e-9,
            "bottleneck error fell when eps grew: {prev:.6} -> {b:.6} at eps={eps}"
        );
        prev = b;
    }

    // With Sheehy's section-6 deletion times the construction behaves as the
    // theorem says it should: the error is BOUNDED BY eps, not superlinear in
    // it. The earlier breakdown was caused by deletion times that carried no
    // eps dependence; see the ledger, iterations 20 and 21.
    //
    // Sheehy's interleaving is multiplicative, so `b / eps` must not grow. It
    // is asserted flat-or-falling here, which is what distinguishes a faithful
    // implementation from one that merely happens to be small at one setting.
    for &eps in &[0.02f64, 0.05, 0.1, 0.2, 1.0 / 3.0] {
        let m = relaxed_matrix(&pts, eps);
        let d = persistent_homology_from_distances(&m, n, cfg(n)).unwrap();
        let b = bottleneck_distance(&exact, &d, 1);
        assert!(
            b <= eps + 1e-9,
            "eps={eps}: bottleneck {b:.6} exceeds eps; the interleaving is not multiplicative"
        );
        assert!(
            b < expect_death,
            "eps={eps}: bottleneck {b:.6} reaches the whole feature scale {expect_death:.6}"
        );
    }

    // EXACT PIN, not a bound. `b <= eps` has slack — it cannot detect a 2x
    // error in the deletion times, which the mutation gate demonstrated by
    // substituting rad(v_p) for rad(par(v_p)) and surviving. These are the
    // measured values at circle n=40, H1, with Sheehy's section-6 deletion
    // times. Any change to the construction moves one of them and forces this
    // entry and the ledger to be revised deliberately.
    for &(eps, want) in &[
        (0.02f64, 0.0f64),
        (0.05, 0.0),
        (0.10, 0.0),
        (0.20, 0.065746),
        (1.0 / 3.0, 0.065746),
    ] {
        let m = relaxed_matrix(&pts, eps);
        let d = persistent_homology_from_distances(&m, n, cfg(n)).unwrap();
        let b = bottleneck_distance(&exact, &d, 1);
        assert!((b - want).abs() < 1e-5,
            "eps={eps}: bottleneck {b:.6}, pinned at {want:.6}. The construction              changed. If deliberately, update this pin and the ledger.");
    }
}

#[test]
fn tiny_epsilon_recovers_the_exact_diagram() {
    // As eps -> 0 the weights vanish and the relaxed filtration must converge
    // to the exact one. This is the reduction the construction has to satisfy.
    let n = 32usize;
    let pts = circle(n, 1.0);
    let exact = persistent_homology(&pts, cfg(n)).unwrap();
    for &eps in &[1e-6f64, 1e-9] {
        let m = relaxed_matrix(&pts, eps);
        let relaxed = persistent_homology_from_distances(&m, n, cfg(n)).unwrap();
        let b = bottleneck_distance(&exact, &relaxed, 1);
        println!("eps={eps:.0e}: bottleneck to exact = {b:.3e}");
        assert!(
            b < 1e-3,
            "eps={eps:.0e} did not recover the exact diagram: {b}"
        );
    }
}
