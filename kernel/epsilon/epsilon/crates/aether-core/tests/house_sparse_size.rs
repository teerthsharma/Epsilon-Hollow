//! Theorem 9.3 claims the sparse filtration has O(n) simplices where exact Rips
//! has O(n^2) edges. This measures the edge count directly.
//!
//! An edge (p,q) is in the sparse filtration Q_alpha exactly when both
//! endpoints are still alive at its entry time: `a < min(t_p, t_q)`, where
//! `N_alpha = {p : t_p > alpha}` is Sheehy's open net (section 5).
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::{relaxed_entry_time, NetTree};

fn circle(n: usize, r: f64) -> Vec<ManifoldPoint<2>> {
    (0..n).map(|i| {
        let t = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
        ManifoldPoint::new([r * t.cos(), r * t.sin()])
    }).collect()
}

fn sphere(n: usize) -> Vec<ManifoldPoint<3>> {
    let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n).map(|i| {
        let y = 1.0 - 2.0 * (i as f64) / ((n.max(2) - 1) as f64);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let t = ga * (i as f64);
        ManifoldPoint::new([t.cos() * r, y, t.sin() * r])
    }).collect()
}

/// (sparse edges, dense edges)
fn edge_counts<const D: usize>(pts: &[ManifoldPoint<D>], eps: f64) -> (usize, usize) {
    let n = pts.len();
    let t = NetTree::build(pts, 2.0).deletion_times(n, eps);
    let mut sparse = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i].distance(&pts[j]);
            let a = relaxed_entry_time(d, t[i], t[j], eps);
            // Sheehy's open net: the edge exists only while BOTH endpoints live.
            if a < t[i].min(t[j]) {
                sparse += 1;
            }
        }
    }
    (sparse, n * (n - 1) / 2)
}

/// Least-squares slope of log(count) against log(n).
fn exponent(points: &[(f64, f64)]) -> f64 {
    let k = points.len() as f64;
    let (mx, my) = (
        points.iter().map(|p| p.0.ln()).sum::<f64>() / k,
        points.iter().map(|p| p.1.ln()).sum::<f64>() / k,
    );
    let num: f64 = points.iter().map(|p| (p.0.ln() - mx) * (p.1.ln() - my)).sum();
    let den: f64 = points.iter().map(|p| (p.0.ln() - mx).powi(2)).sum();
    num / den
}

#[test]
fn sparse_edge_count_grows_linearly_where_dense_grows_quadratically() {
    for &eps in &[0.1f64, 0.2, 1.0 / 3.0] {
        println!("=== eps = {eps:.4}, circle ===");
        println!("{:>6} {:>10} {:>10} {:>10}", "n", "sparse", "dense", "ratio");
        let mut sp = Vec::new();
        let mut dn = Vec::new();
        for &n in &[64usize, 128, 256, 512, 1024] {
            let (s, d) = edge_counts(&circle(n, 1.0), eps);
            println!("{n:>6} {s:>10} {d:>10} {:>10.4}", s as f64 / d as f64);
            sp.push((n as f64, s.max(1) as f64));
            dn.push((n as f64, d as f64));
        }
        let (es, ed) = (exponent(&sp), exponent(&dn));
        println!("  fitted exponent: sparse {es:.3}, dense {ed:.3}");
        assert!((ed - 2.0).abs() < 0.05, "dense edges must grow like n^2, got n^{ed:.3}");
        assert!(es < 1.35,
            "eps={eps}: sparse edges grew like n^{es:.3}; Theorem 9.3 claims linear");
        println!();
    }
}

#[test]
fn the_same_holds_on_a_two_dimensional_sample() {
    // A circle has intrinsic dimension 1. The sphere is 2, where the constant
    // (1/eps)^O(kd) is larger and the linear claim is harder to meet.
    let eps = 1.0 / 3.0;
    println!("=== eps = {eps:.4}, sphere S2 ===");
    println!("{:>6} {:>10} {:>10} {:>10}", "n", "sparse", "dense", "ratio");
    let mut sp = Vec::new();
    for &n in &[64usize, 128, 256, 512] {
        let (s, d) = edge_counts(&sphere(n), eps);
        println!("{n:>6} {s:>10} {d:>10} {:>10.4}", s as f64 / d as f64);
        sp.push((n as f64, s.max(1) as f64));
    }
    let es = exponent(&sp);
    println!("  fitted exponent: sparse {es:.3}");
    assert!(es < 1.6, "sphere: sparse edges grew like n^{es:.3}");
}
