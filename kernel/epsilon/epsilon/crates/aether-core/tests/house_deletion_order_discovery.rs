//! Neighbour discovery ordered by deletion time rather than by proximity.
//!
//! E(p) = {q : t_q > t_p, entry(p,q) < t_p}. The selectivity is the deletion
//! ordering, not the metric: at eps = 1/3 over 40% of points have t_p above the
//! sphere's own diameter, and at eps = 0.05 every point does, so a distance
//! range query returns everything.
//!
//! Processing points in DECREASING t_p makes the two effects complementary: a
//! large t_p has a ball covering everything but only a few longer-lived points
//! to compare against, and a small t_p has many candidates but a tight ball.
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::{relaxed_entry_time, NetTree};
use std::time::Instant;

fn sphere(n: usize) -> Vec<ManifoldPoint<3>> {
    let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64) / ((n.max(2) - 1) as f64);
            let r = (1.0 - y * y).max(0.0).sqrt();
            let t = ga * (i as f64);
            ManifoldPoint::new([t.cos() * r, y, t.sin() * r])
        })
        .collect()
}

/// Baseline: every pair.
fn quadratic(pts: &[ManifoldPoint<3>], t: &[f64], eps: f64) -> usize {
    let n = pts.len();
    let mut c = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i].distance(&pts[j]);
            let lim = t[i].min(t[j]);
            if d < lim && relaxed_entry_time(d, t[i], t[j], eps) < lim {
                c += 1;
            }
        }
    }
    c
}

/// Deletion-ordered: for each p in decreasing t_p, compare only against the
/// longer-lived points already seen.
fn deletion_ordered(pts: &[ManifoldPoint<3>], t: &[f64], eps: f64) -> usize {
    let n = pts.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| t[b].total_cmp(&t[a])); // decreasing t
    let mut seen: Vec<usize> = Vec::with_capacity(n);
    let mut c = 0;
    for &p in &order {
        for &q in &seen {
            // q was seen first, so t_q >= t_p: the limit is t_p.
            let d = pts[p].distance(&pts[q]);
            if d < t[p] && relaxed_entry_time(d, t[p], t[q], eps) < t[p] {
                c += 1;
            }
        }
        seen.push(p);
    }
    c
}

#[test]
fn deletion_ordered_discovery_matches_the_quadratic_scan() {
    let eps = 1.0 / 3.0;
    println!(
        "{:>6} {:>12} {:>12} {:>10} {:>12}",
        "n", "quadratic", "ordered", "speedup", "counts agree"
    );
    for &n in &[256usize, 512, 1024, 2048, 4096] {
        let pts = sphere(n);
        let t = NetTree::build(&pts, 2.0).deletion_times(n, eps);

        let a = Instant::now();
        let q = quadratic(&pts, &t, eps);
        let q_ms = a.elapsed().as_secs_f64() * 1e3;

        let b = Instant::now();
        let o = deletion_ordered(&pts, &t, eps);
        let o_ms = b.elapsed().as_secs_f64() * 1e3;

        println!(
            "{n:>6} {q_ms:>12.2} {o_ms:>12.2} {:>9.2}x {:>12}",
            q_ms / o_ms.max(1e-9),
            q == o
        );
        assert_eq!(
            q, o,
            "n={n}: deletion-ordered discovery found {o} edges, scan found {q}"
        );
    }
}
