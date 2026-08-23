//! Sheehy's Lemma 9.2 bounds |E(p)| where E(p) is the set of neighbours of p in
//! Q_{t_p} — the complex at p's OWN deletion time. Since Q_alpha lives on
//! N_alpha = {r : t_r > alpha}, membership requires t_q > t_p.
//!
//! E(p) therefore contains only neighbours that OUTLIVE p. Each edge is charged
//! once, to its shorter-lived endpoint. Iteration 27 measured the undirected
//! degree, which counts both directions and is not what the theorem bounds.
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::{relaxed_entry_time, NetTree};

fn sphere(n: usize) -> Vec<ManifoldPoint<3>> {
    let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n).map(|i| {
        let y = 1.0 - 2.0 * (i as f64) / ((n.max(2) - 1) as f64);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let t = ga * (i as f64);
        ManifoldPoint::new([t.cos() * r, y, t.sin() * r])
    }).collect()
}

/// (max undirected degree, max |E(p)| as Sheehy defines it, mean |E(p)|)
fn degrees(pts: &[ManifoldPoint<3>], eps: f64) -> (usize, usize, f64) {
    let n = pts.len();
    let t = NetTree::build(pts, 2.0).deletion_times(n, eps);
    let mut undirected = vec![0usize; n];
    let mut sheehy = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i].distance(&pts[j]);
            let lim = t[i].min(t[j]);
            if d >= lim { continue; }
            if relaxed_entry_time(d, t[i], t[j], eps) >= lim { continue; }
            undirected[i] += 1;
            undirected[j] += 1;
            // Charge the edge to the SHORTER-lived endpoint: q must outlive p.
            if t[i] < t[j] { sheehy[i] += 1; } else if t[j] < t[i] { sheehy[j] += 1; }
            else { sheehy[i] += 1; }   // tie: charge one side, arbitrarily but consistently
        }
    }
    (
        undirected.iter().copied().max().unwrap_or(0),
        sheehy.iter().copied().max().unwrap_or(0),
        sheehy.iter().sum::<usize>() as f64 / n as f64,
    )
}

#[test]
fn sheehy_degree_versus_undirected_degree() {
    let eps = 1.0 / 3.0;
    println!("=== sphere S2, eps = {eps:.4} ===");
    println!("{:>6} {:>16} {:>16} {:>16}", "n", "MAX undirected", "MAX |E(p)|", "mean |E(p)|");
    let mut sh = Vec::new();
    for &n in &[64usize, 128, 256, 512, 1024, 2048] {
        let (u, s, m) = degrees(&sphere(n), eps);
        println!("{n:>6} {u:>16} {s:>16} {m:>16.2}");
        sh.push((n, s));
    }
    println!();
    let a = sh[1].1 as f64;
    let b = sh[sh.len() - 1].1 as f64;
    println!("MAX |E(p)| from n=128 to n=2048 (16x more points): {a} -> {b}, ratio {:.3}", b / a);
    println!("(iteration 27 measured the undirected column: 127 -> 513, ratio 4.039)");
}
