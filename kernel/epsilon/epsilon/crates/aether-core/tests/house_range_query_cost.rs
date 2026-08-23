//! Does the range query prune, or does it merely reach the right answer by
//! examining everything? Measured as wall clock for FULL edge discovery:
//! every point queried at its own deletion radius, versus the O(n^2) scan.
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::NetTree;
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

#[test]
fn does_the_range_query_prune() {
    let eps = 1.0 / 3.0;
    println!("Full neighbour discovery: every p queried at radius t_p.");
    println!(
        "{:>6} {:>12} {:>12} {:>10} {:>14}",
        "n", "scan (ms)", "query (ms)", "speedup", "hits agree"
    );
    for &n in &[128usize, 256, 512, 1024, 2048] {
        let pts = sphere(n);
        let tree = NetTree::build(&pts, 2.0);
        let t = tree.deletion_times(n, eps);
        let cover = tree.covering(&pts);

        let t0 = Instant::now();
        let mut scan_total = 0usize;
        for q in 0..n {
            scan_total += (0..n)
                .filter(|&p| p != q && pts[q].distance(&pts[p]) <= t[q])
                .count();
        }
        let scan_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let mut q_total = 0usize;
        for (q, &radius) in t.iter().enumerate().take(n) {
            q_total += tree.range_query(&pts, &cover, q, radius).len();
        }
        let q_ms = t1.elapsed().as_secs_f64() * 1e3;

        println!(
            "{n:>6} {scan_ms:>12.2} {q_ms:>12.2} {:>10.2}x {:>14}",
            scan_ms / q_ms.max(1e-9),
            scan_total == q_total
        );
        assert_eq!(
            scan_total, q_total,
            "n={n}: hit counts differ, the query is not exact"
        );
    }
}
