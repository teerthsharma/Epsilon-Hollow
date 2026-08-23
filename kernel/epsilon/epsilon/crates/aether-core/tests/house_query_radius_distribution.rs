//! Why does a spatial index fail here? Measure the query radius against the
//! data's own diameter. A range query whose radius exceeds the diameter must
//! return everything, and no index beats a scan at that.
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::NetTree;

fn sphere(n: usize) -> Vec<ManifoldPoint<3>> {
    let ga = core::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n).map(|i| {
        let y = 1.0 - 2.0 * (i as f64) / ((n.max(2) - 1) as f64);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let t = ga * (i as f64);
        ManifoldPoint::new([t.cos() * r, y, t.sin() * r])
    }).collect()
}

#[test]
fn how_large_is_the_query_radius_relative_to_the_data() {
    println!("Chordal diameter of the unit sphere is 2.0.");
    println!("{:>6} {:>8} {:>10} {:>10} {:>10} {:>14}", "n", "eps", "min t_p", "median", "max t_p", "% t_p >= 2.0");
    for &eps in &[1.0 / 3.0, 0.1, 0.05] {
        for &n in &[256usize, 1024, 4096] {
            let pts = sphere(n);
            let mut t = NetTree::build(&pts, 2.0).deletion_times(n, eps);
            let over = t.iter().filter(|&&x| x >= 2.0).count();
            t.sort_by(f64::total_cmp);
            println!("{n:>6} {eps:>8.4} {:>10.4} {:>10.4} {:>10.4} {:>13.1}%",
                t[0], t[t.len() / 2], t[t.len() - 1],
                100.0 * over as f64 / n as f64);
        }
    }
    println!();
    println!("A query radius at or above 2.0 covers the whole sphere: the answer");
    println!("is every point, and the selectivity of E(p) comes from the");
    println!("t_q > t_p and entry-time conditions, not from proximity.");
}
