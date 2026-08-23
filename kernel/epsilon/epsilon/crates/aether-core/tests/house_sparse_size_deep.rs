//! Settling iteration 22's two open questions with more evidence:
//!   (a) is the sphere's 1.286 exponent an asymptotic artefact?
//!   (b) does the linear claim survive at k = 2, where Theorem 9.3 lives?
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

fn circle(n: usize) -> Vec<ManifoldPoint<2>> {
    (0..n).map(|i| {
        let t = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
        ManifoldPoint::new([t.cos(), t.sin()])
    }).collect()
}

/// Entry times for every pair, plus deletion times.
fn tables<const D: usize>(pts: &[ManifoldPoint<D>], eps: f64) -> (Vec<f64>, Vec<f64>) {
    let n = pts.len();
    let t = NetTree::build(pts, 2.0).deletion_times(n, eps);
    let mut a = vec![f64::INFINITY; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let e = relaxed_entry_time(pts[i].distance(&pts[j]), t[i], t[j], eps);
            a[i * n + j] = e;
            a[j * n + i] = e;
        }
    }
    (a, t)
}

fn sparse_edges(a: &[f64], t: &[f64], n: usize) -> usize {
    let mut c = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if a[i * n + j] < t[i].min(t[j]) { c += 1; }
        }
    }
    c
}

/// A triangle enters at the max of its edge entry times and needs all three
/// vertices alive then.
fn sparse_triangles(a: &[f64], t: &[f64], n: usize) -> usize {
    let mut c = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let aij = a[i * n + j];
            if aij >= t[i].min(t[j]) { continue; }
            for k in (j + 1)..n {
                let e = aij.max(a[i * n + k]).max(a[j * n + k]);
                if e < t[i].min(t[j]).min(t[k]) { c += 1; }
            }
        }
    }
    c
}

fn exponent(p: &[(f64, f64)]) -> f64 {
    let k = p.len() as f64;
    let mx = p.iter().map(|q| q.0.ln()).sum::<f64>() / k;
    let my = p.iter().map(|q| q.1.ln()).sum::<f64>() / k;
    let num: f64 = p.iter().map(|q| (q.0.ln() - mx) * (q.1.ln() - my)).sum();
    let den: f64 = p.iter().map(|q| (q.0.ln() - mx).powi(2)).sum();
    num / den
}

#[test]
fn question_a_does_the_sphere_exponent_fall_with_n() {
    let eps = 1.0 / 3.0;
    println!("=== sphere S2, eps = {eps:.4} ===");
    println!("{:>6} {:>12} {:>12}", "n", "sparse E", "E/n");
    let mut pts = Vec::new();
    for &n in &[64usize, 128, 256, 512, 1024, 2048] {
        let p = sphere(n);
        let (a, t) = tables(&p, eps);
        let e = sparse_edges(&a, &t, n);
        println!("{n:>6} {e:>12} {:>12.3}", e as f64 / n as f64);
        pts.push((n as f64, e.max(1) as f64));
    }
    let all = exponent(&pts);
    let tail = exponent(&pts[pts.len() - 3..]);
    println!("  exponent over all n: {all:.3}");
    println!("  exponent over the last three: {tail:.3}");
    // The question is whether the TAIL is closer to 1 than the whole range.
    // Report either way; assert only that it does not diverge upward.
    assert!(tail <= all + 0.05,
        "the tail exponent {tail:.3} exceeds the full-range {all:.3}: not converging");
}

#[test]
fn question_b_does_the_linear_claim_survive_at_k_equals_two() {
    let eps = 1.0 / 3.0;
    for (name, build) in [("circle", 0usize), ("sphere", 1usize)] {
        println!("=== {name}, triangles, eps = {eps:.4} ===");
        println!("{:>6} {:>12} {:>12} {:>12}", "n", "sparse tri", "dense tri", "tri/n");
        let mut sp = Vec::new();
        for &n in &[48usize, 64, 96, 128, 192] {
            let (a, t) = if build == 0 {
                let p = circle(n); tables(&p, eps)
            } else {
                let p = sphere(n); tables(&p, eps)
            };
            let tri = sparse_triangles(&a, &t, n);
            let dense = n * (n - 1) * (n - 2) / 6;
            println!("{n:>6} {tri:>12} {dense:>12} {:>12.2}", tri as f64 / n as f64);
            sp.push((n as f64, tri.max(1) as f64));
        }
        println!("  fitted exponent: {:.3}", exponent(&sp));
        println!();
    }
}
