//! Iteration 23 stalled at n=192 because triangles were enumerated over all
//! C(n,3) candidates. A sparse triangle's entry time is the max of its three
//! edge entry times, and it needs `max < min(t_i,t_j,t_k)`. That forces each
//! edge's own entry time below the same bound, so EVERY edge of a sparse
//! triangle is itself a sparse edge. Enumerating from the sparse adjacency is
//! therefore exact, not an approximation.
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::{relaxed_entry_time, NetTree};

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

/// Sparse adjacency: `adj[i]` lists j > i with the edge in the filtration,
/// paired with its entry time.
fn sparse_adjacency<const D: usize>(
    pts: &[ManifoldPoint<D>],
    eps: f64,
) -> (Vec<Vec<(usize, f64)>>, Vec<f64>) {
    let n = pts.len();
    let t = NetTree::build(pts, 2.0).deletion_times(n, eps);
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i].distance(&pts[j]);
            let lim = t[i].min(t[j]);
            // d_alpha >= d, so entry >= d: a cheap exact prefilter that skips
            // the bisection for pairs that cannot possibly qualify.
            if d >= lim {
                continue;
            }
            let a = relaxed_entry_time(d, t[i], t[j], eps);
            if a < lim {
                adj[i].push((j, a));
            }
        }
    }
    (adj, t)
}

fn count_triangles(adj: &[Vec<(usize, f64)>], t: &[f64]) -> usize {
    let n = adj.len();
    let mut entry = vec![f64::INFINITY; n * n];
    for (i, row) in adj.iter().enumerate() {
        for &(j, a) in row {
            entry[i * n + j] = a;
            entry[j * n + i] = a;
        }
    }
    let mut c = 0usize;
    for i in 0..n {
        for &(j, aij) in &adj[i] {
            for &(k, aik) in &adj[i] {
                if k <= j {
                    continue;
                }
                let ajk = entry[j * n + k];
                if !ajk.is_finite() {
                    continue;
                }
                let e = aij.max(aik).max(ajk);
                if e < t[i].min(t[j]).min(t[k]) {
                    c += 1;
                }
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
fn sphere_triangles_at_scale() {
    let eps = 1.0 / 3.0;
    println!("=== sphere S2 triangles, eps = {eps:.4} ===");
    println!(
        "{:>6} {:>12} {:>12} {:>12}",
        "n", "sparse tri", "tri/n", "edges/n"
    );
    let mut tri = Vec::new();
    for &n in &[64usize, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let p = sphere(n);
        let (adj, t) = sparse_adjacency(&p, eps);
        let e: usize = adj.iter().map(|r| r.len()).sum();
        let c = count_triangles(&adj, &t);
        println!(
            "{n:>6} {c:>12} {:>12.2} {:>12.2}",
            c as f64 / n as f64,
            e as f64 / n as f64
        );
        tri.push((n as f64, c.max(1) as f64));
    }
    let all = exponent(&tri);
    let tail = exponent(&tri[tri.len() - 3..]);
    println!("  exponent over all n: {all:.3}");
    println!("  exponent over the last three: {tail:.3}");
    assert!(
        tail <= all + 0.05,
        "tail exponent {tail:.3} exceeds full-range {all:.3}: diverging, not converging"
    );
}

#[test]
fn the_adjacency_reformulation_equals_brute_force() {
    // The reformulation is justified by an argument: every edge of a sparse
    // triangle is itself a sparse edge. An argument is not a check. This
    // compares it against the O(n^3) enumeration over ALL C(n,3) candidates,
    // wherever brute force is still affordable.
    let eps = 1.0 / 3.0;
    for &n in &[32usize, 48, 64, 96, 128] {
        let pts = sphere(n);
        let t = NetTree::build(&pts, 2.0).deletion_times(n, eps);

        // Brute force: every pair, every triple, no pruning at all.
        let mut a = vec![f64::INFINITY; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let e = relaxed_entry_time(pts[i].distance(&pts[j]), t[i], t[j], eps);
                a[i * n + j] = e;
                a[j * n + i] = e;
            }
        }
        let mut brute = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let e = a[i * n + j].max(a[i * n + k]).max(a[j * n + k]);
                    if e < t[i].min(t[j]).min(t[k]) {
                        brute += 1;
                    }
                }
            }
        }

        let (adj, t2) = sparse_adjacency(&pts, eps);
        let fast = count_triangles(&adj, &t2);
        println!("n={n:4}: brute {brute:8}  adjacency {fast:8}");
        assert_eq!(
            brute, fast,
            "n={n}: the adjacency reformulation disagrees with brute force"
        );
    }
}
