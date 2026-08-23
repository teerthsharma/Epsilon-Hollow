//! Is MAX |E(p)| bounded or growing? 63 -> 93 over 16x in n does not settle it.
//! Extending the range does. Also measures the separation half of Lemma 9.2 on
//! the CORRECTED set — E(p) = neighbours that outlive p — since iteration 27's
//! figures were computed over the undirected neighbourhood.
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

struct R {
    max_e: usize,
    mean_e: f64,
    min_sep_ratio: f64,
    worst_reach: f64,
}

fn measure(pts: &[ManifoldPoint<3>], eps: f64) -> R {
    let n = pts.len();
    let t = NetTree::build(pts, 2.0).deletion_times(n, eps);
    // E(p): neighbours q with t_q > t_p, edge present.
    let mut e: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i].distance(&pts[j]);
            let lim = t[i].min(t[j]);
            if d >= lim {
                continue;
            }
            if relaxed_entry_time(d, t[i], t[j], eps) >= lim {
                continue;
            }
            if t[i] < t[j] {
                e[i].push(j);
            } else {
                e[j].push(i);
            }
        }
    }
    // Separation within E(p), relative to t_p. Sheehy needs this bounded BELOW
    // by K_p*eps*(1-2eps); at eps=1/3 that is K_p/9, so ~0.11 for K_p ~ 1.
    let mut min_sep = f64::INFINITY;
    let mut worst_reach = 0.0f64;
    for (p, list) in e.iter().enumerate() {
        if t[p] <= 0.0 {
            continue;
        }
        for &q in list {
            let r = pts[p].distance(&pts[q]) / t[p];
            if r > worst_reach {
                worst_reach = r;
            }
        }
        if list.len() < 2 {
            continue;
        }
        let mut m = f64::INFINITY;
        for a in 0..list.len() {
            for b in (a + 1)..list.len() {
                let d = pts[list[a]].distance(&pts[list[b]]);
                if d < m {
                    m = d;
                }
            }
        }
        let r = m / t[p];
        if r < min_sep {
            min_sep = r;
        }
    }
    let tot: usize = e.iter().map(|v| v.len()).sum();
    R {
        max_e: e.iter().map(|v| v.len()).max().unwrap_or(0),
        mean_e: tot as f64 / n as f64,
        min_sep_ratio: min_sep,
        worst_reach,
    }
}

#[test]
fn does_max_e_of_p_flatten() {
    let eps = 1.0 / 3.0;
    println!("=== sphere S2, eps = {eps:.4}. E(p) = neighbours that OUTLIVE p. ===");
    println!(
        "Sheehy 9.2: d(p,q) <= t_p for q in E(p), and E(p) separated by >= K*eps*(1-2eps)*t_p."
    );
    println!("At eps=1/3 that separation floor is K/9 ~ 0.11 for K ~ 1.");
    println!();
    println!(
        "{:>6} {:>10} {:>10} {:>14} {:>14}",
        "n", "MAX E(p)", "mean E(p)", "max d/t_p", "min sep/t_p"
    );
    let mut rows = Vec::new();
    for &n in &[256usize, 512, 1024, 2048, 4096, 8192] {
        let r = measure(&sphere(n), eps);
        println!(
            "{n:>6} {:>10} {:>10.2} {:>14.4} {:>14.5}",
            r.max_e, r.mean_e, r.worst_reach, r.min_sep_ratio
        );
        rows.push((n as f64, r.max_e as f64));
    }
    println!();
    for w in rows.windows(2) {
        println!(
            "  n {:>5.0} -> {:>5.0}: MAX E(p) x{:.3}",
            w[0].0,
            w[1].0,
            w[1].1 / w[0].1
        );
    }
    let first = rows[0].1;
    let last = rows[rows.len() - 1].1;
    println!();
    println!(
        "MAX E(p) over the full 32x range in n: {first} -> {last}, x{:.3}",
        last / first
    );
    // Report, do not assert a verdict. The point of this iteration is the number.
    assert!(
        last >= first,
        "MAX E(p) fell, which would be a different finding entirely"
    );
}
