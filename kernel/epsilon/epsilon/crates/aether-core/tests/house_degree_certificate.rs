//! Dr House's decisive measurement. Sheehy section 9.2 bounds |E(p)| by a
//! constant via two per-point facts, both decidable NOW with no asymptotics:
//!   (i)  every q in E(p) lies within 2*t_p of p;
//!   (ii) E(p) is pairwise separated, so a packing bound caps its cardinality.
//!
//! Fitting an exponent to six points cannot separate "bounded degree with a
//! large constant" from "degree growing with n". MAX degree can.
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

struct Report {
    mean_deg: f64,
    max_deg: usize,
    worst_reach_ratio: f64, // max over edges of d(p,q) / (2*max(t_p,t_q))
    min_sep_ratio: f64,     // min over points of (min pairwise sep in E(p)) / t_p
}

fn certify(pts: &[ManifoldPoint<3>], eps: f64, t_scale: f64) -> Report {
    let n = pts.len();
    let mut t = NetTree::build(pts, 2.0).deletion_times(n, eps);
    for v in t.iter_mut() {
        *v *= t_scale;
    } // t_scale = 1.0 normally; 2.0 is the misfire control

    let mut nbr: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut worst_reach = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = pts[i].distance(&pts[j]);
            let lim = t[i].min(t[j]);
            if d >= lim {
                continue;
            }
            if relaxed_entry_time(d, t[i], t[j], eps) < lim {
                nbr[i].push(j);
                nbr[j].push(i);
                let reach = d / (2.0 * t[i].max(t[j]));
                if reach > worst_reach {
                    worst_reach = reach;
                }
            }
        }
    }

    let degs: Vec<usize> = nbr.iter().map(|v| v.len()).collect();
    let mut min_sep_ratio = f64::INFINITY;
    for (p, list) in nbr.iter().enumerate() {
        if list.len() < 2 || t[p] <= 0.0 {
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
        if r < min_sep_ratio {
            min_sep_ratio = r;
        }
    }

    Report {
        mean_deg: degs.iter().sum::<usize>() as f64 / n as f64,
        max_deg: degs.iter().copied().max().unwrap_or(0),
        worst_reach_ratio: worst_reach,
        min_sep_ratio,
    }
}

#[test]
fn max_degree_separates_bounded_from_growing() {
    let eps = 1.0 / 3.0;
    println!("=== sphere S2, eps = {eps:.4}. Sheehy 9.2 predicts MAX degree bounded. ===");
    println!(
        "{:>6} {:>10} {:>10} {:>14} {:>14}",
        "n", "mean deg", "MAX deg", "reach ratio", "sep/t_p"
    );
    let mut maxes = Vec::new();
    for &n in &[64usize, 128, 256, 512, 1024, 2048] {
        let r = certify(&sphere(n), eps, 1.0);
        println!(
            "{n:>6} {:>10.2} {:>10} {:>14.4} {:>14.5}",
            r.mean_deg, r.max_deg, r.worst_reach_ratio, r.min_sep_ratio
        );
        // The containment ratio is REPORTED, never asserted. An edge is
        // accepted only when d < min(t_p,t_q) <= max(t_p,t_q), so
        // d / (2 max(t_p,t_q)) < 1/2 by construction. Asserting it would be an
        // identity — see `the_containment_ratio_is_bounded_by_construction`.
        maxes.push((n, r.max_deg));
    }
    let first = maxes[1].1 as f64;
    let last = maxes[maxes.len() - 1].1 as f64;
    println!();
    println!(
        "MAX degree n=128 -> n=2048 (16x more points): {first} -> {last}, ratio {:.3}",
        last / first
    );
}

#[test]
fn the_containment_ratio_is_bounded_by_construction_so_it_cannot_be_a_check() {
    // Iteration 27 asserted "every accepted edge lies within 2*t_p" and treated
    // it as a certificate. It is an identity. The acceptance predicate is
    // `d < min(t_p, t_q)`, and min <= max, so
    //
    //     d / (2 * max(t_p, t_q))  <  min / (2 * max)  <=  1/2
    //
    // for every accepted edge, at every n, for any deletion times whatsoever.
    // Its required-misfire control — inflating all t_p by 2x — did not trip it,
    // which is how the vacuity surfaced.
    //
    // This test pins the reason, so the identity is not re-added as a check.
    let eps = 1.0 / 3.0;
    for scale in [0.5f64, 1.0, 2.0, 10.0] {
        let r = certify(&sphere(256), eps, scale);
        println!(
            "t_p scaled {scale:>5}: reach ratio {:.6} (structurally < 0.5)",
            r.worst_reach_ratio
        );
        assert!(
            r.worst_reach_ratio < 0.5,
            "scale {scale}: ratio {:.6} reached 0.5, which the acceptance predicate forbids",
            r.worst_reach_ratio
        );
    }
    // The ratio DOES vary slightly with the deletion times — scaling t_p
    // changes which edges are accepted, so the maximum is taken over a
    // different set (0.166650 at scale 1, 0.166611 at scale 2, 0.088541 at
    // scale 10). An earlier form of this test asserted exact invariance and was
    // wrong. What makes the original check vacuous is not invariance but the
    // structural ceiling: no deletion times whatsoever can push it to 1.0, so a
    // check against 1.0 has no failing input.
    let doubled = certify(&sphere(256), eps, 2.0).worst_reach_ratio;
    let tenfold = certify(&sphere(256), eps, 10.0).worst_reach_ratio;
    assert!(doubled < 0.5 && tenfold < 0.5,
        "a 2x or 10x change in every deletion time pushed the ratio to          {doubled:.6} / {tenfold:.6}; the structural ceiling of 0.5 does not hold          and the vacuity finding needs revisiting");
}
