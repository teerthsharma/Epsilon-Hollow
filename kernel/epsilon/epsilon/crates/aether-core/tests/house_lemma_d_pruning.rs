//! How much does Lemma D actually prune, and is the filter exact?
//!
//! Lemma D says the entry time is never below the true distance, so a pair can
//! appear in a filtration truncated at `alpha_max` only when `d <= alpha_max`.
//! Two separate things follow and only one of them is a theorem:
//!
//! * **exactness** - the filter drops no admitted pair. That is proved, and the
//!   first test here is the executable form of it.
//! * **usefulness** - the filter drops *many* pairs. That is an empirical claim
//!   about this fixture at this `alpha_max`, and the second test measures it
//!   rather than asserting it.
//!
//! Reporting the second without the first would be the characteristic defect
//! this repository keeps finding in itself: a hypothesis that is not
//! load-bearing. A filter that prunes nothing is exact too.

use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::{relaxed_entry_time_exact, NetTree};

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

/// The theorem. Every pair admitted at or below `alpha_max` satisfies
/// `d <= alpha_max`, so screening candidates by distance alone loses nothing.
///
/// A counterexample here would mean the fixed-radius reformulation of edge
/// discovery is unsound, not merely unhelpful.
#[test]
fn lemma_d_filter_never_drops_an_admitted_pair() {
    let mut admitted_total = 0usize;
    for &eps in &[1.0 / 3.0, 0.1] {
        for &n in &[128usize, 512] {
            let pts = sphere(n);
            let t = NetTree::build(&pts, 2.0).deletion_times(n, eps);
            for &alpha_max in &[0.05, 0.2, 0.75, 2.0] {
                for i in 0..n {
                    for j in (i + 1)..n {
                        let d = pts[i].distance(&pts[j]);
                        let entry = relaxed_entry_time_exact(d, t[i], t[j], eps);
                        if entry <= alpha_max {
                            admitted_total += 1;
                            assert!(
                                d <= alpha_max + 1e-12,
                                "pair ({i},{j}) is admitted at {entry} <= {alpha_max} \
                                 but its true distance is {d}, so the fixed-radius \
                                 filter would have dropped it (n={n} eps={eps})"
                            );
                        }
                    }
                }
            }
        }
    }
    println!("Lemma D checked against {admitted_total} admitted pairs");
    assert!(
        admitted_total > 1000,
        "only {admitted_total} pairs were admitted anywhere in the sweep, so this \
         test proves almost nothing; widen alpha_max"
    );
}

/// The measurement. Exactness is free; selectivity is not, and it decides
/// whether building a spatial index over `alpha_max` is worth anything.
///
/// Reported, not asserted, except for the one outcome that would make the whole
/// reformulation pointless: the filter must actually remove pairs.
#[test]
fn how_much_does_the_lemma_d_filter_remove() {
    println!(
        "{:>6} {:>8} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "n", "eps", "alpha_max", "pairs", "d<=a_max", "admitted", "kept %"
    );
    let mut worst_kept = 0.0f64;
    for &eps in &[1.0 / 3.0, 0.1] {
        for &n in &[256usize, 1024] {
            let pts = sphere(n);
            let t = NetTree::build(&pts, 2.0).deletion_times(n, eps);
            for &alpha_max in &[0.05, 0.2, 0.75] {
                let total = n * (n - 1) / 2;
                let mut within = 0usize;
                let mut admitted = 0usize;
                for i in 0..n {
                    for j in (i + 1)..n {
                        let d = pts[i].distance(&pts[j]);
                        if d <= alpha_max {
                            within += 1;
                        }
                        if relaxed_entry_time_exact(d, t[i], t[j], eps) <= alpha_max {
                            admitted += 1;
                        }
                    }
                }
                let kept = 100.0 * within as f64 / total as f64;
                worst_kept = worst_kept.max(kept);
                println!(
                    "{n:>6} {eps:>8.4} {alpha_max:>10.2} {total:>12} {within:>12} \
                     {admitted:>10} {kept:>9.2}%"
                );
            }
        }
    }
    println!();
    println!("`kept %` is the share of all pairs a fixed-radius index would still");
    println!("have to examine. The gap between that column and `admitted` is the");
    println!("work the filter cannot remove and the entry-time test still must do.");
    assert!(
        worst_kept < 100.0,
        "the distance filter kept every pair at every setting, so it removes \
         nothing and the fixed-radius reformulation buys nothing"
    );
}
