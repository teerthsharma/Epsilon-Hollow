//! The range query must return EXACTLY what a linear scan returns. It is an
//! oracle the fast path must equal, never beat — the gate design `foliation`
//! already uses against Belady.
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::NetTree;

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

fn linear_scan<const D: usize>(pts: &[ManifoldPoint<D>], q: usize, r: f64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..pts.len())
        .filter(|&p| p != q && pts[q].distance(&pts[p]) <= r)
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn range_query_equals_linear_scan_exactly() {
    for &n in &[32usize, 64, 128, 256] {
        let pts = sphere(n);
        let tree = NetTree::build(&pts, 2.0);
        let cover = tree.covering(&pts);
        let mut checked = 0usize;
        for q in (0..n).step_by(if n > 64 { 7 } else { 1 }) {
            for &r in &[0.05f64, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0] {
                let fast = tree.range_query(&pts, &cover, q, r);
                let slow = linear_scan(&pts, q, r);
                assert_eq!(
                    fast,
                    slow,
                    "n={n} q={q} r={r}: range query returned {} points, scan returned {}",
                    fast.len(),
                    slow.len()
                );
                checked += 1;
            }
        }
        println!("n={n:4}: {checked} (query, radius) pairs agree exactly");
    }
}

#[test]
fn degenerate_inputs_agree_too() {
    // Duplicates, a two-point set, and a radius of zero.
    let dup = vec![ManifoldPoint::<3>::new([1.0, 0.0, 0.0]); 8];
    let tree = NetTree::build(&dup, 2.0);
    let cover = tree.covering(&dup);
    for r in [0.0f64, 1e-9, 1.0] {
        assert_eq!(
            tree.range_query(&dup, &cover, 0, r),
            linear_scan(&dup, 0, r),
            "duplicates disagree at radius {r}"
        );
    }
    let two = sphere(2);
    let t2 = NetTree::build(&two, 2.0);
    let c2 = t2.covering(&two);
    for r in [0.0f64, 0.5, 5.0] {
        assert_eq!(
            t2.range_query(&two, &c2, 0, r),
            linear_scan(&two, 0, r),
            "two-point set disagrees at radius {r}"
        );
    }
}
