//! Iteration 7 — the two invariants a net-tree must satisfy, per level.
//! Written before the implementation. Gate: one violation at any level fails.
use aether_core::manifold::ManifoldPoint;
use aether_core::nettree::NetTree;

/// Fibonacci sphere sample — a well-spread S2 point set.
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

/// Deliberately clustered: three tight clumps. Stresses packing.
fn clumps(per: usize) -> Vec<ManifoldPoint<3>> {
    let centres = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut s: u64 = 0xABCDEF0123456789;
    let mut nx = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let mut v = Vec::new();
    for c in centres {
        for _ in 0..per {
            let p = [c[0] + 0.02 * nx(), c[1] + 0.02 * nx(), c[2] + 0.02 * nx()];
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            v.push(ManifoldPoint::new([p[0] / n, p[1] / n, p[2] / n]));
        }
    }
    v
}

fn check(points: &[ManifoldPoint<3>], label: &str) {
    let tree = NetTree::build(points, 2.0);
    println!("{label}: n={} levels={}", points.len(), tree.levels());
    assert!(
        tree.levels() >= 1,
        "{label}: net-tree must have at least one level"
    );

    for l in 0..tree.levels() {
        let net = tree.net(l);
        let r = tree.radius(l);
        assert!(!net.is_empty(), "{label} level {l}: empty net");

        // PACKING: distinct net points at this level are more than r apart.
        for (a, &i) in net.iter().enumerate() {
            for &j in net.iter().skip(a + 1) {
                let d = points[i].distance(&points[j]);
                assert!(
                    d > r,
                    "{label} level {l}: PACKING violated, d({i},{j})={d:.6} <= r={r:.6}"
                );
            }
        }

        // COVERING: every point of the previous level is within r of this net.
        let prev: Vec<usize> = if l == 0 {
            (0..points.len()).collect()
        } else {
            tree.net(l - 1).to_vec()
        };
        for &p in &prev {
            let best = net
                .iter()
                .map(|&q| points[p].distance(&points[q]))
                .fold(f64::INFINITY, f64::min);
            assert!(best <= r,
                "{label} level {l}: COVERING violated, point {p} is {best:.6} from the net, r={r:.6}");
        }
        println!(
            "  level {l}: |net|={:4} r={:.5}  packing OK  covering OK",
            net.len(),
            r
        );
    }

    // Top level must be a single point: the hierarchy terminates.
    assert_eq!(
        tree.net(tree.levels() - 1).len(),
        1,
        "{label}: top level must be a single root"
    );
}

#[test]
fn net_tree_invariants_on_a_sphere_sample() {
    check(&sphere(200), "sphere-200");
}

#[test]
fn net_tree_invariants_on_clustered_data() {
    check(&clumps(40), "clumps-120");
}

#[test]
fn net_tree_invariants_on_degenerate_inputs() {
    check(&sphere(2), "sphere-2");
    // exact duplicates: packing must still hold, duplicates collapse into one net point
    let dup = vec![ManifoldPoint::<3>::new([1.0, 0.0, 0.0]); 10];
    check(&dup, "ten-duplicates");
}

#[test]
fn level_zero_covers_every_input_point() {
    let pts = sphere(120);
    let tree = NetTree::build(&pts, 2.0);
    for (i, p) in pts.iter().enumerate() {
        let best = tree
            .net(0)
            .iter()
            .map(|&q| p.distance(&pts[q]))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best <= tree.radius(0),
            "point {i} uncovered at level 0: {best}"
        );
    }
}
