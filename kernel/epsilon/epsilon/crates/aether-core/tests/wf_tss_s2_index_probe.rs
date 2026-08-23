// Scratch probe: observe tss-s2-index values. Read-only, no src/ modified.
use aether_core::tss::*;
use std::f64::consts::PI;

/// Exact inline copy of tss.rs:23 great_circle_distance.
fn gcd_as_written(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let cos_d = t1.sin() * t2.sin() + t1.cos() * t2.cos() * (p1 - p2).cos();
    cos_d.clamp(-1.0, 1.0).acos()
}

/// True angle between the SAME two (theta,phi) under tss.rs:200
/// spherical_to_unit_vector (colatitude / physics convention).
fn gcd_true_colatitude(t1: f64, p1: f64, t2: f64, p2: f64) -> f64 {
    let a = [t1.sin() * p1.cos(), t1.sin() * p1.sin(), t1.cos()];
    let b = [t2.sin() * p2.cos(), t2.sin() * p2.sin(), t2.cos()];
    (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        .clamp(-1.0, 1.0)
        .acos()
}

fn k8() -> [(f64, f64); 8] {
    let mut c = [(0.0, 0.0); 8];
    for i in 0..8 {
        c[i] = (PI / 2.0, i as f64 * (2.0 * PI / 8.0));
    }
    c
}

#[test]
fn probe_a_equatorial_fixture_makes_distance_constant() {
    let c = k8();
    // The fixture used by tss.rs tests AND by tests/proptest_tss.rs.
    for &(qt, qp) in &[(0.3, 0.7), (1.9, 4.4), (2.7, 0.1), (0.9, 5.9)] {
        let ds: Vec<f64> = c.iter().map(|&(t, p)| gcd_as_written(qt, qp, t, p)).collect();
        let spread = ds.iter().cloned().fold(f64::MIN, f64::max)
            - ds.iter().cloned().fold(f64::MAX, f64::min);
        let truth: Vec<f64> = c
            .iter()
            .map(|&(t, p)| gcd_true_colatitude(qt, qp, t, p))
            .collect();
        let tspread = truth.iter().cloned().fold(f64::MIN, f64::max)
            - truth.iter().cloned().fold(f64::MAX, f64::min);
        println!(
            "q=({qt},{qp}) as-written spread={spread:.3e} d[0]={:.6}  |  true spread={tspread:.6}",
            ds[0]
        );
    }
}

#[test]
fn probe_b_locate_always_returns_zero_on_the_fixture() {
    let idx = SphericalVoronoiIndex::<8>::new(k8());
    let mut hist = [0usize; 8];
    let mut q = 0u64;
    for _ in 0..20000 {
        q = q.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let theta = ((q >> 33) as f64 / (1u64 << 31) as f64) * PI;
        let phi = ((q & 0xFFFF_FFFF) as f64 / (1u64 << 32) as f64) * 2.0 * PI;
        hist[idx.locate((theta, phi))] += 1;
    }
    println!("locate histogram over 20000 uniform S2 queries: {hist:?}");
    println!("betti_0() reports {}", idx.betti_0());
    let nonempty = hist.iter().filter(|&&n| n > 0).count();
    println!("nonempty Voronoi cells actually observed: {nonempty}");
}

#[test]
fn probe_c_betti0_with_duplicate_centroids() {
    // Four slots, but only TWO geometrically distinct sites.
    let dup = [(0.5, 0.0), (0.5, 0.0), (2.0, 3.0), (2.0, 3.0)];
    let idx = SphericalVoronoiIndex::<4>::new(dup);
    let mut hist = [0usize; 4];
    let mut q = 0u64;
    for _ in 0..20000 {
        q = q.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let theta = ((q >> 33) as f64 / (1u64 << 31) as f64) * PI;
        let phi = ((q & 0xFFFF_FFFF) as f64 / (1u64 << 32) as f64) * 2.0 * PI;
        hist[idx.locate((theta, phi))] += 1;
    }
    println!("duplicate-centroid histogram: {hist:?}");
    println!("betti_0() = {} (nonempty cells = {})", idx.betti_0(),
             hist.iter().filter(|&&n| n > 0).count());
}

#[test]
fn probe_d_grid_hash_phi_clamp_collapses_cells() {
    let c = k8();
    let g = SphericalGridHashIndex::<8>::new(c);
    let s = g.stats();
    println!("K=8 equatorial fixture grid stats: {s:?}");
    println!(
        "n_theta={} n_phi={} total_cells={} occupied={} max_cell_size={}",
        s.n_theta, s.n_phi, s.total_cells, s.occupied_cells, s.max_cell_size
    );
    // phi values fed to hash():
    for (i, &(_, p)) in c.iter().enumerate() {
        println!("  centroid {i}: phi={p:.4}  (hash clamps phi to [-PI,PI]; PI={:.4})", PI);
    }
}

#[test]
fn probe_e_grid_and_voronoi_disagree() {
    // Generic, non-degenerate centroids.
    let c: [(f64, f64); 8] = [
        (0.4, 0.2), (1.1, 1.3), (2.2, 2.5), (2.9, 0.9),
        (0.8, 4.0), (1.7, 5.2), (2.5, 3.6), (1.3, 2.9),
    ];
    let vor = SphericalVoronoiIndex::<8>::new(c);
    let mut grid = SphericalGridHashIndex::<8>::new(c);
    let (mut disagree, mut grid_wrong, mut n) = (0usize, 0usize, 0usize);
    let mut q = 0u64;
    for _ in 0..5000 {
        q = q.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let theta = ((q >> 33) as f64 / (1u64 << 31) as f64) * PI;
        let phi = ((q & 0xFFFF_FFFF) as f64 / (1u64 << 32) as f64) * 2.0 * PI;
        let v = vor.locate((theta, phi));
        let g = grid.locate((theta, phi));
        // brute-force true nearest under the colatitude metric the grid uses
        let mut best = 0usize;
        let mut bd = f64::MAX;
        for (i, &(t, p)) in c.iter().enumerate() {
            let d = gcd_true_colatitude(theta, phi, t, p);
            if d < bd { bd = d; best = i; }
        }
        if v != g { disagree += 1; }
        if g != best { grid_wrong += 1; }
        n += 1;
    }
    println!("queries={n}  voronoi!=grid: {disagree}  grid!=true-nearest: {grid_wrong}");
    println!("grid o1_hit_rate reported = {:.4}", grid.o1_hit_rate());
}

#[test]
fn probe_f_bounded_retrieval_is_a_tautology() {
    // tss.rs:160  bounded_retrieval = retrieval.epsilon_o1_ops == k*d
    // tss.rs:176  epsilon_o1_ops    = k*d
    for &(n, k, d) in &[(10usize, 1usize, 1usize), (1_000_000, 5, 128), (7, 999, 3)] {
        let r = tss_retrieval_bound(n, 99999, k, d);
        println!(
            "n={n} k={k} d={d} -> epsilon_o1_ops={} k*d={} equal={}  (p field={} , appears in NO op count)",
            r.epsilon_o1_ops, k * d, r.epsilon_o1_ops == k * d, r.p
        );
    }
    // Absurd centroid count P=1e9; the "speedup" is unchanged.
    let a = tss_retrieval_bound(1_000_000, 1_000, 5, 128);
    let b = tss_retrieval_bound(1_000_000, 1_000_000_000, 5, 128);
    println!("P=1e3 speedup_vs_hnsw={:.4}  P=1e9 speedup_vs_hnsw={:.4}", a.speedup_vs_hnsw, b.speedup_vs_hnsw);
    println!("P=1e3 speedup_vs_brute={:.4} P=1e9 speedup_vs_brute={:.4}", a.speedup_vs_brute, b.speedup_vs_brute);

    // And theorem_holds with a deliberately absurd theta_min / packing.
    let v = TssVerifier::<4>::new([(0.5, 0.0), (1.0, 1.5), (2.0, 3.0), (2.5, 4.5)], 0.1);
    let rep = v.full_verification(1_000, 4, 16);
    println!("report: theta_min={} p_max={} sep={} pack={} holds={}",
        rep.theta_min, rep.p_max, rep.separation_holds, rep.packing_holds, rep.theorem_holds);
}

#[test]
fn probe_g_separation_check_uses_the_same_wrong_metric() {
    // Two centroids that are 90 degrees apart on the real sphere but which the
    // as-written formula scores as coincident (or vice versa).
    let cases = [
        ((0.0, 0.0), (0.0, PI / 2.0)),
        ((PI / 2.0, 0.0), (PI / 2.0, PI)),
        ((0.3, 1.0), (0.3, 4.0)),
    ];
    for (a, b) in cases {
        println!(
            "a={a:?} b={b:?}  as-written={:.6}  true-colatitude={:.6}",
            gcd_as_written(a.0, a.1, b.0, b.1),
            gcd_true_colatitude(a.0, a.1, b.0, b.1)
        );
    }
    // verify_separation demanding 0.5 rad on a pair that is truly PI apart:
    let pair = [(PI / 2.0, 0.0), (PI / 2.0, PI)];
    println!("verify_separation(theta_min=0.5) on a truly-antipodal-on-equator pair = {}",
        verify_separation(&pair, 0.5));
}
