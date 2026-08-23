//! Read-only observation probe for the geometry-metrics subsystem.
//! Asserts nothing about correctness; prints observed values.

use aether_core::cross_manifold_alignment::CrossManifoldAligner;
use aether_core::geodesic_consolidation::{great_circle_distance, GeodesicConsolidator};
use aether_core::hyperbolic_capacity::{h100_analysis, hcs_analysis};
use aether_core::hyperbolic_geometry::{norm, PoincareBall};
use aether_core::parallel_riemannian::DistributedRiemannianSgd;
use aether_core::riemannian_optimizer::{ManifoldType, RiemannianSgd};

fn atanh(x: f64) -> f64 {
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

#[test]
fn probe_poincare_distance_ceiling() {
    for c in [1.0_f64, 0.1, 4.0, 100.0, 1e-6] {
        let ball = PoincareBall::new(c);
        let r = ball.max_norm();
        let x = ball.project(&[-r * 2.0, 0.0, 0.0]);
        let y = ball.project(&[r * 2.0, 0.0, 0.0]);
        let d = ball.distance(&x, &y);
        // Closed form for antipodal points at radius r in the ball of curvature -c:
        // |(-x) (+) y| = 2*sqrt(c)*r / (1 + c r^2); d = (2/sqrt c) atanh(sqrt(c)*that/... )
        let sc = c.sqrt();
        let mob = 2.0 * r / (1.0 + c * r * r);
        let true_d = (2.0 / sc) * atanh((sc * mob).min(1.0 - 1e-16));
        println!(
            "c={c:e} max_norm={r:.12} computed_d={d:.15} true_d={true_d:.15} ratio={:.6}",
            true_d / d
        );
    }

    // Does the reported distance move at all as points approach the boundary?
    let ball = PoincareBall::unit();
    for k in [1e-2_f64, 1e-4, 1e-6, 1e-9, 1e-12] {
        let r = 1.0 - k;
        let d = ball.distance(&[-r, 0.0, 0.0], &[r, 0.0, 0.0]);
        let mob = 2.0 * r / (1.0 + r * r);
        println!(
            "unit ball, r=1-{k:e}: computed_d={d:.15}  true_d={:.15}",
            2.0 * atanh(mob.min(1.0 - 1e-16))
        );
    }
}

#[test]
fn probe_rgcs_bound_holds_can_ever_be_false() {
    let mut runs = 0;
    let mut failures = 0;
    for n in [1_u32, 8, 64, 4096] {
        for p in [1_u32, 2, 8, 512] {
            for workers in [1_u32, 2, 8, 1024] {
                for lr in [0.0_f64, -5.0, 1e-9, 0.01, 1e9] {
                    for sync in [0_u32, 1, 7, 100_000] {
                        let r = DistributedRiemannianSgd::new(n, p, workers, lr, sync)
                            .verify_theorem(20);
                        runs += 1;
                        if !r.theorem_holds {
                            failures += 1;
                            println!("FALSE at n={n} p={p} w={workers} lr={lr} sync={sync}");
                        }
                    }
                }
            }
        }
    }
    println!("RGCS runs={runs} theorem_holds==false count={failures}");
    let r = DistributedRiemannianSgd::new(64, 8, 8, 0.01, 1).verify_theorem(20);
    println!(
        "RGCS witness: max_deviation={} bound={} ratio={}",
        r.max_deviation,
        r.theoretical_bound,
        r.max_deviation / r.theoretical_bound
    );
}

#[test]
fn probe_cma_theorem_holds_can_ever_be_false() {
    let mut runs = 0;
    let mut failures = 0;
    for a in [1_u32, 7, 128, 65535] {
        for b in [1_u32, 7, 128, 65535] {
            for c in [1_u32, 7, 128, 65535] {
                for n_ref in [0_u32, 1, 128, 100_000] {
                    let v = CrossManifoldAligner::<3>::new([a, b, c]).verify_theorem(n_ref);
                    runs += 1;
                    if !v.theorem_holds {
                        failures += 1;
                        println!("FALSE at dims=[{a},{b},{c}] n_ref={n_ref}");
                    }
                }
            }
        }
    }
    println!("CMA runs={runs} theorem_holds==false count={failures}");
    let v = CrossManifoldAligner::<3>::new([128, 64, 32]).verify_theorem(128);
    println!(
        "CMA: total_transitive_error={} sum_pairwise_errors={} identical={}",
        v.total_transitive_error,
        v.sum_pairwise_errors,
        v.total_transitive_error == v.sum_pairwise_errors
    );
    for i in 0..v.n_pairs {
        let p = v.pairwise_results[i];
        println!(
            "  pair{i}: empirical={} bound={} ratio={}",
            p.empirical_error,
            p.theoretical_bound,
            p.empirical_error / p.theoretical_bound
        );
    }
}

#[test]
fn probe_gmc_theorem_holds_can_ever_be_false() {
    // Deterministic LCG so the sweep is reproducible.
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let mut runs = 0;
    let mut failures = 0;
    for _ in 0..4000 {
        let mut cent = [(0.0_f64, 0.0_f64); 6];
        let mut sizes = [0_u32; 6];
        for k in 0..6 {
            let t = (next() % 100_000) as f64 / 100_000.0 * core::f64::consts::PI
                - core::f64::consts::FRAC_PI_2;
            let p = (next() % 100_000) as f64 / 100_000.0 * 2.0 * core::f64::consts::PI;
            cent[k] = (t, p);
            sizes[k] = next() % 50;
        }
        for thr in [0.0_f64, 0.05, 0.5, 3.5, 1e9] {
            let r = GeodesicConsolidator::new(thr).consolidate(&cent, &sizes);
            runs += 1;
            if !r.theorem_holds {
                failures += 1;
                println!("FALSE thr={thr} cent={cent:?} sizes={sizes:?}");
            }
        }
    }
    println!("GMC runs={runs} theorem_holds==false count={failures}");
}

#[test]
fn probe_gmc_merge_is_coordinate_mean_not_geodesic() {
    // Two points near the phi=0 / phi=2pi seam, on the equator.
    let a = (0.0_f64, 0.05_f64);
    let b = (0.0_f64, 6.23_f64);
    println!("geodesic distance a..b = {}", great_circle_distance(a, b));

    let r = GeodesicConsolidator::new(0.2).consolidate(&[a, b], &[1_u32, 1]);
    let m = r.new_centroids[0];
    println!(
        "merges={} merged_centroid={:?}  d(merged,a)={}  d(merged,b)={}",
        r.merges_performed,
        m,
        great_circle_distance(m, a),
        great_circle_distance(m, b)
    );
    println!("theorem_holds={}", r.theorem_holds);
}

#[test]
fn probe_stiefel_vector_path_is_euclidean() {
    let x = [1.0_f64, 0.0];
    let g = [0.0_f64, 1.0];
    let mut sgd_default: RiemannianSgd<2> = RiemannianSgd::default();
    let y_default = sgd_default.step_vector(x, g);
    let mut sgd_stiefel = RiemannianSgd::<2>::new(0.1, 0.9, ManifoldType::Stiefel);
    let mut sgd_euclid = RiemannianSgd::<2>::new(0.1, 0.9, ManifoldType::Euclidean);
    let mut sgd_sphere = RiemannianSgd::<2>::new(0.1, 0.9, ManifoldType::Sphere);
    let ys = sgd_stiefel.step_vector(x, g);
    let ye = sgd_euclid.step_vector(x, g);
    let yp = sgd_sphere.step_vector(x, g);
    println!("default(Stiefel) y={y_default:?} |y|={}", norm(&y_default));
    println!("Stiefel   y={ys:?} |y|={}", norm(&ys));
    println!("Euclidean y={ye:?} |y|={}", norm(&ye));
    println!("Sphere    y={yp:?} |y|={}", norm(&yp));
    println!("stiefel_equals_euclidean={}", ys == ye);
}

#[test]
fn probe_hcs_depth_cap() {
    let full = h100_analysis();
    let capped = hcs_analysis(4096, 0.1, 64, 20);
    println!(
        "h100(depth=50): depth_field={} nodes={} memory_gb={:e} fits={}",
        full.depth, full.total_tree_nodes, full.memory_gb, full.fits_h100_80gb
    );
    println!(
        "hcs(depth=20):  depth_field={} nodes={} memory_gb={:e} fits={}",
        capped.depth, capped.total_tree_nodes, capped.memory_gb, capped.fits_h100_80gb
    );
    println!(
        "same_nodes={} same_memory={}",
        full.total_tree_nodes == capped.total_tree_nodes,
        full.memory_gb == capped.memory_gb
    );
}
