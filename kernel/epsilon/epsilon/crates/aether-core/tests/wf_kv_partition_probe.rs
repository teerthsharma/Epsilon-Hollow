//! Scratch probe. Observation only.

use aether_core::attention::{
    dense_dot_cost, routing_plan, select_mask, selection_dot_cost, single_linkage_clusters,
    Selector,
};
use aether_core::persistent_kv_partition::BettiGuidedPartitioner;

#[test]
fn probe_perfect_locality_is_structurally_forced() {
    // Wildly different cluster shapes, tiny capacities, huge capacities.
    let cases: [([u64; 4], f64, f64, u64); 4] = [
        ([400, 300, 200, 100], 0.00008, 0.0001, 256),
        ([1, 1, 1, 1], 0.0, 0.0, 1 << 30),
        ([1_000_000, 1, 1, 1], 1e9, 1e9, 4096),
        ([7, 7, 7, 7], 1e-12, 1e-12, 1 << 40),
    ];
    for (sizes, hbm, ddr, bpk) in cases {
        let r = BettiGuidedPartitioner::new(hbm, ddr).partition(&sizes, bpk);
        println!(
            "sizes={:?} hbm={hbm} ddr={ddr} -> locality={} perfect={} assign={:?} betti={:.3} random={:.3} speedup={:.3}",
            sizes, r.topological_locality, r.perfect_locality, r.assignments,
            r.latency_betti_ns, r.latency_random_ns, r.speedup
        );
    }
}

#[test]
fn probe_latency_random_is_input_independent() {
    let a = BettiGuidedPartitioner::new(80.0, 512.0).partition(&[1_u64, 2, 3, 4], 1);
    let b = BettiGuidedPartitioner::new(80.0, 512.0).partition(&[9_u64; 4], 1 << 30);
    println!(
        "random_a={:.6} random_b={:.6}",
        a.latency_random_ns, b.latency_random_ns
    );
    println!(
        "betti_a={:.6} betti_b={:.6}",
        a.latency_betti_ns, b.latency_betti_ns
    );
}

// ── attention: does the reported cost match the path actually run? ────────────

fn structured_keys(seq: usize, head_dim: usize, groups: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let centers: Vec<Vec<f64>> = (0..groups)
        .map(|_| (0..head_dim).map(|_| next() * 8.0).collect())
        .collect();
    let mut k = vec![0.0; seq * head_dim];
    for t in 0..seq {
        let c = &centers[t % groups];
        for d in 0..head_dim {
            k[t * head_dim + d] = c[d] + next() * 0.05;
        }
    }
    k
}

/// Recompute what `select_mask` does for TopologicalRouted, counting the dot
/// products it actually performs: `cluster_count` centroid dots plus one per
/// candidate. Mirrors src/attention.rs lines 310-341 and 426-468.
fn actual_routed_cost(
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    clusters: usize,
    budget: usize,
    causal: bool,
) -> f64 {
    let (assignment, _) = single_linkage_clusters(k, seq, head_dim, clusters, true);
    let cluster_count = assignment.iter().copied().max().map_or(0, |m| m + 1);
    let mut centroids = vec![0.0f64; cluster_count * head_dim];
    let mut members = vec![0usize; cluster_count];
    for (t, &label) in assignment.iter().enumerate() {
        let norm: f64 = (0..head_dim)
            .map(|d| k[t * head_dim + d] * k[t * head_dim + d])
            .sum::<f64>()
            .sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        for d in 0..head_dim {
            centroids[label * head_dim + d] += k[t * head_dim + d] * inv;
        }
        members[label] += 1;
    }
    for label in 0..cluster_count {
        let c = members[label].max(1) as f64;
        for d in 0..head_dim {
            centroids[label * head_dim + d] /= c;
        }
    }

    let mut total = 0usize;
    for i in 0..seq {
        let legal_end = if causal { i + 1 } else { seq };
        let mut ranked: Vec<(usize, f64)> = (0..cluster_count)
            .map(|label| {
                let s: f64 = (0..head_dim)
                    .map(|d| q[i * head_dim + d] * centroids[label * head_dim + d])
                    .sum();
                (label, s)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut candidates = 0usize;
        for (label, _) in ranked {
            if candidates >= budget {
                break;
            }
            candidates += (0..legal_end).filter(|&j| assignment[j] == label).count();
        }
        total += cluster_count + candidates;
    }
    total as f64 / seq as f64
}

#[test]
fn probe_reported_cost_vs_path_actually_run() {
    let (seq, head_dim, budget, clusters) = (48usize, 8usize, 6usize, 6usize);
    let dense = dense_dot_cost(seq, true);
    for groups in [1usize, 2, 4, 6, 12, 48] {
        let k = structured_keys(seq, head_dim, groups, 5000 + groups as u64);
        let q = structured_keys(seq, head_dim, groups.max(2), 99_000 + groups as u64);
        let plan = routing_plan(&q, &k, seq, head_dim, clusters, budget, true);
        let reported = selection_dot_cost(
            Selector::TopologicalRouted { budget, clusters },
            &q,
            &k,
            seq,
            head_dim,
            true,
        ) / dense;
        let actual = actual_routed_cost(&q, &k, seq, head_dim, clusters, budget, true) / dense;
        // sanity: the mask is non-degenerate
        let m = select_mask(
            Selector::TopologicalRouted { budget, clusters },
            &q,
            &k,
            &k,
            seq,
            head_dim,
            true,
        );
        let picked = m.iter().filter(|b| **b).count();
        println!(
            "groups={groups}: plan.cost_ratio={:.6} reported={:.6} actual_alignment_order={:.6} delta={:.6} mask_true={picked}",
            plan.cost_ratio, reported, actual, actual - reported
        );
    }
}
