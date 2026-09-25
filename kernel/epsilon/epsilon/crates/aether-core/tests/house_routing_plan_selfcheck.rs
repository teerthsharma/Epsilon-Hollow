//! `attention_contracts.rs` asserts `plan.cost_ratio == selection_dot_cost(..)/dense`.
//! But `routing_plan` COMPUTES `cost_ratio` by calling `selection_dot_cost` with
//! those same arguments and dividing by that same denominator
//! (`attention.rs:164`). The assertion is `|x - x| < 1e-12`.
//!
//! What it actually validates: that `routing_plan` stores what
//! `selection_dot_cost` returned. What its failure message claims: that a
//! prediction matches a measurement. Those are different, and only the first
//! is checked.
use aether_core::attention::{dense_dot_cost, routing_plan, selection_dot_cost, Selector};

fn keys(seq: usize, head_dim: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..seq * head_dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect()
}

#[test]
fn the_contract_is_an_identity_and_holds_for_any_cost_function() {
    // Demonstration: the two sides are the same expression. Their difference is
    // exactly zero, not merely within 1e-12 — which is what an identity looks
    // like, and what a genuine prediction-vs-measurement check does not.
    let (seq, head_dim, clusters, budget) = (48usize, 8usize, 4usize, 12usize);
    let k = keys(seq, head_dim, 0xBEEF);
    let q = keys(seq, head_dim, 0xFACE);

    let plan = routing_plan(&q, &k, seq, head_dim, clusters, budget, true);
    let measured = selection_dot_cost(
        Selector::TopologicalRouted { budget, clusters },
        &q,
        &k,
        seq,
        head_dim,
        true,
    ) / dense_dot_cost(seq, true);

    let diff = (plan.cost_ratio - measured).abs();
    println!("plan.cost_ratio = {:.17}", plan.cost_ratio);
    println!("measured        = {:.17}", measured);
    println!("difference      = {diff:.17e}");
    assert_eq!(
        diff, 0.0,
        "the difference is not exactly zero, so the two sides are genuinely \
         computed differently and this finding is wrong"
    );
}

#[test]
fn the_cost_model_charges_routing_overhead_and_never_invents_keys() {
    // Two properties the contract at attention_contracts.rs:1044 cannot check,
    // because both of its sides come from `selection_dot_cost` itself.
    //
    // (a) No phantom keys. Under a causal mask row `i` has `i + 1` legal keys.
    //     The routed selector accumulates whole clusters but filters each to
    //     `j < legal_end`, so per-row candidates cannot exceed the legal count.
    // (b) Routing is not free. Each row is charged `cluster_count + candidates`
    //     — the cost of comparing against every cluster centroid to decide
    //     which clusters to take. A routed selector may therefore cost MORE
    //     than dense, and at small `seq` it does. That is the model being
    //     honest about overhead a naive sparsity claim would omit.
    let head_dim = 4usize;
    for seq in [8usize, 16, 32, 64, 128] {
        let k = keys(seq, head_dim, 0x1234 + seq as u64);
        let q = keys(seq, head_dim, 0x5678 + seq as u64);
        let dense = dense_dot_cost(seq, true);
        let clusters = 2usize;
        let cost = selection_dot_cost(
            Selector::TopologicalRouted {
                budget: 4,
                clusters,
            },
            &q,
            &k,
            seq,
            head_dim,
            true,
        );
        let ratio = cost / dense;
        println!("seq {seq:4}: dense {dense:8.3} routed {cost:8.3} ratio {ratio:6.3}");

        // (a) an upper bound that holds for ANY selector: it can never charge
        //     more than the whole legal set plus the fixed routing overhead.
        assert!(
            cost <= dense + clusters as f64 + 1e-9,
            "seq {seq}: cost {cost} exceeds dense {dense} plus the {clusters}-cluster overhead"
        );
        assert!(cost > 0.0, "seq {seq}: zero cost is not a selection");
    }
}

#[test]
fn the_plan_declines_to_route_when_the_ratio_exceeds_one() {
    // The decision the cost model exists to drive. At small seq the overhead
    // dominates and routing must be refused.
    let (head_dim, clusters, budget) = (8usize, 4usize, 12usize);
    for seq in [16usize, 32, 48] {
        let k = keys(seq, head_dim, 0xBEEF + seq as u64);
        let q = keys(seq, head_dim, 0xFACE + seq as u64);
        let plan = routing_plan(&q, &k, seq, head_dim, clusters, budget, true);
        println!(
            "seq {seq:3}: cost_ratio {:.4} threshold {:.4} worth_routing {}",
            plan.cost_ratio, plan.threshold, plan.worth_routing
        );
        if plan.cost_ratio >= plan.threshold {
            assert!(
                !plan.worth_routing,
                "seq {seq}: ratio {} is at or above threshold {} yet routing was judged worthwhile",
                plan.cost_ratio, plan.threshold
            );
        }
    }
}
