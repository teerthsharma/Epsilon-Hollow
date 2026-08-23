//! Wilson probe: observe values, assert nothing. Run with --nocapture.
use aether_core::compute_betti_1;

#[test]
fn wilson_observes_betti_1_growth() {
    // Pattern that DOES satisfy the fn's own loop rule: |w[0]-w[3]| <= 5
    // and |w[0]-w[1]| > 5. Period 4, returns to start.
    println!("--- compute_betti_1 on [0,60,120,0] repeated ---");
    for reps in [1usize, 2, 4, 8, 16, 32] {
        let mut v = Vec::new();
        for _ in 0..reps {
            v.extend_from_slice(&[0u8, 60, 120, 0]);
        }
        println!("len {:>4} -> beta_1 {}", v.len(), compute_betti_1(&v));
    }
    println!("--- same point SET, order reversed (permutation test) ---");
    let fwd = [0u8, 60, 120, 0, 0, 60, 120, 0];
    let mut rev = fwd.to_vec();
    rev.reverse();
    println!("forward  {:?} -> {}", fwd, compute_betti_1(&fwd));
    println!("reversed {:?} -> {}", rev, compute_betti_1(&rev));
    println!("--- scale equivariance: multiply every byte by 2 ---");
    let base = [0u8, 30, 60, 0, 0, 30, 60, 0];
    let scaled: Vec<u8> = base.iter().map(|b| b * 2).collect();
    println!("base   -> {}", compute_betti_1(&base));
    println!("scaled -> {}", compute_betti_1(&scaled));
}
