// scratch probe: observation only, no assertions about intended behaviour
use aether_core::spectral_entropy::*;
use aether_core::thermodynamic_plasticity::*;
use aether_core::world_model_horizon::*;

#[test]
fn probe_lr_agreement_is_forced() {
    let a = ThermodynamicAnalyzer::h100_default();
    for dist in [
        vec![0.25, 0.25, 0.25, 0.25],
        vec![0.97, 0.01, 0.01, 0.01],
        vec![-5.0, 3.0, 0.0, 1e9, 2.0],
        vec![1.0, 2.0, 3.0],
    ] {
        let r = a.lr_analysis(&dist, 0.01);
        println!(
            "n={} H={:.6} Hmax={:.6} S={:.6e} Smax={:.6e} eta_sh={:.17e} eta_gi={:.17e} diff={:.3e} agree={} dF={:.6e} favorable={}",
            dist.len(), r.h_shannon, r.h_max, r.s_gibbs, r.s_max,
            r.eta_shannon, r.eta_gibbs, (r.eta_shannon - r.eta_gibbs).abs(),
            r.lr_agreement, r.delta_f, r.learning_favorable
        );
    }
    // largest possible entropy term vs the hardcoded delta_u = 0.001 J
    let big = vec![1.0; 100_000];
    let r = a.lr_analysis(&big, 0.01);
    println!(
        "n=1e5 Smax={:.6e} T*dS_max={:.6e} dF={:.17e} favorable={}",
        r.s_max,
        a.temperature_k * r.s_max,
        r.delta_f,
        r.learning_favorable
    );
}

#[test]
fn probe_world_model_gate() {
    for rate in [0.5, 5.0, 100.0] {
        let w = WorldModelAnalyzer::new(rate);
        let v = w.verify_theorem();
        let s = w.three_model_stack();
        println!(
            "rate={} adv={:.9} expected={} ratio_ok={} max_ind={:.6e} combined={:.6e} bonus={:.6e} comb_ok={} holds={}",
            rate, v.topological_advantage_ratio, v.expected_ratio, v.ratio_correct,
            s.individual_horizons.iter().cloned().fold(f64::MIN, f64::max),
            s.combined_horizon,
            s.combined_horizon - s.individual_horizons.iter().cloned().fold(f64::MIN, f64::max),
            v.combined_exceeds_individuals, v.theorem_holds
        );
    }
    // advantage vs. nodes/clusters, over a wide sweep
    for (n, c, d, eq) in [
        (100_000u64, 1000u64, 128u32, 1e-4),
        (10, 9, 1, 0.5),
        (1_000_000, 2, 8192, 1e-9),
    ] {
        let adv = topological_advantage(n, c, d, eq);
        println!(
            "n={} c={} d={} adv={:.9} n/c={:.9} rel_dev={:.3e}",
            n,
            c,
            d,
            adv,
            n as f64 / c as f64,
            (adv - n as f64 / c as f64).abs() / (n as f64 / c as f64)
        );
    }
}

#[test]
fn probe_spectral_degenerate_and_kl() {
    let constant = [3.0, 3.0, 3.0, 3.0];
    let periodic = [1.0, -1.0, 1.0, -1.0];
    let noise = [0.3, -1.2, 0.7, 2.1, -0.5, 0.9, -1.8, 0.4];
    println!(
        "coh(constant)={} coh(periodic)={} coh(noise8)={}",
        spectral_coherence(&constant),
        spectral_coherence(&periodic),
        spectral_coherence(&noise)
    );
    println!(
        "ent(constant)={} ent(periodic)={} ent(noise8)={}",
        spectral_entropy(&constant),
        spectral_entropy(&periodic),
        spectral_entropy(&noise)
    );
    let near_constant = [3.0, 3.0 + 1e-9, 3.0, 3.0 - 1e-9];
    println!("coh(near_constant)={}", spectral_coherence(&near_constant));
    // tracker on a pure constant stream
    let mut t = SpectralCoherenceTracker::new(8, 0.9);
    let mut last = SpectralUpdate::default();
    for _ in 0..10 {
        last = t.update(7.0, 0.9);
    }
    println!(
        "constant stream: coh={} ent={} holds={} rate={}",
        last.coherence, last.spectral_entropy, last.invariant_holds, last.violation_rate
    );
    // does the invariant ever fire on white noise with epsilon_t = epsilon_max?
    let mut t2 = SpectralCoherenceTracker::new(16, 0.9);
    let mut fired = 0;
    let mut x = 1u64;
    for _ in 0..200 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((x >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
        let u = t2.update(v, 0.0);
        if !u.invariant_holds {
            fired += 1;
        }
    }
    println!(
        "noise, epsilon_t=0 (strictest bound): violations={} rate={}",
        fired,
        t2.violation_rate()
    );
    println!("kl(self)={}", spectral_kl_divergence(&noise, &noise));
    let other = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    println!(
        "kl(noise,other)={} kl(other,noise)={}",
        spectral_kl_divergence(&noise, &other),
        spectral_kl_divergence(&other, &noise)
    );
    // Haar orthonormality: energy preservation
    let tr = haar_wavelet_transform(&noise);
    let mut e: f64 = tr.approximation.iter().map(|v| v * v).sum();
    for d in &tr.details {
        e += d.iter().map(|v| v * v).sum::<f64>();
    }
    let e0: f64 = noise.iter().map(|v| v * v).sum();
    println!(
        "haar energy in={:.17e} out={:.17e} diff={:.3e}",
        e0,
        e,
        (e - e0).abs()
    );
    // Landauer chain
    println!(
        "landauer(300K)={:.6e} J/bit  k_B*300*ln2={:.6e}",
        landauer_energy_per_bit(300.0),
        1.380_649e-23 * 300.0 * core::f64::consts::LN_2
    );
}
