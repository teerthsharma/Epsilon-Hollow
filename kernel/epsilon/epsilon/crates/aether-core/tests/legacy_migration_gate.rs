use aether_core::governor::{
    GovernorConvergenceAnalyzer, GovernorSimulationHistory, GovernorTheoremVerification,
};
use aether_core::hyperbolic_capacity::{
    euclidean_distortion_bound, h100_analysis, hyperbolic_distortion_bound, separation_ratio,
    HcsVerifier,
};
use aether_core::hyperbolic_geometry::{
    verify_plasticity_bound, AngularMomentumTracker, PoincareBall,
};
use aether_core::meta_controller::{
    Action, ConstitutionalSafetyFilter, DecisionReason, DecisionTarget, MetaController, MetaInput,
    ToolDecision, ToolRegistry, ToolSpec,
};
use aether_core::scm::SpectralContractionVerifier;
use aether_core::scm::{LatentPredictor, TelemetryOperator};
use aether_core::spectral_entropy::{
    haar_wavelet_transform, spectral_entropy, wavelet_entropy_per_scale, HaarWaveletTransform,
    SpectralCoherenceTracker,
};
use aether_core::thermodynamic_plasticity::{
    gibbs_entropy, helmholtz_free_energy_change, landauer_energy_per_bit,
    max_sustainable_hot_ratio, min_energy_per_update, thermodynamic_lr, ThermodynamicAnalyzer,
    K_BOLTZMANN, LN2,
};
use aether_core::tss::{
    auto_sized_dimensions, epsilon_from_chebyshev, tss_retrieval_bound, SphericalGridHashIndex,
    TssVerifier, EMPTY_LOCATE,
};

#[test]
fn legacy_agent_safety_filter_is_rust_backed() {
    let mut safety = ConstitutionalSafetyFilter::new(0.3);

    assert!(!safety.check_compliance_text(&[1.0, 0.0, 0.0, 0.0], "rm -rf /"));
    assert!(!safety.check_compliance_text(&[1.0, 0.0, 0.0, 0.0], "reason"));
    assert!(safety.check_compliance_text(&[0.25; 4], "reason"));
    assert_eq!(safety.entropy_threshold(), 0.3);
    assert_eq!(safety.stats().violations_blocked, 2);

    let mut controller = MetaController::<4>::new(7, 0.8);
    let decision = controller.decide(MetaInput {
        latent: &[0.25; 4],
        context_len: 1,
        step: 1.0,
        tool_count: 1,
        safety_compliant: true,
    });
    assert!(matches!(
        decision.action,
        Action::Execute | Action::Reason | Action::Explore | Action::Refusal
    ));
    assert!(matches!(
        decision.reason,
        DecisionReason::PolicySelected
            | DecisionReason::NoToolAvailable
            | DecisionReason::SafetyViolation
    ));
    assert!(matches!(
        decision.target,
        DecisionTarget::None
            | DecisionTarget::InternalModel
            | DecisionTarget::MemoryFrontier
            | DecisionTarget::Tool(_)
    ));

    controller.clear_weights();
    controller.set_bias(Action::Execute, 10.0);
    controller.set_bias(Action::Reason, -10.0);
    controller.set_bias(Action::Explore, -10.0);
    controller.set_bias(Action::Refusal, -10.0);

    let tools = [
        ToolSpec::new("danger", "rm -rf /tmp/cache", "shell.command", false, 1),
        ToolSpec::new("lookup", "inspect manifold index", "seal.query", true, 2),
    ];
    let registry = ToolRegistry::new(&tools).unwrap();
    let tool_decision: ToolDecision<'_> =
        controller.decide_from_state_with_tools(&[0.5, 0.5, 0.5, 0.5], false, true, registry);
    let payload = tool_decision.payload.expect("execute must carry payload");
    assert_eq!(tool_decision.decision.action, Action::Execute);
    assert_eq!(
        tool_decision.decision.reason,
        DecisionReason::PolicySelected
    );
    assert_eq!(payload.tool_name, "lookup");

    let dangerous = [
        ToolSpec::new("eval", "eval(user_code)", "script.eval", false, 1),
        ToolSpec::new("sql", "drop table users", "sql.write", false, 1),
    ];
    let refused = controller.decide_from_state_with_tools(
        &[0.5, 0.5, 0.5, 0.5],
        false,
        true,
        ToolRegistry::new(&dangerous).unwrap(),
    );
    assert_eq!(refused.decision.action, Action::Refusal);
    assert_eq!(refused.decision.reason, DecisionReason::NoCompliantTool);
    assert!(refused.payload.is_none());

    let direct = controller.decide_with_tools(
        MetaInput {
            latent: &[0.5, 0.5, 0.5, 0.5],
            context_len: 1,
            step: 2.0,
            tool_count: tools.len(),
            safety_compliant: false,
        },
        ToolRegistry::new(&tools).unwrap(),
    );
    assert_eq!(direct.decision.action, Action::Refusal);
    assert_eq!(direct.decision.reason, DecisionReason::SafetyViolation);
}

#[test]
fn legacy_memory_tss_index_is_rust_backed() {
    let centroids = [(0.1, 0.1), (0.8, 1.2), (1.6, 2.4), (2.4, 3.1)];
    let mut grid = SphericalGridHashIndex::<4>::new(centroids);
    grid.insert(2, 42);
    grid.insert_centroid(1, (2.8, 4.8), 99);
    assert_eq!(auto_sized_dimensions(17), (5, 10));

    let (slot, payload) = grid.locate_with_payload(centroids[2]);
    assert_eq!(slot, 2);
    assert_eq!(payload, 42);
    let (slot, payload) = grid.locate_with_payload((2.8, 4.8));
    assert_eq!(slot, 1);
    assert_eq!(payload, 99);
    assert!(grid.stats().total_centroids >= 4);
    grid.build(centroids);
    assert_eq!(grid.locate_legacy(centroids[0]), 0);

    let mut empty = SphericalGridHashIndex::<0>::new([]);
    assert_eq!(empty.locate_legacy((0.1, 0.2)), EMPTY_LOCATE);

    let eps = epsilon_from_chebyshev(0.2, 0.1, 2.0);
    assert!(eps > 0.2);
    let bound = tss_retrieval_bound(1_000_000, 1_000, 5, 128);
    assert!(bound.epsilon_o1_ops > 0);
    assert!(bound.speedup_vs_brute > 1.0);

    let verifier = TssVerifier::<4>::new(centroids, 0.1);
    assert!(verifier.full_verification(1_000_000, 5, 128).theorem_holds);
}

#[test]
fn legacy_world_model_math_is_rust_backed() {
    let operator = TelemetryOperator::<4>::new(0.06, 0.2, 0.1, 0.9);
    assert!(operator.adaptive_gain(0.45) > 0.0);
    let scm_report = SpectralContractionVerifier::<4>::new(0.2)
        .full_verification([10.0; 4], [0.0; 4], 200, 1e-6);
    assert!(scm_report.theorem_holds);

    let predictor = LatentPredictor::<4, 2>::new([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]], 0.1);
    let projected = predictor.project(&[1.0, 1.0]);
    let stepped = predictor.step(&[0.0; 4], &[1.0, 1.0], &projected);
    assert!(stepped.iter().any(|value| value.abs() > 0.0));

    let gov = GovernorConvergenceAnalyzer::new(0.01, 0.05, 1.0, 0.05, 1.0, 0.5);
    assert!(gov.theoretical_analysis().gain_margin_stable);
    let gov_report: GovernorTheoremVerification = gov.verify_theorem(200, 0.5, 0.2);
    assert!(gov_report.theorem_holds);
    let history: GovernorSimulationHistory =
        gov.simulate_measurements(0.5, &[0.2, 0.25, 0.15, 0.2]);
    assert_eq!(history.errors.len(), 4);
    assert_eq!(history.epsilons.len(), 5);
    let default_history = gov.simulate_default_history(4, 0.5);
    assert_eq!(default_history.errors.len(), 4);

    let signal = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
    assert!(spectral_entropy(&signal).is_finite());
    let transform: HaarWaveletTransform = haar_wavelet_transform(&signal);
    assert_eq!(transform.approximation.len(), 1);
    assert!(!transform.details.is_empty());
    let scale_entropy = wavelet_entropy_per_scale(&signal);
    assert!(!scale_entropy.is_empty());

    let mut coherence = SpectralCoherenceTracker::new(16, 1.0);
    coherence.update(1.0, 0.1);
    let update = coherence.update(0.5, 0.1);
    assert!(update.coherence >= 0.0);
    assert!(coherence.predict_betti_survival(8, 1.0, 0.5, 0.25) > 0.0);

    let ball = PoincareBall::new(1.0);
    let origin = [0.0, 0.0];
    let point = [0.1, 0.0];
    assert!(ball.distance(&origin, &point) > 0.0);

    let angular_value = AngularMomentumTracker::compute(
        &[0.1, 0.0],
        &[[0.1, 0.0], [0.0, 0.1]],
        &[1.0, 0.5],
        &[true, true],
    );
    assert!(angular_value >= 0.0);
    assert!(verify_plasticity_bound(0.009, 0.01, 0.0, 10.0, 0.01));
}

#[test]
fn legacy_hyperbolic_capacity_is_rust_backed() {
    let hyp = hyperbolic_distortion_bound(1.0, 128, 10);
    let euc = euclidean_distortion_bound(4, 128, 10);
    let ratio = separation_ratio(1.0, 4, 10);

    assert!((hyp - 0.0015625).abs() < 1e-12);
    assert!(euc > 0.09 && euc < 0.10);
    assert!(ratio > 62.0 && ratio < 63.0);

    let h100 = h100_analysis();
    assert_eq!(h100.dim, 4096);
    assert_eq!(h100.depth, 50);
    assert!(!h100.fits_h100_80gb);

    let report = HcsVerifier::new(128, 1.0).verify_theorem(4, 8);
    assert_eq!(report.branching_factor, 4);
    assert_eq!(report.depth, 8);
    assert!(report.tree_nodes > 0);
    assert!(report.separation_ratio_theory > 38.0);
    assert!(report.hyperbolic_better);
}

#[test]
fn legacy_thermodynamic_plasticity_is_rust_backed() {
    let bit_floor = landauer_energy_per_bit(300.0);
    assert!((bit_floor - (K_BOLTZMANN * 300.0 * LN2)).abs() < 1e-35);

    let update_floor = min_energy_per_update(350_000_000, 16, 300.0);
    assert!(update_floor > 1.6e-11 && update_floor < 1.7e-11);

    let hot_ratio = max_sustainable_hot_ratio(70_000_000_000, 700.0, 20.0, 16, 300.0);
    assert!(hot_ratio > 1.0e10);

    let peaked_entropy = gibbs_entropy(&[1.0, 0.0, 0.0, 0.0]);
    let uniform_entropy = gibbs_entropy(&[1.0, 1.0, 1.0, 1.0]);
    assert!(peaked_entropy < uniform_entropy);

    let lr = thermodynamic_lr(0.01, peaked_entropy, uniform_entropy);
    assert!(lr > 0.009);

    let delta_f = helmholtz_free_energy_change(0.0, 300.0, uniform_entropy - peaked_entropy);
    assert!(delta_f < 0.0);

    let analyzer = ThermodynamicAnalyzer::h100_default();
    let energy = analyzer.energy_analysis(20.0);
    assert_eq!(energy.hot_params, 350_000_000);
    assert!(energy.orders_above_landauer > 12.0);
    assert!(energy.thermodynamic_headroom > 1.0e12);

    let cluster = analyzer.h100_cluster_analysis(8);
    assert_eq!(cluster.n_gpus, 8);
    assert!(cluster.max_hot_ratio_cluster > energy.max_hot_ratio_thermodynamic);

    let lr_report = analyzer.lr_analysis(&[1.0, 1.0, 1.0, 1.0], 0.01);
    assert!(lr_report.lr_agreement);
    assert!(lr_report.eta_gibbs < 1e-12);
}
