//! The falsifiable claim, and the control that lets it fail.
//!
//! Claim under test: over N repetitions of an identical workload, the
//! cumulative cost per repetition is monotone non-increasing after a warm-up
//! window of exactly `candidates.len()` repetitions.
//!
//! That claim, stated with `<=`, is worthless on its own — a perfectly flat
//! series satisfies it. So the discriminating statistic used throughout this
//! file is *strict* decrease of the running mean at every repetition after
//! warm-up, and every assertion of it is paired with the same assertion run
//! against a control that retains nothing. The control must fail the predicate.
//! If it ever starts passing, the predicate has stopped measuring the memo.
//!
//! Cost model: the library measures nothing, so "cost" here is a synthetic
//! per-choice constant supplied by this file. The cost of *searching* is not a
//! separate line item — it is the cost of having executed a suboptimal choice
//! during exploration, which is what autotuning actually costs in practice.

use seal_jit::{Memo, PlanChoice, PlanKey, Rung, Transition};

const K: usize = 8;
const REPS: usize = 1000;
const KEY: PlanKey = PlanKey(0x5EA1_0000_0000_0001);

const CANDIDATES: [PlanChoice; K] = [
    PlanChoice(0),
    PlanChoice(1),
    PlanChoice(2),
    PlanChoice(3),
    PlanChoice(4),
    PlanChoice(5),
    PlanChoice(6),
    PlanChoice(7),
];

/// Baseline workload. Cheapest choice is index 4 at 480; the spread is wide
/// enough that picking wrong is expensive and the ordering is deliberately not
/// monotone, so a memo that simply keeps the first or last probe fails.
const COST_A: [u64; K] = [900, 1400, 620, 1100, 480, 1500, 730, 1250];

/// Drifted workload. The old winner (4) becomes the second *worst* choice and
/// a previously mediocre choice (6) becomes the cheapest by a wide margin.
const COST_B: [u64; K] = [900, 1400, 620, 1100, 2000, 1500, 300, 1250];

/// Rotating-working-set workload: `COST_A` with the cost of candidate 0 and
/// candidate 5 exchanged, so the total, the mean and the minimum are unchanged
/// and candidate 0 is the single *most expensive* choice.
///
/// That last property is the whole point of a separate fixture. A memo that
/// retains nothing always hands back candidate 0, because a key with no record
/// gets the first candidate. Under `COST_A` candidate 0 costs 900 against a
/// mean of 997.5, so a memo that had degraded to no memo at all would still
/// beat the untuned baseline — by accident, on a fixture detail. Here the same
/// degradation costs 1500 against the same mean, so the baseline comparison
/// measures retention rather than the ordering of the array.
const COST_C: [u64; K] = [1500, 1400, 620, 1100, 480, 900, 730, 1250];

fn cost_of(costs: &[u64; K], choice: PlanChoice) -> u64 {
    costs.get(choice.0 as usize).copied().unwrap_or(u64::MAX)
}

fn cheapest(costs: &[u64; K]) -> (PlanChoice, u64) {
    let mut best = (PlanChoice(0), costs[0]);
    for (index, &cost) in costs.iter().enumerate() {
        if cost < best.1 {
            best = (PlanChoice(index as u32), cost);
        }
    }
    best
}

fn total(costs: &[u64; K]) -> u64 {
    costs.iter().sum()
}

/// Exact rational test: is `sum(0..=i)/(i+1)` strictly less than
/// `sum(0..=i-1)/i` for every `i > warmup`? Cross-multiplied in `u128` so there
/// is no floating point anywhere in the decision.
fn running_mean_strictly_decreasing_after(per_rep: &[u64], warmup: usize) -> bool {
    let mut running: u128 = per_rep
        .iter()
        .take(warmup + 1)
        .map(|&c| u128::from(c))
        .sum();
    for (index, &cost) in per_rep.iter().enumerate().skip(warmup + 1) {
        let previous = running;
        running += u128::from(cost);
        // running/(index+1) < previous/index
        if running * (index as u128) >= previous * ((index + 1) as u128) {
            return false;
        }
    }
    true
}

fn running_mean(per_rep: &[u64], upto: usize) -> f64 {
    let sum: u64 = per_rep.iter().take(upto + 1).sum();
    sum as f64 / (upto + 1) as f64
}

/// The boosted arm: one persistent memo, one select/observe per repetition.
fn run_boosted(reps: usize, costs: &[u64; K]) -> Vec<u64> {
    let mut memo = Memo::new(64);
    let mut per_rep = Vec::with_capacity(reps);
    for _ in 0..reps {
        let decision = memo
            .select(KEY, &CANDIDATES)
            .expect("candidate set is non-empty");
        let cost = cost_of(costs, decision.choice);
        memo.observe(KEY, decision.choice, cost);
        per_rep.push(cost);
    }
    per_rep
}

/// The control arm: the memo is disabled by discarding it between repetitions.
///
/// An autotuner with no persistence has to re-derive the winner every time it
/// runs, and deriving it means executing every candidate. So each repetition is
/// charged a full sweep. Both arms drive the same library code; the only
/// difference is whether anything survives the repetition boundary.
fn run_control(reps: usize, costs: &[u64; K]) -> Vec<u64> {
    let mut per_rep = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut memo = Memo::new(64);
        let mut swept = 0u64;
        for _ in 0..K {
            let decision = memo
                .select(KEY, &CANDIDATES)
                .expect("candidate set is non-empty");
            let cost = cost_of(costs, decision.choice);
            memo.observe(KEY, decision.choice, cost);
            swept += cost;
        }
        per_rep.push(swept);
    }
    per_rep
}

/// The untuned arm: pick a candidate without tuning, one execution per
/// repetition, no sweep.
///
/// # Why this baseline is required
///
/// `run_control` charges a full sweep per repetition, so beating it only shows
/// that the memo eliminated a *search*. It does not show that the memo found a
/// choice worth having — a memo that reliably converged to the *worst*
/// candidate would still beat it by a wide margin.
///
/// This arm answers the separate question: was the tuning worth doing at all,
/// versus picking a candidate arbitrarily and executing it once? Round-robin is
/// used instead of an RNG because over `REPS` a multiple of `K` its total is
/// exactly `REPS/K · Σcosts` — the closed-form expectation of uniform random
/// selection — so the baseline is deterministic and the assertion cannot flake
/// on a seed.
fn run_untuned_roundrobin(reps: usize, costs: &[u64; K]) -> Vec<u64> {
    (0..reps).map(|r| costs[r % K]).collect()
}

#[test]
fn booster_beats_an_untuned_arbitrary_choice_not_just_the_search() {
    let boosted: u64 = run_boosted(REPS, &COST_A).iter().sum();
    let untuned: u64 = run_untuned_roundrobin(REPS, &COST_A).iter().sum();
    let floor = REPS as u64 * COST_A.iter().copied().min().expect("non-empty");
    let worst = REPS as u64 * COST_A.iter().copied().max().expect("non-empty");

    // Round-robin over a whole number of cycles is exactly the uniform-random
    // expectation. Assert that rather than trusting the arithmetic.
    assert_eq!(
        REPS % K,
        0,
        "round-robin is only the random expectation over whole cycles"
    );
    assert_eq!(
        untuned,
        (REPS / K) as u64 * COST_A.iter().sum::<u64>(),
        "untuned arm is not REPS/K full cycles"
    );

    println!("\n=== booster vs UNTUNED arbitrary choice (the decisive baseline) ===");
    println!("  boosted total           {boosted}");
    println!("  untuned total           {untuned}   (= uniform-random expectation)");
    println!("  oracle floor            {floor}");
    println!("  worst-choice ceiling    {worst}");
    println!(
        "  speedup vs untuned      {:.3}x",
        untuned as f64 / boosted as f64
    );
    println!(
        "  fraction of the floor-to-random gap captured: {:.4}",
        (untuned - boosted) as f64 / (untuned - floor) as f64
    );

    assert!(
        boosted < untuned,
        "tuning must beat an arbitrary choice, else the memo only removed a search it created"
    );
    // The honest target: land essentially on the oracle floor, not merely below
    // the random mean. Warm-up is the only permitted excess.
    let warmup_excess: u64 =
        COST_A.iter().sum::<u64>() - K as u64 * COST_A.iter().copied().min().expect("non-empty");
    assert_eq!(
        boosted,
        floor + warmup_excess,
        "boosted cost must be the oracle floor plus exactly one warm-up sweep of excess"
    );
}

#[test]
fn booster_beats_the_no_memo_control_over_1000_repetitions() {
    let boosted = run_boosted(REPS, &COST_A);
    let control = run_control(REPS, &COST_A);

    let (best_choice, best_cost) = cheapest(&COST_A);
    let boosted_total: u64 = boosted.iter().sum();
    let control_total: u64 = control.iter().sum();
    let floor_total = best_cost * REPS as u64;

    // Closed-form prediction of the schedule: one sweep of every candidate,
    // then the winner forever. Asserted exactly, so a change to the budget
    // schedule cannot slip through unnoticed.
    let predicted = total(&COST_A) + (REPS as u64 - K as u64) * best_cost;
    assert_eq!(
        boosted_total, predicted,
        "boosted total departed from the schedule prediction"
    );
    assert_eq!(
        control_total,
        total(&COST_A) * REPS as u64,
        "control total is not N full sweeps"
    );

    println!("\n=== seal-jit booster vs control, N={REPS} repetitions, K={K} candidates ===");
    println!("  per-choice costs        {COST_A:?}");
    println!("  cheapest choice         {best_choice:?} at cost {best_cost}");
    println!("  boosted total cost      {boosted_total}");
    println!("  control total cost      {control_total}   (memo discarded each repetition)");
    println!("  oracle floor            {floor_total}   (cheapest choice from repetition 0)");
    println!(
        "  speedup vs control      {:.3}x",
        control_total as f64 / boosted_total as f64
    );
    println!(
        "  overhead vs floor       {:.4}x  ({} excess cost, all of it warm-up)",
        boosted_total as f64 / floor_total as f64,
        boosted_total - floor_total
    );
    println!("  running mean rep 0      {:.2}", running_mean(&boosted, 0));
    println!(
        "  running mean rep {:<6} {:.2}",
        K - 1,
        running_mean(&boosted, K - 1)
    );
    println!(
        "  running mean rep {:<6} {:.2}",
        REPS - 1,
        running_mean(&boosted, REPS - 1)
    );
    println!(
        "  control running mean    {:.2} at rep 0, {:.2} at rep {}  (delta {:.2})",
        running_mean(&control, 0),
        running_mean(&control, REPS - 1),
        REPS - 1,
        running_mean(&control, REPS - 1) - running_mean(&control, 0)
    );

    assert!(
        boosted_total < control_total / 10,
        "booster must be more than 10x under the control"
    );
    assert_eq!(
        boosted_total - floor_total,
        total(&COST_A) - K as u64 * best_cost
    );
}

#[test]
fn per_repetition_cost_is_monotone_non_increasing_after_the_warm_up_window() {
    let boosted = run_boosted(REPS, &COST_A);

    // Warm-up window is exactly K repetitions: `explore_budget` permits one
    // probe per candidate and then closes. Documented, not fudged.
    for index in K..REPS {
        let previous = boosted[index - 1];
        let current = boosted[index];
        assert!(
            current <= previous,
            "cost rose at repetition {index}: {previous} -> {current}"
        );
    }
    assert!(boosted[K..].iter().all(|&c| c == cheapest(&COST_A).1));
}

#[test]
fn running_mean_strictly_falls_for_the_booster_and_is_exactly_flat_for_the_control() {
    let boosted = run_boosted(REPS, &COST_A);
    let control = run_control(REPS, &COST_A);
    let warmup = K - 1;

    assert!(
        running_mean_strictly_decreasing_after(&boosted, warmup),
        "booster running mean failed to fall at every repetition after warm-up"
    );

    // The negative control. A flat series satisfies "non-increasing", which is
    // why the predicate above is strict. It must reject the control.
    assert!(
        !running_mean_strictly_decreasing_after(&control, warmup),
        "the control passed the discriminating predicate, so the predicate is not measuring the memo"
    );

    // Quantify the difference rather than asserting a direction.
    let first = control[0];
    assert!(
        control.iter().all(|&c| c == first),
        "control per-repetition cost is not constant"
    );
    let control_delta = running_mean(&control, REPS - 1) - running_mean(&control, 0);
    let boosted_delta = running_mean(&boosted, REPS - 1) - running_mean(&boosted, warmup);
    assert_eq!(control_delta, 0.0, "control trend must be exactly zero");
    assert!(
        boosted_delta < -400.0,
        "booster running mean fell by only {boosted_delta:.2}"
    );
    println!(
        "\n  trend over {REPS} reps: booster {boosted_delta:+.2} cost/rep, control {control_delta:+.2} cost/rep"
    );
}

#[test]
fn converges_to_the_genuinely_cheapest_choice_not_merely_to_a_choice() {
    let (best_choice, best_cost) = cheapest(&COST_A);
    let mut memo = Memo::new(64);
    for _ in 0..REPS {
        let decision = memo
            .select(KEY, &CANDIDATES)
            .expect("candidate set is non-empty");
        memo.observe(KEY, decision.choice, cost_of(&COST_A, decision.choice));
    }

    let entry = memo.entry(KEY).expect("key was observed");
    assert_eq!(
        entry.best, best_choice,
        "converged to a choice that is not the cheapest"
    );
    assert_eq!(entry.best_cost, best_cost);
    // Independent of the memo: nothing in the candidate set is cheaper.
    assert!(CANDIDATES.iter().all(|&c| cost_of(&COST_A, c) >= best_cost));
    assert_eq!(
        entry.rung,
        Rung::Folded,
        "a stable winner must reach the top rung"
    );

    // Exploration really did stop: the last repetition is a replay.
    let decision = memo
        .select(KEY, &CANDIDATES)
        .expect("candidate set is non-empty");
    assert!(!decision.exploring);
    assert_eq!(decision.choice, best_choice);
}

#[test]
fn mid_run_drift_triggers_demotion_and_re_search_instead_of_locking_in_the_stale_winner() {
    const DRIFT_AT: usize = 500;
    let (stale_choice, _) = cheapest(&COST_A);
    let (drifted_choice, drifted_cost) = cheapest(&COST_B);
    assert_ne!(
        stale_choice, drifted_choice,
        "the fixture must actually drift"
    );

    let mut memo = Memo::new(64);
    let mut per_rep = Vec::with_capacity(REPS);
    let mut demotions = 0usize;
    let mut demoted_at = None;
    let mut explorations_after_drift = 0usize;

    for rep in 0..REPS {
        let costs = if rep < DRIFT_AT { &COST_A } else { &COST_B };
        let decision = memo
            .select(KEY, &CANDIDATES)
            .expect("candidate set is non-empty");
        if rep >= DRIFT_AT && decision.exploring {
            explorations_after_drift += 1;
        }
        let cost = cost_of(costs, decision.choice);
        if memo.observe(KEY, decision.choice, cost) == Transition::Demoted {
            demotions += 1;
            demoted_at.get_or_insert(rep);
        }
        per_rep.push(cost);
    }

    assert_eq!(demotions, 1, "expected exactly one demotion across the run");
    assert_eq!(
        demoted_at,
        Some(DRIFT_AT),
        "demotion did not fire on the first drifted repetition"
    );
    assert_eq!(
        explorations_after_drift, K,
        "re-search did not sweep the full candidate set"
    );

    let entry = memo.entry(KEY).expect("key was observed");
    assert_eq!(
        entry.best, drifted_choice,
        "memo kept the stale winner after drift"
    );
    assert_eq!(entry.best_cost, drifted_cost);

    // Negative control: what staying locked on the stale winner would have cost.
    let post_drift: u64 = per_rep[DRIFT_AT..].iter().sum();
    let locked = cost_of(&COST_B, stale_choice) * (REPS - DRIFT_AT) as u64;
    assert!(
        post_drift < locked,
        "re-searching was not cheaper than staying locked"
    );

    println!("\n=== drift at repetition {DRIFT_AT}: {stale_choice:?} -> {drifted_choice:?} ===");
    println!(
        "  stale winner's new cost   {}",
        cost_of(&COST_B, stale_choice)
    );
    println!("  new cheapest              {drifted_choice:?} at cost {drifted_cost}");
    println!("  cost of reps {DRIFT_AT}..{REPS}     {post_drift}   (demote, re-sweep {K}, replay)");
    println!("  cost if locked stale      {locked}");
    println!(
        "  saved by demotion         {:.3}x",
        locked as f64 / post_drift as f64
    );
    println!("  final rung                {:?}", entry.rung);
}

#[test]
fn a_memo_persisted_and_reloaded_resumes_without_re_searching() {
    let mut memo = Memo::new(64);
    for _ in 0..REPS / 2 {
        let decision = memo
            .select(KEY, &CANDIDATES)
            .expect("candidate set is non-empty");
        memo.observe(KEY, decision.choice, cost_of(&COST_A, decision.choice));
    }

    let bytes = memo.encode();
    let reloaded = Memo::decode(&bytes).expect("round trip through the wire format");
    let decision = reloaded
        .select(KEY, &CANDIDATES)
        .expect("candidate set is non-empty");

    assert!(!decision.exploring, "a reloaded memo re-opened the search");
    assert_eq!(decision.choice, cheapest(&COST_A).0);
    assert_eq!(reloaded.entry(KEY), memo.entry(KEY));
    println!(
        "\n  persisted memo is {} bytes for {} key(s)",
        bytes.len(),
        memo.len()
    );
}

/// Distinct keys in each half of the rotating workload, and the memo capacity.
/// One set fits exactly, so the second set can only be tuned by displacing the
/// first.
const SET: usize = 8;

/// Repetitions spent establishing the first set before the working set rotates.
/// A multiple of `SET`, so every established key is observed `ESTABLISH / SET`
/// times and the rotation starts on a set boundary.
const ESTABLISH: usize = 400;

/// The established set is numerically *above* the newcomers, so the eviction
/// tie-break — lowest key first — points at the newcomers. The policy has to
/// work against the tie-break rather than be rescued by it.
const ESTABLISHED_BASE: u64 = 0x5EA1_0000_0002_0000;
const NEWCOMER_BASE: u64 = 0x5EA1_0000_0001_0000;

fn key_at(rep: usize) -> PlanKey {
    let base = if rep < ESTABLISH {
        ESTABLISHED_BASE
    } else {
        NEWCOMER_BASE
    };
    PlanKey(base + (rep % SET) as u64)
}

/// One memo, capacity `SET`, driven round-robin over `SET` keys that are
/// replaced wholesale at repetition `ESTABLISH`.
fn run_rotating(reps: usize, costs: &[u64; K]) -> (Vec<u64>, Memo) {
    let mut memo = Memo::new(SET);
    let mut per_rep = Vec::with_capacity(reps);
    for rep in 0..reps {
        let key = key_at(rep);
        let decision = memo
            .select(key, &CANDIDATES)
            .expect("candidate set is non-empty");
        let cost = cost_of(costs, decision.choice);
        memo.observe(key, decision.choice, cost);
        per_rep.push(cost);
    }
    (per_rep, memo)
}

#[test]
fn a_rotated_working_set_is_tuned_rather_than_thrashed_out_of_the_table() {
    let (per_rep, memo) = run_rotating(REPS, &COST_C);
    let (best_choice, best_cost) = cheapest(&COST_C);

    let after: u64 = per_rep[ESTABLISH..].iter().sum();
    let rotated_reps = REPS - ESTABLISH;
    let untuned: u64 = run_untuned_roundrobin(rotated_reps, &COST_C).iter().sum();
    let floor = rotated_reps as u64 * best_cost;
    let thrashed = rotated_reps as u64 * COST_C[0];

    assert_eq!(
        rotated_reps % K,
        0,
        "round-robin is only the random expectation over whole cycles"
    );
    assert_eq!(
        untuned,
        (rotated_reps / K) as u64 * COST_C.iter().sum::<u64>(),
        "untuned arm is not rotated_reps/K full cycles"
    );

    println!(
        "\n=== rotating working set: {SET} keys evicted by {SET} new keys, capacity {SET} ==="
    );
    println!("  repetitions after the rotation  {rotated_reps}");
    println!("  cost after the rotation         {after}");
    println!("  untuned round-robin             {untuned}   (= uniform-random expectation)");
    println!("  cost if the table never retains {thrashed}   (every key a miss, candidate 0)");
    println!("  oracle floor                    {floor}");
    println!(
        "  speedup vs untuned              {:.3}x",
        untuned as f64 / after as f64
    );
    println!(
        "  fraction of the floor-to-random gap captured: {:.4}",
        (untuned as f64 - after as f64) / (untuned - floor) as f64
    );

    // The discriminating assertion. Without aging, every post-rotation key is
    // the strict argmin of the eviction rank the instant it is inserted, so it
    // is dropped before its next repetition: every repetition after the
    // rotation is a first sight of the key, which costs candidate 0 forever.
    assert!(
        after < untuned,
        "a rotated working set must still beat an arbitrary choice: {after} vs {untuned}"
    );
    // Stated separately so the failure mode is named, not merely implied.
    assert!(
        after < thrashed / 2,
        "post-rotation cost {after} is within 2x of retaining nothing at all ({thrashed})"
    );

    // The direct statement of what has to have happened: the newcomers are
    // resident, and each one is tuned rather than merely present.
    for offset in 0..SET as u64 {
        let key = PlanKey(NEWCOMER_BASE + offset);
        let entry = memo
            .entry(key)
            .unwrap_or_else(|| panic!("newcomer {key:?} never displaced a stale key"));
        assert_eq!(
            entry.best, best_choice,
            "newcomer {key:?} is resident but was never tuned"
        );
        assert_eq!(entry.best_cost, best_cost);
    }
    // And the stale set really is gone, so this is displacement rather than a
    // capacity clamp quietly handing out extra slots.
    assert_eq!(memo.len(), SET);
    for offset in 0..SET as u64 {
        assert!(
            memo.entry(PlanKey(ESTABLISHED_BASE + offset)).is_none(),
            "a stale key survived a full rotation of the working set"
        );
    }
}

#[test]
fn independent_keys_do_not_share_tuning_results() {
    let other = PlanKey(0x5EA1_0000_0000_0002);
    let mut memo = Memo::new(64);
    for _ in 0..REPS {
        let decision = memo
            .select(KEY, &CANDIDATES)
            .expect("candidate set is non-empty");
        memo.observe(KEY, decision.choice, cost_of(&COST_A, decision.choice));
    }
    assert_eq!(memo.best(KEY), Some(cheapest(&COST_A).0));
    assert_eq!(memo.best(other), None);

    let decision = memo
        .select(other, &CANDIDATES)
        .expect("candidate set is non-empty");
    assert!(
        decision.exploring,
        "an unseen key inherited another key's convergence"
    );
    assert_eq!(decision.rung, Rung::Generic);
}
