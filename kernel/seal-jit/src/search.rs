//! Shrinking-budget allocator: how much exploration a key is still owed.
//!
//! # The schedule
//!
//! One function, [`explore_budget`], defines the whole policy:
//!
//! ```text
//! explore_budget(k, n) = k.saturating_sub(n)
//! ```
//!
//! where `k` is the candidate-set size and `n` is the number of observations
//! recorded for the key since it was last demoted. It has exactly the three
//! properties the rest of the crate depends on, and each one is a test:
//!
//! - **Non-increasing in `n`.** `budget(k, n+1) <= budget(k, n)` for all `n`.
//!   This is the property the whole "every loop is a little faster" claim rests
//!   on: work spent searching can only ever shrink.
//! - **Absorbing at zero.** Once `n >= k` the budget is 0 and stays 0. The
//!   thousandth repetition explores nothing.
//! - **Complete.** Summed over the first `k` observations it equals
//!   `k + (k-1) + ... + 1`, but more usefully, observation `n < k` probes
//!   candidate index `n`, so every candidate is probed exactly once before the
//!   budget closes. Without this the memo could converge to *a* choice rather
//!   than the *cheapest* choice.
//!
//! # Why not multi-sweep successive halving
//!
//! Successive halving earns its keep when probes come at multiple fidelities:
//! give every arm a small budget, keep the better half, give the survivors a
//! larger budget, repeat. The extra sweeps buy a *better estimate* of an arm
//! already probed.
//!
//! This crate has one fidelity. A probe is a full execution and the caller
//! hands back the cost of it; there is no cheaper look and no more expensive
//! one. Re-probing a survivor under a deterministic cost source returns the
//! number already on file, so every sweep after the first is pure waste that
//! shows up directly in the metric this crate exists to improve. Under a
//! single-fidelity oracle successive halving degenerates to "sweep once, take
//! the argmin", and that is what is implemented, deliberately and by that name.
//!
//! A geometric schedule such as `k >> n` would shrink faster but would close
//! the budget before every candidate had been probed, forfeiting the
//! completeness property above. Linear decay is the fastest schedule that keeps
//! it.

use crate::memo::Entry;
use crate::{Decision, Error, PlanChoice, Rung};

/// Exploratory probes still permitted for a key with `candidates` candidates
/// that has been observed `observations` times since its last demotion.
///
/// See the module docs for the three properties this schedule guarantees.
#[must_use]
pub const fn explore_budget(candidates: u32, observations: u32) -> u32 {
    candidates.saturating_sub(observations)
}

/// Pick a choice for the next repetition of `key`'s workload.
///
/// `entry` is the memo's current record for the key, or `None` if it has never
/// been seen.
pub(crate) fn select(entry: Option<&Entry>, candidates: &[PlanChoice]) -> Result<Decision, Error> {
    let Some(&first) = candidates.first() else {
        return Err(Error::EmptyCandidateSet);
    };
    let len = u32::try_from(candidates.len()).unwrap_or(u32::MAX);

    let Some(entry) = entry else {
        // Never seen: full budget, probe candidate 0.
        return Ok(Decision {
            choice: first,
            exploring: true,
            rung: Rung::Generic,
        });
    };

    if explore_budget(len, entry.observations) > 0 {
        // Observation n probes candidate n. The modulo is unreachable while the
        // budget is open (`observations < len`) and is present only so that a
        // caller who shrinks a candidate set cannot make this index out of range.
        let index = (entry.observations as usize) % candidates.len();
        let choice = candidates.get(index).copied().unwrap_or(first);
        return Ok(Decision {
            choice,
            exploring: true,
            rung: entry.rung,
        });
    }

    if candidates.contains(&entry.best) {
        return Ok(Decision {
            choice: entry.best,
            exploring: false,
            rung: entry.rung,
        });
    }

    // Contract violation: the candidate set changed under a stable key, so the
    // recorded incumbent is not runnable. Sweep the new set rather than hand
    // back a choice the caller cannot execute. `Memo::forget` is the supported
    // way to do this on purpose.
    let index = (entry.observations as usize) % candidates.len();
    let choice = candidates.get(index).copied().unwrap_or(first);
    Ok(Decision {
        choice,
        exploring: true,
        rung: entry.rung,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Memo, PlanKey};
    use alloc::vec::Vec;

    #[test]
    fn budget_is_non_increasing_in_observations() {
        for k in 0..64u32 {
            for n in 0..256u32 {
                assert!(
                    explore_budget(k, n + 1) <= explore_budget(k, n),
                    "budget rose from n={n} to n={} at k={k}",
                    n + 1
                );
            }
        }
    }

    #[test]
    fn budget_reaches_zero_at_k_and_stays_there() {
        let k = 8;
        assert_eq!(explore_budget(k, 0), k);
        assert_eq!(explore_budget(k, k - 1), 1);
        for n in k..10_000 {
            assert_eq!(explore_budget(k, n), 0, "budget reopened at n={n}");
        }
        assert_eq!(explore_budget(k, u32::MAX), 0);
    }

    #[test]
    fn the_sweep_probes_every_candidate_exactly_once() {
        let candidates: Vec<PlanChoice> = (0..8).map(PlanChoice).collect();
        let key = PlanKey(1);
        let mut memo = Memo::new(4);
        let mut probed = Vec::new();

        for _ in 0..candidates.len() {
            let d = memo.select(key, &candidates).expect("non-empty");
            assert!(d.exploring, "budget closed early");
            probed.push(d.choice);
            memo.observe(key, d.choice, 1000);
        }
        assert_eq!(probed, candidates);

        let d = memo.select(key, &candidates).expect("non-empty");
        assert!(!d.exploring, "budget must be closed after a full sweep");
    }

    #[test]
    fn an_empty_candidate_set_is_refused() {
        let memo = Memo::new(4);
        assert_eq!(memo.select(PlanKey(1), &[]), Err(Error::EmptyCandidateSet));
    }

    #[test]
    fn a_shrunken_candidate_set_never_returns_an_unrunnable_choice() {
        let wide: Vec<PlanChoice> = (0..8).map(PlanChoice).collect();
        let key = PlanKey(1);
        let mut memo = Memo::new(4);
        for i in 0..8u64 {
            let d = memo.select(key, &wide).expect("non-empty");
            memo.observe(key, d.choice, 100 - i); // last candidate wins
        }
        assert_eq!(memo.best(key), Some(PlanChoice(7)));

        // Caller violates the stable-candidate-set contract.
        let narrow = [PlanChoice(0), PlanChoice(1)];
        for _ in 0..16 {
            let d = memo.select(key, &narrow).expect("non-empty");
            assert!(
                narrow.contains(&d.choice),
                "handed back a choice not in the set"
            );
            memo.observe(key, d.choice, 5000);
        }
    }
}
