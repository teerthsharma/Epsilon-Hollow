//! Specialisation ladder: how much the memo is willing to commit to a plan.
//!
//! A rung is a statement about accumulated evidence, not about generated code.
//! This crate generates no code; the rung is a label the caller may use to
//! decide how aggressively *it* wants to specialise. The ladder's only job is
//! to raise that label when evidence accumulates and drop it when evidence is
//! contradicted.
//!
//! # The rule, stated exactly
//!
//! Let `best` be the recorded incumbent choice for a key, `best_cost` its
//! recorded cost, and `confirmations` a counter reset to 0 on every incumbent
//! change and on every demotion. On each [`Memo::observe`](crate::Memo::observe)
//! of `(choice, cost)` for an existing entry:
//!
//! - **Regression → demote.** If `choice == best` and
//!   `cost > best_cost + (best_cost >> REGRESSION_SHIFT)` — that is, the
//!   incumbent re-measured more than 25% worse than its own record — the entry
//!   drops to [`Rung::Generic`], `confirmations` resets to 0, `best_cost` is
//!   overwritten with the freshly measured `cost`, and the observation counter
//!   that drives the search budget resets to 0. The reset of the observation
//!   counter is the part that matters: it is what re-opens the exploration
//!   budget so the next repetitions re-sweep the candidate set instead of
//!   replaying a stale winner forever.
//!
//!   Demotion goes to `Generic` in one step, not one rung at a time. The
//!   evidence that justified *every* rung above `Generic` was the single
//!   proposition "this choice keeps winning at this cost". A regression
//!   falsifies that proposition outright, so every rung built on it falls at
//!   once.
//!
//! - **Confirmation → promote.** If `choice == best` and the cost is within the
//!   regression band, `confirmations` increments; `best_cost` also ratchets
//!   down if the new cost is lower. When `confirmations` reaches
//!   [`CONFIRMATIONS_PER_RUNG`] the entry moves up one rung and `confirmations`
//!   resets to 0. At the top rung `confirmations` saturates at the threshold
//!   rather than wrapping.
//!
//! - **New winner → reset.** If `choice != best` and `cost < best_cost`, the
//!   challenger becomes the incumbent, and the entry returns to
//!   [`Rung::Generic`] with `confirmations` at 0. A new plan has earned no
//!   specialisation yet.
//!
//! - Anything else holds.
//!
//! # What the ladder does not do
//!
//! It detects drift only by re-measuring the incumbent. A workload change that
//! leaves the incumbent's cost untouched while making a different candidate
//! cheaper produces no regression and therefore no re-search. Choosing to
//! detect that would require periodic re-probing of non-incumbents, which would
//! break the monotone-cost property this crate is built to demonstrate. The
//! trade is deliberate and it is a real limitation, not a simplification.

use crate::memo::Entry;
use crate::PlanChoice;

/// Consecutive confirmations of the incumbent required to climb one rung.
///
/// Four, because that is the smallest count that outlives a single anomalous
/// measurement in both directions and still promotes inside a warm-up window
/// of practical size. It is a constant rather than a knob: nothing in this
/// crate varies it, and a config field for a value nobody changes is a lie
/// about flexibility.
pub const CONFIRMATIONS_PER_RUNG: u16 = 4;

/// Right-shift applied to the recorded best cost to form the regression band.
///
/// `2` gives a 25% tolerance: a re-measurement of the incumbent is treated as a
/// regression only once it exceeds `best_cost * 1.25`. Set above typical
/// measurement jitter so that noise does not repeatedly demote a healthy plan;
/// a caller whose cost source is noisier than that will see spurious
/// re-searches, and this is the number to revisit.
pub const REGRESSION_SHIFT: u32 = 2;

/// How far the memo has committed to a plan for one key.
///
/// The names describe what a caller *may* do at each level. This crate does
/// none of it — it only says when the evidence supports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rung {
    /// No commitment. The choice is still being searched or has just changed.
    Generic,
    /// The same choice has won consistently for this key, which encodes fixed
    /// shapes and dtypes. A caller may specialise on those.
    ShapeLocked,
    /// Top rung. The choice has survived a further round of confirmations; a
    /// caller may fold constants and drop dispatch entirely.
    Folded,
}

impl Rung {
    /// The next rung up, or `None` at the top.
    #[must_use]
    pub const fn next(self) -> Option<Self> {
        match self {
            Self::Generic => Some(Self::ShapeLocked),
            Self::ShapeLocked => Some(Self::Folded),
            Self::Folded => None,
        }
    }

    /// Wire discriminant used by the persisted memo format.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Generic => 0,
            Self::ShapeLocked => 1,
            Self::Folded => 2,
        }
    }

    /// Inverse of [`Rung::to_u8`]. Returns `None` for undefined discriminants
    /// so a decoder can refuse them by name instead of guessing.
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Generic),
            1 => Some(Self::ShapeLocked),
            2 => Some(Self::Folded),
            _ => None,
        }
    }
}

/// What one observation did to an entry's ladder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// Nothing changed: a confirmation short of a promotion, or a challenger
    /// that failed to beat the incumbent.
    Held,
    /// A challenger beat the incumbent and replaced it. Also reported for the
    /// very first observation of a key.
    NewBest,
    /// Enough confirmations accumulated; the entry climbed one rung.
    Promoted,
    /// The incumbent re-measured outside its regression band. The entry fell to
    /// [`Rung::Generic`] and its exploration budget was re-opened.
    Demoted,
    /// A first sight of a key was not admitted: the table was full and no
    /// resident entry had decayed far enough to be displaced. Nothing was
    /// recorded, so the next repetition of this key starts from nothing again.
    /// Reported rather than swallowed, because a caller that sees this
    /// persistently is running a working set the memo cannot hold. See
    /// [`Memo::observe`](crate::Memo::observe).
    Refused,
}

/// Upper edge of the regression band for a recorded cost.
///
/// Saturating, so a pathological `best_cost` near `u64::MAX` cannot wrap into a
/// band that reports every observation as a regression.
#[must_use]
pub const fn regression_ceiling(best_cost: u64) -> u64 {
    best_cost.saturating_add(best_cost >> REGRESSION_SHIFT)
}

/// Apply one observation to an existing entry. See the module docs for the rule.
///
/// Insertion of a brand-new key is [`Memo::observe`](crate::Memo::observe)'s
/// job, not this function's: a first observation has no incumbent to compare
/// against and is unconditionally [`Transition::NewBest`].
pub(crate) fn record(entry: &mut Entry, choice: PlanChoice, cost: u64) -> Transition {
    entry.credit = entry.credit.saturating_add(1);
    entry.observations = entry.observations.saturating_add(1);

    if choice != entry.best {
        if cost < entry.best_cost {
            entry.best = choice;
            entry.best_cost = cost;
            entry.confirmations = 0;
            entry.rung = Rung::Generic;
            return Transition::NewBest;
        }
        return Transition::Held;
    }

    if cost > regression_ceiling(entry.best_cost) {
        entry.best_cost = cost;
        entry.confirmations = 0;
        entry.observations = 0;
        entry.rung = Rung::Generic;
        return Transition::Demoted;
    }

    if cost < entry.best_cost {
        entry.best_cost = cost;
    }
    entry.confirmations = entry.confirmations.saturating_add(1);
    if entry.confirmations >= CONFIRMATIONS_PER_RUNG {
        if let Some(next) = entry.rung.next() {
            entry.rung = next;
            entry.confirmations = 0;
            return Transition::Promoted;
        }
        entry.confirmations = CONFIRMATIONS_PER_RUNG;
    }
    Transition::Held
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Memo, PlanKey};

    fn seeded(best_cost: u64) -> Entry {
        Entry {
            key: PlanKey(1),
            best: PlanChoice(7),
            best_cost,
            observations: 0,
            credit: 0,
            confirmations: 0,
            rung: Rung::Generic,
        }
    }

    #[test]
    fn promotion_requires_the_full_confirmation_count() {
        let mut entry = seeded(100);
        for _ in 0..CONFIRMATIONS_PER_RUNG - 1 {
            assert_eq!(record(&mut entry, PlanChoice(7), 100), Transition::Held);
            assert_eq!(entry.rung, Rung::Generic);
        }
        assert_eq!(record(&mut entry, PlanChoice(7), 100), Transition::Promoted);
        assert_eq!(entry.rung, Rung::ShapeLocked);
        assert_eq!(entry.confirmations, 0);
    }

    #[test]
    fn ladder_saturates_at_the_top_rung() {
        let mut entry = seeded(100);
        for _ in 0..40 {
            record(&mut entry, PlanChoice(7), 100);
        }
        assert_eq!(entry.rung, Rung::Folded);
        assert_eq!(entry.confirmations, CONFIRMATIONS_PER_RUNG);
        assert_eq!(record(&mut entry, PlanChoice(7), 100), Transition::Held);
        assert_eq!(entry.rung, Rung::Folded);
    }

    #[test]
    fn regression_demotes_to_generic_in_one_step_and_reopens_the_budget() {
        let mut entry = seeded(100);
        for _ in 0..12 {
            record(&mut entry, PlanChoice(7), 100);
        }
        assert_eq!(entry.rung, Rung::Folded);
        assert!(entry.observations > 0);

        // 125 is exactly the ceiling and must NOT demote; 126 must.
        assert_eq!(regression_ceiling(100), 125);
        assert_eq!(record(&mut entry, PlanChoice(7), 125), Transition::Held);
        assert_eq!(entry.rung, Rung::Folded);

        assert_eq!(record(&mut entry, PlanChoice(7), 126), Transition::Demoted);
        assert_eq!(entry.rung, Rung::Generic);
        assert_eq!(entry.observations, 0, "budget must reopen on demotion");
        assert_eq!(entry.best_cost, 126, "record must track the new reality");
        assert_eq!(
            entry.credit, 14,
            "the eviction credit must survive a demotion"
        );
    }

    #[test]
    fn a_new_winner_resets_the_ladder() {
        let mut entry = seeded(100);
        for _ in 0..8 {
            record(&mut entry, PlanChoice(7), 100);
        }
        assert_eq!(entry.rung, Rung::Folded);
        assert_eq!(record(&mut entry, PlanChoice(3), 50), Transition::NewBest);
        assert_eq!(entry.rung, Rung::Generic);
        assert_eq!(entry.best, PlanChoice(3));
        assert_eq!(entry.best_cost, 50);
    }

    #[test]
    fn a_losing_challenger_changes_nothing_but_the_counters() {
        let mut entry = seeded(100);
        assert_eq!(record(&mut entry, PlanChoice(3), 400), Transition::Held);
        assert_eq!(entry.best, PlanChoice(7));
        assert_eq!(entry.best_cost, 100);
        assert_eq!(entry.observations, 1);
    }

    #[test]
    fn regression_ceiling_saturates_instead_of_wrapping() {
        assert_eq!(regression_ceiling(u64::MAX), u64::MAX);
        assert_eq!(regression_ceiling(0), 0);
    }

    #[test]
    fn first_observation_of_a_key_is_a_new_best() {
        let mut memo = Memo::new(4);
        assert_eq!(
            memo.observe(PlanKey(9), PlanChoice(1), 10),
            Transition::NewBest
        );
    }
}
