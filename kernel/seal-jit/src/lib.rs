//! `seal-jit` — a persistent autotune memo with a shrinking search budget.
//!
//! # What this crate claims
//!
//! Given a workload that is executed repeatedly under the same identity, this
//! crate makes the *later* repetitions cheaper than the earlier ones by doing
//! two things and only two things:
//!
//! 1. **It remembers.** The measured cost of every configuration it has been
//!    told about is kept in a bounded table keyed by an opaque content hash.
//!    A repetition that has already been searched is not searched again.
//! 2. **It stops looking.** The number of exploratory probes permitted for a
//!    key is a non-increasing function of how many times that key has been
//!    observed ([`search::explore_budget`]). The first encounter sweeps the
//!    whole candidate set; the thousandth explores nothing and replays the
//!    winner.
//!
//! The falsifiable claim, which `tests/booster.rs` exercises against a
//! mandatory negative control: over N repetitions of an identical workload the
//! per-repetition cost is monotone non-increasing after a warm-up window of
//! exactly `candidates.len()` repetitions, and a control that retains nothing
//! shows a perfectly flat trend over the same N.
//!
//! # What this crate is not
//!
//! - **It is not a compiler.** It emits no code, parses nothing, and has no
//!   notion of what a kernel, a tile, or an instruction is. A [`PlanChoice`] is
//!   an opaque `u32` whose meaning belongs entirely to the caller.
//! - **It does not rewrite itself.** Nothing here is self-modifying. The
//!   "getting faster" is a search that terminates, not a program that improves.
//! - **It does not measure anything.** Cost is supplied by the caller as a
//!   `u64`. There is no clock read, no RNG, and no I/O anywhere in the library.
//!   This is what makes it deterministic, testable, and usable inside a kernel
//!   where neither a clock nor an allocator-backed timer is available.
//! - **It cannot see a drift that leaves the incumbent unchanged.** Regression
//!   is detected only by re-measuring the *incumbent*. A workload change that
//!   makes some other candidate better while the incumbent stays exactly as
//!   fast is invisible to this design. See [`ladder`] for the precise rule.
//! - **It does not admit a new key for free.** Eviction is by a decayed
//!   observation credit, so a newly hot key displaces an established one only
//!   after the established one has failed to be observed across a decay window.
//!   Until then its observations are refused and its repetitions run untuned.
//!   A working set that rotates faster than the decay is tuned only in part.
//!   See [`memo::Memo::observe`].
//!
//! # Usage
//!
//! ```
//! use seal_jit::{Memo, PlanChoice, PlanKey};
//!
//! let mut memo = Memo::new(64);
//! let key = PlanKey(0xdead_beef);
//! let candidates = [PlanChoice(0), PlanChoice(1), PlanChoice(2)];
//!
//! for _ in 0..100 {
//!     let decision = memo.select(key, &candidates).expect("non-empty candidate set");
//!     let cost = run_workload(decision.choice); // caller measures, crate does not
//!     memo.observe(key, decision.choice, cost);
//! }
//! assert_eq!(memo.best(key), Some(PlanChoice(0)));
//!
//! # fn run_workload(c: PlanChoice) -> u64 { u64::from(c.0) + 1 }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

use core::fmt;

pub mod ladder;
pub mod memo;
pub mod search;

pub use ladder::{Rung, Transition};
pub use memo::{Entry, Memo};

/// Opaque identity of a plan: a content hash of whatever the caller considers
/// to define "the same workload" — kernel identity, shapes, dtypes, target
/// architecture.
///
/// This crate never inspects the value. Two workloads that should share tuning
/// results must hash equal; two that must not share results must hash unequal.
/// Getting that wrong is a caller bug this crate cannot detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanKey(pub u64);

/// Opaque identity of one configuration in a candidate set: tile size, unroll
/// factor, warp count — whatever the caller means by it.
///
/// The candidate set supplied to [`Memo::select`] must be a stable function of
/// the [`PlanKey`]. If it is not, a recorded incumbent may no longer be a
/// member of the set; `select` detects that and falls back to sweeping, but the
/// tuning history for that key is then meaningless. Call [`Memo::forget`]
/// instead when a candidate set genuinely changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanChoice(pub u32);

/// What [`Memo::select`] decided to run this repetition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// The configuration the caller should execute.
    pub choice: PlanChoice,
    /// `true` when this repetition spends exploration budget on a candidate
    /// that is not (yet) the incumbent. `false` when it replays a known winner.
    pub exploring: bool,
    /// The specialisation rung the key currently sits on.
    pub rung: Rung,
}

/// Every way a call into this crate can fail.
///
/// Decode failures are deliberately fine-grained: the memo is persisted to
/// storage and read back as untrusted bytes, so a rejection must say which
/// invariant the bytes violated rather than collapsing to "malformed".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// [`Memo::select`] was called with an empty candidate set. There is
    /// nothing to choose and no defensible default.
    EmptyCandidateSet,
    /// Encoded memo is shorter than a header plus a checksum trailer.
    Truncated,
    /// Leading magic is not `SJM1`.
    BadMagic,
    /// Format version is not one this build understands.
    UnsupportedVersion(u16),
    /// A field reserved for future use was non-zero. Refused rather than
    /// ignored, so a future writer cannot be silently misread by this reader.
    ReservedFieldSet,
    /// Trailing FNV-1a digest does not match the payload.
    ChecksumMismatch,
    /// Declared capacity is zero or exceeds [`memo::MAX_ENTRIES`].
    CapacityOutOfRange,
    /// Declared entry count exceeds the declared capacity.
    CountExceedsCapacity,
    /// Payload length is not exactly the declared entry count times the entry
    /// stride. Refused in both directions: short buffers and trailing bytes.
    LengthMismatch,
    /// An entry's rung discriminant is not a defined [`Rung`].
    BadRung(u8),
    /// Entries are not in strictly ascending key order. The invariant is load
    /// bearing: lookup is a binary search, and a duplicate key would make one
    /// of the two copies permanently unreachable.
    UnsortedEntries,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => f.write_str("candidate set is empty"),
            Self::Truncated => f.write_str("encoded memo is truncated"),
            Self::BadMagic => f.write_str("encoded memo has wrong magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported memo version {v}"),
            Self::ReservedFieldSet => f.write_str("reserved header field is non-zero"),
            Self::ChecksumMismatch => f.write_str("memo checksum does not match payload"),
            Self::CapacityOutOfRange => f.write_str("declared capacity out of range"),
            Self::CountExceedsCapacity => f.write_str("declared count exceeds declared capacity"),
            Self::LengthMismatch => f.write_str("payload length does not match declared count"),
            Self::BadRung(b) => write!(f, "undefined ladder rung discriminant {b}"),
            Self::UnsortedEntries => f.write_str("entries are not strictly ascending by key"),
        }
    }
}

impl core::error::Error for Error {}
