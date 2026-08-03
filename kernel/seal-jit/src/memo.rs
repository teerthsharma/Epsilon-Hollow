//! Persistent measurement store: `key -> (best choice, cost, observations, rung)`.
//!
//! The table is bounded, sorted by key, and round-trips through a byte format
//! so the kernel can park it in ManifoldFS and read it back after a reboot.
//!
//! # Decoding is a trust boundary
//!
//! Bytes handed to [`Memo::decode`] are untrusted, and are treated the way this
//! repository treats every other untrusted input:
//!
//! - every multi-byte read is bounds-checked through `slice::get`, with
//!   `checked_add` on the offsets, so no read can panic or wrap;
//! - the FNV-1a trailer is verified *before* any field is interpreted, so a
//!   corrupt buffer is rejected without ever being structurally parsed;
//! - the declared entry count is validated against the declared capacity and
//!   against the actual remaining length before a single byte is allocated —
//!   a header claiming four billion entries allocates nothing;
//! - the payload length must match the declared count exactly, so trailing
//!   bytes are refused rather than ignored;
//! - the rung discriminant and the reserved header field are refused when
//!   undefined rather than defaulted;
//! - keys must be strictly ascending, which rejects duplicates and unsorted
//!   input, because lookup is a binary search and a duplicate key would make
//!   one of the two records permanently unreachable.
//!
//! There is no `unwrap`, `expect`, `panic!`, or slice-index operator anywhere
//! on the decode path.
//!
//! # Wire format, version 1
//!
//! ```text
//! header   16 bytes  magic "SJM1" | version u16 | reserved u16 | capacity u32 | count u32
//! entry    36 bytes  key u64 | best u32 | best_cost u64 | observations u32
//!                    | credit u64 | confirmations u16 | rung u8 | pad u8
//! trailer   8 bytes  FNV-1a-64 of every preceding byte
//! ```
//!
//! All integers little-endian. Entries appear exactly `count` times, strictly
//! ascending by key.
//!
//! The `credit` field replaced a never-decayed lifetime observation count at
//! the same offset and width, so the version stays 1: a memo written by the
//! older build decodes unchanged and its credits simply begin to age.

use alloc::vec::Vec;

use crate::ladder::{self, Rung, Transition};
use crate::{Decision, Error, PlanChoice, PlanKey};

/// Hard ceiling on entries, applied both to [`Memo::new`] and to a decoded
/// capacity field. Bounds the worst-case table at
/// `MAX_ENTRIES * ENTRY_LEN` = 2.25 MiB, which is a defensible resident cost
/// for a kernel and small enough that a hostile header cannot ask for more.
pub const MAX_ENTRIES: u32 = 65_536;

const MAGIC: [u8; 4] = *b"SJM1";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 16;
const ENTRY_LEN: usize = 36;
const TRAILER_LEN: usize = 8;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    let end = at.checked_add(2)?;
    bytes.get(at..end)?.try_into().ok().map(u16::from_le_bytes)
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    bytes.get(at..end)?.try_into().ok().map(u32::from_le_bytes)
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    bytes.get(at..end)?.try_into().ok().map(u64::from_le_bytes)
}

/// One key's tuning record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// The workload identity this record belongs to.
    pub key: PlanKey,
    /// Cheapest configuration measured so far.
    pub best: PlanChoice,
    /// Cost recorded for [`Entry::best`]. Overwritten on a regression so the
    /// regression band always tracks the current reality rather than an
    /// unreachable historical number.
    pub best_cost: u64,
    /// Observations since the last demotion. Drives
    /// [`explore_budget`](crate::search::explore_budget); reset to 0 by a
    /// demotion, which is what re-opens the search.
    pub observations: u32,
    /// Decayed observation count: `+1` per observation, halved for every entry
    /// whenever the table has absorbed a table's worth of eviction pressure.
    /// Used only as the eviction rank — see [`Memo::observe`]. Not a lifetime
    /// total and not comparable across memos with different capacities.
    pub credit: u64,
    /// Consecutive confirmations of the incumbent at the current rung.
    pub confirmations: u16,
    /// Current specialisation rung.
    pub rung: Rung,
}

/// Bounded, persistable store of tuning records.
///
/// Deliberately not [`Default`]: a zero-capacity memo would accept every
/// `observe` call and retain nothing, which is a silent no-op dressed as a
/// working cache. Capacity must be chosen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Memo {
    entries: Vec<Entry>,
    capacity: usize,
    /// Insertions attempted against a full table since the last decay. Not
    /// persisted: it is a phase within one run's decay cycle, meaningless to
    /// the run that reads the bytes back, and dropping it costs at most one
    /// deferred halving.
    pressure: usize,
}

impl Memo {
    /// Create an empty memo holding at most `capacity` keys.
    ///
    /// `capacity` is clamped to `1..=MAX_ENTRIES`. The clamp is stated rather
    /// than silent because a caller asking for zero entries has a bug, and a
    /// zero-capacity memo would evict every key immediately while still
    /// reporting success.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity: capacity.clamp(1, MAX_ENTRIES as usize),
            pressure: 0,
        }
    }

    /// Maximum number of keys this memo will retain.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of keys currently recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any key is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every record, ascending by key.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The record for `key`, if any.
    #[must_use]
    pub fn entry(&self, key: PlanKey) -> Option<&Entry> {
        let index = self.entries.binary_search_by_key(&key, |e| e.key).ok()?;
        self.entries.get(index)
    }

    /// The cheapest configuration measured so far for `key`.
    #[must_use]
    pub fn best(&self, key: PlanKey) -> Option<PlanChoice> {
        self.entry(key).map(|e| e.best)
    }

    /// Choose the configuration to run for the next repetition of `key`.
    ///
    /// Does not mutate: a caller that selects but never observes leaves no
    /// trace. `candidates` must be a stable function of `key` — see
    /// [`PlanChoice`].
    ///
    /// # Errors
    ///
    /// [`Error::EmptyCandidateSet`] if `candidates` is empty.
    pub fn select(&self, key: PlanKey, candidates: &[PlanChoice]) -> Result<Decision, Error> {
        crate::search::select(self.entry(key), candidates)
    }

    /// Record the measured `cost` of running `choice` for `key`.
    ///
    /// This is the only mutating path. For a key already present the ladder
    /// rule in [`ladder`] applies verbatim. For a new key the record is
    /// inserted with `choice` as its incumbent and [`Transition::NewBest`] is
    /// reported.
    ///
    /// # Eviction, aging, and admission
    ///
    /// Each entry carries a [`credit`](Entry::credit): `+1` per observation,
    /// halved for every entry each time the table absorbs `capacity`
    /// insertions against a full table. Aging is driven by that pressure
    /// counter and by nothing else, so a working set that fits is never aged —
    /// a training loop with a fixed set of shapes decays no faster than it did
    /// before this rule existed.
    ///
    /// A first sight of a key is admitted only by displacing an entry whose
    /// credit has reached 0, meaning it produced no observation across a full
    /// decay window. When no entry has, the key is refused: the table is left
    /// untouched, nothing is recorded, and [`Transition::Refused`] is returned.
    /// Among zero-credit entries the lowest key goes, so the choice is
    /// deterministic.
    ///
    /// Refusal rather than displacement is what makes this scan-resistant. A
    /// newcomer is worth exactly one observation and cannot outrank a resident
    /// key that is still being used, so a burst of one-shot keys cannot flush a
    /// training loop's tuning results. What it *can* do is spend the pressure
    /// that ages them: `log2(credit)` refusals per halving, and an established
    /// key that has genuinely stopped being used falls to 0 and is taken.
    ///
    /// # Limits
    ///
    /// Admission costs a newcomer roughly `capacity * log2(incumbent credit)`
    /// refused observations before the first slot opens, and those repetitions
    /// run untuned. A working set that rotates faster than that is still tuned
    /// only in part. A working set that is permanently larger than the capacity
    /// converges to retaining a stable subset — which is the intended outcome,
    /// not a repaired one: a cyclic overload has no residency that helps every
    /// key, and thrashing the whole table to be fair to all of them retains
    /// nothing for any of them.
    pub fn observe(&mut self, key: PlanKey, choice: PlanChoice, cost: u64) -> Transition {
        match self.entries.binary_search_by_key(&key, |e| e.key) {
            Ok(index) => match self.entries.get_mut(index) {
                Some(entry) => ladder::record(entry, choice, cost),
                // Unreachable: `binary_search_by_key` returned an in-range index.
                // Handled rather than unwrapped so this path cannot panic.
                None => Transition::Held,
            },
            Err(index) => {
                let mut at = index;
                if self.entries.len() >= self.capacity {
                    self.age();
                    let Some(victim) = self.eviction_target() else {
                        return Transition::Refused;
                    };
                    self.entries.remove(victim);
                    if victim < at {
                        at -= 1;
                    }
                }
                if self.entries.len() < self.capacity {
                    self.entries.insert(
                        at.min(self.entries.len()),
                        Entry {
                            key,
                            best: choice,
                            best_cost: cost,
                            observations: 1,
                            credit: 1,
                            confirmations: 0,
                            rung: Rung::Generic,
                        },
                    );
                }
                Transition::NewBest
            }
        }
    }

    /// Charge one unit of eviction pressure, halving every credit once a
    /// table's worth has accumulated.
    ///
    /// The rate limit is the load-bearing half of the rule. Halving on *every*
    /// pressure event would drive the whole table to zero faster than a
    /// resident key can re-earn credit, and the eviction order would then be
    /// decided by the tie-break rather than by evidence. One halving per
    /// `capacity` events leaves every resident key at least one observation in
    /// which to prove it is still in use.
    fn age(&mut self) {
        self.pressure = self.pressure.saturating_add(1);
        if self.pressure < self.capacity {
            return;
        }
        self.pressure = 0;
        for entry in &mut self.entries {
            entry.credit >>= 1;
        }
    }

    /// Index of the entry to drop when making room, or `None` when nothing has
    /// decayed far enough to be displaced by a first-time key.
    fn eviction_target(&self) -> Option<usize> {
        self.entries.iter().position(|entry| entry.credit == 0)
    }

    /// Drop the record for `key`, returning whether one was present.
    ///
    /// The supported way to invalidate tuning results when a key's candidate
    /// set genuinely changes.
    pub fn forget(&mut self, key: PlanKey) -> bool {
        match self.entries.binary_search_by_key(&key, |e| e.key) {
            Ok(index) => {
                self.entries.remove(index);
                true
            }
            Err(_) => false,
        }
    }

    /// Serialise to the version 1 wire format described in the module docs.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let count = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        let capacity = u32::try_from(self.capacity).unwrap_or(MAX_ENTRIES);
        let mut out = Vec::with_capacity(HEADER_LEN + self.entries.len() * ENTRY_LEN + TRAILER_LEN);

        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // reserved
        out.extend_from_slice(&capacity.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());

        for entry in &self.entries {
            out.extend_from_slice(&entry.key.0.to_le_bytes());
            out.extend_from_slice(&entry.best.0.to_le_bytes());
            out.extend_from_slice(&entry.best_cost.to_le_bytes());
            out.extend_from_slice(&entry.observations.to_le_bytes());
            out.extend_from_slice(&entry.credit.to_le_bytes());
            out.extend_from_slice(&entry.confirmations.to_le_bytes());
            out.push(entry.rung.to_u8());
            out.push(0); // pad
        }

        let digest = fnv1a(&out);
        out.extend_from_slice(&digest.to_le_bytes());
        out
    }

    /// Parse the version 1 wire format. See the module docs for the checks
    /// applied and the order they are applied in.
    ///
    /// # Errors
    ///
    /// One [`Error`] variant per violated invariant; never a panic, and never
    /// an allocation sized by a field that has not been validated first.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let Some(split) = bytes.len().checked_sub(TRAILER_LEN) else {
            return Err(Error::Truncated);
        };
        if bytes.len() < HEADER_LEN + TRAILER_LEN {
            return Err(Error::Truncated);
        }
        let Some((payload, trailer)) = bytes.split_at_checked(split) else {
            return Err(Error::Truncated);
        };

        // Integrity before interpretation.
        let Some(declared_digest) = read_u64(trailer, 0) else {
            return Err(Error::Truncated);
        };
        if declared_digest != fnv1a(payload) {
            return Err(Error::ChecksumMismatch);
        }

        if payload.get(0..4) != Some(&MAGIC[..]) {
            return Err(Error::BadMagic);
        }
        let version = read_u16(payload, 4).ok_or(Error::Truncated)?;
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        if read_u16(payload, 6).ok_or(Error::Truncated)? != 0 {
            return Err(Error::ReservedFieldSet);
        }
        let capacity = read_u32(payload, 8).ok_or(Error::Truncated)?;
        if capacity == 0 || capacity > MAX_ENTRIES {
            return Err(Error::CapacityOutOfRange);
        }
        let count = read_u32(payload, 12).ok_or(Error::Truncated)?;
        if count > capacity {
            return Err(Error::CountExceedsCapacity);
        }

        // `count <= capacity <= MAX_ENTRIES`, so this product cannot overflow.
        let body_len = (count as usize) * ENTRY_LEN;
        let body = payload.get(HEADER_LEN..).ok_or(Error::Truncated)?;
        if body.len() != body_len {
            return Err(Error::LengthMismatch);
        }

        // Only now, with `count` validated against both the capacity ceiling
        // and the actual byte length, is it safe to size an allocation by it.
        let mut entries = Vec::with_capacity(count as usize);
        let mut previous: Option<PlanKey> = None;
        for index in 0..count as usize {
            let at = index * ENTRY_LEN;
            let key = PlanKey(read_u64(body, at).ok_or(Error::Truncated)?);
            if let Some(prev) = previous {
                if key <= prev {
                    return Err(Error::UnsortedEntries);
                }
            }
            previous = Some(key);

            let best = PlanChoice(read_u32(body, at + 8).ok_or(Error::Truncated)?);
            let best_cost = read_u64(body, at + 12).ok_or(Error::Truncated)?;
            let observations = read_u32(body, at + 20).ok_or(Error::Truncated)?;
            let credit = read_u64(body, at + 24).ok_or(Error::Truncated)?;
            let confirmations = read_u16(body, at + 32).ok_or(Error::Truncated)?;
            let rung_byte = *body.get(at + 34).ok_or(Error::Truncated)?;
            let rung = Rung::from_u8(rung_byte).ok_or(Error::BadRung(rung_byte))?;
            if *body.get(at + 35).ok_or(Error::Truncated)? != 0 {
                return Err(Error::ReservedFieldSet);
            }

            entries.push(Entry {
                key,
                best,
                best_cost,
                observations,
                credit,
                confirmations,
                rung,
            });
        }

        Ok(Self {
            entries,
            capacity: capacity as usize,
            pressure: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn populated() -> Memo {
        let mut memo = Memo::new(8);
        for k in [7u64, 2, 9, 4] {
            for i in 0..(k as u32 + 1) {
                memo.observe(PlanKey(k), PlanChoice(i), 500 - u64::from(i));
            }
        }
        memo
    }

    #[test]
    fn entries_stay_sorted_and_deduplicated() {
        let memo = populated();
        assert_eq!(memo.len(), 4);
        let keys: Vec<PlanKey> = memo.entries().iter().map(|e| e.key).collect();
        assert_eq!(keys, [PlanKey(2), PlanKey(4), PlanKey(7), PlanKey(9)]);
    }

    #[test]
    fn round_trip_is_exact() {
        let memo = populated();
        let bytes = memo.encode();
        let back = Memo::decode(&bytes).expect("round trip");
        assert_eq!(back.entries(), memo.entries());
        assert_eq!(back.capacity(), memo.capacity());
        assert_eq!(back.encode(), bytes);
    }

    #[test]
    fn empty_memo_round_trips() {
        let memo = Memo::new(4);
        let back = Memo::decode(&memo.encode()).expect("round trip");
        assert!(back.is_empty());
        assert_eq!(back.capacity(), 4);
    }

    #[test]
    fn decode_refuses_a_truncated_buffer_at_every_length() {
        let bytes = populated().encode();
        for cut in 0..bytes.len() {
            let err = Memo::decode(&bytes[..cut]).expect_err("short buffer accepted");
            assert!(
                matches!(
                    err,
                    Error::Truncated
                        | Error::ChecksumMismatch
                        | Error::LengthMismatch
                        | Error::BadMagic
                ),
                "unexpected error {err:?} at cut={cut}"
            );
        }
    }

    #[test]
    fn decode_refuses_trailing_bytes() {
        let mut bytes = populated().encode();
        bytes.push(0);
        assert_eq!(Memo::decode(&bytes), Err(Error::ChecksumMismatch));
    }

    #[test]
    fn decode_refuses_a_flipped_bit_anywhere() {
        let bytes = populated().encode();
        for index in 0..bytes.len() {
            let mut corrupt = bytes.clone();
            if let Some(byte) = corrupt.get_mut(index) {
                *byte ^= 0x40;
            }
            assert_eq!(
                Memo::decode(&corrupt),
                Err(Error::ChecksumMismatch),
                "bit flip at byte {index} survived"
            );
        }
    }

    /// The checksum test above would pass even if every structural check were
    /// deleted, because a mutation always breaks the digest. These re-checksum
    /// after mutating, so only the structural check can reject them.
    fn reseal(mut payload: Vec<u8>) -> Vec<u8> {
        let keep = payload.len() - TRAILER_LEN;
        payload.truncate(keep);
        let digest = fnv1a(&payload);
        payload.extend_from_slice(&digest.to_le_bytes());
        payload
    }

    #[test]
    fn decode_refuses_bad_magic_even_when_resealed() {
        let mut bytes = populated().encode();
        bytes[0] = b'X';
        assert_eq!(Memo::decode(&reseal(bytes)), Err(Error::BadMagic));
    }

    #[test]
    fn decode_refuses_an_unknown_version_even_when_resealed() {
        let mut bytes = populated().encode();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        assert_eq!(
            Memo::decode(&reseal(bytes)),
            Err(Error::UnsupportedVersion(99))
        );
    }

    #[test]
    fn decode_refuses_a_set_reserved_field_even_when_resealed() {
        let mut bytes = populated().encode();
        bytes[6..8].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(Memo::decode(&reseal(bytes)), Err(Error::ReservedFieldSet));
    }

    #[test]
    fn decode_refuses_an_absurd_count_without_allocating() {
        let mut bytes = populated().encode();
        bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Memo::decode(&reseal(bytes)),
            Err(Error::CountExceedsCapacity)
        );
    }

    #[test]
    fn decode_refuses_an_out_of_range_capacity() {
        let mut bytes = populated().encode();
        bytes[8..12].copy_from_slice(&(MAX_ENTRIES + 1).to_le_bytes());
        assert_eq!(Memo::decode(&reseal(bytes)), Err(Error::CapacityOutOfRange));

        let mut zero = populated().encode();
        zero[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(Memo::decode(&reseal(zero)), Err(Error::CapacityOutOfRange));
    }

    #[test]
    fn decode_refuses_a_count_that_does_not_match_the_payload() {
        let mut bytes = populated().encode();
        bytes[12..16].copy_from_slice(&3u32.to_le_bytes()); // real count is 4
        assert_eq!(Memo::decode(&reseal(bytes)), Err(Error::LengthMismatch));
    }

    #[test]
    fn decode_refuses_an_undefined_rung_even_when_resealed() {
        let mut bytes = populated().encode();
        bytes[HEADER_LEN + 34] = 3;
        assert_eq!(Memo::decode(&reseal(bytes)), Err(Error::BadRung(3)));
    }

    #[test]
    fn decode_refuses_unsorted_or_duplicated_keys_even_when_resealed() {
        let mut bytes = populated().encode();
        // Overwrite the second entry's key with the first entry's key.
        let first = bytes[HEADER_LEN..HEADER_LEN + 8].to_vec();
        bytes[HEADER_LEN + ENTRY_LEN..HEADER_LEN + ENTRY_LEN + 8].copy_from_slice(&first);
        assert_eq!(Memo::decode(&reseal(bytes)), Err(Error::UnsortedEntries));
    }

    #[test]
    fn eviction_drops_the_coldest_key_once_pressure_has_aged_it() {
        let mut memo = Memo::new(3);
        for _ in 0..10 {
            memo.observe(PlanKey(1), PlanChoice(0), 100);
        }
        for _ in 0..5 {
            memo.observe(PlanKey(2), PlanChoice(0), 100);
        }
        memo.observe(PlanKey(3), PlanChoice(0), 100); // 1 observation, coldest
        assert_eq!(memo.len(), 3);

        // Credits are 10, 5, 1 and the table is full, so the first two sightings
        // of a new key are refused; each one charges pressure. The third reaches
        // `capacity` and halves every credit to 5, 2, 0 — and a zero credit is
        // what admission takes.
        assert_eq!(
            memo.observe(PlanKey(4), PlanChoice(0), 100),
            Transition::Refused
        );
        assert_eq!(
            memo.observe(PlanKey(4), PlanChoice(0), 100),
            Transition::Refused
        );
        assert!(memo.entry(PlanKey(4)).is_none(), "a refusal recorded a key");
        assert_eq!(
            memo.observe(PlanKey(4), PlanChoice(0), 100),
            Transition::NewBest
        );

        assert_eq!(memo.len(), 3);
        assert!(
            memo.entry(PlanKey(3)).is_none(),
            "coldest key survived eviction"
        );
        assert!(memo.entry(PlanKey(1)).is_some());
        assert!(memo.entry(PlanKey(2)).is_some());
        assert!(memo.entry(PlanKey(4)).is_some());

        let keys: Vec<PlanKey> = memo.entries().iter().map(|e| e.key).collect();
        assert_eq!(
            keys,
            [PlanKey(1), PlanKey(2), PlanKey(4)],
            "sort order survived eviction"
        );
    }

    /// The defect this policy exists to close: with an un-aged rank a fresh
    /// entry is the strict minimum the instant it is inserted, so it is dropped
    /// before its own next repetition and can never accumulate anything.
    #[test]
    fn a_new_key_displaces_an_established_one_that_has_stopped_being_used() {
        let mut memo = Memo::new(2);
        for _ in 0..64 {
            memo.observe(PlanKey(20), PlanChoice(0), 100);
            memo.observe(PlanKey(21), PlanChoice(0), 100);
        }
        assert_eq!(memo.entry(PlanKey(20)).map(|e| e.credit), Some(64));

        // Keys deliberately *below* the incumbents, so the lowest-key tie-break
        // among zero-credit entries points at the newcomer rather than rescuing
        // it. Two newcomers, interleaved, so neither is ever the only new key
        // in flight — the shape that starved under the old rank.
        let mut refusals = 0usize;
        for round in 0..64 {
            for key in [PlanKey(1), PlanKey(2)] {
                if memo.observe(key, PlanChoice(0), 100) == Transition::Refused {
                    refusals += 1;
                }
            }
            if memo.entry(PlanKey(1)).is_some() && memo.entry(PlanKey(2)).is_some() {
                assert!(round < 32, "admission took {round} rounds");
                break;
            }
        }

        assert_eq!(memo.len(), 2);
        assert!(memo.entry(PlanKey(1)).is_some(), "newcomer never admitted");
        assert!(memo.entry(PlanKey(2)).is_some(), "newcomer never admitted");
        assert!(
            memo.entry(PlanKey(20)).is_none() && memo.entry(PlanKey(21)).is_none(),
            "a stale key outlived the working set that replaced it"
        );
        // Bounded, not merely eventual. Seven halvings take a credit of 64 to
        // 0 and each one costs `capacity` = 2 pressure events, so the 14th
        // sighting of a newcomer is admitted and the 13 before it are refused.
        // The second newcomer then takes the other zero-credit slot for free.
        assert_eq!(refusals, 13);
    }

    /// The other half of the trade: pressure ages the table, but a key that is
    /// still being observed re-earns its credit faster than the decay removes
    /// it, so a stream of one-shot keys cannot flush a live working set.
    #[test]
    fn a_scan_of_one_shot_keys_does_not_flush_a_working_set_still_in_use() {
        let mut memo = Memo::new(4);
        let live = [PlanKey(100), PlanKey(101), PlanKey(102)];
        for _ in 0..8 {
            for &key in &live {
                memo.observe(key, PlanChoice(0), 100);
            }
        }
        for scan in 0..1000u64 {
            memo.observe(PlanKey(scan), PlanChoice(0), 100);
            for &key in &live {
                memo.observe(key, PlanChoice(0), 100);
            }
        }
        for &key in &live {
            assert!(
                memo.entry(key).is_some(),
                "{key:?} was flushed by one-shot traffic"
            );
        }
    }

    #[test]
    fn aging_is_driven_by_pressure_alone_so_a_fitting_working_set_never_decays() {
        let mut memo = Memo::new(4);
        for _ in 0..1000 {
            for key in 0..4u64 {
                memo.observe(PlanKey(key), PlanChoice(0), 100);
            }
        }
        assert!(
            memo.entries().iter().all(|e| e.credit == 1000),
            "a working set that fits was aged anyway"
        );
    }

    #[test]
    fn capacity_is_clamped_at_both_ends() {
        assert_eq!(Memo::new(0).capacity(), 1);
        assert_eq!(Memo::new(usize::MAX).capacity(), MAX_ENTRIES as usize);
    }

    #[test]
    fn forget_removes_exactly_one_key() {
        let mut memo = populated();
        assert!(memo.forget(PlanKey(7)));
        assert!(!memo.forget(PlanKey(7)));
        assert_eq!(memo.len(), 3);
        assert!(memo.entry(PlanKey(9)).is_some());
    }

    #[test]
    fn decode_of_arbitrary_short_garbage_never_panics() {
        for len in 0..64usize {
            let garbage: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37)).collect();
            let _ = Memo::decode(&garbage);
        }
    }
}
