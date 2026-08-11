// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! The journal region, and why exactly one writer owns it.
//!
//! LBAs `journal_start ..= journal_start + journal_blocks - 1` — 1..=1024, as
//! both `Superblock::new` and `seal-mkimage::create_manifold_superblock` lay
//! them out — belong to the write-ahead log in `fs/block_store.rs` and to
//! nothing else. That log writes one `JournalEntry` sector per metadata update
//! at `journal_start + (journal_pos % journal_blocks)`, commits the region in
//! `commit_journal`, and replays the committed entries in `replay_journal`.
//!
//! This module held a second journal, `TopologicalJournal`, which wrote a
//! `TOPJ` header at `journal_start` and its own entries immediately after —
//! the same sectors, derived from the same two superblock fields. It ran
//! second, from `flush_all`, and neither writer read before writing, so every
//! write-ahead entry was overwritten before `commit_journal` could scan for
//! it. No entry ever parsed, none was ever marked committed, and no remount
//! ever replayed one for as long as both existed.
//!
//! The write-ahead log kept the region because it is the one ordered to work.
//! `write_journal_entry` runs before the inode table write it describes, and
//! `commit_journal` now runs before that write too, so a crash in between
//! leaves a committed entry a remount can apply. The topological journal was
//! ordered the other way round: `record_change` was called from
//! `write_data_extent`, which wrote the data blocks immediately, and the
//! images reached the disk later in `flush_all`. Every image it committed
//! described a write that was already durable, so it could recover nothing —
//! while a torn commit left `replay` reading entry headers that were never
//! written and copying their contents to an unvalidated `block_num`.
//!
//! Splitting the region rather than deleting a writer does not fit inside it:
//! one topological entry for a maximal `MAX_EXTENT_BLOCKS` extent needs
//! `1 + 256 + 256 = 513` sectors, so two of them exceed the 1024 the
//! superblock reserves and `commit` returns `InvalidLba`, which `flush_all`
//! propagates as a failed sync.
