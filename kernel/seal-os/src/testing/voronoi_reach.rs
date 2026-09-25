// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-kernel checks of the ManifoldFS Voronoi reachability invariant: every
//! stored id is in the bucket that `VoronoiCap::locate` returns for its
//! payload, so a content lookup can always reach it.
//!
//! The checks are written to hold whatever cell distribution
//! `SphericalVoronoiIndex::locate` produces: the split check inserts enough
//! files that some cell must overflow by pigeonhole.

use alloc::format;
use alloc::vec::Vec;

use crate::fs::encoder::{self, ManifoldPayload};
use crate::fs::voronoi_cap::VoronoiCap;
use crate::testing::TestResult;

const CELLS: usize = 8;
/// More than `CELLS * 64`, so at least one cell must exceed the
/// 64-file occupancy cap and split, whatever `locate` does.
const SPLIT_FILES: u64 = (CELLS as u64) * 64 + 8;

fn payload(i: u64) -> ManifoldPayload {
    encoder::encode_text(&format!("voronoi-reach-{}", i))
}

/// True when `id` appears exactly once in the bucket `locate(p)` names.
fn reachable_once(vc: &VoronoiCap, id: u64, p: &ManifoldPayload) -> bool {
    let (cell, sub) = vc.locate(p);
    vc.files_in_bucket(cell, sub)
        .iter()
        .filter(|&&x| x == id)
        .count()
        == 1
}

fn held_anywhere(vc: &VoronoiCap, id: u64) -> bool {
    (0..CELLS).any(|c| vc.all_files_in_cell(c).contains(&id))
}

/// After a split, every id must still be reachable from `locate`, and
/// removing an id must take it out of whichever subcell holds it.
fn test_split_keeps_every_id_reachable() -> TestResult {
    let mut vc = VoronoiCap::new();
    let payloads: Vec<ManifoldPayload> = (0..SPLIT_FILES).map(payload).collect();
    for (i, p) in payloads.iter().enumerate() {
        vc.insert(i as u64 + 1, p);
    }
    test_assert!(vc.split_count() >= 1, "no cell split after 8*64+8 inserts");
    for (i, p) in payloads.iter().enumerate() {
        test_assert!(
            reachable_once(&vc, i as u64 + 1, p),
            "an id is not exactly once in its locate() bucket after a split"
        );
    }

    // Remove every id that routes to subcell 1 of a split cell.
    let doomed: Vec<u64> = payloads
        .iter()
        .enumerate()
        .filter(|(_, p)| vc.locate(p).1 == 1)
        .map(|(i, _)| i as u64 + 1)
        .collect();
    test_assert!(!doomed.is_empty(), "no id routed to subcell 1");
    for &id in &doomed {
        vc.remove(id);
    }
    for (i, p) in payloads.iter().enumerate() {
        let id = i as u64 + 1;
        if doomed.contains(&id) {
            test_assert!(!held_anywhere(&vc, id), "removed id is still held");
        } else {
            test_assert!(
                reachable_once(&vc, id, p),
                "a surviving id is unreachable after removals"
            );
        }
    }
    TestResult::Pass
}

pub fn register_all() {
    crate::testing::register_test(
        "filesystem::voronoi_split_keeps_every_id_reachable",
        test_split_keeps_every_id_reachable,
    );
}
