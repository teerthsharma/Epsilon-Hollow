// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! In-kernel checks of the ManifoldFS Voronoi reachability invariant: every
//! stored id is in the bucket that `VoronoiCap::locate` returns for its
//! payload, so a content lookup can always reach it.
//!
//! Both checks are written to hold whatever cell distribution
//! `SphericalVoronoiIndex::locate` produces: the split check inserts enough
//! files that some cell must overflow by pigeonhole, and the merge check
//! spreads files by reading back the cell each one landed in.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::fs::encoder::{self, ManifoldPayload};
use crate::fs::manifold_fs::ManifoldFS;
use crate::fs::voronoi_cap::VoronoiCap;
use crate::testing::TestResult;
use crate::serial_println;

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

/// Merging a cell into a split cell, and a split cell into another, must
/// leave every id reachable from `locate`.
fn test_merge_cells_keeps_every_id_reachable() -> TestResult {
    let mut vc = VoronoiCap::new();
    let payloads: Vec<ManifoldPayload> = (0..SPLIT_FILES).map(payload).collect();
    for (i, p) in payloads.iter().enumerate() {
        vc.insert(i as u64 + 1, p);
    }
    let sizes = vc.cell_sizes();
    let split = match (0..CELLS).find(|&c| sizes[c] > 64) {
        Some(c) => c,
        None => return TestResult::Fail("no cell split after 8*64+8 inserts"),
    };
    let other = (split + 1) % CELLS;
    let third = (split + 2) % CELLS;
    let moved = vc.merge_cells(other, split);
    test_assert!(moved.len() == sizes[other], "merge moved the wrong ids");
    let moved = vc.merge_cells(split, third);
    test_assert!(
        moved.len() == sizes[other] + sizes[split],
        "second merge moved the wrong ids"
    );
    for (i, p) in payloads.iter().enumerate() {
        test_assert!(
            reachable_once(&vc, i as u64 + 1, p),
            "an id is not exactly once in its locate() bucket after a merge"
        );
    }
    TestResult::Pass
}

/// After a T3/GMC entropy merge, `find(content)` must still return every
/// stored file.
fn test_entropy_merge_keeps_every_file_findable() -> TestResult {
    const QUOTA: usize = 4;
    const CANDIDATES: u64 = 4000;

    let mut fs = ManifoldFS::new_ramfs();
    let root = fs.root_id();
    let src = fs.mkdir("src", root).unwrap();
    let dst = fs.mkdir("dst", root).unwrap();

    // Keep at most QUOTA files per cell until entropy passes the 2.0 merge
    // threshold; `voronoi_cell` reports where `store` put each file.
    let mut per_cell = [0usize; CELLS];
    let mut kept: Vec<(u64, String, String)> = Vec::new();
    let mut i = 0;
    while fs.stats().current_entropy <= 2.0 && i < CANDIDATES {
        let name = format!("m{}", i);
        let content = format!("merge-{}", i);
        i += 1;
        let id = fs.store_text(&name, &content, src).unwrap();
        let cell = fs.inode(id).unwrap().voronoi_cell;
        if per_cell[cell] >= QUOTA {
            fs.delete(&name, src).unwrap();
        } else {
            per_cell[cell] += 1;
            kept.push((id, name, content));
        }
    }
    if fs.stats().current_entropy <= 2.0 {
        serial_println!("[voronoi_reach] per-cell after {} candidates: {:?}", i, per_cell);
        return TestResult::Fail("locate() reached too few cells to pass entropy 2.0");
    }

    let before: Vec<usize> = kept
        .iter()
        .map(|(id, _, _)| fs.inode(*id).unwrap().voronoi_cell)
        .collect();
    fs.teleport(&kept[0].1, src, dst).unwrap();
    let merged = kept
        .iter()
        .zip(before.iter())
        .any(|((id, _, _), &c)| fs.inode(*id).unwrap().voronoi_cell != c);
    test_assert!(merged, "teleport above entropy 2.0 did not merge a cell");

    for (id, _, content) in &kept {
        let hits = fs.find(content);
        test_assert!(
            hits.iter().any(|r| r.inode_id == *id),
            "find(content) lost a file after the entropy merge"
        );
    }
    TestResult::Pass
}

pub fn register_all() {
    crate::testing::register_test(
        "filesystem::voronoi_split_keeps_every_id_reachable",
        test_split_keeps_every_id_reachable,
    );
    crate::testing::register_test(
        "filesystem::voronoi_merge_cells_keeps_every_id_reachable",
        test_merge_cells_keeps_every_id_reachable,
    );
    crate::testing::register_test(
        "filesystem::voronoi_entropy_merge_keeps_every_file_findable",
        test_entropy_merge_keeps_every_file_findable,
    );
}
