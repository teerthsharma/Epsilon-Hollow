// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Voronoi cells over S², each split in two along its axis of greatest
//! variance once it holds more than 64 files, and merged back below 24.
//!
//! ponytail: a cell splits once; its two subcells are never split again, so
//! a subcell is unbounded and `find` over a crowded region is a linear scan
//! of it. Payloads with identical first points (every empty file encodes to
//! the origin) cannot be separated by any coordinate split, so a recursive
//! k-d split would still need a depth or size floor. Upgrade path: recurse in
//! `split_cell` with a depth limit.

use aether_core::tss::SphericalVoronoiIndex;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

const VORONOI_CELLS: usize = 8;
const MAX_CELL_OCCUPANCY: usize = 64;
const MERGE_THRESHOLD: usize = 24;

#[derive(Debug, Clone)]
struct Subcell {
    files: Vec<u64>,
    boundary: f64,
    /// Index into the first point's coordinates: 0 = x, 1 = y, 2 = z.
    axis: usize,
}

#[derive(Debug, Clone)]
struct CellState {
    files: Vec<u64>,
    subcells: Option<[Subcell; 2]>,
}

/// Buckets of inode ids keyed by the Voronoi cell (and, once split, the
/// subcell) of each payload's first point on S².
///
/// Invariant: every stored id is reachable from `locate(payload)`. It
/// appears exactly once, in `files_in_bucket(locate(payload))`, where
/// `payload` is the one it was inserted with. Insertion, splitting, removal
/// and cell merging all go through `route`, which reads only the point
/// recorded in `points` and the `merged_into` map, so none of them can
/// disagree about where an id lives.
pub struct VoronoiCap {
    cells: Vec<CellState>,
    voronoi: SphericalVoronoiIndex<VORONOI_CELLS>,
    /// First point of each stored id's payload, the only input to `route`.
    points: BTreeMap<u64, [f64; 3]>,
    /// The live cell each Voronoi cell's payloads route to: itself until
    /// `merge_cells` folds it into another. Always fully resolved, so one
    /// lookup suffices.
    merged_into: [usize; VORONOI_CELLS],
    splits: u64,
    merges: u64,
}

impl VoronoiCap {
    pub fn new() -> Self {
        let default_centroids = Self::default_centroids();
        let mut cells = Vec::with_capacity(VORONOI_CELLS);
        for _ in 0..VORONOI_CELLS {
            cells.push(CellState {
                files: Vec::new(),
                subcells: None,
            });
        }
        Self {
            cells,
            voronoi: SphericalVoronoiIndex::<VORONOI_CELLS>::new(default_centroids),
            points: BTreeMap::new(),
            merged_into: core::array::from_fn(|i| i),
            splits: 0,
            merges: 0,
        }
    }

    fn default_centroids() -> [(f64, f64); VORONOI_CELLS] {
        let mut c = [(0.0, 0.0); VORONOI_CELLS];
        for (i, slot) in c.iter_mut().enumerate() {
            *slot = (
                core::f64::consts::FRAC_PI_2,
                (i as f64) * (2.0 * core::f64::consts::PI / VORONOI_CELLS as f64),
            );
        }
        c
    }

    pub fn locate(&self, payload: &super::encoder::ManifoldPayload) -> (usize, usize) {
        self.route(&first_point(payload))
    }

    pub fn insert(
        &mut self,
        inode_id: u64,
        payload: &super::encoder::ManifoldPayload,
    ) -> (usize, usize) {
        let pt = first_point(payload);
        self.points.insert(inode_id, pt);
        self.place(inode_id, &pt);
        self.route(&pt)
    }

    /// Push `inode_id` into the bucket `route(pt)` names, splitting the cell
    /// if that takes it over the occupancy cap.
    fn place(&mut self, inode_id: u64, pt: &[f64; 3]) {
        let (cell, sub) = self.route(pt);
        match self.cells[cell].subcells.as_mut() {
            Some(subs) => subs[sub].files.push(inode_id),
            None => {
                self.cells[cell].files.push(inode_id);
                if self.cells[cell].files.len() > MAX_CELL_OCCUPANCY {
                    self.split_cell(cell);
                }
            }
        }
    }

    /// Remove `inode_id` from whichever bucket holds it. The bucket is found
    /// from the point recorded at insertion, not supplied by the caller.
    pub fn remove(&mut self, inode_id: u64) {
        let Some(pt) = self.points.remove(&inode_id) else {
            return;
        };
        let (cell, sub) = self.route(&pt);
        let state = &mut self.cells[cell];
        if let Some(ref mut subs) = state.subcells {
            subs[sub].files.retain(|&id| id != inode_id);
            let total: usize = subs.iter().map(|s| s.files.len()).sum();
            if total < MERGE_THRESHOLD {
                self.merge_cell(cell);
            }
        } else {
            state.files.retain(|&id| id != inode_id);
        }
    }

    pub fn files_in_bucket(&self, cell: usize, subcell: usize) -> &[u64] {
        if cell >= self.cells.len() {
            return &[];
        }
        match &self.cells[cell].subcells {
            None => &self.cells[cell].files,
            Some(subs) => {
                if subcell < subs.len() {
                    &subs[subcell].files
                } else {
                    &subs[0].files
                }
            }
        }
    }

    pub fn cell_sizes(&self) -> Vec<usize> {
        self.cells
            .iter()
            .map(|c| match &c.subcells {
                None => c.files.len(),
                Some(subs) => subs.iter().map(|s| s.files.len()).sum(),
            })
            .collect()
    }

    pub fn all_files_in_cell(&self, cell: usize) -> Vec<u64> {
        if cell >= self.cells.len() {
            return Vec::new();
        }
        match &self.cells[cell].subcells {
            None => self.cells[cell].files.clone(),
            Some(subs) => {
                let mut all = Vec::new();
                for s in subs.iter() {
                    all.extend_from_slice(&s.files);
                }
                all
            }
        }
    }

    /// Fold cell `src` into cell `dst` for good: every payload that located
    /// to `src` now locates to `dst`, and each id `src` held is re-bucketed
    /// in `dst` by its recorded point. Returns the moved ids, empty when the
    /// two already route to the same cell or either index is out of range.
    pub fn merge_cells(&mut self, src: usize, dst: usize) -> Vec<u64> {
        if src >= VORONOI_CELLS || dst >= VORONOI_CELLS {
            return Vec::new();
        }
        let (src, dst) = (self.merged_into[src], self.merged_into[dst]);
        if src == dst {
            return Vec::new();
        }
        let moved = self.all_files_in_cell(src);
        self.cells[src].files.clear();
        self.cells[src].subcells = None;
        for target in self.merged_into.iter_mut() {
            if *target == src {
                *target = dst;
            }
        }
        for &id in &moved {
            let pt = self.points.get(&id).copied().unwrap_or([0.0; 3]);
            self.place(id, &pt);
        }
        moved
    }

    pub fn split_count(&self) -> u64 {
        self.splits
    }

    pub fn merge_count(&self) -> u64 {
        self.merges
    }

    /// The bucket for a first point: its Voronoi cell, then the side of that
    /// cell's split plane when the cell is split.
    fn route(&self, pt: &[f64; 3]) -> (usize, usize) {
        let cell = self.assign_cell(pt);
        match &self.cells[cell].subcells {
            None => (cell, 0),
            Some(subs) => (cell, side(pt, subs[0].axis, subs[0].boundary)),
        }
    }

    fn assign_cell(&self, pt: &[f64; 3]) -> usize {
        let r = libm::sqrt(pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]);
        let raw = if r < 1e-12 {
            0
        } else {
            let theta = libm::acos((pt[2] / r).clamp(-1.0, 1.0));
            let phi = libm::atan2(pt[1], pt[0]);
            self.voronoi.locate((theta, phi))
        };
        self.merged_into[raw]
    }

    /// Split `cell` at the mean of its files' first points along the axis of
    /// greatest variance, partitioning with the same `side` that `route` uses.
    fn split_cell(&mut self, cell: usize) {
        let state = &mut self.cells[cell];
        if state.files.len() <= MAX_CELL_OCCUPANCY {
            return;
        }
        let pts: Vec<[f64; 3]> = state
            .files
            .iter()
            .map(|id| self.points.get(id).copied().unwrap_or([0.0; 3]))
            .collect();
        let n = pts.len() as f64;
        let mut mean = [0.0f64; 3];
        for p in &pts {
            for k in 0..3 {
                mean[k] += p[k] / n;
            }
        }
        let mut var = [0.0f64; 3];
        for p in &pts {
            for k in 0..3 {
                var[k] += (p[k] - mean[k]) * (p[k] - mean[k]);
            }
        }
        let axis = if var[0] >= var[1] && var[0] >= var[2] {
            0
        } else if var[1] >= var[2] {
            1
        } else {
            2
        };
        let boundary = mean[axis];

        let mut halves = [Vec::new(), Vec::new()];
        for (&id, p) in state.files.iter().zip(pts.iter()) {
            halves[side(p, axis, boundary)].push(id);
        }
        let [sub0, sub1] = halves;
        state.subcells = Some([
            Subcell {
                files: sub0,
                boundary,
                axis,
            },
            Subcell {
                files: sub1,
                boundary,
                axis,
            },
        ]);
        state.files.clear();
        self.splits += 1;
    }

    fn merge_cell(&mut self, cell: usize) {
        let state = &mut self.cells[cell];
        if let Some(ref subs) = state.subcells {
            for s in subs.iter() {
                state.files.extend_from_slice(&s.files);
            }
            state.subcells = None;
            self.merges += 1;
        }
    }
}

/// First point of a payload, or the origin for an empty one, which
/// `assign_cell` sends to cell 0.
fn first_point(payload: &super::encoder::ManifoldPayload) -> [f64; 3] {
    payload.points.first().map(|p| p.coords).unwrap_or([0.0; 3])
}

/// Which side of a split plane a point falls on: 1 above `boundary`, else 0.
fn side(pt: &[f64; 3], axis: usize, boundary: f64) -> usize {
    usize::from(pt[axis] > boundary)
}

impl Default for VoronoiCap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::super::encoder;
    use super::*;
    use crate::testing::TestResult;
    use crate::{test_assert, test_assert_eq};

    fn test_basic_insert_locate() -> TestResult {
        let mut vc = VoronoiCap::new();
        let payload = encoder::encode_text("hello");
        let (cell, sub) = vc.insert(1, &payload);
        test_assert!(cell < VORONOI_CELLS);
        let bucket = vc.files_in_bucket(cell, sub);
        test_assert!(bucket.contains(&1));
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("voronoi_cap::basic_insert_locate", test_basic_insert_locate);
    }
}
