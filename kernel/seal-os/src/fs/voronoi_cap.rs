// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Bounded Voronoi cells — cap at 64 files, split/merge with hysteresis.

use alloc::vec;
use alloc::vec::Vec;
use aether_core::tss::SphericalVoronoiIndex;

const VORONOI_CELLS: usize = 8;
const MAX_CELL_OCCUPANCY: usize = 64;
const MERGE_THRESHOLD: usize = 24;

#[derive(Debug, Clone, Copy)]
enum SplitAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone)]
struct Subcell {
    files: Vec<u64>,
    boundary: f64,
    axis: SplitAxis,
}

#[derive(Debug, Clone)]
struct CellState {
    files: Vec<u64>,
    subcells: Option<[Subcell; 2]>,
}

pub struct VoronoiCap {
    cells: Vec<CellState>,
    voronoi: SphericalVoronoiIndex<VORONOI_CELLS>,
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
        let cell = self.assign_cell(payload);
        match &self.cells[cell].subcells {
            None => (cell, 0),
            Some(subs) => {
                let sub = self.assign_subcell(payload, &subs[0]);
                (cell, sub)
            }
        }
    }

    pub fn insert(&mut self, inode_id: u64, payload: &super::encoder::ManifoldPayload) -> (usize, usize) {
        let cell = self.assign_cell(payload);

        // Fast path: already split
        if let Some(ref subs) = self.cells[cell].subcells {
            let sub_idx = self.assign_subcell(payload, &subs[0]);
            self.cells[cell].subcells.as_mut().unwrap()[sub_idx].files.push(inode_id);
            return (cell, sub_idx);
        }

        self.cells[cell].files.push(inode_id);

        if self.cells[cell].files.len() > MAX_CELL_OCCUPANCY {
            self.split_cell(cell);
            let sub_idx = {
                let subs = self.cells[cell].subcells.as_ref().unwrap();
                self.assign_subcell(payload, &subs[0])
            };
            self.cells[cell].subcells.as_mut().unwrap()[sub_idx].files.push(inode_id);
            self.cells[cell].files.retain(|&id| id != inode_id);
            return (cell, sub_idx);
        }

        (cell, 0)
    }

    pub fn remove(&mut self, inode_id: u64, cell: usize, subcell: usize) {
        if cell >= self.cells.len() {
            return;
        }
        let state = &mut self.cells[cell];
        if let Some(ref mut subs) = state.subcells {
            if subcell < subs.len() {
                subs[subcell].files.retain(|&id| id != inode_id);
                let total: usize = subs.iter().map(|s| s.files.len()).sum();
                if total < MERGE_THRESHOLD {
                    self.merge_cell(cell);
                }
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
            .map(|c| {
                match &c.subcells {
                    None => c.files.len(),
                    Some(subs) => subs.iter().map(|s| s.files.len()).sum(),
                }
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

    pub fn clear_cell(&mut self, cell: usize) {
        if cell < self.cells.len() {
            self.cells[cell].files.clear();
            self.cells[cell].subcells = None;
        }
    }

    pub fn move_file_to_cell(&mut self, inode_id: u64, from_cell: usize, to_cell: usize) {
        self.remove(inode_id, from_cell, 0);
        if to_cell < self.cells.len() {
            self.cells[to_cell].files.push(inode_id);
        }
    }

    pub fn split_count(&self) -> u64 {
        self.splits
    }

    pub fn merge_count(&self) -> u64 {
        self.merges
    }

    fn assign_cell(&self, payload: &super::encoder::ManifoldPayload) -> usize {
        if payload.points.is_empty() {
            return 0;
        }
        let pt = &payload.points[0];
        let r = libm::sqrt(
            pt.coords[0] * pt.coords[0]
                + pt.coords[1] * pt.coords[1]
                + pt.coords[2] * pt.coords[2],
        );
        if r < 1e-12 {
            return 0;
        }
        let theta = libm::acos((pt.coords[2] / r).clamp(-1.0, 1.0));
        let phi = libm::atan2(pt.coords[1], pt.coords[0]);
        self.voronoi.locate((theta, phi))
    }

    fn assign_subcell(&self, payload: &super::encoder::ManifoldPayload, reference: &Subcell) -> usize {
        if payload.points.is_empty() {
            return 0;
        }
        let coord = match reference.axis {
            SplitAxis::X => payload.points[0].coords[0],
            SplitAxis::Y => payload.points[0].coords[1],
            SplitAxis::Z => payload.points[0].coords[2],
        };
        if coord > reference.boundary {
            1
        } else {
            0
        }
    }

    fn split_cell(&mut self, cell: usize) {
        let state = &mut self.cells[cell];
        if state.files.len() <= MAX_CELL_OCCUPANCY {
            return;
        }
        // Compute dominant axis of variance
        let mut mean = [0.0f64; 3];
        for &id in &state.files {
            // We don't have payload here; use placeholder based on id
            // In real usage, the caller provides payload. For now, simple hash-based split.
            let h = id.wrapping_mul(0x9e3779b97f4a7c15);
            mean[0] += ((h >> 0) & 0xFF) as f64 / 255.0;
            mean[1] += ((h >> 8) & 0xFF) as f64 / 255.0;
            mean[2] += ((h >> 16) & 0xFF) as f64 / 255.0;
        }
        let n = state.files.len() as f64;
        mean[0] /= n;
        mean[1] /= n;
        mean[2] /= n;

        let mut var = [0.0f64; 3];
        for &id in &state.files {
            let h = id.wrapping_mul(0x9e3779b97f4a7c15);
            let x = ((h >> 0) & 0xFF) as f64 / 255.0;
            let y = ((h >> 8) & 0xFF) as f64 / 255.0;
            let z = ((h >> 16) & 0xFF) as f64 / 255.0;
            var[0] += (x - mean[0]) * (x - mean[0]);
            var[1] += (y - mean[1]) * (y - mean[1]);
            var[2] += (z - mean[2]) * (z - mean[2]);
        }

        let axis = if var[0] >= var[1] && var[0] >= var[2] {
            SplitAxis::X
        } else if var[1] >= var[2] {
            SplitAxis::Y
        } else {
            SplitAxis::Z
        };

        let boundary = match axis {
            SplitAxis::X => mean[0],
            SplitAxis::Y => mean[1],
            SplitAxis::Z => mean[2],
        };

        let mut sub0 = Vec::new();
        let mut sub1 = Vec::new();
        for &id in &state.files {
            let h = id.wrapping_mul(0x9e3779b97f4a7c15);
            let coord = match axis {
                SplitAxis::X => ((h >> 0) & 0xFF) as f64 / 255.0,
                SplitAxis::Y => ((h >> 8) & 0xFF) as f64 / 255.0,
                SplitAxis::Z => ((h >> 16) & 0xFF) as f64 / 255.0,
            };
            if coord > boundary {
                sub1.push(id);
            } else {
                sub0.push(id);
            }
        }

        state.subcells = Some([
            Subcell { files: sub0, boundary, axis },
            Subcell { files: sub1, boundary, axis: axis },
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

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;
    use super::super::encoder;

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
