// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Task (process) representation with manifold embedding.

use alloc::string::String;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Dead,
}

pub struct Task {
    pub id: u64,
    pub name: String,
    pub state: TaskState,
    pub manifold_embedding: [f64; 8],
    pub voronoi_cell: usize,
    pub priority: u8,
    pub ticks_used: u64,
    pub entry: fn(),
}

impl Task {
    pub fn new(id: u64, name: &str, priority: u8, entry: fn()) -> Self {
        let embedding = Self::compute_embedding(id);
        Self {
            id,
            name: String::from(name),
            state: TaskState::Ready,
            manifold_embedding: embedding,
            voronoi_cell: (id as usize) % 8,
            priority,
            ticks_used: 0,
            entry,
        }
    }

    fn compute_embedding(id: u64) -> [f64; 8] {
        let mut e = [0.0f64; 8];
        let mut hash = id.wrapping_mul(0x517cc1b727220a95);
        for slot in e.iter_mut() {
            hash = hash.wrapping_mul(0x6c62272e07bb0142).wrapping_add(1);
            *slot = ((hash >> 32) as f64) / (u32::MAX as f64);
        }
        let norm = libm::sqrt(e.iter().map(|x| x * x).sum::<f64>());
        if norm > 1e-12 {
            for slot in e.iter_mut() {
                *slot /= norm;
            }
        }
        e
    }
}
