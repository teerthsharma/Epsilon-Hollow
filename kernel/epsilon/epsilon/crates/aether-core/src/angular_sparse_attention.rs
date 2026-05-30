#![allow(clippy::needless_range_loop)]

// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Angular sparse attention in `no_std` Rust.
//!
//! This replaces the legacy `epsilon_core/angular_sparse_attention.py`
//! reference path with fixed-buffer primitives for centroid generation,
//! cosine-threshold pruning, and online-softmax attention over active KV
//! blocks.

use libm::{exp, sqrt};

const EPS: f64 = 1e-12;

/// Compute a normalized centroid for a KV block.
pub fn normalized_block_centroid<const D: usize>(block: &[[f64; D]]) -> [f64; D] {
    let mut centroid = [0.0; D];
    if block.is_empty() {
        return centroid;
    }

    for row in block {
        for dim in 0..D {
            centroid[dim] += row[dim];
        }
    }

    let inv_len = 1.0 / block.len() as f64;
    for value in &mut centroid {
        *value *= inv_len;
    }
    normalize(&mut centroid);
    centroid
}

/// Cosine similarity for two fixed-size vectors.
pub fn cosine_similarity<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut dot = 0.0;
    let mut a_norm = 0.0;
    let mut b_norm = 0.0;
    for idx in 0..D {
        dot += a[idx] * b[idx];
        a_norm += a[idx] * a[idx];
        b_norm += b[idx] * b[idx];
    }
    if a_norm < EPS || b_norm < EPS {
        return 0.0;
    }
    dot / (sqrt(a_norm) * sqrt(b_norm))
}

/// Block-sparse attention state with angular pruning counters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngularSparseAttention {
    block_size: usize,
    cos_threshold: f64,
    head_dim: u32,
    total_blocks_checked: u64,
    total_blocks_active: u64,
}

impl AngularSparseAttention {
    /// Create a fixed-buffer angular sparse attention engine.
    pub const fn new(block_size: usize, cos_threshold: f64, head_dim: u32) -> Self {
        Self {
            block_size,
            cos_threshold,
            head_dim,
            total_blocks_checked: 0,
            total_blocks_active: 0,
        }
    }

    /// Number of tokens per KV block.
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Cosine threshold used for angular pruning.
    pub const fn cos_threshold(&self) -> f64 {
        self.cos_threshold
    }

    /// Total blocks checked by `generate_active_indices`.
    pub const fn total_blocks_checked(&self) -> u64 {
        self.total_blocks_checked
    }

    /// Total active blocks emitted by `generate_active_indices`.
    pub const fn total_blocks_active(&self) -> u64 {
        self.total_blocks_active
    }

    /// Generate active block indices whose centroid passes the cosine threshold.
    ///
    /// Returns the number of active entries written into `out`.
    pub fn generate_active_indices<const B: usize, const D: usize>(
        &mut self,
        query: &[f64; D],
        centroids: &[[f64; D]; B],
        out: &mut [usize; B],
    ) -> usize {
        self.total_blocks_checked += B as u64;
        for slot in out.iter_mut() {
            *slot = usize::MAX;
        }

        let mut q_norm = 0.0;
        for value in query {
            q_norm += value * value;
        }
        if q_norm < EPS {
            for (idx, slot) in out.iter_mut().enumerate() {
                *slot = idx;
            }
            self.total_blocks_active += B as u64;
            return B;
        }

        let mut active_len = 0;
        for (idx, centroid) in centroids.iter().enumerate() {
            if cosine_similarity(query, centroid) >= self.cos_threshold {
                out[active_len] = idx;
                active_len += 1;
            }
        }
        self.total_blocks_active += active_len as u64;
        active_len
    }

    /// Compute online-softmax attention over active KV blocks.
    pub fn sparse_flash_forward<const TOKENS: usize, const D: usize, const B: usize>(
        &self,
        query: &[f64; D],
        keys: &[[f64; D]; TOKENS],
        values: &[[f64; D]; TOKENS],
        active_indices: &[usize; B],
        active_len: usize,
        out: &mut [f64; D],
    ) {
        for value in out.iter_mut() {
            *value = 0.0;
        }

        let scale = 1.0 / sqrt(f64::from(self.head_dim.max(1)));
        let mut max_score = f64::NEG_INFINITY;
        let mut normalizer = 0.0;
        let bounded_active = active_len.min(B);

        for &block_idx in active_indices.iter().take(bounded_active) {
            if block_idx == usize::MAX {
                continue;
            }
            let start = block_idx.saturating_mul(self.block_size);
            let end = start.saturating_add(self.block_size).min(TOKENS);
            for token_idx in start..end {
                let score = dot(query, &keys[token_idx]) * scale;
                let new_max = max_score.max(score);
                let alpha = if max_score.is_finite() {
                    exp(max_score - new_max)
                } else {
                    0.0
                };
                let beta = exp(score - new_max);

                normalizer = normalizer * alpha + beta;
                for dim in 0..D {
                    out[dim] = out[dim] * alpha + beta * values[token_idx][dim];
                }
                max_score = new_max;
            }
        }

        if normalizer > EPS {
            for value in out.iter_mut() {
                *value /= normalizer;
            }
        }
    }

    /// Return fraction of pruned blocks across all generated indices.
    pub fn sparsity_ratio(&self) -> f64 {
        if self.total_blocks_checked == 0 {
            return 0.0;
        }
        1.0 - self.total_blocks_active as f64 / self.total_blocks_checked as f64
    }

    /// Reset accumulated pruning statistics.
    pub fn reset_stats(&mut self) {
        self.total_blocks_checked = 0;
        self.total_blocks_active = 0;
    }
}

fn normalize<const D: usize>(vector: &mut [f64; D]) {
    let mut norm = 0.0;
    for value in vector.iter() {
        norm += value * value;
    }
    if norm < EPS {
        return;
    }
    let inv_norm = 1.0 / sqrt(norm);
    for value in vector.iter_mut() {
        *value *= inv_norm;
    }
}

fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut total = 0.0;
    for idx in 0..D {
        total += a[idx] * b[idx];
    }
    total
}
