// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT
// Ported from epsilon-os encoder.rs to no_std.

//! Data → Manifold encoder. Converts bytes into ManifoldPayloads on S².

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

const MAX_PAYLOAD_POINTS: usize = 64;
const PROJECTION_DIM: usize = 3;
const HASH_DIM: usize = 128;
const BLOCK_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy)]
pub struct SpherePoint {
    pub coords: [f64; PROJECTION_DIM],
}

impl SpherePoint {
    pub fn zero() -> Self {
        Self {
            coords: [0.0; PROJECTION_DIM],
        }
    }

    pub fn distance_sq(&self, other: &Self) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct ManifoldPayload {
    pub points: Vec<SpherePoint>,
    pub point_count: usize,
    pub betti_0: usize,
    pub original_size: u64,
    pub content_hash: u64,
}

struct ProjectionMatrix {
    weights: [[f64; PROJECTION_DIM]; HASH_DIM],
}

impl ProjectionMatrix {
    fn new(seed: u64) -> Self {
        let mut weights = [[0.0f64; PROJECTION_DIM]; HASH_DIM];
        let mut rng = Xoshiro256::new(seed);
        let scale = 1.0 / libm::sqrt(PROJECTION_DIM as f64);
        for row in weights.iter_mut() {
            for val in row.iter_mut() {
                *val = rng.next_gaussian() * scale;
            }
        }
        Self { weights }
    }

    fn project(&self, high_dim: &[f64; HASH_DIM]) -> [f64; PROJECTION_DIM] {
        let mut out = [0.0f64; PROJECTION_DIM];
        for (i, &h) in high_dim.iter().enumerate() {
            if h.abs() < 1e-15 {
                continue;
            }
            for (j, o) in out.iter_mut().enumerate() {
                *o += h * self.weights[i][j];
            }
        }
        out
    }
}

struct Xoshiro256 {
    state: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        let mut s = Self {
            state: [
                seed,
                seed ^ 0x6a09e667,
                seed.wrapping_mul(3),
                seed ^ 0xbb67ae85,
            ],
        };
        for _ in 0..8 {
            s.next_u64();
        }
        s
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * core::f64::consts::PI * u2)
    }
}

pub fn encode_data(data: &[u8]) -> ManifoldPayload {
    let content_hash = hash_bytes(data);
    let original_size = data.len() as u64;

    if data.is_empty() {
        return ManifoldPayload {
            points: vec![SpherePoint::zero()],
            point_count: 1,
            betti_0: 1,
            original_size: 0,
            content_hash,
        };
    }

    let proj = ProjectionMatrix::new(0xE951_10A7);
    let num_blocks = data.len().div_ceil(BLOCK_SIZE);
    let stride = if num_blocks <= MAX_PAYLOAD_POINTS {
        1
    } else {
        num_blocks.div_ceil(MAX_PAYLOAD_POINTS)
    };

    let mut points = Vec::with_capacity(MAX_PAYLOAD_POINTS.min(num_blocks));

    for block_idx in (0..num_blocks).step_by(stride) {
        if points.len() >= MAX_PAYLOAD_POINTS {
            break;
        }
        let start = block_idx * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(data.len());
        let block = &data[start..end];

        let high_dim = block_to_hash_vector(block);
        let projected = proj.project(&high_dim);
        let normalized = l2_normalize_3d(projected);
        points.push(SpherePoint { coords: normalized });
    }

    let betti_0 = compute_betti_0(&points);

    ManifoldPayload {
        point_count: points.len(),
        points,
        betti_0,
        original_size,
        content_hash,
    }
}

pub fn encode_text(text: &str) -> ManifoldPayload {
    encode_data(text.as_bytes())
}

fn compute_betti_0(points: &[SpherePoint]) -> usize {
    if points.is_empty() {
        return 0;
    }
    let epsilon_sq = 0.5 * 0.5;
    let n = points.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank = vec![0u8; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if points[i].distance_sq(&points[j]) < epsilon_sq {
                union(&mut parent, &mut rank, i, j);
            }
        }
    }

    let mut roots = BTreeSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    roots.len()
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    match rank[ra].cmp(&rank[rb]) {
        core::cmp::Ordering::Less => parent[ra] = rb,
        core::cmp::Ordering::Greater => parent[rb] = ra,
        core::cmp::Ordering::Equal => {
            parent[rb] = ra;
            rank[ra] += 1;
        }
    }
}

fn block_to_hash_vector(block: &[u8]) -> [f64; HASH_DIM] {
    let mut vec = [0.0f64; HASH_DIM];
    for i in 0..block.len().saturating_sub(2) {
        let h = (block[i] as u64).wrapping_mul(31)
            ^ (block[i + 1] as u64).wrapping_mul(37)
            ^ (block[i + 2] as u64).wrapping_mul(41);
        let idx = (h as usize) % HASH_DIM;
        let sign = if (h / HASH_DIM as u64) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        vec[idx] += sign;
    }
    for &b in block {
        let idx = ((b as usize) * 7 + 3) % HASH_DIM;
        vec[idx] += 0.5;
    }
    vec
}

fn l2_normalize_3d(v: [f64; PROJECTION_DIM]) -> [f64; PROJECTION_DIM] {
    let norm = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if norm < 1e-12 {
        return [1.0, 0.0, 0.0];
    }
    [v[0] / norm, v[1] / norm, v[2] / norm]
}

pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
