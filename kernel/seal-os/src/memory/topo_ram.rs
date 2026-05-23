// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological RAM driver — T1–T5 theorem integration over physical frames.
//!
//! Wraps the physical bitmap allocator with Voronoi cells, spectral
//! contraction, entropy tracking, pressure governance, and hyperbolic
//! lifetime classification.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;
use x86_64::PhysAddr;

use crate::memory::phys::{self, USABLE_FRAMES};
use crate::THEOREM_STATES;

const VORONOI_SEEDS: usize = 16;
const RECENT_ALLOC_LEN: usize = 256;
const PREFETCH_COUNT: usize = 8;

/// Per-frame topological metadata.
#[derive(Clone, Copy)]
pub struct TopoFrame {
    /// 32 quantized angles = 16 pairs on S².
    pub embedding: [u16; 32],
    /// Bitfield of last 64 access ticks.
    pub access_history: u64,
    /// Voronoi cell assignment.
    pub cell_id: u16,
    /// 0=ephemeral, 1=medium, 2=permanent.
    pub lifetime_class: u8,
    /// Cached entropy score.
    pub entropy_score: f32,
}

/// Global topological RAM state.
pub struct TopoRam {
    frame_meta: &'static mut [TopoFrame],
    voronoi_seeds: [[u16; 32]; VORONOI_SEEDS],
    pressure: f32,
    tick: u64,
    alloc_count: u64,
    free_count: u64,
    recent_allocs: [usize; RECENT_ALLOC_LEN],
    recent_alloc_head: usize,
    eigenvectors: [[f32; 32]; 3],
}

static mut FRAME_META: [TopoFrame; USABLE_FRAMES] = [TopoFrame {
    embedding: [0; 32],
    access_history: 0,
    cell_id: 0,
    lifetime_class: 1,
    entropy_score: 0.0,
}; USABLE_FRAMES];

static TOPO_RAM: Mutex<Option<TopoRam>> = Mutex::new(None);

/// Run a closure against the global `TopoRam`, if initialized.
pub fn with_topo_ram<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut TopoRam) -> R,
{
    let mut guard = TOPO_RAM.lock();
    guard.as_mut().map(f)
}

// ---------------------------------------------------------------------------
// T1 — Voronoi Partitioning
// ---------------------------------------------------------------------------

fn embedding_distance_sq(a: &[u16; 32], b: &[u16; 32]) -> u32 {
    let mut sum = 0u32;
    for i in 0..32 {
        let d = (a[i] as i32) - (b[i] as i32);
        sum += (d * d) as u32;
    }
    sum
}

fn voronoi_cell_of(embedding: &[u16; 32], seeds: &[[u16; 32]; VORONOI_SEEDS]) -> u16 {
    let mut best = 0usize;
    let mut best_dist = embedding_distance_sq(embedding, &seeds[0]);
    for i in 1..VORONOI_SEEDS {
        let dist = embedding_distance_sq(embedding, &seeds[i]);
        if dist < best_dist {
            best = i;
            best_dist = dist;
        }
    }
    best as u16
}

fn reseed_voronoi(topo: &mut TopoRam) {
    if !THEOREM_STATES[0].load(Ordering::Relaxed) {
        return;
    }

    let mut new_seeds = [[0u16; 32]; VORONOI_SEEDS];
    let mut picked = 0usize;

    for i in 0..RECENT_ALLOC_LEN {
        let frame_idx = topo.recent_allocs[i];
        if frame_idx < USABLE_FRAMES && picked < VORONOI_SEEDS {
            new_seeds[picked] = topo.frame_meta[frame_idx].embedding;
            picked += 1;
        }
        if picked >= VORONOI_SEEDS {
            break;
        }
    }

    // Fill remaining seeds with deterministic frames.
    if picked < VORONOI_SEEDS {
        for i in picked..VORONOI_SEEDS {
            let idx = (i.wrapping_mul(65537).wrapping_add(7)) % USABLE_FRAMES;
            new_seeds[i] = topo.frame_meta[idx].embedding;
        }
    }

    topo.voronoi_seeds = new_seeds;

    for i in 0..USABLE_FRAMES {
        let cell = voronoi_cell_of(&topo.frame_meta[i].embedding, &topo.voronoi_seeds);
        topo.frame_meta[i].cell_id = cell;
    }
}

// ---------------------------------------------------------------------------
// T2 — Spectral Contraction
// ---------------------------------------------------------------------------

fn update_spectral(topo: &mut TopoRam) {
    if !THEOREM_STATES[1].load(Ordering::Relaxed) {
        return;
    }

    let mut cov = [[0.0f32; 32]; 32];
    let mut mean = [0.0f32; 32];

    let n = RECENT_ALLOC_LEN;
    for i in 0..n {
        let frame_idx = topo.recent_allocs[i];
        if frame_idx < USABLE_FRAMES {
            let emb = &topo.frame_meta[frame_idx].embedding;
            for j in 0..32 {
                mean[j] += emb[j] as f32;
            }
        }
    }
    for j in 0..32 {
        mean[j] /= n as f32;
    }

    for i in 0..n {
        let frame_idx = topo.recent_allocs[i];
        if frame_idx < USABLE_FRAMES {
            let emb = &topo.frame_meta[frame_idx].embedding;
            for a in 0..32 {
                let da = emb[a] as f32 - mean[a];
                for b in 0..32 {
                    let db = emb[b] as f32 - mean[b];
                    cov[a][b] += da * db;
                }
            }
        }
    }
    for a in 0..32 {
        for b in 0..32 {
            cov[a][b] /= n as f32;
        }
    }

    // Power iteration for top 3 eigenvectors.
    for ev_idx in 0..3 {
        let mut vec = [0.0f32; 32];
        vec[ev_idx % 32] = 1.0;

        for _ in 0..20 {
            let mut new_vec = [0.0f32; 32];
            for a in 0..32 {
                let mut sum = 0.0f32;
                for b in 0..32 {
                    sum += cov[a][b] * vec[b];
                }
                new_vec[a] = sum;
            }

            let mut norm_sq = 0.0f32;
            for a in 0..32 {
                norm_sq += new_vec[a] * new_vec[a];
            }
            let norm = libm::sqrtf(norm_sq);
            if norm > 0.0 {
                for a in 0..32 {
                    new_vec[a] /= norm;
                }
            }
            vec = new_vec;
        }

        topo.eigenvectors[ev_idx] = vec;

        // Deflate covariance matrix.
        for a in 0..32 {
            for b in 0..32 {
                cov[a][b] -= vec[a] * vec[b] * cov[a][b];
            }
        }
    }
}

fn prefetch_hint_inner(topo: &mut TopoRam, addr: PhysAddr) -> Vec<PhysAddr> {
    if !THEOREM_STATES[1].load(Ordering::Relaxed) {
        return Vec::new();
    }
    if topo.pressure > 0.6 {
        return Vec::new();
    }

    let frame_idx = (addr.as_u64() / 4096) as usize;
    if frame_idx >= USABLE_FRAMES {
        return Vec::new();
    }

    let addr_emb = &topo.frame_meta[frame_idx].embedding;
    let principal = topo.eigenvectors[0];

    let mut candidates: [(usize, f32); PREFETCH_COUNT] = [(0, f32::MIN); PREFETCH_COUNT];

    phys::with_bitmap_mut(|bitmap| {
        for i in 0..USABLE_FRAMES {
            let word = i / 64;
            let bit = i % 64;
            let free = (bitmap[word] >> bit) & 1 == 0;
            if !free {
                continue;
            }

            let emb = &topo.frame_meta[i].embedding;
            let mut score = 0.0f32;
            for j in 0..32 {
                score += (emb[j] as f32 - addr_emb[j] as f32) * principal[j];
            }

            let mut min_idx = 0usize;
            let mut min_score = candidates[0].1;
            for k in 1..PREFETCH_COUNT {
                if candidates[k].1 < min_score {
                    min_score = candidates[k].1;
                    min_idx = k;
                }
            }
            if score > min_score {
                candidates[min_idx] = (i, score);
            }
        }
    });

    let mut result = Vec::with_capacity(PREFETCH_COUNT);
    for (idx, _) in candidates.iter() {
        if *idx < USABLE_FRAMES {
            result.push(PhysAddr::new(*idx as u64 * 4096));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// T3 — Entropy / Fragmentation
// ---------------------------------------------------------------------------

/// Scan the physical bitmap and compute Betti-0 entropy of the free region.
pub fn fragmentation_entropy() -> f32 {
    if !THEOREM_STATES[2].load(Ordering::Relaxed) {
        return 0.0;
    }

    let total_free = phys::free_count();
    if total_free == 0 {
        return 0.0;
    }

    let mut components = 0usize;
    let mut in_run = false;

    phys::with_bitmap_mut(|bitmap| {
        for i in 0..USABLE_FRAMES {
            let word = i / 64;
            let bit = i % 64;
            let free = (bitmap[word] >> bit) & 1 == 0;
            if free && !in_run {
                components += 1;
                in_run = true;
            } else if !free {
                in_run = false;
            }
        }
    });

    let num = libm::log2f((components + 1) as f32);
    let den = libm::log2f((total_free + 1) as f32);
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

// ---------------------------------------------------------------------------
// T4 — PD Governor
// ---------------------------------------------------------------------------

fn update_pressure(topo: &mut TopoRam) {
    if !THEOREM_STATES[3].load(Ordering::Relaxed) {
        return;
    }

    let alloc_rate = topo.alloc_count as f32 / 1024.0;
    let free_rate = topo.free_count as f32 / 1024.0;
    let mut p = (alloc_rate - free_rate) / (alloc_rate + free_rate + 1.0);
    if p < 0.0 {
        p = 0.0;
    } else if p > 1.0 {
        p = 1.0;
    }
    topo.pressure = p;
    topo.alloc_count = 0;
    topo.free_count = 0;
}

/// External API to override pressure directly.
pub fn set_pressure(p: f32) {
    let mut guard = TOPO_RAM.lock();
    if let Some(topo) = guard.as_mut() {
        let mut clamped = p;
        if clamped < 0.0 {
            clamped = 0.0;
        } else if clamped > 1.0 {
            clamped = 1.0;
        }
        topo.pressure = clamped;
    }
}

// ---------------------------------------------------------------------------
// T5 — Hyperbolic Curvature
// ---------------------------------------------------------------------------

fn hyperbolic_class(access_pattern: u64) -> u8 {
    let bits = access_pattern.count_ones();
    let density = bits as f32 / 64.0;
    if density > 0.5 {
        2 // permanent — center of Poincaré disk
    } else if density > 0.1 {
        1 // medium
    } else {
        0 // ephemeral — boundary
    }
}

fn propagate_class(topo: &mut TopoRam, center: usize, class: u8) {
    let offsets: [isize; 8] = [-1, 1, -8, 8, -64, 64, -512, 512];
    for &off in &offsets {
        let ni = center as isize + off;
        if ni >= 0 && (ni as usize) < USABLE_FRAMES {
            topo.frame_meta[ni as usize].lifetime_class = class;
        }
    }
}

// ---------------------------------------------------------------------------
// TopoRam helpers
// ---------------------------------------------------------------------------

impl TopoRam {
    fn push_recent_alloc(&mut self, frame_idx: usize) {
        self.recent_allocs[self.recent_alloc_head] = frame_idx;
        self.recent_alloc_head = (self.recent_alloc_head + 1) % RECENT_ALLOC_LEN;
        if self.recent_alloc_head == 0 {
            update_spectral(self);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// One-time initialisation of the topological RAM metadata.
pub fn init() {
    unsafe {
        let meta = &mut FRAME_META[..];

        let mut seeds = [[0u16; 32]; VORONOI_SEEDS];
        for s in 0..VORONOI_SEEDS {
            for a in 0..32 {
                seeds[s][a] = ((s.wrapping_mul(7919)
                    .wrapping_add(a.wrapping_mul(104729)))
                    % 65536) as u16;
            }
        }

        for i in 0..USABLE_FRAMES {
            let mut embedding = [0u16; 32];
            for a in 0..32 {
                embedding[a] = ((i.wrapping_mul(1103515245)
                    .wrapping_add(12345)
                    .wrapping_add(a.wrapping_mul(65537)))
                    % 65536) as u16;
            }
            let cell = voronoi_cell_of(&embedding, &seeds);
            meta[i] = TopoFrame {
                embedding,
                access_history: 0,
                cell_id: cell,
                lifetime_class: 1,
                entropy_score: 0.0,
            };
        }

        let topo = TopoRam {
            frame_meta: meta,
            voronoi_seeds: seeds,
            pressure: 0.0,
            tick: 0,
            alloc_count: 0,
            free_count: 0,
            recent_allocs: [0; RECENT_ALLOC_LEN],
            recent_alloc_head: 0,
            eigenvectors: [[0.0f32; 32]; 3],
        };

        *TOPO_RAM.lock() = Some(topo);
    }
}

/// Allocate `count` frames, optionally biased toward `hint` embedding.
pub fn alloc_frames(count: usize, hint: Option<&[u16; 32]>) -> Option<PhysAddr> {
    if count == 0 {
        return None;
    }

    let mut guard = TOPO_RAM.lock();
    let topo = guard.as_mut()?;

    topo.tick = topo.tick.wrapping_add(1);
    if topo.tick % 1024 == 0 {
        update_pressure(topo);
    }

    let target_cell = hint.map(|h| voronoi_cell_of(h, &topo.voronoi_seeds));
    let prefer_contiguous = topo.pressure > 0.6 && count > 1;

    // -----------------------------------------------------------------
    // High-pressure path: try contiguous first.
    // -----------------------------------------------------------------
    if prefer_contiguous {
        for _ in 0..3 {
            let mut frames = Vec::with_capacity(count);
            let mut ok = true;
            for _ in 0..count {
                match phys::alloc_frame() {
                    Some(f) => frames.push(f),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && frames.len() == count {
                let contiguous = frames.windows(2).all(|w| w[1].as_u64() == w[0].as_u64() + 4096);
                if contiguous {
                    let first = frames[0];
                    for f in &frames {
                        let idx = (f.as_u64() / 4096) as usize;
                        topo.frame_meta[idx].access_history = 0;
                        topo.push_recent_alloc(idx);
                        topo.alloc_count += 1;
                    }
                    return Some(first);
                }
            }
            // Free partial or non-contiguous batch.
            for f in &frames {
                unsafe { phys::free_frame(*f); }
            }
            if !ok {
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Normal / fallback path: cell-aware search.
    // -----------------------------------------------------------------
    let mut frames = Vec::with_capacity(count);

    phys::with_bitmap_mut(|bitmap| {
        // First pass: target cell.
        if let Some(cell) = target_cell {
            for i in 0..USABLE_FRAMES {
                if frames.len() >= count {
                    break;
                }
                let word = i / 64;
                let bit = i % 64;
                let free = (bitmap[word] >> bit) & 1 == 0;
                if free && topo.frame_meta[i].cell_id == cell {
                    bitmap[word] |= 1 << bit;
                    frames.push(i);
                }
            }
        }

        // Second pass: any free frame.
        if frames.len() < count {
            for i in 0..USABLE_FRAMES {
                if frames.len() >= count {
                    break;
                }
                let word = i / 64;
                let bit = i % 64;
                let free = (bitmap[word] >> bit) & 1 == 0;
                if free {
                    bitmap[word] |= 1 << bit;
                    frames.push(i);
                }
            }
        }
    });

    if frames.len() < count {
        // Roll back partial allocation.
        phys::with_bitmap_mut(|bitmap| {
            for &idx in &frames {
                let word = idx / 64;
                let bit = idx % 64;
                bitmap[word] &= !(1 << bit);
            }
        });
        return None;
    }

    phys::decr_free_count(frames.len());

    // Mark used and update metadata.
    let first = PhysAddr::new(frames[0] as u64 * 4096);
    for &idx in &frames {
        topo.frame_meta[idx].access_history = 0;
        topo.push_recent_alloc(idx);
        topo.alloc_count += 1;
    }

    // Trigger reseed if fragmentation is high.
    let entropy = fragmentation_entropy();
    if entropy > 0.7 {
        reseed_voronoi(topo);
    }

    Some(first)
}

/// Free `count` frames starting at `addr`.
pub fn free_frames(addr: PhysAddr, count: usize) {
    if count == 0 {
        return;
    }

    let mut guard = TOPO_RAM.lock();
    let topo = match guard.as_mut() {
        Some(t) => t,
        None => return,
    };

    topo.tick = topo.tick.wrapping_add(1);
    if topo.tick % 1024 == 0 {
        update_pressure(topo);
    }

    for i in 0..count {
        let frame_addr = PhysAddr::new(addr.as_u64() + (i as u64 * 4096));
        let idx = (frame_addr.as_u64() / 4096) as usize;
        if idx < USABLE_FRAMES {
            let class = hyperbolic_class(topo.frame_meta[idx].access_history);
            topo.frame_meta[idx].lifetime_class = class;
            propagate_class(topo, idx, class);
        }
        unsafe { phys::free_frame(frame_addr); }
        topo.free_count += 1;
    }
}

/// Record an access tick for `addr`.
pub fn access_frame(addr: PhysAddr) {
    let mut guard = TOPO_RAM.lock();
    let topo = match guard.as_mut() {
        Some(t) => t,
        None => return,
    };

    let idx = (addr.as_u64() / 4096) as usize;
    if idx >= USABLE_FRAMES {
        return;
    }

    topo.frame_meta[idx].access_history =
        (topo.frame_meta[idx].access_history << 1) | 1;
    topo.tick = topo.tick.wrapping_add(1);

    if topo.tick % 1024 == 0 {
        update_pressure(topo);
    }
}

/// Return up to 8 prefetch candidates spatially correlated with `addr`.
pub fn prefetch_hint(addr: PhysAddr) -> Vec<PhysAddr> {
    let mut guard = TOPO_RAM.lock();
    let topo = match guard.as_mut() {
        Some(t) => t,
        None => return Vec::new(),
    };
    prefetch_hint_inner(topo, addr)
}
