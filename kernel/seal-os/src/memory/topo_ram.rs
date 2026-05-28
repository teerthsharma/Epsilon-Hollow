// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Topological RAM driver — T1–T5 theorem integration over physical frames.
//!
//! Memory is divided into three Voronoi zones:
//!   LOW      : 0 – 4 GiB DRAM
//!   HIGH     : 4+ GiB DRAM
//!   PCIE_BAR : device memory (metadata only, longest-lived per T5)
//!
//! Each zone carries its own TopoRam instance with 8 seeds so that
//! spectral prefetch, entropy tracking, and hyperbolic lifetime work
//! per-zone.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;
use x86_64::PhysAddr;

use crate::memory::phys::{self, LOW_FRAME_LIMIT};
use crate::THEOREM_STATES;

const VORONOI_SEEDS: usize = 8;
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

/// Zone hint for topology-aware allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoneHint {
    Low,
    High,
    PcieBar,
}

/// Topological RAM state for a single zone.
pub struct TopoRam {
    frame_meta: Vec<TopoFrame>,
    base_frame: usize,
    voronoi_seeds: [[u16; 32]; VORONOI_SEEDS],
    pressure: f32,
    tick: u64,
    alloc_count: u64,
    free_count: u64,
    recent_allocs: [usize; RECENT_ALLOC_LEN],
    recent_alloc_head: usize,
    eigenvectors: [[f32; 32]; 3],
}

/// Global zone container.
pub struct Zones {
    pub low: TopoRam,
    pub high: TopoRam,
    pub pcie_bar: TopoRam,
}

static ZONES: Mutex<Option<Zones>> = Mutex::new(None);

impl TopoRam {
    fn new(base_frame: usize, count: usize) -> Self {
        let mut seeds = [[0u16; 32]; VORONOI_SEEDS];
        for s in 0..VORONOI_SEEDS {
            for a in 0..32usize {
                let tmp: usize = s.wrapping_mul(7919).wrapping_add(a.wrapping_mul(104729));
                seeds[s][a] = (tmp % 65536) as u16;
            }
        }

        let mut meta = Vec::with_capacity(count);
        for i in 0..count {
            let mut embedding = [0u16; 32];
            for a in 0..32usize {
                let tmp: usize = (base_frame + i)
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345)
                    .wrapping_add(a.wrapping_mul(65537));
                embedding[a] = (tmp % 65536) as u16;
            }
            let cell = voronoi_cell_of(&embedding, &seeds);
            meta.push(TopoFrame {
                embedding,
                access_history: 0,
                cell_id: cell,
                lifetime_class: 1,
                entropy_score: 0.0,
            });
        }

        Self {
            frame_meta: meta,
            base_frame,
            voronoi_seeds: seeds,
            pressure: 0.0,
            tick: 0,
            alloc_count: 0,
            free_count: 0,
            recent_allocs: [0; RECENT_ALLOC_LEN],
            recent_alloc_head: 0,
            eigenvectors: [[0.0f32; 32]; 3],
        }
    }

    fn push_recent_alloc(&mut self, frame_idx: usize) {
        self.recent_allocs[self.recent_alloc_head] = frame_idx;
        self.recent_alloc_head = (self.recent_alloc_head + 1) % RECENT_ALLOC_LEN;
        if self.recent_alloc_head == 0 {
            update_spectral(self);
        }
    }
}
fn zone_for_global_frame(global: usize) -> ZoneHint {
    if global < LOW_FRAME_LIMIT {
        ZoneHint::Low
    } else if global < phys::total_usable_frames() {
        ZoneHint::High
    } else {
        ZoneHint::PcieBar
    }
}

// ---------------------------------------------------------------------------
// T1 — Voronoi Partitioning (per-zone)
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
        if frame_idx < topo.frame_meta.len() && picked < VORONOI_SEEDS {
            new_seeds[picked] = topo.frame_meta[frame_idx].embedding;
            picked += 1;
        }
        if picked >= VORONOI_SEEDS {
            break;
        }
    }

    if picked < VORONOI_SEEDS {
        for i in picked..VORONOI_SEEDS {
            let idx = (i.wrapping_mul(65537).wrapping_add(7)) % topo.frame_meta.len().max(1);
            new_seeds[i] = topo.frame_meta[idx].embedding;
        }
    }

    topo.voronoi_seeds = new_seeds;

    for i in 0..topo.frame_meta.len() {
        let cell = voronoi_cell_of(&topo.frame_meta[i].embedding, &topo.voronoi_seeds);
        topo.frame_meta[i].cell_id = cell;
    }
}

// ---------------------------------------------------------------------------
// T2 — Spectral Contraction (per-zone)
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
        if frame_idx < topo.frame_meta.len() {
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
        if frame_idx < topo.frame_meta.len() {
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

    let global_idx = (addr.as_u64() / 4096) as usize;
    if global_idx < topo.base_frame || global_idx >= topo.base_frame + topo.frame_meta.len() {
        return Vec::new();
    }
    let ref_idx = global_idx - topo.base_frame;
    let addr_emb = &topo.frame_meta[ref_idx].embedding;
    let principal = topo.eigenvectors[0];

    let mut candidates: [(usize, f32); PREFETCH_COUNT] = [(0, f32::MIN); PREFETCH_COUNT];

    phys::with_bitmap_mut(|bitmap| {
        for local_idx in 0..topo.frame_meta.len() {
            let gidx = topo.base_frame + local_idx;
            let word = gidx / 64;
            let bit = gidx % 64;
            let free = (bitmap[word] >> bit) & 1 == 0;
            if !free {
                continue;
            }

            let emb = &topo.frame_meta[local_idx].embedding;
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
                candidates[min_idx] = (local_idx, score);
            }
        }
    });

    let mut result = Vec::with_capacity(PREFETCH_COUNT);
    for (local_idx, _) in candidates.iter() {
        if *local_idx < topo.frame_meta.len() {
            result.push(PhysAddr::new((topo.base_frame + *local_idx) as u64 * 4096));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// T3 — Entropy / Fragmentation
// ---------------------------------------------------------------------------

fn fragmentation_entropy_inner(_topo: &TopoRam, zone_start: usize, zone_end: usize) -> f32 {
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
        for i in zone_start..zone_end {
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

/// Global fragmentation entropy (all zones).
pub fn fragmentation_entropy() -> f32 {
    if !THEOREM_STATES[2].load(Ordering::Relaxed) {
        return 0.0;
    }
    let total_free = phys::free_count();
    if total_free == 0 {
        return 0.0;
    }
    let limit = phys::total_usable_frames();
    let mut components = 0usize;
    let mut in_run = false;
    phys::with_bitmap_mut(|bitmap| {
        for i in 0..limit {
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

/// External API to override pressure directly (applies to all zones).
pub fn set_pressure(p: f32) {
    let mut guard = ZONES.lock();
    if let Some(zones) = guard.as_mut() {
        let clamped = p.clamp(0.0, 1.0);
        zones.low.pressure = clamped;
        zones.high.pressure = clamped;
        zones.pcie_bar.pressure = clamped;
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
        if ni >= 0 && (ni as usize) < topo.frame_meta.len() {
            topo.frame_meta[ni as usize].lifetime_class = class;
        }
    }
}

// ---------------------------------------------------------------------------
// Zone helpers
// ---------------------------------------------------------------------------

fn alloc_frames_in_zone(
    topo: &mut TopoRam,
    count: usize,
    hint: Option<&[u16; 32]>,
) -> Option<PhysAddr> {
    let limit = phys::total_usable_frames();
    let zone_start = topo.base_frame;
    let zone_end = zone_start + topo.frame_meta.len();

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
                let contiguous = frames
                    .windows(2)
                    .all(|w| w[1].as_u64() == w[0].as_u64() + 4096);
                if contiguous {
                    let first = frames[0];
                    for f in &frames {
                        let gidx = (f.as_u64() / 4096) as usize;
                        let lidx = gidx.saturating_sub(topo.base_frame);
                        if let Some(meta) = topo.frame_meta.get_mut(lidx) {
                            meta.access_history = 0;
                        }
                        topo.push_recent_alloc(lidx);
                        topo.alloc_count += 1;
                    }
                    return Some(first);
                }
            }
            for f in &frames {
                unsafe {
                    phys::free_frame(*f);
                }
            }
            if !ok {
                break;
            }
        }
    }

    // -----------------------------------------------------------------
    // Normal / fallback path: cell-aware search within zone range.
    // -----------------------------------------------------------------
    let mut frames = Vec::with_capacity(count);

    phys::with_bitmap_mut(|bitmap| {
        if let Some(cell) = target_cell {
            for gidx in zone_start..zone_end {
                if frames.len() >= count {
                    break;
                }
                if gidx >= limit {
                    break;
                }
                let word = gidx / 64;
                let bit = gidx % 64;
                let free = (bitmap[word] >> bit) & 1 == 0;
                let lidx = gidx - zone_start;
                if free && topo.frame_meta[lidx].cell_id == cell {
                    bitmap[word] |= 1 << bit;
                    frames.push(gidx);
                }
            }
        }

        if frames.len() < count {
            for gidx in zone_start..zone_end {
                if frames.len() >= count {
                    break;
                }
                if gidx >= limit {
                    break;
                }
                let word = gidx / 64;
                let bit = gidx % 64;
                let free = (bitmap[word] >> bit) & 1 == 0;
                if free {
                    bitmap[word] |= 1 << bit;
                    frames.push(gidx);
                }
            }
        }
    });

    if frames.len() < count {
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

    let first = PhysAddr::new(frames[0] as u64 * 4096);
    for &gidx in &frames {
        let lidx = gidx - zone_start;
        if let Some(meta) = topo.frame_meta.get_mut(lidx) {
            meta.access_history = 0;
        }
        topo.push_recent_alloc(lidx);
        topo.alloc_count += 1;
    }

    let entropy = fragmentation_entropy_inner(topo, zone_start, zone_end);
    if entropy > 0.7 {
        reseed_voronoi(topo);
    }

    Some(first)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// One-time initialisation of the topological RAM metadata (per-zone).
pub fn init() {
    let total = phys::total_usable_frames();
    let low_count = total.min(LOW_FRAME_LIMIT);
    let high_count = total.saturating_sub(LOW_FRAME_LIMIT);

    let low = TopoRam::new(0, low_count);
    let high = TopoRam::new(LOW_FRAME_LIMIT, high_count);
    let pcie_bar = TopoRam::new(0, 0); // empty until BARs are registered

    *ZONES.lock() = Some(Zones {
        low,
        high,
        pcie_bar,
    });
}

/// Register a PCI BAR range as the PCIE_BAR zone.
pub fn register_pcie_bar(start: PhysAddr, pages: usize) {
    let mut guard = ZONES.lock();
    if let Some(zones) = guard.as_mut() {
        let base_frame = (start.as_u64() / 4096) as usize;
        let mut new_zone = TopoRam::new(base_frame, pages);
        // T5: PCIE_BAR frames are longest-lived (permanent).
        for meta in &mut new_zone.frame_meta {
            meta.lifetime_class = 2;
        }
        zones.pcie_bar = new_zone;
    }
}

/// Allocate `count` frames, optionally biased toward `hint` embedding.
pub fn alloc_frames(count: usize, zone: ZoneHint, hint: Option<&[u16; 32]>) -> Option<PhysAddr> {
    if count == 0 {
        return None;
    }

    let mut guard = ZONES.lock();
    let zones = guard.as_mut()?;

    match zone {
        ZoneHint::Low => {
            if let Some(addr) = alloc_frames_in_zone(&mut zones.low, count, hint) {
                return Some(addr);
            }
            alloc_frames_in_zone(&mut zones.high, count, hint)
        }
        ZoneHint::High => {
            if let Some(addr) = alloc_frames_in_zone(&mut zones.high, count, hint) {
                return Some(addr);
            }
            alloc_frames_in_zone(&mut zones.low, count, hint)
        }
        ZoneHint::PcieBar => {
            // PCIE_BAR is not allocatable from the physical allocator;
            // fallback to High then Low.
            if let Some(addr) = alloc_frames_in_zone(&mut zones.high, count, hint) {
                return Some(addr);
            }
            alloc_frames_in_zone(&mut zones.low, count, hint)
        }
    }
}

/// Free `count` frames starting at `addr`.
pub fn free_frames(addr: PhysAddr, count: usize) {
    if count == 0 {
        return;
    }

    let mut guard = ZONES.lock();
    let zones = match guard.as_mut() {
        Some(z) => z,
        None => return,
    };

    for i in 0..count {
        let frame_addr = PhysAddr::new(addr.as_u64() + (i as u64 * 4096));
        let gidx = (frame_addr.as_u64() / 4096) as usize;
        let zone = zone_for_global_frame(gidx);
        let topo = match zone {
            ZoneHint::Low => &mut zones.low,
            ZoneHint::High => &mut zones.high,
            ZoneHint::PcieBar => &mut zones.pcie_bar,
        };
        let lidx = gidx.saturating_sub(topo.base_frame);
        if let Some(meta) = topo.frame_meta.get_mut(lidx) {
            let class = hyperbolic_class(meta.access_history);
            meta.lifetime_class = class;
            propagate_class(topo, lidx, class);
        }
        unsafe {
            phys::free_frame(frame_addr);
        }
        topo.free_count += 1;
        topo.tick = topo.tick.wrapping_add(1);
        if topo.tick % 1024 == 0 {
            update_pressure(topo);
        }
    }
}

/// Record an access tick for `addr`.
pub fn access_frame(addr: PhysAddr) {
    let mut guard = ZONES.lock();
    let zones = match guard.as_mut() {
        Some(z) => z,
        None => return,
    };

    let gidx = (addr.as_u64() / 4096) as usize;
    let zone = zone_for_global_frame(gidx);
    let topo = match zone {
        ZoneHint::Low => &mut zones.low,
        ZoneHint::High => &mut zones.high,
        ZoneHint::PcieBar => &mut zones.pcie_bar,
    };
    let lidx = gidx.saturating_sub(topo.base_frame);
    if let Some(meta) = topo.frame_meta.get_mut(lidx) {
        meta.access_history = (meta.access_history << 1) | 1;
        topo.tick = topo.tick.wrapping_add(1);
        if topo.tick % 1024 == 0 {
            update_pressure(topo);
        }
    }
}

/// Return up to 8 prefetch candidates spatially correlated with `addr`.
pub fn prefetch_hint(addr: PhysAddr) -> Vec<PhysAddr> {
    let mut guard = ZONES.lock();
    let zones = match guard.as_mut() {
        Some(z) => z,
        None => return Vec::new(),
    };
    let gidx = (addr.as_u64() / 4096) as usize;
    let zone = zone_for_global_frame(gidx);
    let topo = match zone {
        ZoneHint::Low => &mut zones.low,
        ZoneHint::High => &mut zones.high,
        ZoneHint::PcieBar => &mut zones.pcie_bar,
    };
    prefetch_hint_inner(topo, addr)
}

/// Return the access-history bit density for a physical frame (used by swap daemon).
pub fn frame_access_density(addr: PhysAddr) -> Option<u32> {
    let mut guard = ZONES.lock();
    let zones = guard.as_mut()?;
    let gidx = (addr.as_u64() / 4096) as usize;
    let zone = zone_for_global_frame(gidx);
    let topo = match zone {
        ZoneHint::Low => &mut zones.low,
        ZoneHint::High => &mut zones.high,
        ZoneHint::PcieBar => &mut zones.pcie_bar,
    };
    let lidx = gidx.saturating_sub(topo.base_frame);
    topo.frame_meta
        .get(lidx)
        .map(|m| m.access_history.count_ones())
}

/// Find candidate frames for swap-out (T5: high bit-density = short-lived).
pub fn find_swap_candidates(count: usize) -> Vec<PhysAddr> {
    let mut guard = ZONES.lock();
    let zones = match guard.as_mut() {
        Some(z) => z,
        None => return Vec::new(),
    };
    let limit = phys::total_usable_frames();
    let mut candidates: Vec<(PhysAddr, u32)> = Vec::with_capacity(count * 2);

    let scan_low = zones.low.frame_meta.len().min(4096);
    let scan_high = zones.high.frame_meta.len().min(4096);

    phys::with_bitmap_mut(|bitmap| {
        for local_idx in 0..scan_low {
            let gidx = zones.low.base_frame + local_idx;
            if gidx >= limit {
                break;
            }
            let word = gidx / 64;
            let bit = gidx % 64;
            if (bitmap[word] >> bit) & 1 != 0 {
                let density = zones.low.frame_meta[local_idx].access_history.count_ones();
                candidates.push((PhysAddr::new(gidx as u64 * 4096), density));
            }
        }
        for local_idx in 0..scan_high {
            let gidx = zones.high.base_frame + local_idx;
            if gidx >= limit {
                break;
            }
            let word = gidx / 64;
            let bit = gidx % 64;
            if (bitmap[word] >> bit) & 1 != 0 {
                let density = zones.high.frame_meta[local_idx].access_history.count_ones();
                candidates.push((PhysAddr::new(gidx as u64 * 4096), density));
            }
        }
    });

    // T5: high density = short-lived = swap first.
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.truncate(count);
    candidates.into_iter().map(|(addr, _)| addr).collect()
}
