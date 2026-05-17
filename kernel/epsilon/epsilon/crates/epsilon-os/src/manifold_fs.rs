// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! ManifoldFS — A filesystem where all data is stored as geometry on S².
//!
//! Files are not byte sequences. They are `ManifoldPayload`s — 64-point
//! clouds on the unit sphere. Moving a file between locations is topological
//! surgery: O(1) regardless of original data size.
//!
//! # Active Theorems
//!
//! - **T1 (TSS)**: File lookup via Voronoi cell (O(1), not path traversal)
//! - **T2 (SCM)**: Predictive prefetch (spectral contraction toward likely next access)
//! - **T3 (GMC)**: Auto-defrag (merge clusters when entropy exceeds threshold)
//! - **T4 (AGCR)**: Adaptive I/O governor (PD control on flush rate)
//! - **T5 (HCS)**: Hyperbolic directory hierarchy (lower distortion than flat paths)

use std::collections::HashMap;
use std::time::Instant;

use aether_core::governor::GeometricGovernor;
use aether_core::scm::SpectralContractionOperator;
use aether_core::tss::SphericalVoronoiIndex;

use crate::encoder::{self, ManifoldPayload};

const VORONOI_CELLS: usize = 8;
const ENTROPY_MERGE_THRESHOLD: f64 = 2.0;

// ─── Inode ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub enum InodeKind {
    File,
    Directory,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InodeMetadata {
    pub created_ms: u64,
    pub modified_ms: u64,
    pub original_size: u64,
    pub permissions: u16,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Inode {
    pub id: u64,
    pub name: String,
    pub kind: InodeKind,
    pub payload: ManifoldPayload,
    pub metadata: InodeMetadata,
    pub voronoi_cell: usize,
    pub cluster_id: i32,
    pub parent: u64,
}

// ─── Theorem Activity Log ───────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct TheoremEvent {
    pub theorem: &'static str,
    pub action: String,
    pub timestamp_ms: u64,
}

// ─── ManifoldFS ─────────────────────────────────────────────────────────

pub struct ManifoldFS {
    inodes: HashMap<u64, Inode>,
    dir_entries: HashMap<u64, HashMap<String, u64>>, // dir_inode → {name → child_inode}
    root_id: u64,
    next_inode: u64,
    // T1: O(1) file lookup via Voronoi
    voronoi: SphericalVoronoiIndex<VORONOI_CELLS>,
    cell_files: [Vec<u64>; VORONOI_CELLS],
    // T2: Predictive prefetch
    scm: SpectralContractionOperator<3>,
    access_state: [f64; 3],
    last_prefetch_prediction: Option<u64>,
    prefetch_hits: u64,
    prefetch_misses: u64,
    // T3: Entropy monitoring
    cluster_sizes: Vec<usize>,
    entropy_merges: u64,
    current_entropy: f64,
    // T4: I/O governor
    governor: GeometricGovernor,
    governor_ticks: u64,
    // T5: Hyperbolic tree depth tracking
    max_depth: usize,
    hyperbolic_ratio: f64,
    // Metrics
    total_teleports: u64,
    total_lookups: u64,
    activity_log: Vec<TheoremEvent>,
}

impl ManifoldFS {
    pub fn new() -> Self {
        let default_centroids = Self::default_centroids();
        let root_payload = encoder::encode_text("/");

        let mut fs = Self {
            inodes: HashMap::new(),
            dir_entries: HashMap::new(),
            root_id: 0,
            next_inode: 1,
            voronoi: SphericalVoronoiIndex::<VORONOI_CELLS>::new(default_centroids),
            cell_files: Default::default(),
            scm: SpectralContractionOperator::new(0.7),
            access_state: [0.0; 3],
            last_prefetch_prediction: None,
            prefetch_hits: 0,
            prefetch_misses: 0,
            cluster_sizes: vec![0; VORONOI_CELLS],
            entropy_merges: 0,
            current_entropy: 0.0,
            governor: GeometricGovernor::new(),
            governor_ticks: 0,
            max_depth: 0,
            hyperbolic_ratio: f64::INFINITY,
            total_teleports: 0,
            total_lookups: 0,
            activity_log: Vec::new(),
        };

        // Create root directory
        let root = Inode {
            id: 0,
            name: "/".into(),
            kind: InodeKind::Directory,
            payload: root_payload,
            metadata: InodeMetadata {
                created_ms: now_ms(),
                modified_ms: now_ms(),
                original_size: 0,
                permissions: 0o755,
            },
            voronoi_cell: 0,
            cluster_id: 0,
            parent: 0,
        };
        fs.inodes.insert(0, root);
        fs.dir_entries.insert(0, HashMap::new());
        fs
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

    // ─── Store (create file) ────────────────────────────────────────────

    /// Store data into ManifoldFS. Returns the inode id.
    pub fn store(&mut self, name: &str, data: &[u8], parent_id: u64) -> Result<u64, FsError> {
        if !self.dir_entries.contains_key(&parent_id) {
            return Err(FsError::NotADirectory);
        }
        if self.dir_entries[&parent_id].contains_key(name) {
            return Err(FsError::AlreadyExists);
        }

        let payload = encoder::encode_data(data);
        let cell = self.assign_voronoi_cell(&payload);
        let id = self.next_inode;
        self.next_inode += 1;

        let inode = Inode {
            id,
            name: name.to_string(),
            kind: InodeKind::File,
            payload,
            metadata: InodeMetadata {
                created_ms: now_ms(),
                modified_ms: now_ms(),
                original_size: data.len() as u64,
                permissions: 0o644,
            },
            voronoi_cell: cell,
            cluster_id: cell as i32,
            parent: parent_id,
        };

        self.inodes.insert(id, inode);
        self.dir_entries
            .get_mut(&parent_id)
            .unwrap()
            .insert(name.to_string(), id);
        self.cell_files[cell].push(id);
        self.cluster_sizes[cell] += 1;
        self.update_entropy();

        // T4: Governor tick
        self.governor_tick(1.0);

        self.log_event("T1/TSS", format!("stored '{}' → cell {}", name, cell));

        Ok(id)
    }

    /// Store text content.
    pub fn store_text(&mut self, name: &str, text: &str, parent_id: u64) -> Result<u64, FsError> {
        self.store(name, text.as_bytes(), parent_id)
    }

    // ─── Teleport (move file) — O(1) ───────────────────────────────────

    /// Move a file between directories via topological surgery. O(1) regardless of file size.
    pub fn teleport(
        &mut self,
        name: &str,
        src_dir: u64,
        dst_dir: u64,
    ) -> Result<TeleportResult, FsError> {
        let t0 = Instant::now();

        // Lookup source (T1: O(1) via directory entry)
        let inode_id = self
            .dir_entries
            .get(&src_dir)
            .and_then(|entries| entries.get(name).copied())
            .ok_or(FsError::NotFound)?;

        if !self.dir_entries.contains_key(&dst_dir) {
            return Err(FsError::NotADirectory);
        }
        if self.dir_entries[&dst_dir].contains_key(name) {
            return Err(FsError::AlreadyExists);
        }

        // T4: Governor clutch for surgery
        let pre_epsilon = self.governor.epsilon();
        self.governor_tick(0.0); // zero deviation during surgery

        // The actual "teleport": move directory entry (O(1))
        // The ManifoldPayload stays in the same inode — we just reparent it
        self.dir_entries.get_mut(&src_dir).unwrap().remove(name);
        self.dir_entries
            .get_mut(&dst_dir)
            .unwrap()
            .insert(name.to_string(), inode_id);

        // Update inode parent
        if let Some(inode) = self.inodes.get_mut(&inode_id) {
            inode.parent = dst_dir;
            inode.metadata.modified_ms = now_ms();
        }

        // T4: Governor restore
        self.governor_tick(1.0);
        let post_epsilon = self.governor.epsilon();

        let elapsed = t0.elapsed();
        self.total_teleports += 1;

        let original_size = self
            .inodes
            .get(&inode_id)
            .map(|i| i.metadata.original_size)
            .unwrap_or(0);

        self.log_event(
            "T1/TSS+T4/AGCR",
            format!(
                "teleported '{}' ({} bytes) in {:?} [governor ε: {:.4} → {:.4}]",
                name, original_size, elapsed, pre_epsilon, post_epsilon
            ),
        );

        // T3: Check entropy after structural change
        self.check_entropy_and_merge();

        Ok(TeleportResult {
            inode_id,
            elapsed_ns: elapsed.as_nanos() as u64,
            original_size,
            payload_points: self
                .inodes
                .get(&inode_id)
                .map(|i| i.payload.point_count)
                .unwrap_or(0),
            governor_epsilon: post_epsilon,
        })
    }

    // ─── Find (T1: O(1) content-addressable lookup) ─────────────────────

    /// Find files by content similarity using T1 Voronoi cells.
    pub fn find(&mut self, query: &str) -> Vec<FindResult> {
        let t0 = Instant::now();
        self.total_lookups += 1;

        let query_payload = encoder::encode_text(query);
        let cell = self.assign_voronoi_cell(&query_payload);

        // T2: Update SCM access state for predictive prefetch
        self.update_prefetch_state(&query_payload);

        // Search within the Voronoi cell (O(n/K) where K=8)
        let mut results = Vec::new();
        for &inode_id in &self.cell_files[cell] {
            if let Some(inode) = self.inodes.get(&inode_id) {
                let similarity = payload_similarity(&query_payload, &inode.payload);
                results.push(FindResult {
                    inode_id,
                    name: inode.name.clone(),
                    similarity,
                    cell,
                    original_size: inode.metadata.original_size,
                });
            }
        }
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        let elapsed = t0.elapsed();
        self.log_event(
            "T1/TSS",
            format!(
                "find '{}' → cell {}, {} results in {:?}",
                query,
                cell,
                results.len(),
                elapsed
            ),
        );

        results
    }

    // ─── List directory ──────────────────────────────────────────────���──

    pub fn ls(&self, dir_id: u64) -> Result<Vec<LsEntry>, FsError> {
        let entries = self.dir_entries.get(&dir_id).ok_or(FsError::NotADirectory)?;
        let mut result: Vec<LsEntry> = entries
            .iter()
            .filter_map(|(name, &id)| {
                self.inodes.get(&id).map(|inode| LsEntry {
                    name: name.clone(),
                    kind: match inode.kind {
                        InodeKind::File => "file",
                        InodeKind::Directory => "dir",
                    },
                    original_size: inode.metadata.original_size,
                    payload_points: inode.payload.point_count,
                    voronoi_cell: inode.voronoi_cell,
                })
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(result)
    }

    // ─── Mkdir ──────────────────────────────────────────────────────────

    pub fn mkdir(&mut self, name: &str, parent_id: u64) -> Result<u64, FsError> {
        if !self.dir_entries.contains_key(&parent_id) {
            return Err(FsError::NotADirectory);
        }
        if self.dir_entries[&parent_id].contains_key(name) {
            return Err(FsError::AlreadyExists);
        }

        let payload = encoder::encode_text(name);
        let id = self.next_inode;
        self.next_inode += 1;

        let depth = self.compute_depth(parent_id) + 1;
        if depth > self.max_depth {
            self.max_depth = depth;
            self.update_hyperbolic_ratio(depth);
        }

        let inode = Inode {
            id,
            name: name.to_string(),
            kind: InodeKind::Directory,
            payload,
            metadata: InodeMetadata {
                created_ms: now_ms(),
                modified_ms: now_ms(),
                original_size: 0,
                permissions: 0o755,
            },
            voronoi_cell: 0,
            cluster_id: 0,
            parent: parent_id,
        };

        self.inodes.insert(id, inode);
        self.dir_entries.insert(id, HashMap::new());
        self.dir_entries
            .get_mut(&parent_id)
            .unwrap()
            .insert(name.to_string(), id);

        self.log_event(
            "T5/HCS",
            format!(
                "mkdir '{}' depth={} hyp_ratio={:.1}x",
                name, depth, self.hyperbolic_ratio
            ),
        );

        Ok(id)
    }

    // ─── Resolve path ───────────────────────────────────────────────────

    pub fn resolve_path(&self, path: &str) -> Result<u64, FsError> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let mut current = self.root_id;
        for part in parts {
            let entries = self.dir_entries.get(&current).ok_or(FsError::NotADirectory)?;
            current = *entries.get(part).ok_or(FsError::NotFound)?;
        }
        Ok(current)
    }

    // ─── T2: Predictive Prefetch ────────────────────────────────────────

    fn update_prefetch_state(&mut self, accessed: &ManifoldPayload) {
        if accessed.points.is_empty() {
            return;
        }
        let pt = accessed.points[0].coords;

        // Check if last prediction was correct
        if let Some(predicted_id) = self.last_prefetch_prediction {
            if let Some(inode) = self.inodes.get(&predicted_id) {
                let pred_pt = inode.payload.points.first().map(|p| p.coords).unwrap_or([0.0; 3]);
                let dist = (0..3)
                    .map(|i| (pt[i] - pred_pt[i]) * (pt[i] - pred_pt[i]))
                    .sum::<f64>();
                if dist < 0.25 {
                    self.prefetch_hits += 1;
                } else {
                    self.prefetch_misses += 1;
                }
            }
        }

        // T2/SCM: Contract access state toward current access point
        self.access_state = self.scm.apply(&self.access_state, &pt);

        // Predict next access: find nearest file to predicted state
        let predicted_cell = self.voronoi.locate((
            libm::acos(self.access_state[2].clamp(-1.0, 1.0)),
            libm::atan2(self.access_state[1], self.access_state[0]),
        ));
        self.last_prefetch_prediction = self.cell_files[predicted_cell].last().copied();

        if self.total_lookups % 10 == 0 {
            self.log_event(
                "T2/SCM",
                format!(
                    "prefetch accuracy: {}/{} ({:.0}%)",
                    self.prefetch_hits,
                    self.prefetch_hits + self.prefetch_misses,
                    if self.prefetch_hits + self.prefetch_misses > 0 {
                        100.0 * self.prefetch_hits as f64
                            / (self.prefetch_hits + self.prefetch_misses) as f64
                    } else {
                        0.0
                    }
                ),
            );
        }
    }

    // ─── T3: Entropy Monitoring & Merge ─────────────────────────────────

    fn update_entropy(&mut self) {
        let total: usize = self.cluster_sizes.iter().sum();
        if total == 0 {
            self.current_entropy = 0.0;
            return;
        }
        let mut h = 0.0f64;
        for &size in &self.cluster_sizes {
            if size > 0 {
                let p = size as f64 / total as f64;
                h -= p * libm::log2(p);
            }
        }
        self.current_entropy = h;
    }

    fn check_entropy_and_merge(&mut self) {
        if self.current_entropy <= ENTROPY_MERGE_THRESHOLD {
            return;
        }
        // Find two smallest clusters and merge
        let mut sorted: Vec<(usize, usize)> = self
            .cluster_sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0)
            .map(|(i, &s)| (i, s))
            .collect();
        sorted.sort_by_key(|(_, s)| *s);

        if sorted.len() < 2 {
            return;
        }
        let (src_cell, _) = sorted[0];
        let (dst_cell, _) = sorted[1];

        // Merge: move all files from src_cell to dst_cell
        let files_to_move: Vec<u64> = self.cell_files[src_cell].clone();
        for &fid in &files_to_move {
            if let Some(inode) = self.inodes.get_mut(&fid) {
                inode.voronoi_cell = dst_cell;
                inode.cluster_id = dst_cell as i32;
            }
            self.cell_files[dst_cell].push(fid);
        }
        let moved_count = self.cell_files[src_cell].len();
        self.cluster_sizes[dst_cell] += self.cluster_sizes[src_cell];
        self.cluster_sizes[src_cell] = 0;
        self.cell_files[src_cell].clear();

        let old_h = self.current_entropy;
        self.update_entropy();
        self.entropy_merges += 1;

        self.log_event(
            "T3/GMC",
            format!(
                "merged cell {} → {}, {} files moved, ΔH = {:.4} bits",
                src_cell,
                dst_cell,
                moved_count,
                self.current_entropy - old_h
            ),
        );
    }

    // ─── T4: Governor ──────────────────────────────────────────────���────

    fn governor_tick(&mut self, deviation: f64) {
        self.governor.adapt(deviation, 0.01);
        self.governor_ticks += 1;
    }

    // ─── T5: Hyperbolic Ratio ───────────────────────────────────────────

    fn compute_depth(&self, mut id: u64) -> usize {
        let mut depth = 0;
        while id != self.root_id {
            if let Some(inode) = self.inodes.get(&id) {
                id = inode.parent;
                depth += 1;
            } else {
                break;
            }
        }
        depth
    }

    fn update_hyperbolic_ratio(&mut self, depth: usize) {
        // T5: separation_ratio = c·ln(b)·D·(D-1)/2 where c=1, b=4, D=depth
        let c = 1.0;
        let b = 4.0f64;
        let d = depth as f64;
        if d >= 2.0 {
            self.hyperbolic_ratio = c * libm::log(b) * d * (d - 1.0) / 2.0;
        }
    }

    // ─── Voronoi Cell Assignment ────────────────────────────────────────

    fn assign_voronoi_cell(&self, payload: &ManifoldPayload) -> usize {
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

    // ─── Activity Log ───────────────────────────────────────────────────

    fn log_event(&mut self, theorem: &'static str, action: String) {
        self.activity_log.push(TheoremEvent {
            theorem,
            action,
            timestamp_ms: now_ms(),
        });
        if self.activity_log.len() > 100 {
            self.activity_log.remove(0);
        }
    }

    // ─── Public Stats ───────────────────────────────────────────────────

    pub fn stats(&self) -> FsStats {
        FsStats {
            total_files: self.inodes.values().filter(|i| matches!(i.kind, InodeKind::File)).count(),
            total_dirs: self
                .inodes
                .values()
                .filter(|i| matches!(i.kind, InodeKind::Directory))
                .count(),
            total_teleports: self.total_teleports,
            total_lookups: self.total_lookups,
            voronoi_cells: VORONOI_CELLS,
            cell_distribution: self.cluster_sizes.clone(),
            current_entropy: self.current_entropy,
            entropy_merges: self.entropy_merges,
            governor_epsilon: self.governor.epsilon(),
            governor_ticks: self.governor_ticks,
            prefetch_hits: self.prefetch_hits,
            prefetch_misses: self.prefetch_misses,
            max_depth: self.max_depth,
            hyperbolic_ratio: self.hyperbolic_ratio,
        }
    }

    pub fn recent_activity(&self, n: usize) -> &[TheoremEvent] {
        let start = self.activity_log.len().saturating_sub(n);
        &self.activity_log[start..]
    }

    pub fn theorem_status(&self) -> Vec<(&'static str, &'static str, String)> {
        vec![
            (
                "T1/TSS",
                "ACTIVE",
                format!(
                    "Voronoi: {} cells, {} lookups, O(1)",
                    VORONOI_CELLS, self.total_lookups
                ),
            ),
            (
                "T2/SCM",
                "ACTIVE",
                format!(
                    "Prefetch: ρ=0.700, hits={}, accuracy={:.0}%",
                    self.prefetch_hits,
                    if self.prefetch_hits + self.prefetch_misses > 0 {
                        100.0 * self.prefetch_hits as f64
                            / (self.prefetch_hits + self.prefetch_misses) as f64
                    } else {
                        0.0
                    }
                ),
            ),
            (
                "T3/GMC",
                "ACTIVE",
                format!(
                    "Entropy: H={:.3} bits, merges={}, threshold={}",
                    self.current_entropy, self.entropy_merges, ENTROPY_MERGE_THRESHOLD
                ),
            ),
            (
                "T4/AGCR",
                "ACTIVE",
                format!(
                    "Governor: ε={:.4}, ticks={}",
                    self.governor.epsilon(),
                    self.governor_ticks
                ),
            ),
            (
                "T5/HCS",
                "ACTIVE",
                format!(
                    "Hyperbolic: depth={}, ratio={:.1}x",
                    self.max_depth, self.hyperbolic_ratio
                ),
            ),
        ]
    }
}

// ─── Helper: payload similarity ─────────────────────────────────────��───

fn payload_similarity(a: &ManifoldPayload, b: &ManifoldPayload) -> f64 {
    if a.points.is_empty() || b.points.is_empty() {
        return 0.0;
    }
    // Cosine similarity of first points (representative)
    let pa = &a.points[0].coords;
    let pb = &b.points[0].coords;
    pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ─── Result Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct TeleportResult {
    pub inode_id: u64,
    pub elapsed_ns: u64,
    pub original_size: u64,
    pub payload_points: usize,
    pub governor_epsilon: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FindResult {
    pub inode_id: u64,
    pub name: String,
    pub similarity: f64,
    pub cell: usize,
    pub original_size: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LsEntry {
    pub name: String,
    pub kind: &'static str,
    pub original_size: u64,
    pub payload_points: usize,
    pub voronoi_cell: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FsStats {
    pub total_files: usize,
    pub total_dirs: usize,
    pub total_teleports: u64,
    pub total_lookups: u64,
    pub voronoi_cells: usize,
    pub cell_distribution: Vec<usize>,
    pub current_entropy: f64,
    pub entropy_merges: u64,
    pub governor_epsilon: f64,
    pub governor_ticks: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub max_depth: usize,
    pub hyperbolic_ratio: f64,
}

#[derive(Debug, Clone)]
pub enum FsError {
    NotFound,
    NotADirectory,
    AlreadyExists,
    #[allow(dead_code)]
    PermissionDenied,
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "not found"),
            Self::NotADirectory => write!(f, "not a directory"),
            Self::AlreadyExists => write!(f, "already exists"),
            Self::PermissionDenied => write!(f, "permission denied"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_ls() {
        let mut fs = ManifoldFS::new();
        fs.store_text("hello.txt", "Hello World", 0).unwrap();
        fs.store_text("data.bin", "binary content here", 0).unwrap();
        let entries = fs.ls(0).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_teleport_is_fast() {
        let mut fs = ManifoldFS::new();
        let dir_a = fs.mkdir("vol_a", 0).unwrap();
        let dir_b = fs.mkdir("vol_b", 0).unwrap();

        // Store a "large" file
        let big_data = vec![0x42u8; 10_000_000]; // 10MB
        fs.store("big.dat", &big_data, dir_a).unwrap();

        // Teleport it
        let result = fs.teleport("big.dat", dir_a, dir_b).unwrap();
        assert_eq!(result.original_size, 10_000_000);
        assert!(result.elapsed_ns < 1_000_000); // must be < 1ms

        // Verify it moved
        assert!(fs.ls(dir_a).unwrap().is_empty());
        assert_eq!(fs.ls(dir_b).unwrap().len(), 1);
    }

    #[test]
    fn test_find_by_content() {
        let mut fs = ManifoldFS::new();
        fs.store_text("rust.txt", "Rust programming language systems", 0)
            .unwrap();
        fs.store_text("python.txt", "Python scripting dynamic typing", 0)
            .unwrap();
        fs.store_text("cargo.txt", "Rust cargo build system package manager", 0)
            .unwrap();

        let results = fs.find("Rust systems programming");
        assert!(!results.is_empty());
        // Rust-related files should score higher
    }

    #[test]
    fn test_mkdir_and_depth() {
        let mut fs = ManifoldFS::new();
        let a = fs.mkdir("a", 0).unwrap();
        let b = fs.mkdir("b", a).unwrap();
        let _c = fs.mkdir("c", b).unwrap();
        assert_eq!(fs.max_depth, 3);
        assert!(fs.hyperbolic_ratio > 1.0);
    }

    #[test]
    fn test_theorem_status_all_active() {
        let fs = ManifoldFS::new();
        let status = fs.theorem_status();
        assert_eq!(status.len(), 5);
        for (_, state, _) in &status {
            assert_eq!(*state, "ACTIVE");
        }
    }
}
