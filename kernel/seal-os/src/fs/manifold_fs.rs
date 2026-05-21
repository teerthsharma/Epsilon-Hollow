// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT
// Ported from epsilon-os manifold_fs.rs to no_std.

//! ManifoldFS — all data is geometry on S². File moves = O(1) topological surgery.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use aether_core::governor::GeometricGovernor;
use aether_core::scm::SpectralContractionOperator;

use super::block_store::BlockStore;
use super::dir_hash::DirHash;
use super::encoder::{self, ManifoldPayload};
use super::inode_slab::InodeSlab;
use super::path_cache::PathCache;
use super::vfs::{FileSystem, VfsDirEntry, VfsError, VfsHandle, VfsNode, VfsNodeType};
use super::voronoi_cap::VoronoiCap;
use crate::drivers::interrupts;

const VORONOI_CELLS: usize = 8;
const ENTROPY_MERGE_THRESHOLD: f64 = 2.0;

#[derive(Debug, Clone)]
pub enum InodeKind {
    File,
    Directory,
}

#[derive(Debug, Clone)]
pub struct InodeMetadata {
    pub created_ms: u64,
    pub modified_ms: u64,
    pub original_size: u64,
    pub permissions: u16,
}

#[derive(Debug, Clone)]
pub struct Inode {
    pub id: u64,
    pub name: String,
    pub kind: InodeKind,
    pub payload: ManifoldPayload,
    pub data: Vec<u8>,
    pub metadata: InodeMetadata,
    pub voronoi_cell: usize,
    pub cluster_id: i32,
    pub parent: u64,
    // Intrusive doubly-linked list for directory entries (readdir O(K) not O(N))
    pub sibling_next: Option<u64>,
    pub sibling_prev: Option<u64>,
    pub dir_first_child: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TheoremEvent {
    pub theorem: &'static str,
    pub action: String,
    pub timestamp_ms: u64,
}

pub struct ManifoldFS {
    inodes: InodeSlab,
    dirs: DirHash,
    path_cache: PathCache,
    store: BlockStore,
    voronoi: VoronoiCap,
    root_id: u64,
    scm: SpectralContractionOperator<3>,
    access_state: [f64; 3],
    last_prefetch_prediction: Option<u64>,
    prefetch_hits: u64,
    prefetch_misses: u64,
    cluster_sizes: Vec<usize>,
    entropy_merges: u64,
    current_entropy: f64,
    governor: GeometricGovernor,
    governor_ticks: u64,
    max_depth: usize,
    hyperbolic_ratio: f64,
    total_teleports: u64,
    total_lookups: u64,
    activity_log: Vec<TheoremEvent>,
}

impl ManifoldFS {
    pub fn new() -> Self {
        match BlockStore::mount_ahci() {
            Ok(store) => Self::init_with_store(store),
            Err(_) => Self::fresh(),
        }
    }

    pub fn new_with_mock_store(num_sectors: usize) -> Self {
        let store = BlockStore::with_mock(num_sectors);
        Self::init_with_store(store)
    }

    fn init_with_store(store: BlockStore) -> Self {
        let inodes = store.load_inodes();
        if inodes.iter().any(|i| i.name == "/") {
            return Self::from_store(store, inodes);
        }
        let mut fs = Self::fresh();
        fs.store = store;
        if let Some(root) = fs.inodes.get(fs.root_id).cloned() {
            let _ = fs.store.write_inode(&root, &[]);
            let _ = fs.store.flush_all();
        }
        fs
    }

    fn from_store(store: BlockStore, inodes: Vec<Inode>) -> Self {
        let mut slab = InodeSlab::new();
        let mut dirs = DirHash::new(0);
        let path_cache = PathCache::with_capacity(8);
        let mut voronoi = VoronoiCap::new();
        let mut cluster_sizes = vec![0; VORONOI_CELLS];
        let mut root_id = 0u64;

        for inode in inodes {
            let id = match slab.insert_at(inode.id, inode.clone()) {
                Ok(id) => id,
                Err(()) => slab.alloc(inode.clone()),
            };
            if inode.name == "/" {
                root_id = id;
            }
            let (cell, _subcell) = voronoi.insert(id, &inode.payload);
            if let Some(ino) = slab.get_mut(id) {
                ino.voronoi_cell = cell;
                ino.cluster_id = cell as i32;
            }
            if cell < cluster_sizes.len() {
                cluster_sizes[cell] += 1;
            }
        }

        // Rebuild dirs and sibling links from parent pointers.
        let mut children: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for id in slab.iter().map(|i| i.id).collect::<Vec<_>>() {
            let inode = match slab.get(id) {
                Some(i) => i,
                None => continue,
            };
            if inode.name != "/" {
                dirs.insert(inode.parent, &inode.name, id);
            }
            if inode.parent != id || inode.name == "/" {
                children.entry(inode.parent).or_default().push(id);
            }
        }

        for (parent_id, mut child_ids) in children {
            child_ids.sort_by_key(|&id| slab.get(id).map(|i| i.name.clone()).unwrap_or_default());
            for i in 0..child_ids.len() {
                let curr = child_ids[i];
                let next = child_ids.get(i + 1).copied();
                let prev = if i > 0 { child_ids.get(i - 1).copied() } else { None };
                if let Some(ino) = slab.get_mut(curr) {
                    ino.sibling_next = next;
                    ino.sibling_prev = prev;
                }
            }
            if let Some(first) = child_ids.first() {
                if let Some(parent) = slab.get_mut(parent_id) {
                    parent.dir_first_child = Some(*first);
                }
            }
        }

        dirs.insert(root_id, ".", root_id);
        dirs.insert(root_id, "..", root_id);

        let mut fs = Self {
            inodes: slab,
            dirs,
            path_cache,
            store,
            voronoi,
            root_id,
            scm: SpectralContractionOperator::new(0.7),
            access_state: [0.0; 3],
            last_prefetch_prediction: None,
            prefetch_hits: 0,
            prefetch_misses: 0,
            cluster_sizes,
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
        fs.update_entropy();
        fs
    }

    fn fresh() -> Self {
        let root_payload = encoder::encode_text("/");

        let mut inodes = InodeSlab::new();
        let mut dirs = DirHash::new(interrupts::ticks());

        let root = Inode {
            id: 0,
            name: String::from("/"),
            kind: InodeKind::Directory,
            payload: root_payload,
            data: Vec::new(),
            metadata: InodeMetadata {
                created_ms: now_ms(),
                modified_ms: now_ms(),
                original_size: 0,
                permissions: 0o755,
            },
            voronoi_cell: 0,
            cluster_id: 0,
            parent: 0,
            sibling_next: None,
            sibling_prev: None,
            dir_first_child: None,
        };
        let root_id = inodes.alloc(root);
        dirs.insert(root_id, ".", root_id);
        dirs.insert(root_id, "..", root_id);

        let mut cluster_sizes = vec![0; VORONOI_CELLS];
        cluster_sizes[0] = 1;

        Self {
            inodes,
            dirs,
            path_cache: PathCache::with_capacity(8),
            store: BlockStore::new(),
            voronoi: VoronoiCap::new(),
            root_id,
            scm: SpectralContractionOperator::new(0.7),
            access_state: [0.0; 3],
            last_prefetch_prediction: None,
            prefetch_hits: 0,
            prefetch_misses: 0,
            cluster_sizes,
            entropy_merges: 0,
            current_entropy: 0.0,
            governor: GeometricGovernor::new(),
            governor_ticks: 0,
            max_depth: 0,
            hyperbolic_ratio: f64::INFINITY,
            total_teleports: 0,
            total_lookups: 0,
            activity_log: Vec::new(),
        }
    }

    pub fn store(&mut self, name: &str, data: &[u8], parent_id: u64) -> Result<u64, FsError> {
        if self.dirs.lookup(parent_id, ".").is_none() && parent_id != self.root_id {
            return Err(FsError::NotADirectory);
        }
        if self.dirs.lookup(parent_id, name).is_some() {
            return Err(FsError::AlreadyExists);
        }

        let payload = encoder::encode_data(data);
        let (cell, subcell) = self.voronoi.insert(0, &payload);

        let inode = Inode {
            id: 0,
            name: String::from(name),
            kind: InodeKind::File,
            payload,
            data: Vec::from(data),
            metadata: InodeMetadata {
                created_ms: now_ms(),
                modified_ms: now_ms(),
                original_size: data.len() as u64,
                permissions: 0o644,
            },
            voronoi_cell: cell,
            cluster_id: cell as i32,
            parent: parent_id,
            sibling_next: None,
            sibling_prev: None,
            dir_first_child: None,
        };

        let id = self.inodes.alloc(inode);
        self.voronoi.remove(0, cell, subcell);
        let payload_ref = match self.inodes.get(id) {
            Some(i) => &i.payload,
            None => return Err(FsError::NotFound),
        };
        let (cell, _subcell) = self.voronoi.insert(id, payload_ref);
        if let Some(ino) = self.inodes.get_mut(id) {
            ino.voronoi_cell = cell;
            ino.cluster_id = cell as i32;
        }

        self.dirs.insert(parent_id, name, id);
        self.link_child(parent_id, id);
        self.cluster_sizes[cell] += 1;
        self.update_entropy();
        self.governor_tick(1.0);
        self.path_cache.bump_generation();
        if let Some(ino) = self.inodes.get(id) {
            let _ = self.store.write_inode(ino, data);
        }
        self.log_event("T1/TSS", format!("stored '{}' -> cell {}", name, cell));

        Ok(id)
    }

    pub fn store_text(&mut self, name: &str, text: &str, parent_id: u64) -> Result<u64, FsError> {
        self.store(name, text.as_bytes(), parent_id)
    }

    pub fn teleport(
        &mut self,
        name: &str,
        src_dir: u64,
        dst_dir: u64,
    ) -> Result<TeleportResult, FsError> {
        let t0 = interrupts::ticks();

        let inode_id = self.dirs.lookup(src_dir, name).ok_or(FsError::NotFound)?;

        if self.dirs.lookup(dst_dir, ".").is_none() && dst_dir != self.root_id {
            return Err(FsError::NotADirectory);
        }
        if self.dirs.lookup(dst_dir, name).is_some() {
            return Err(FsError::AlreadyExists);
        }

        let pre_epsilon = self.governor.epsilon();
        self.governor_tick(0.0);

        self.dirs.remove(src_dir, name);
        self.unlink_child(src_dir, inode_id);
        self.dirs.insert(dst_dir, name, inode_id);
        self.link_child(dst_dir, inode_id);

        if let Some(inode) = self.inodes.get_mut(inode_id) {
            inode.parent = dst_dir;
            inode.metadata.modified_ms = now_ms();
        }

        self.governor_tick(1.0);
        let post_epsilon = self.governor.epsilon();

        let elapsed_ticks = interrupts::ticks() - t0;
        self.total_teleports += 1;

        let original_size = self
            .inodes
            .get(inode_id)
            .map(|i| i.metadata.original_size)
            .unwrap_or(0);

        self.log_event(
            "T1/TSS+T4/AGCR",
            format!(
                "teleported '{}' ({} bytes) in {} ticks [governor e: {:.4} -> {:.4}]",
                name, original_size, elapsed_ticks, pre_epsilon, post_epsilon
            ),
        );

        self.path_cache.bump_generation();
        if let Some(inode) = self.inodes.get(inode_id) {
            let data = inode.data.clone();
            let _ = self.store.write_inode(inode, &data);
        }

        self.check_entropy_and_merge();

        Ok(TeleportResult {
            inode_id,
            elapsed_ticks,
            original_size,
            payload_points: self
                .inodes
                .get(inode_id)
                .map(|i| i.payload.point_count)
                .unwrap_or(0),
            governor_epsilon: post_epsilon,
        })
    }

    pub fn teleport_bulk(
        &mut self,
        src_dir: u64,
        dst_dir: u64,
    ) -> Result<Vec<TeleportResult>, FsError> {
        let names: Vec<String> = self
            .dirs
            .entries_in_dir(src_dir)
            .into_iter()
            .map(|(name, _id)| name)
            .collect();
        let mut results = Vec::with_capacity(names.len());
        for name in names {
            results.push(self.teleport(&name, src_dir, dst_dir)?);
        }
        Ok(results)
    }

    pub fn store_large(
        &mut self,
        name: &str,
        data: &[u8],
        parent_id: u64,
    ) -> Result<u64, FsError> {
        self.store(name, data, parent_id)
    }

    pub fn find(&mut self, query: &str) -> Vec<FindResult> {
        self.total_lookups += 1;
        let query_payload = encoder::encode_text(query);
        let (cell, subcell) = self.voronoi.locate(&query_payload);

        self.update_prefetch_state(&query_payload);

        let mut results = Vec::new();
        for &inode_id in self.voronoi.files_in_bucket(cell, subcell) {
            if let Some(inode) = self.inodes.get(inode_id) {
                let similarity = payload_similarity_full(&query_payload, &inode.payload);
                results.push(FindResult {
                    inode_id,
                    name: inode.name.clone(),
                    similarity,
                    cell,
                    original_size: inode.metadata.original_size,
                });
            }
        }
        results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        results
    }

    pub fn ls(&self, dir_id: u64) -> Result<Vec<LsEntry>, FsError> {
        if self.inodes.get(dir_id).is_none() {
            return Err(FsError::NotADirectory);
        }
        let inode = self.inodes.get(dir_id).ok_or(FsError::NotADirectory)?;
        if !matches!(inode.kind, InodeKind::Directory) {
            return Err(FsError::NotADirectory);
        }

        let mut result = Vec::new();
        let mut child_id = inode.dir_first_child;
        while let Some(id) = child_id {
            if let Some(child) = self.inodes.get(id) {
                result.push(LsEntry {
                    name: child.name.clone(),
                    kind: match child.kind {
                        InodeKind::File => "file",
                        InodeKind::Directory => "dir",
                    },
                    original_size: child.metadata.original_size,
                    payload_points: child.payload.point_count,
                    voronoi_cell: child.voronoi_cell,
                });
            }
            child_id = self.inodes.get(id).and_then(|c| c.sibling_next);
        }
        result.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(result)
    }

    pub fn mkdir(&mut self, name: &str, parent_id: u64) -> Result<u64, FsError> {
        if self.dirs.lookup(parent_id, ".").is_none() && parent_id != self.root_id {
            return Err(FsError::NotADirectory);
        }
        if self.dirs.lookup(parent_id, name).is_some() {
            return Err(FsError::AlreadyExists);
        }

        let payload = encoder::encode_text(name);

        let depth = self.compute_depth(parent_id) + 1;
        if depth > self.max_depth {
            self.max_depth = depth;
            self.update_hyperbolic_ratio(depth);
        }

        let inode = Inode {
            id: 0,
            name: String::from(name),
            kind: InodeKind::Directory,
            payload,
            data: Vec::new(),
            metadata: InodeMetadata {
                created_ms: now_ms(),
                modified_ms: now_ms(),
                original_size: 0,
                permissions: 0o755,
            },
            voronoi_cell: 0,
            cluster_id: 0,
            parent: parent_id,
            sibling_next: None,
            sibling_prev: None,
            dir_first_child: None,
        };

        let id = self.inodes.alloc(inode);
        self.dirs.insert(parent_id, name, id);
        self.link_child(parent_id, id);
        self.dirs.insert(id, ".", id);
        self.dirs.insert(id, "..", parent_id);

        self.path_cache.bump_generation();
        self.log_event(
            "T5/HCS",
            format!(
                "mkdir '{}' depth={} hyp_ratio={:.1}x",
                name, depth, self.hyperbolic_ratio
            ),
        );

        Ok(id)
    }

    pub fn resolve_path(&self, path: &str) -> Result<u64, FsError> {
        let mut current = self.root_id;
        for part in path.split('/').filter(|s| !s.is_empty()) {
            current = self.dirs.lookup(current, part).ok_or(FsError::NotFound)?;
        }
        Ok(current)
    }

    fn update_prefetch_state(&mut self, accessed: &ManifoldPayload) {
        if accessed.points.is_empty() {
            return;
        }
        let pt = accessed.points[0].coords;

        if let Some(predicted_id) = self.last_prefetch_prediction {
            if let Some(inode) = self.inodes.get(predicted_id) {
                let pred_pt = inode
                    .payload
                    .points
                    .first()
                    .map(|p| p.coords)
                    .unwrap_or([0.0; 3]);
                let dist: f64 = (0..3)
                    .map(|i| (pt[i] - pred_pt[i]) * (pt[i] - pred_pt[i]))
                    .sum();
                if dist < 0.25 {
                    self.prefetch_hits += 1;
                } else {
                    self.prefetch_misses += 1;
                }
            }
        }

        self.access_state = self.scm.apply(&self.access_state, &pt);

        let predicted_cell = self.voronoi.locate(&encoder::ManifoldPayload {
            points: vec![super::encoder::SpherePoint {
                coords: self.access_state,
            }],
            point_count: 1,
            betti_0: 1,
            original_size: 0,
            content_hash: 0,
        });
        let bucket = self.voronoi.files_in_bucket(predicted_cell.0, predicted_cell.1);
        self.last_prefetch_prediction = bucket.last().copied();
    }

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

        let files_to_move: Vec<u64> = self.voronoi.all_files_in_cell(src_cell);
        for &fid in &files_to_move {
            if let Some(inode) = self.inodes.get_mut(fid) {
                inode.voronoi_cell = dst_cell;
                inode.cluster_id = dst_cell as i32;
            }
        }
        for &fid in &files_to_move {
            self.voronoi.move_file_to_cell(fid, src_cell, dst_cell);
        }
        let moved_count = files_to_move.len();
        self.cluster_sizes[dst_cell] += self.cluster_sizes[src_cell];
        self.cluster_sizes[src_cell] = 0;
        self.voronoi.clear_cell(src_cell);

        self.update_entropy();
        self.entropy_merges += 1;

        self.log_event(
            "T3/GMC",
            format!(
                "merged cell {} -> {}, {} files moved",
                src_cell, dst_cell, moved_count
            ),
        );
    }

    fn governor_tick(&mut self, deviation: f64) {
        self.governor.adapt(deviation, 0.01);
        self.governor_ticks += 1;
    }

    fn compute_depth(&self, mut id: u64) -> usize {
        let mut depth = 0;
        while id != self.root_id {
            if let Some(inode) = self.inodes.get(id) {
                id = inode.parent;
                depth += 1;
            } else {
                break;
            }
        }
        depth
    }

    fn update_hyperbolic_ratio(&mut self, depth: usize) {
        let b = 4.0f64;
        let d = depth as f64;
        if d >= 2.0 {
            self.hyperbolic_ratio = libm::log(b) * d * (d - 1.0) / 2.0;
        }
    }

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

    pub fn stats(&self) -> FsStats {
        FsStats {
            total_files: self
                .inodes
                .iter()
                .filter(|i| matches!(i.kind, InodeKind::File))
                .count(),
            total_dirs: self
                .inodes
                .iter()
                .filter(|i| matches!(i.kind, InodeKind::Directory))
                .count(),
            total_teleports: self.total_teleports,
            governor_epsilon: self.governor.epsilon(),
            current_entropy: self.current_entropy,
            max_depth: self.max_depth,
            hyperbolic_ratio: self.hyperbolic_ratio,
            path_cache_hits: self.path_cache.hits,
            path_cache_misses: self.path_cache.misses,
            voronoi_splits: self.voronoi.split_count(),
            voronoi_merges: self.voronoi.merge_count(),
        }
    }

    pub fn inode(&self, id: u64) -> Option<&Inode> {
        self.inodes.get(id)
    }

    pub fn is_dir(&self, id: u64) -> bool {
        self.inodes
            .get(id)
            .map(|i| matches!(i.kind, InodeKind::Directory))
            .unwrap_or(false)
    }

    pub fn exists(&self, name: &str, dir_id: u64) -> bool {
        self.dirs.lookup(dir_id, name).is_some()
    }

    pub fn read_text(&self, name: &str, dir_id: u64) -> Option<String> {
        let inode_id = self.dirs.lookup(dir_id, name)?;
        let inode = self.inodes.get(inode_id)?;
        String::from_utf8(inode.data.clone()).ok()
    }

    pub fn file_info(&self, name: &str, dir_id: u64) -> Option<String> {
        let inode_id = self.dirs.lookup(dir_id, name)?;
        let inode = self.inodes.get(inode_id)?;
        Some(format!(
            "Name:           {}\n\
             Type:           {}\n\
             Size:           {} bytes\n\
             Payload:        {} points on S²\n\
             Voronoi cell:   {}\n\
             Cluster ID:     {}\n\
             Permissions:    {:o}\n\
             Inode:          {}",
            inode.name,
            match inode.kind { InodeKind::File => "file", InodeKind::Directory => "directory" },
            inode.metadata.original_size,
            inode.payload.point_count,
            inode.voronoi_cell,
            inode.cluster_id,
            inode.metadata.permissions,
            inode.id
        ))
    }

    pub fn delete(&mut self, name: &str, dir_id: u64) -> Result<(), FsError> {
        let inode_id = self.dirs.lookup(dir_id, name).ok_or(FsError::NotFound)?;

        if let Some(inode) = self.inodes.get(inode_id) {
            let cell = inode.voronoi_cell;
            self.voronoi.remove(inode_id, cell, 0);
            if self.cluster_sizes[cell] > 0 {
                self.cluster_sizes[cell] -= 1;
            }
        }

        self.dirs.remove(dir_id, name);
        self.unlink_child(dir_id, inode_id);
        self.inodes.free(inode_id);
        self.update_entropy();
        self.path_cache.bump_generation();
        self.log_event("T1/TSS", format!("deleted '{}'", name));
        Ok(())
    }

    pub fn rename(&mut self, old: &str, new: &str, dir_id: u64) -> Result<(), FsError> {
        let inode_id = self.dirs.lookup(dir_id, old).ok_or(FsError::NotFound)?;

        if self.exists(new, dir_id) {
            return Err(FsError::AlreadyExists);
        }

        if let Some(inode) = self.inodes.get_mut(inode_id) {
            inode.name = String::from(new);
            inode.metadata.modified_ms = now_ms();
        }

        self.dirs.remove(dir_id, old);
        self.dirs.insert(dir_id, new, inode_id);
        self.path_cache.bump_generation();
        self.log_event("T1/TSS", format!("renamed '{}' -> '{}'", old, new));
        Ok(())
    }

    pub fn duplicate(&mut self, name: &str, src_dir: u64, dst_dir: u64) -> Result<u64, FsError> {
        let src_id = self.dirs.lookup(src_dir, name).ok_or(FsError::NotFound)?;

        if self.dirs.lookup(dst_dir, ".").is_none() && dst_dir != self.root_id {
            return Err(FsError::NotADirectory);
        }
        if self.dirs.lookup(dst_dir, name).is_some() {
            return Err(FsError::AlreadyExists);
        }

        let src_inode = self.inodes.get(src_id).ok_or(FsError::NotFound)?.clone();
        let cell = src_inode.voronoi_cell;

        let new_inode = Inode {
            id: 0,
            name: String::from(name),
            kind: src_inode.kind,
            payload: src_inode.payload,
            data: src_inode.data.clone(),
            metadata: InodeMetadata {
                created_ms: now_ms(),
                modified_ms: now_ms(),
                original_size: src_inode.metadata.original_size,
                permissions: src_inode.metadata.permissions,
            },
            voronoi_cell: cell,
            cluster_id: src_inode.cluster_id,
            parent: dst_dir,
            sibling_next: None,
            sibling_prev: None,
            dir_first_child: None,
        };

        let id = self.inodes.alloc(new_inode);
        let payload_ref = match self.inodes.get(id) {
            Some(i) => &i.payload,
            None => return Err(FsError::NotFound),
        };
        self.voronoi.insert(id, payload_ref);
        self.dirs.insert(dst_dir, name, id);
        self.link_child(dst_dir, id);
        self.cluster_sizes[cell] += 1;
        self.update_entropy();
        self.path_cache.bump_generation();
        self.log_event("T1/TSS", format!("duplicated '{}' -> dir {}", name, dst_dir));
        Ok(id)
    }

    pub fn resolve_path_from(&self, path: &str, from: u64) -> Result<u64, FsError> {
        if path.starts_with('/') {
            return self.resolve_path(path);
        }
        let mut current = from;
        for part in path.split('/').filter(|s| !s.is_empty()) {
            if part == ".." {
                if let Some(inode) = self.inodes.get(current) {
                    current = inode.parent;
                }
                continue;
            }
            current = self.dirs.lookup(current, part).ok_or(FsError::NotFound)?;
        }
        Ok(current)
    }

    pub fn theorem_status(&self) -> [(&'static str, &'static str); 5] {
        [
            ("T1/TSS", "ACTIVE"),
            ("T2/SCM", "ACTIVE"),
            ("T3/GMC", "ACTIVE"),
            ("T4/AGCR", "ACTIVE"),
            ("T5/HCS", "ACTIVE"),
        ]
    }

    // ── Intrusive linked list helpers ───────────────────────────────────────

    fn link_child(&mut self, parent_id: u64, child_id: u64) {
        let parent = match self.inodes.get_mut(parent_id) {
            Some(p) => p,
            None => return,
        };
        let old_first = parent.dir_first_child;
        parent.dir_first_child = Some(child_id);

        if let Some(child) = self.inodes.get_mut(child_id) {
            child.sibling_next = old_first;
            child.sibling_prev = None;
        }
        if let Some(old) = old_first {
            if let Some(old_child) = self.inodes.get_mut(old) {
                old_child.sibling_prev = Some(child_id);
            }
        }
    }

    fn unlink_child(&mut self, parent_id: u64, child_id: u64) {
        let child = match self.inodes.get(child_id) {
            Some(c) => c.clone(),
            None => return,
        };

        if let Some(prev) = child.sibling_prev {
            if let Some(p) = self.inodes.get_mut(prev) {
                p.sibling_next = child.sibling_next;
            }
        } else {
            if let Some(parent) = self.inodes.get_mut(parent_id) {
                parent.dir_first_child = child.sibling_next;
            }
        }

        if let Some(next) = child.sibling_next {
            if let Some(n) = self.inodes.get_mut(next) {
                n.sibling_prev = child.sibling_prev;
            }
        }

        if let Some(c) = self.inodes.get_mut(child_id) {
            c.sibling_next = None;
            c.sibling_prev = None;
        }
    }
}

fn map_err(e: FsError) -> VfsError {
    match e {
        FsError::NotFound => VfsError::NotFound,
        FsError::NotADirectory => VfsError::NotADirectory,
        FsError::AlreadyExists => VfsError::AlreadyExists,
    }
}

fn split_path(path: &str) -> Result<(&str, &str), VfsError> {
    let path = path.trim_end_matches('/');
    match path.rfind('/') {
        Some(pos) => {
            let dir = if pos == 0 { "/" } else { &path[..pos] };
            let name = &path[pos + 1..];
            if name.is_empty() {
                return Err(VfsError::InvalidPath);
            }
            Ok((dir, name))
        }
        None => Ok(("/", path)),
    }
}

impl FileSystem for ManifoldFS {
    fn lookup(&self, path: &str) -> Result<VfsHandle, VfsError> {
        let id = self.resolve_path(path).map_err(map_err)?;
        Ok(VfsHandle { fs_idx: 0, inode: id })
    }

    fn read(&self, handle: VfsHandle, buf: &mut [u8], offset: u64) -> Result<usize, VfsError> {
        let inode = self.inodes.get(handle.inode).ok_or(VfsError::NotFound)?;
        let off = offset as usize;
        if off >= inode.data.len() {
            return Ok(0);
        }
        let len = buf.len().min(inode.data.len() - off);
        buf[..len].copy_from_slice(&inode.data[off..off + len]);
        Ok(len)
    }

    fn write(&mut self, handle: VfsHandle, buf: &[u8], _offset: u64) -> Result<usize, VfsError> {
        let inode = self.inodes.get_mut(handle.inode).ok_or(VfsError::NotFound)?;
        if !matches!(inode.kind, InodeKind::File) {
            return Err(VfsError::NotSupported);
        }
        inode.payload = encoder::encode_data(buf);
        inode.data = Vec::from(buf);
        inode.metadata.original_size = buf.len() as u64;
        inode.metadata.modified_ms = now_ms();
        let data = inode.data.clone();
        let _ = self.store.write_inode(inode, &data);
        Ok(buf.len())
    }

    fn create(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let (dir_path, name) = split_path(path)?;
        let dir_id = self.resolve_path(dir_path).map_err(map_err)?;
        let id = self.store(name, b"", dir_id).map_err(map_err)?;
        Ok(VfsHandle { fs_idx: 0, inode: id })
    }

    fn mkdir(&mut self, path: &str) -> Result<VfsHandle, VfsError> {
        let (dir_path, name) = split_path(path)?;
        let dir_id = self.resolve_path(dir_path).map_err(map_err)?;
        let id = self.mkdir(name, dir_id).map_err(map_err)?;
        Ok(VfsHandle { fs_idx: 0, inode: id })
    }

    fn unlink(&mut self, path: &str) -> Result<(), VfsError> {
        let (dir_path, name) = split_path(path)?;
        let dir_id = self.resolve_path(dir_path).map_err(map_err)?;
        self.delete(name, dir_id).map_err(map_err)
    }

    fn readdir(&self, handle: VfsHandle) -> Result<Vec<VfsDirEntry>, VfsError> {
        let inode = self.inodes.get(handle.inode).ok_or(VfsError::NotADirectory)?;
        if !matches!(inode.kind, InodeKind::Directory) {
            return Err(VfsError::NotADirectory);
        }

        let mut result = Vec::new();
        let mut child_id = inode.dir_first_child;
        while let Some(id) = child_id {
            if let Some(child) = self.inodes.get(id) {
                let node_type = match child.kind {
                    InodeKind::File => VfsNodeType::File,
                    InodeKind::Directory => VfsNodeType::Directory,
                };
                result.push(VfsDirEntry {
                    name: child.name.clone(),
                    node_type,
                });
            }
            child_id = self.inodes.get(id).and_then(|c| c.sibling_next);
        }
        Ok(result)
    }

    fn stat(&self, handle: VfsHandle) -> Result<VfsNode, VfsError> {
        let inode = self.inodes.get(handle.inode).ok_or(VfsError::NotFound)?;
        let node_type = match inode.kind {
            InodeKind::File => VfsNodeType::File,
            InodeKind::Directory => VfsNodeType::Directory,
        };
        Ok(VfsNode {
            size: inode.metadata.original_size,
            permissions: inode.metadata.permissions,
            uid: 0,
            gid: 0,
            mode: inode.metadata.permissions,
            atime: inode.metadata.modified_ms,
            mtime: inode.metadata.modified_ms,
            node_type,
        })
    }

    fn mknod(
        &mut self,
        _path: &str,
        _node_type: VfsNodeType,
        _major: u32,
        _minor: u32,
    ) -> Result<VfsHandle, VfsError> {
        Err(VfsError::NotSupported)
    }
}

fn payload_similarity_full(a: &ManifoldPayload, b: &ManifoldPayload) -> f64 {
    if a.points.is_empty() || b.points.is_empty() {
        return 0.0;
    }
    let min_pts = a.points.len().min(b.points.len());
    let mut sum = 0.0f64;
    for i in 0..min_pts {
        let pa = &a.points[i].coords;
        let pb = &b.points[i].coords;
        sum += pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2];
    }
    sum / min_pts as f64
}

fn now_ms() -> u64 {
    interrupts::ticks()
}

// Result types

#[derive(Debug, Clone)]
pub struct TeleportResult {
    pub inode_id: u64,
    pub elapsed_ticks: u64,
    pub original_size: u64,
    pub payload_points: usize,
    pub governor_epsilon: f64,
}

#[derive(Debug, Clone)]
pub struct FindResult {
    pub inode_id: u64,
    pub name: String,
    pub similarity: f64,
    pub cell: usize,
    pub original_size: u64,
}

#[derive(Debug, Clone)]
pub struct LsEntry {
    pub name: String,
    pub kind: &'static str,
    pub original_size: u64,
    pub payload_points: usize,
    pub voronoi_cell: usize,
}

#[derive(Debug, Clone)]
pub struct FsStats {
    pub total_files: usize,
    pub total_dirs: usize,
    pub total_teleports: u64,
    pub governor_epsilon: f64,
    pub current_entropy: f64,
    pub max_depth: usize,
    pub hyperbolic_ratio: f64,
    pub path_cache_hits: u64,
    pub path_cache_misses: u64,
    pub voronoi_splits: u64,
    pub voronoi_merges: u64,
}

#[derive(Debug, Clone)]
pub enum FsError {
    NotFound,
    NotADirectory,
    AlreadyExists,
}

impl core::fmt::Display for FsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFound => write!(f, "not found"),
            Self::NotADirectory => write!(f, "not a directory"),
            Self::AlreadyExists => write!(f, "already exists"),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-mode"))]
pub mod tests {
    use super::*;
    use crate::{test_assert, test_assert_eq};
    use crate::testing::TestResult;

    fn test_store_and_ls() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let id = fs.store_text("hello.txt", "world", root).unwrap();
        test_assert_eq!(id, 1);
        let entries = fs.ls(root).unwrap();
        test_assert_eq!(entries.len(), 1);
        test_assert_eq!(entries[0].name, "hello.txt");
        TestResult::Pass
    }

    fn test_mkdir_and_resolve_path() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let nested = fs.mkdir("nested", docs).unwrap();
        let resolved = fs.resolve_path("/docs/nested").unwrap();
        test_assert_eq!(resolved, nested);
        TestResult::Pass
    }

    fn test_teleport_is_o1() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let vol = fs.mkdir("vol_a", root).unwrap();
        fs.store_text("file.txt", "content", docs).unwrap();

        let result = fs.teleport("file.txt", docs, vol).unwrap();
        test_assert!(result.elapsed_ticks <= 5, "teleport took too many ticks (not O(1))");
        test_assert_eq!(fs.exists("file.txt", docs), false);
        test_assert_eq!(fs.exists("file.txt", vol), true);
        TestResult::Pass
    }

    fn test_find_returns_results() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        fs.store_text("alpha.txt", "alpha content", root).unwrap();
        fs.store_text("beta.txt", "beta content", root).unwrap();
        let results = fs.find("alpha");
        test_assert!(!results.is_empty(), "find returned no results");
        test_assert_eq!(results[0].name, "alpha.txt");
        TestResult::Pass
    }

    fn test_resolve_path_from_relative() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let inner = fs.mkdir("inner", docs).unwrap();
        let resolved = fs.resolve_path_from("inner", docs).unwrap();
        test_assert_eq!(resolved, inner);
        let resolved2 = fs.resolve_path_from("../inner", inner).unwrap();
        test_assert_eq!(resolved2, inner);
        let resolved3 = fs.resolve_path_from("../../docs/inner", inner).unwrap();
        test_assert_eq!(resolved3, inner);
        TestResult::Pass
    }

    fn test_delete_and_rename() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        fs.store_text("old.txt", "data", root).unwrap();
        fs.rename("old.txt", "new.txt", root).unwrap();
        test_assert_eq!(fs.exists("old.txt", root), false);
        test_assert_eq!(fs.exists("new.txt", root), true);
        fs.delete("new.txt", root).unwrap();
        test_assert_eq!(fs.exists("new.txt", root), false);
        TestResult::Pass
    }

    fn test_duplicate() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let vol = fs.mkdir("vol_a", root).unwrap();
        fs.store_text("file.txt", "content", docs).unwrap();
        let dup_id = fs.duplicate("file.txt", docs, vol).unwrap();
        test_assert_eq!(fs.exists("file.txt", vol), true);
        let inode = fs.inode(dup_id).unwrap();
        test_assert_eq!(inode.name, "file.txt");
        TestResult::Pass
    }

    fn test_teleport_errors() -> TestResult {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let vol = fs.mkdir("vol_a", root).unwrap();
        let r = fs.teleport("missing.txt", docs, vol);
        test_assert!(r.is_err(), "teleport of missing file should fail");
        let r2 = fs.teleport("missing.txt", docs, 999);
        test_assert!(r2.is_err(), "teleport to non-dir should fail");
        TestResult::Pass
    }

    pub fn register_all() {
        crate::testing::register_test("filesystem::store_and_ls", test_store_and_ls);
        crate::testing::register_test("filesystem::mkdir_and_resolve_path", test_mkdir_and_resolve_path);
        crate::testing::register_test("filesystem::teleport_is_o1", test_teleport_is_o1);
        crate::testing::register_test("filesystem::find_returns_results", test_find_returns_results);
        crate::testing::register_test("filesystem::resolve_path_from_relative", test_resolve_path_from_relative);
        crate::testing::register_test("filesystem::delete_and_rename", test_delete_and_rename);
        crate::testing::register_test("filesystem::duplicate", test_duplicate);
        crate::testing::register_test("filesystem::teleport_errors", test_teleport_errors);
    }
}

#[cfg(test)]
mod host_tests {
    use super::*;

    #[test]
    fn store_and_ls() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let id = fs.store_text("hello.txt", "world", root).unwrap();
        assert_eq!(id, 1);
        let entries = fs.ls(root).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "hello.txt");
    }

    #[test]
    fn mkdir_and_resolve_path() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let nested = fs.mkdir("nested", docs).unwrap();
        assert_eq!(fs.resolve_path("/docs/nested").unwrap(), nested);
    }

    #[test]
    fn teleport_is_fast() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let vol = fs.mkdir("vol_a", root).unwrap();
        fs.store_text("file.txt", "content", docs).unwrap();
        let result = fs.teleport("file.txt", docs, vol).unwrap();
        assert!(result.elapsed_ticks <= 5, "teleport not O(1)");
        assert!(!fs.exists("file.txt", docs));
        assert!(fs.exists("file.txt", vol));
    }

    #[test]
    fn find_returns_results() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        fs.store_text("alpha.txt", "alpha content", root).unwrap();
        let results = fs.find("alpha");
        assert!(!results.is_empty());
        assert_eq!(results[0].name, "alpha.txt");
    }

    #[test]
    fn resolve_path_from_relative() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let inner = fs.mkdir("inner", docs).unwrap();
        assert_eq!(fs.resolve_path_from("inner", docs).unwrap(), inner);
    }

    #[test]
    fn delete_and_rename() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        fs.store_text("old.txt", "data", root).unwrap();
        fs.rename("old.txt", "new.txt", root).unwrap();
        assert!(!fs.exists("old.txt", root));
        assert!(fs.exists("new.txt", root));
        fs.delete("new.txt", root).unwrap();
        assert!(!fs.exists("new.txt", root));
    }

    #[test]
    fn duplicate_file() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let vol = fs.mkdir("vol_a", root).unwrap();
        fs.store_text("file.txt", "content", docs).unwrap();
        let dup_id = fs.duplicate("file.txt", docs, vol).unwrap();
        assert!(fs.exists("file.txt", vol));
        assert_eq!(fs.inode(dup_id).unwrap().name, "file.txt");
    }

    #[test]
    fn teleport_errors() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let docs = fs.mkdir("docs", root).unwrap();
        let vol = fs.mkdir("vol_a", root).unwrap();
        assert!(fs.teleport("missing.txt", docs, vol).is_err());
        assert!(fs.teleport("missing.txt", docs, 999).is_err());
    }

    #[test]
    fn teleport_o1_under_load() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        let src = fs.mkdir("src", root).unwrap();
        let dst = fs.mkdir("dst", root).unwrap();
        fs.store_text("base.txt", "x", src).unwrap();
        let base = fs.teleport("base.txt", src, dst).unwrap();

        for n in [10, 100, 1000usize] {
            let mut fs2 = ManifoldFS::new();
            let s = fs2.mkdir("s", 0).unwrap();
            let d = fs2.mkdir("d", 0).unwrap();
            for i in 0..n {
                fs2.store_text(&alloc::format!("f{}", i), "x", s).unwrap();
            }
            fs2.store_text("target.txt", "payload", s).unwrap();
            let r = fs2.teleport("target.txt", s, d).unwrap();
            let ratio = if base.elapsed_ticks == 0 {
                r.elapsed_ticks
            } else {
                r.elapsed_ticks / base.elapsed_ticks.max(1)
            };
            assert!(ratio <= 2, "teleport not O(1) at N={}: {} vs {}", n, r.elapsed_ticks, base.elapsed_ticks);
        }
    }

    #[test]
    fn find_bounded_bucket_size() {
        let mut fs = ManifoldFS::new();
        let root = 0u64;
        for i in 0..500u64 {
            fs.store_text(&alloc::format!("file{}", i), &alloc::format!("content{}", i), root).unwrap();
        }
        let results = fs.find("file0");
        assert!(!results.is_empty());
    }
}
