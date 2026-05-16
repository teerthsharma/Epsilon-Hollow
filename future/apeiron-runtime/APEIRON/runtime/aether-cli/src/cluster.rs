// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

use serde::{Deserialize, Serialize};
use aho_corasick::AhoCorasick;

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct SeedData {
    pub meta: SeedMeta,
    pub manifold: Vec<ClusterData>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct SeedMeta {
    pub version: String,
    pub cluster_count: usize,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ClusterData {
    pub cluster_id: u64,
    pub archetype: String,
    pub centroid_hint: Vec<f64>,
    pub inputs: Vec<String>,
    pub responses: Vec<String>,
}

pub struct ClusterManager {
    clusters: Vec<ClusterData>,
    automaton: Option<AhoCorasick>,
    pattern_map: Vec<usize>, // pattern_id -> cluster_index
    pending_updates: Vec<usize>, // Indices of clusters not yet in automaton
    next_id: u64,
}

impl ClusterManager {
    pub fn new(clusters: Vec<ClusterData>) -> Self {
        let next_id = clusters.iter().map(|c| c.cluster_id).max().unwrap_or(0) + 1;
        let mut manager = Self {
            clusters,
            automaton: None,
            pattern_map: Vec::new(),
            pending_updates: Vec::new(),
            next_id,
        };
        manager.rebuild_index();
        manager
    }

    fn rebuild_index(&mut self) {
        self.pattern_map.clear();
        self.pending_updates.clear();

        // 1. Populate pattern_map first
        // We must ensure the order matches exactly what we'll feed to AhoCorasick
        let mut count = 0;
        for (i, cluster) in self.clusters.iter().enumerate() {
            for _ in &cluster.inputs {
                self.pattern_map.push(i);
                count += 1;
            }
        }

        if count > 0 {
            // 2. Stream patterns directly to AhoCorasick without allocation
            // Optimization: Avoids creating a temporary Vec<String> and cloning all strings
            let pattern_iter = self.clusters.iter().flat_map(|c| c.inputs.iter());

            // We assume patterns in ClusterData are already lowercased.
            match AhoCorasick::new(pattern_iter) {
                Ok(ac) => self.automaton = Some(ac),
                Err(e) => {
                    eprintln!("[ClusterManager] Failed to build Aho-Corasick automaton: {}", e);
                    self.automaton = None;
                }
            }
        } else {
            self.automaton = None;
        }
    }

    pub fn find_match<'a>(&'a self, input: &str) -> Option<&'a ClusterData> {
        let input_lower = input.to_lowercase();
        let mut best_match: Option<&'a ClusterData> = None;

        // 1. Search Automaton (Optimized)
        if let Some(ac) = &self.automaton {
            let mut best_cluster_idx = None;

            for mat in ac.find_iter(&input_lower) {
                let pattern_id = mat.pattern().as_usize();
                if pattern_id < self.pattern_map.len() {
                    let cluster_idx = self.pattern_map[pattern_id];

                    match best_cluster_idx {
                        None => best_cluster_idx = Some(cluster_idx),
                        Some(best) => {
                            if cluster_idx < best {
                                best_cluster_idx = Some(cluster_idx);
                            }
                        }
                    }

                    // Optimization: If we found the first cluster (index 0), we can't do better.
                    if best_cluster_idx == Some(0) {
                        return Some(&self.clusters[0]);
                    }
                }
            }

            if let Some(idx) = best_cluster_idx {
                best_match = Some(&self.clusters[idx]);
            }
        } else {
             // Fallback if automaton is missing (e.g. failed to build or empty)
             // This only happens if we have no clusters or build failed.
             // If build failed, we should probably check everything linearly.
             // But if self.automaton is None, self.clusters might still be non-empty (if build failed).
             // However, strictly speaking, if automaton is None, we assume empty or broken.
             // But we should still check pending?
             // If automaton is None, `rebuild_index` was called. `pending` was cleared.
             // So if `automaton` is None, `pending` is empty unless we added more since then.
        }

        // If we found a match in automaton, return it.
        // Reason: Automaton contains clusters [0..N]. Pending contains [N+1..].
        // Match in 0..N is always preferred over match in N+1.. because we prioritize earlier clusters.
        if best_match.is_some() {
            return best_match;
        }

        // 2. Search Pending Updates (Linear Fallback)
        // Check new clusters that haven't been indexed yet.
        for &idx in &self.pending_updates {
             if idx < self.clusters.len() {
                 let cluster = &self.clusters[idx];
                 for pattern in &cluster.inputs {
                     if input_lower.contains(pattern) {
                         // Found a match in pending. Since pending is ordered by insertion,
                         // and we iterate in order, the first match is the best among pending.
                         return Some(cluster);
                     }
                 }
             }
        }

        // 3. Global Fallback
        // If automaton is missing (e.g., build failed) but we have clusters, scan everything.
        // We check if pending coverage is insufficient (i.e. we have clusters not in pending).
        if self.automaton.is_none() && !self.clusters.is_empty() && self.pending_updates.len() < self.clusters.len() {
             for cluster in &self.clusters {
                for pattern in &cluster.inputs {
                    if input_lower.contains(pattern) {
                        return Some(cluster);
                    }
                }
            }
        }

        None
    }

    pub fn clusters(&self) -> &[ClusterData] {
        &self.clusters
    }

    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    pub fn add_cluster(&mut self, cluster: ClusterData) {
        if cluster.cluster_id >= self.next_id {
            self.next_id = cluster.cluster_id + 1;
        }
        self.clusters.push(cluster);

        // Optimization: Batch updates instead of rebuilding every time.
        self.pending_updates.push(self.clusters.len() - 1);

        // Rebuild only if we have accumulated enough changes
        if self.pending_updates.len() >= 50 {
             self.rebuild_index();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_id_updates() {
        let mut clusters = Vec::new();
        clusters.push(ClusterData {
            cluster_id: 10,
            archetype: "".to_string(),
            centroid_hint: vec![],
            inputs: vec![],
            responses: vec![],
        });

        let mut manager = ClusterManager::new(clusters);
        assert_eq!(manager.next_id(), 11);

        manager.add_cluster(ClusterData {
            cluster_id: 20,
            archetype: "".to_string(),
            centroid_hint: vec![],
            inputs: vec![],
            responses: vec![],
        });

        assert_eq!(manager.next_id(), 21);

        // Test normal increment scenario (what main.rs does)
        let new_id = manager.next_id();
        manager.add_cluster(ClusterData {
            cluster_id: new_id,
            archetype: "".to_string(),
            centroid_hint: vec![],
            inputs: vec![],
            responses: vec![],
        });
        assert_eq!(manager.next_id(), new_id + 1);
    }

    #[test]
    fn test_find_match_mixed() {
        let mut clusters = Vec::new();
        clusters.push(ClusterData {
            cluster_id: 1,
            archetype: "fruit".to_string(),
            centroid_hint: vec![],
            inputs: vec!["apple".to_string()],
            responses: vec![],
        });

        let mut manager = ClusterManager::new(clusters);

        // Initial state: "apple" in automaton
        assert!(manager.automaton.is_some());
        assert!(manager.pending_updates.is_empty());
        assert_eq!(manager.find_match("I like apple pie").unwrap().cluster_id, 1);

        // Add "banana" (pending)
        manager.add_cluster(ClusterData {
            cluster_id: 2,
            archetype: "fruit".to_string(),
            centroid_hint: vec![],
            inputs: vec!["banana".to_string()],
            responses: vec![],
        });

        assert_eq!(manager.pending_updates.len(), 1);
        // Verify we can find pending
        assert_eq!(manager.find_match("Banana split").unwrap().cluster_id, 2);
        // Verify we can still find automaton
        assert_eq!(manager.find_match("Apple cider").unwrap().cluster_id, 1);

        // Force rebuild by adding 50 clusters
        for i in 0..55 {
            manager.add_cluster(ClusterData {
                cluster_id: 100 + i,
                archetype: "filler".to_string(),
                centroid_hint: vec![],
                inputs: vec![format!("filler_{}", i)],
                responses: vec![],
            });
        }

        // Now pending should be small (rebuilt at 50, now has 55-50=5 pending + maybe banana included in rebuild)
        // Rebuild happens when pending >= 50.
        // We added 1 (banana). pending=1.
        // Loop 0..49: adds 49 items. pending=50.
        // Loop 49 (i=49): adds 1 item. pending=51? No, logic is:
        // push; if len >= 50 { rebuild }
        // So at i=48 (total 50 in pending including banana), we rebuild. pending becomes 0.
        // Then remaining loop items are added to pending.

        // "banana" should now be in automaton.
        assert_eq!(manager.find_match("Banana bread").unwrap().cluster_id, 2);

        // Check new pending items
        assert!(manager.find_match("This is filler_54").is_some());
    }
}
