// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

//! Topological World Model — Rust reference implementation.
//!
//! Episodic memory with cosine-similarity retrieval (O(n) linear scan),
//! Betti-0 tracking via Union-Find, spectral contraction dynamics,
//! theorem verification (T1-T10), and Minimax LLM bridge.
//!
//! The O(1) retrieval bound from T1 applies to [`aether_core::tss::SphericalVoronoiIndex`];
//! this module uses a simpler linear scan suitable for the reference CLI.

use std::collections::{HashMap, VecDeque};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use aether_core::scm::SpectralContractionOperator;

// ─── Memory Episode ──────────────────────────────────────────────────

/// A single memory episode stored in the topological manifold.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Episode {
    pub vector: Vec<f64>,
    pub text: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
    pub reinforcement_count: u32,
    pub cluster_id: i32,
}

// ─── Topological Memory ──────────────────────────────────────────────

/// Union-Find for Betti-0 computation.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    components: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            components: n,
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => {
                self.parent[ra] = rb;
            }
            std::cmp::Ordering::Greater => {
                self.parent[rb] = ra;
            }
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        self.components -= 1;
        true
    }
}

/// Topological manifold memory with L2-normalized vectors and Betti-0 clustering.
pub struct ManifoldMemory {
    pub episodes: VecDeque<Episode>,
    pub dim: usize,
    pub capacity: usize,
    chebyshev_k: f64,
    cluster_radius: f64,
}

impl ManifoldMemory {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            episodes: VecDeque::new(),
            dim,
            capacity,
            chebyshev_k: 2.0,
            cluster_radius: 0.5,
        }
    }

    /// Encode text into a deterministic f64 vector via character trigram hashing.
    pub fn encode_text(&self, text: &str) -> Vec<f64> {
        let mut vec = vec![0.0f64; self.dim];
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return self.l2_normalize(&vec);
        }
        // Trigram hash spread
        for i in 0..bytes.len().saturating_sub(2) {
            let h = ((bytes[i] as u64).wrapping_mul(31)
                ^ (bytes[i + 1] as u64).wrapping_mul(37)
                ^ (bytes[i + 2] as u64).wrapping_mul(41)) as usize;
            let idx = h % self.dim;
            let sign = if (h / self.dim) % 2 == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign;
        }
        // Also scatter unigrams for short strings
        for &b in bytes {
            let idx = (b as usize * 7 + 3) % self.dim;
            vec[idx] += 0.5;
        }
        self.l2_normalize(&vec)
    }

    fn l2_normalize(&self, v: &[f64]) -> Vec<f64> {
        let norm = libm::sqrt(v.iter().map(|x| x * x).sum::<f64>());
        if norm < 1e-12 {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
    }

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Store an episode. Returns its index.
    pub fn store(
        &mut self,
        vector: Vec<f64>,
        text: String,
        metadata: HashMap<String, String>,
    ) -> usize {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        let cluster_id = self.assign_cluster(&vector);
        let idx = self.episodes.len();

        if self.episodes.len() >= self.capacity {
            self.episodes.pop_front();
        }

        self.episodes.push_back(Episode {
            vector,
            text,
            metadata,
            timestamp: ts,
            reinforcement_count: 0,
            cluster_id,
        });
        idx
    }

    fn assign_cluster(&self, vector: &[f64]) -> i32 {
        if self.episodes.is_empty() {
            return 0;
        }
        let mut best_cluster = 0i32;
        let mut best_sim = f64::NEG_INFINITY;
        for ep in &self.episodes {
            let sim = Self::cosine_similarity(vector, &ep.vector);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = ep.cluster_id;
            }
        }
        if best_sim < self.cluster_radius {
            // New cluster
            self.episodes
                .iter()
                .map(|e| e.cluster_id)
                .max()
                .unwrap_or(-1)
                + 1
        } else {
            best_cluster
        }
    }

    /// Retrieve top-k episodes by cosine similarity (O(n) scan).
    pub fn retrieve(&self, query: &[f64], k: usize) -> Vec<(usize, f64, &Episode)> {
        let mut scored: Vec<(usize, f64)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| (i, Self::cosine_similarity(query, &ep.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(i, s)| (i, s, &self.episodes[i]))
            .collect()
    }

    /// Reinforce an episode (boost retrieval priority).
    #[allow(dead_code)]
    pub fn reinforce(&mut self, idx: usize) {
        if let Some(ep) = self.episodes.get_mut(idx) {
            ep.reinforcement_count += 1;
        }
    }

    /// Compute Betti-0 (connected components) using Union-Find on cosine threshold.
    pub fn betti_0(&self) -> usize {
        let n = self.episodes.len();
        if n == 0 {
            return 0;
        }
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let sim =
                    Self::cosine_similarity(&self.episodes[i].vector, &self.episodes[j].vector);
                if sim > self.cluster_radius {
                    uf.union(i, j);
                }
            }
        }
        uf.components
    }

    /// Memory statistics.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            size: self.episodes.len(),
            dim: self.dim,
            capacity: self.capacity,
            betti_0: self.betti_0(),
            chebyshev_k: self.chebyshev_k,
        }
    }

    pub fn reset(&mut self) {
        self.episodes.clear();
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryStats {
    pub size: usize,
    pub dim: usize,
    pub capacity: usize,
    pub betti_0: usize,
    pub chebyshev_k: f64,
}

// ─── Latent Predictor (SCM dynamics) ─────────────────────────────────

/// Forward dynamics: s_{t+1} = SCM(s_t + W·a_t)
pub struct LatentPredictor {
    state_dim: usize,
    action_dim: usize,
    w_matrix: Vec<f64>, // action_dim x state_dim flattened
    operator: SpectralContractionOperator<4>,
}

impl LatentPredictor {
    pub fn new(state_dim: usize, action_dim: usize, seed: u64) -> Self {
        // Deterministic pseudo-random weight matrix
        let total = action_dim * state_dim;
        let mut w = vec![0.0f64; total];
        let mut rng_state = seed;
        let scale = libm::sqrt(2.0 / (state_dim + action_dim) as f64);
        for v in w.iter_mut() {
            // LCG PRNG
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
            *v = u * scale;
        }
        Self {
            state_dim,
            action_dim,
            w_matrix: w,
            operator: SpectralContractionOperator::new(0.1),
        }
    }

    /// Project action text into an action vector via hash.
    pub fn hash_action(&self, action: &str) -> Vec<f64> {
        let mut vec = vec![0.0f64; self.action_dim];
        let bytes = action.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            let idx = ((b as usize).wrapping_mul(31).wrapping_add(i * 7)) % self.action_dim;
            vec[idx] += if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        // Normalize
        let norm = libm::sqrt(vec.iter().map(|x| x * x).sum::<f64>());
        if norm > 1e-12 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
        vec
    }

    /// Step: s' = SCM((s + W^T a), attractor)
    pub fn step(&self, state: &[f64], action: &[f64], attractor: &[f64]) -> Vec<f64> {
        let mut s_prime = state.to_vec();
        // Add W^T * action contribution
        for (j, &action_j) in action.iter().enumerate().take(self.action_dim) {
            for i in 0..self.state_dim.min(s_prime.len()) {
                s_prime[i] += self.w_matrix[j * self.state_dim + i] * action_j * 0.1;
            }
        }
        // Apply SCM contraction toward attractor (componentwise on first 4 dims)
        let alpha = self.operator.alpha;
        for i in 0..s_prime.len().min(attractor.len()) {
            s_prime[i] = (1.0 - alpha) * s_prime[i] + alpha * attractor[i];
        }
        s_prime
    }
}

// ─── LLM Bridge (Minimax 2.7) ───────────────────────────────────────

/// Minimax 2.7 API client.
pub struct MinimaxBridge {
    api_key: String,
    endpoint: String,
    model: String,
}

impl MinimaxBridge {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            endpoint: "https://api.minimax.chat/v1/text/chatcompletion_v2".into(),
            model: "MiniMax-Text-01".into(),
        }
    }

    pub fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }

    /// Call Minimax 2.7 with a user prompt via curl subprocess. Returns the response text.
    pub fn call(&self, system: &str, user: &str) -> Result<String, String> {
        if !self.is_configured() {
            return Err("API key not set".into());
        }
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1024,
        });

        let output = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                &self.endpoint,
                "-H",
                "Content-Type: application/json",
                "-H",
                &format!("Authorization: Bearer {}", self.api_key),
                "-d",
                &body.to_string(),
                "--max-time",
                "30",
            ])
            .output()
            .map_err(|e| format!("curl failed: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("curl error: {stderr}"));
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();
        let parsed: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("JSON parse error: {e}"))?;
        let content = parsed["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or(&text)
            .to_string();
        Ok(content)
    }
}

// ─── Theorem Verification ────────────────────────────────────────────

/// Run all 10 theorems from aether_verified and return pass/fail.
pub fn verify_theorems() -> Vec<(&'static str, bool)> {
    use aether_verified::*;

    let mut results = Vec::new();

    // T1: TSS — spherical packing bound
    {
        let theta = aether_tss::theta_min_from_epsilon(0.1);
        let pm = aether_tss::p_max(0.5);
        let centroids = [(0.0, 0.0), (1.2, 0.0), (0.0, 1.2)];
        let ok = aether_tss::verify_packing_bound(centroids.len(), 0.5)
            && aether_tss::verify_separation(&centroids, 0.5);
        results.push(("T1_TSS", ok && pm > 1.0 && theta > 0.0));
    }

    // T2: SCM — contraction mapping
    {
        let alpha = 0.3;
        let rho = aether_scm::lipschitz_constant(alpha);
        let mut s = 5.0;
        for _ in 0..50 {
            s = aether_scm::apply_operator(s, 0.0, alpha);
        }
        results.push(("T2_SCM", rho < 1.0 && s.abs() < 0.01));
    }

    // T3: GMC — entropy reduction on merge
    {
        let ok1 = aether_gmc::verify_entropy_nonincreasing(100, 50, 1000);
        let ok2 = aether_gmc::verify_entropy_nonincreasing(500, 500, 1000);
        results.push(("T3_GMC", ok1 && ok2));
    }

    // T4: AGCR — governor convergence
    {
        let rho = aether_agcr::contraction_rate(0.01, 0.05, 1.0);
        let hl = aether_agcr::half_life(rho);
        let stable = aether_agcr::gain_margin_stable(0.01, 0.05, 1.0);
        results.push(("T4_AGCR", rho < 1.0 && hl > 0.0 && hl.is_finite() && stable));
    }

    // T5: HCS — hyperbolic beats Euclidean
    {
        let ok = aether_hcs::verify_hcs(1.0, 4, 128, 10);
        let ratio = aether_hcs::separation_ratio(1.0, 4, 10);
        results.push(("T5_HCS", ok && ratio > 50.0));
    }

    // T6: RGCS — tangent deviation bound
    {
        let bound = aether_world::tangent_deviation_bound(0.01, 1.0, 128);
        results.push(("T6_RGCS", bound > 0.0 && bound < 0.01));
    }

    // T7: PHKP — Betti-guided latency
    {
        let lat = aether_world::betti_latency(800, 150, 50);
        let sparse = aether_world::sparse_latency(lat, 0.7);
        results.push(("T7_PHKP", sparse < lat && lat < 5000.0));
    }

    // T8: TEB — Landauer energy bound
    {
        let e = aether_world::landauer_energy_per_bit(300.0);
        results.push(("T8_TEB", e > 2.8e-21 && e < 2.9e-21));
    }

    // T9: CMA — alignment error bound
    {
        let b = aether_world::alignment_error_bound(0.5, 1.0);
        results.push(("T9_CMA", (b - 0.866).abs() < 0.01));
    }

    // T10: WPHB — predictive horizon
    {
        let h = aether_world::predictive_horizon(1000, 128, 1e-4, 10.0);
        results.push(("T10_WPHB", h > 1e5));
    }

    results
}

// ─── World Model ─────────────────────────────────────────────────────

/// The Epsilon-Hollow topological world model.
///
/// Combines:
/// - ManifoldMemory (topological episodic store with Betti-0 tracking)
/// - LatentPredictor (SCM-based forward dynamics)
/// - MinimaxBridge (LLM reasoning cortex)
/// - Theorem verification (T1-T10)
pub struct World {
    pub memory: ManifoldMemory,
    pub predictor: LatentPredictor,
    pub llm: MinimaxBridge,
    pub history: Vec<String>,
    ingest_steps: u64,
}

const SYSTEM_PROMPT: &str = "You are the reasoning cortex of Epsilon-Hollow, a topological OS. \
     You receive retrieved memories from a manifold memory store (with \
     similarity scores and Betti numbers). Synthesize them into a clear, \
     concise answer. If memories are empty, reason from first principles.";

impl World {
    pub fn new(api_key: String, dim: usize, capacity: usize) -> Self {
        Self {
            memory: ManifoldMemory::new(dim, capacity),
            predictor: LatentPredictor::new(dim, 32, 42),
            llm: MinimaxBridge::new(api_key),
            history: Vec::new(),
            ingest_steps: 0,
        }
    }

    /// Ingest text into topological memory.
    pub fn update(&mut self, text: &str) -> UpdateResult {
        let vec = self.memory.encode_text(text);
        let idx = self.memory.store(vec, text.to_string(), HashMap::new());
        self.history.push(text.to_string());
        self.ingest_steps += 1;
        UpdateResult {
            index: idx,
            ingest_step: self.ingest_steps,
            memory_size: self.memory.episodes.len(),
            betti_0: self.memory.betti_0(),
        }
    }

    /// Query memory + optionally call LLM for synthesis.
    pub fn query(&self, text: &str, k: usize) -> QueryResult {
        let vec = self.memory.encode_text(text);
        let t0 = Instant::now();
        let matches = self.memory.retrieve(&vec, k);
        let retrieval_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let top_k: Vec<MatchResult> = matches
            .iter()
            .map(|(idx, score, ep)| MatchResult {
                index: *idx,
                score: *score,
                cluster_id: ep.cluster_id,
                text: ep.text.clone(),
                reinforcement_count: ep.reinforcement_count,
            })
            .collect();

        // Build context for LLM
        let mut llm_response = None;
        if self.llm.is_configured() {
            let ctx = self.format_context(&top_k);
            let prompt = format!("Query: {text}\n\nRetrieved context:\n{ctx}");
            match self.llm.call(SYSTEM_PROMPT, &prompt) {
                Ok(resp) => llm_response = Some(resp),
                // Wrap errors as text so the CLI always has something to display.
                Err(e) => llm_response = Some(format!("[LLM error: {e}]")),
            }
        }

        QueryResult {
            retrieval_ms,
            top_k,
            memory_size: self.memory.episodes.len(),
            betti_0: self.memory.betti_0(),
            llm_response,
        }
    }

    /// Counterfactual rollout through latent dynamics.
    pub fn dream(&self, text: &str, horizon: usize) -> DreamResult {
        let mut state = self.memory.encode_text(text);
        let mut trace = Vec::new();
        let mut total_reward = 0.0;

        for step in 0..horizon {
            let action_str = format!("dream_act_{step}");
            let action = self.predictor.hash_action(&action_str);

            // Use nearest memory as attractor
            let matches = self.memory.retrieve(&state, 1);
            let attractor = if let Some((_, _, ep)) = matches.first() {
                ep.vector.clone()
            } else {
                vec![0.0; self.memory.dim]
            };

            state = self.predictor.step(&state, &action, &attractor);
            let state_norm = libm::sqrt(state.iter().map(|x| x * x).sum::<f64>());

            // Simple reward: novelty (inverse similarity to nearest)
            let novelty = if let Some((_, score, _)) = matches.first() {
                1.0 - score.clamp(0.0, 1.0)
            } else {
                1.0
            };
            total_reward += novelty;

            trace.push(DreamStep {
                step,
                state_norm,
                reward: novelty,
                context_hits: matches.len(),
            });
        }

        DreamResult {
            horizon,
            total_reward,
            trace,
        }
    }

    /// Run T1-T10 theorem verification.
    pub fn verify_theorems(&self) -> Vec<(&'static str, bool)> {
        verify_theorems()
    }

    /// World model status.
    pub fn status(&self) -> WorldStatus {
        WorldStatus {
            dim: self.memory.dim,
            capacity: self.memory.capacity,
            ingest_steps: self.ingest_steps,
            memory: self.memory.stats(),
            history_len: self.history.len(),
            api_key_set: self.llm.is_configured(),
        }
    }

    fn format_context(&self, matches: &[MatchResult]) -> String {
        let mut lines = Vec::new();
        for (i, m) in matches.iter().take(5).enumerate() {
            lines.push(format!(
                "  [{}] score={:.3} cluster={} text=\"{}\"",
                i + 1,
                m.score,
                m.cluster_id,
                if m.text.len() > 60 {
                    format!("{}...", &m.text[..60])
                } else {
                    m.text.clone()
                }
            ));
        }
        let stats = self.memory.stats();
        lines.push(format!(
            "  Memory: size={} betti_0={}",
            stats.size, stats.betti_0
        ));
        lines.join("\n")
    }
}

// ─── Result types ────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
pub struct UpdateResult {
    pub index: usize,
    pub ingest_step: u64,
    pub memory_size: usize,
    pub betti_0: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct MatchResult {
    pub index: usize,
    pub score: f64,
    pub cluster_id: i32,
    pub text: String,
    pub reinforcement_count: u32,
}

#[derive(Debug, serde::Serialize)]
pub struct QueryResult {
    pub retrieval_ms: f64,
    pub top_k: Vec<MatchResult>,
    pub memory_size: usize,
    pub betti_0: usize,
    pub llm_response: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct DreamStep {
    pub step: usize,
    pub state_norm: f64,
    pub reward: f64,
    pub context_hits: usize,
}

#[derive(Debug, serde::Serialize)]
pub struct DreamResult {
    pub horizon: usize,
    pub total_reward: f64,
    pub trace: Vec<DreamStep>,
}

#[derive(Debug, serde::Serialize)]
pub struct WorldStatus {
    pub dim: usize,
    pub capacity: usize,
    pub ingest_steps: u64,
    pub memory: MemoryStats,
    pub history_len: usize,
    pub api_key_set: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Union-Find ──────────────────────────────────────────────

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.components, 5);
        uf.union(0, 1);
        assert_eq!(uf.components, 4);
        uf.union(2, 3);
        assert_eq!(uf.components, 3);
        uf.union(0, 3);
        assert_eq!(uf.components, 2);
        assert_eq!(uf.find(0), uf.find(3));
        assert_eq!(uf.find(1), uf.find(2));
    }

    #[test]
    fn test_union_find_equal_rank_merge() {
        // Regression: equal-rank roots must actually merge.
        let mut uf = UnionFind::new(4);
        uf.union(0, 1); // rank[root] = 1
        uf.union(2, 3); // rank[root] = 1
                        // Both trees have rank 1 — this is the Equal case.
        uf.union(0, 2);
        assert_eq!(uf.components, 1);
        // All four nodes must resolve to the same root.
        let root = uf.find(0);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(3), root);
    }

    #[test]
    fn test_union_find_idempotent() {
        let mut uf = UnionFind::new(3);
        assert!(uf.union(0, 1));
        assert!(!uf.union(0, 1)); // already same component
        assert_eq!(uf.components, 2);
    }

    // ── ManifoldMemory ──────────────────────────────────────────

    #[test]
    fn test_encode_text_deterministic() {
        let mem = ManifoldMemory::new(64, 100);
        let a = mem.encode_text("hello world");
        let b = mem.encode_text("hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_encode_text_unit_norm() {
        let mem = ManifoldMemory::new(128, 100);
        let v = mem.encode_text("topological operating system");
        let norm: f64 = libm::sqrt(v.iter().map(|x| x * x).sum::<f64>());
        assert!((norm - 1.0).abs() < 1e-10, "norm = {}", norm);
    }

    #[test]
    fn test_encode_empty_string() {
        let mem = ManifoldMemory::new(64, 100);
        let v = mem.encode_text("");
        // Empty string produces a zero vector (can't be normalized).
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_store_and_retrieve_ordering() {
        let mut mem = ManifoldMemory::new(128, 1000);
        let v1 = mem.encode_text("the cat sat on the mat");
        let v2 = mem.encode_text("dogs are loyal animals");
        let v3 = mem.encode_text("the cat sat on the rug");
        mem.store(v1, "the cat sat on the mat".into(), HashMap::new());
        mem.store(v2, "dogs are loyal animals".into(), HashMap::new());
        mem.store(v3.clone(), "the cat sat on the rug".into(), HashMap::new());

        let results = mem.retrieve(&v3, 2);
        assert_eq!(results.len(), 2);
        // The query is identical to episode 2, so it must rank first.
        assert!(
            (results[0].1 - 1.0).abs() < 1e-6,
            "self-similarity should be 1.0"
        );
    }

    #[test]
    fn test_betti_0_single_cluster() {
        let mut mem = ManifoldMemory::new(128, 1000);
        // Store the same text three times — high cosine similarity.
        for _ in 0..3 {
            let v = mem.encode_text("identical text");
            mem.store(v, "identical text".into(), HashMap::new());
        }
        assert_eq!(mem.betti_0(), 1);
    }

    #[test]
    fn test_betti_0_separate_clusters() {
        let mut mem = ManifoldMemory::new(128, 1000);
        // Very different texts should form separate components.
        let texts = ["aaa", "zzz999xyz", "!!@@##$$"];
        for t in texts {
            let v = mem.encode_text(t);
            mem.store(v, t.into(), HashMap::new());
        }
        assert!(mem.betti_0() >= 2, "betti_0 = {}", mem.betti_0());
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut mem = ManifoldMemory::new(32, 3);
        for i in 0..5 {
            let v = mem.encode_text(&format!("text {i}"));
            mem.store(v, format!("text {i}"), HashMap::new());
        }
        assert_eq!(mem.episodes.len(), 3);
    }

    #[test]
    fn test_reinforce() {
        let mut mem = ManifoldMemory::new(32, 100);
        let v = mem.encode_text("test");
        mem.store(v, "test".into(), HashMap::new());
        assert_eq!(mem.episodes[0].reinforcement_count, 0);
        mem.reinforce(0);
        assert_eq!(mem.episodes[0].reinforcement_count, 1);
    }

    #[test]
    fn test_reset_clears() {
        let mut mem = ManifoldMemory::new(32, 100);
        let v = mem.encode_text("data");
        mem.store(v, "data".into(), HashMap::new());
        mem.reset();
        assert_eq!(mem.episodes.len(), 0);
    }

    // ── LatentPredictor ─────────────────────────────────────────

    #[test]
    fn test_predictor_deterministic() {
        let p = LatentPredictor::new(64, 16, 42);
        let state = vec![1.0; 64];
        let action = p.hash_action("test action");
        let attractor = vec![0.0; 64];
        let a = p.step(&state, &action, &attractor);
        let b = p.step(&state, &action, &attractor);
        assert_eq!(a, b);
    }

    #[test]
    fn test_predictor_contracts_toward_attractor() {
        let p = LatentPredictor::new(64, 16, 42);
        let attractor = vec![0.0; 64];
        let action = vec![0.0; 16]; // zero action — pure contraction
        let mut state = vec![10.0; 64];
        let initial_dist: f64 = state.iter().map(|x| x * x).sum::<f64>();
        for _ in 0..100 {
            state = p.step(&state, &action, &attractor);
        }
        let final_dist: f64 = state.iter().map(|x| x * x).sum::<f64>();
        assert!(
            final_dist < initial_dist,
            "should contract toward attractor"
        );
    }

    #[test]
    fn test_hash_action_normalized() {
        let p = LatentPredictor::new(64, 32, 42);
        let a = p.hash_action("some action string");
        let norm = libm::sqrt(a.iter().map(|x| x * x).sum::<f64>());
        assert!((norm - 1.0).abs() < 1e-10, "norm = {}", norm);
    }

    // ── World ───────────────────────────────────────────────────

    #[test]
    fn test_world_update() {
        let mut w = World::new(String::new(), 64, 1000);
        let r = w.update("observation one");
        assert_eq!(r.ingest_step, 1);
        assert_eq!(r.memory_size, 1);
        let r2 = w.update("observation two");
        assert_eq!(r2.ingest_step, 2);
        assert_eq!(r2.memory_size, 2);
    }

    #[test]
    fn test_world_query_no_llm() {
        let mut w = World::new(String::new(), 64, 1000);
        w.update("the sky is blue");
        let r = w.query("sky color", 3);
        assert!(r.llm_response.is_none()); // no API key → no LLM call
        assert!(!r.top_k.is_empty());
    }

    #[test]
    fn test_world_dream_trace_length() {
        let mut w = World::new(String::new(), 64, 1000);
        w.update("seed memory");
        let r = w.dream("future", 5);
        assert_eq!(r.trace.len(), 5);
        assert_eq!(r.horizon, 5);
    }

    #[test]
    fn test_verify_theorems_all_pass() {
        let results = verify_theorems();
        let failed: Vec<&str> = results
            .iter()
            .filter(|(_, ok)| !*ok)
            .map(|(name, _)| *name)
            .collect();
        assert!(failed.is_empty(), "failed theorems: {:?}", failed);
        assert_eq!(results.len(), 10);
    }
}
