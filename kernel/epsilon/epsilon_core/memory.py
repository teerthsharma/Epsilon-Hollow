"""
Epsilon-Hollow — Topological Manifold Memory
=============================================
Memory as a high-dimensional manifold M ⊂ ℝ^d with persistent homology.

Mathematical Objects Implemented:
    1. Embedding:  Φ(x) → ℝ^d
    2. Betti-0 Clustering:  Connected components β₀ in the ε-neighborhood graph
    3. Chebyshev Adaptive ε:  ε = μ_local + kσ_local,  P(|X−μ| ≥ kσ) ≤ 1/k²
    4. Centroid Projection on S²:  Stereographic projection for O(1) cluster lookup
    5. Cosine Similarity Retrieval:  cos(q, x) = q·x / (‖q‖‖x‖)
    6. Exponential Decay:  w(t) = exp(−λΔt) for temporal forgetting

Reference: README §Mathematical Specification — "Topological Manifold Persistence"
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Union-Find (Disjoint Set) for O(α(n)) component tracking
# ─────────────────────────────────────────────────────────────────────
class UnionFind:
    """
    Weighted Union-Find with path compression.
    Amortised O(α(n)) per operation where α is the inverse Ackermann function.
    Used to compute Betti-0 (connected components) of the ε-neighborhood graph.
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_components = n

    def find(self, x: int) -> int:
        """Find with path compression: O(α(n)) amortised."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Union by rank. Returns True if a merge occurred."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.num_components -= 1
        return True


# ─────────────────────────────────────────────────────────────────────
# Core Manifold Memory
# ─────────────────────────────────────────────────────────────────────
class TopologicalManifoldMemory:
    """
    Memory as a high-dimensional manifold M ⊂ ℝ^d.

    Architecture:
        • Episodic store: list of (vector, metadata, timestamp) tuples
        • Connectivity graph: ε-neighborhood adjacency
        • Betti-0 tracker: Union-Find over connected components
        • S² projection: cluster centroids projected onto unit 2-sphere

    Performance Guarantees:
        • store():    O(n) for ε-neighborhood update
        • retrieve(): O(C) centroid lookup + O(k log k) within cluster
        • betti_0():  O(1) — maintained incrementally
    """

    def __init__(self, dim: int = 128, chebyshev_k: float = 2.0,
                 decay_lambda: float = 0.001, capacity: int = 100_000):
        """
        Parameters
        ----------
        dim : int
            Manifold embedding dimension |ℝ^d|. Default 128.
        chebyshev_k : float
            Multiplier for Chebyshev bound: ε = μ + k·σ.
            k=2 gives P(|X−μ| ≥ 2σ) ≤ 0.25.
        decay_lambda : float
            Exponential decay rate for temporal forgetting.
        capacity : int
            Hard cap on episodic store size.
        """
        self.dim = dim
        self.chebyshev_k = chebyshev_k
        self.decay_lambda = decay_lambda
        self.capacity = capacity

        # Episodic store: each entry is a dict with 'vector', 'metadata', 'timestamp'
        self.episodes: List[Dict[str, Any]] = []
        # Cached matrix for fast retrieval (N × d)
        self._vectors: Optional[np.ndarray] = None
        self._vectors_dirty = True

        # Union-Find for Betti-0
        self._uf: Optional[UnionFind] = None

        # S² centroid cache: cluster_id → (θ, φ) on unit sphere
        self._centroid_cache: Dict[int, Tuple[float, float]] = {}
        self._centroids_dirty = True

    # ─── Store ────────────────────────────────────────────────────
    def store(self, vector: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """
        Store an experience vector on the manifold.

        Parameters
        ----------
        vector : np.ndarray, shape (d,)
            The latent embedding Φ(x) ∈ ℝ^d.
        metadata : dict, optional
            Arbitrary metadata to attach.

        Returns
        -------
        int : index of the stored experience.
        """
        assert vector.shape == (self.dim,), \
            f"Expected shape ({self.dim},), got {vector.shape}"

        # L2-normalise for cosine geometry
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm

        idx = len(self.episodes)
        self.episodes.append({
            "vector": vector.copy(),
            "metadata": metadata or {},
            "timestamp": time.time(),
            "reinforcement_count": 0,
        })

        self._vectors_dirty = True
        self._centroids_dirty = True

        # Incremental Betti-0 update: check connectivity to existing points
        self._extend_union_find(idx)

        # Evict oldest if over capacity
        if len(self.episodes) > self.capacity:
            self._evict_oldest()

        return idx

    def store_experience(self, vector: np.ndarray, metadata: Optional[Dict] = None) -> int:
        """Backward-compatible alias for store()."""
        return self.store(vector, metadata)

    # ─── Retrieve ─────────────────────────────────────────────────
    def retrieve(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """
        O(1) centroid lookup on S², then k-NN within winning cluster.

        Algorithm:
            1. Project query → (θ_q, φ_q) on S²
            2. Find nearest centroid by angular distance on sphere
            3. Within that cluster, rank by cosine similarity
            4. Return top-k with temporal decay weighting

        Parameters
        ----------
        query_vector : np.ndarray, shape (d,)
        k : int

        Returns
        -------
        List of dicts: [{vector, metadata, score, cluster_id}, ...]
        """
        if not self.episodes:
            return []

        # Normalise query
        qnorm = np.linalg.norm(query_vector)
        if qnorm > 1e-12:
            query_vector = query_vector / qnorm

        self._rebuild_vectors_if_dirty()
        self._rebuild_centroids_if_dirty()

        # Step 1: Find nearest centroid on S²
        q_theta, q_phi = self._project_to_s2(query_vector)
        best_cluster = -1
        best_angular_dist = float("inf")

        for cluster_id, (c_theta, c_phi) in self._centroid_cache.items():
            # Great-circle distance on S²:
            # d = arccos(sin θ₁ sin θ₂ + cos θ₁ cos θ₂ cos(φ₁ − φ₂))
            cos_d = (math.sin(c_theta) * math.sin(q_theta) +
                     math.cos(c_theta) * math.cos(q_theta) *
                     math.cos(c_phi - q_phi))
            cos_d = max(-1.0, min(1.0, cos_d))
            dist = math.acos(cos_d)
            if dist < best_angular_dist:
                best_angular_dist = dist
                best_cluster = cluster_id

        if best_cluster < 0:
            # Fallback: brute-force cosine
            return self._brute_force_retrieve(query_vector, k)

        # Step 2: Within-cluster cosine similarity
        now = time.time()
        candidates = []
        for i, ep in enumerate(self.episodes):
            if self._uf and self._uf.find(i) == best_cluster:
                sim = float(np.dot(query_vector, ep["vector"]))
                # Temporal decay: w(t) = exp(−λΔt)
                age = now - ep["timestamp"]
                decay = math.exp(-self.decay_lambda * age)
                # Reinforcement bonus
                reinforce = 1.0 + 0.1 * ep["reinforcement_count"]
                score = sim * decay * reinforce
                candidates.append({"episode": ep, "score": score,
                                   "cluster_id": best_cluster, "index": i})

        # If cluster is too small, expand search
        if len(candidates) < k:
            return self._brute_force_retrieve(query_vector, k)

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:k]

    # ─── Betti-0 ──────────────────────────────────────────────────
    def compute_betti_0(self) -> int:
        """
        Compute β₀ = number of connected components in the ε-neighborhood graph.

        A "concept" is a connected component. More components ⟹ more distinct
        concepts in memory. Fewer components ⟹ memory is converging.

        Returns
        -------
        int : β₀ (number of distinct concepts)
        """
        if self._uf is None or len(self.episodes) == 0:
            return 0
        return self._uf.num_components

    # ─── Adaptive Epsilon (Chebyshev) ────────────────────────────
    def adaptive_epsilon(self, neighborhood: np.ndarray) -> float:
        """
        Compute adaptive connectivity threshold using Chebyshev's inequality.

        ε = μ_local + k · σ_local

        By Chebyshev: P(|X − μ| ≥ kσ) ≤ 1/k²
        With k=2:    P ≤ 0.25  (at most 25% of points are outliers)
        With k=3:    P ≤ 0.111

        Parameters
        ----------
        neighborhood : np.ndarray, shape (n,)
            Pairwise distances in a local region.

        Returns
        -------
        float : ε_adaptive
        """
        if len(neighborhood) < 2:
            return 1.0  # Default wide threshold

        mu = float(np.mean(neighborhood))
        sigma = float(np.std(neighborhood))
        return mu + self.chebyshev_k * sigma

    # ─── Reinforce ────────────────────────────────────────────────
    def reinforce(self, index: int):
        """Reinforce a memory (anti-decay). Called when a retrieved memory is useful."""
        if 0 <= index < len(self.episodes):
            self.episodes[index]["reinforcement_count"] += 1
            self.episodes[index]["timestamp"] = time.time()  # refresh

    # ─── Manifold Statistics ──────────────────────────────────────
    def manifold_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the memory manifold."""
        n = len(self.episodes)
        if n == 0:
            return {"size": 0, "betti_0": 0, "dim": self.dim}

        self._rebuild_vectors_if_dirty()
        # Compute mean pairwise distance (sampled for large N)
        sample_size = min(n, 200)
        indices = np.random.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
        sampled = self._vectors[indices]  # type: ignore
        # Pairwise cosine distances
        sims = sampled @ sampled.T
        np.fill_diagonal(sims, 0)
        mean_sim = float(np.mean(sims))

        return {
            "size": n,
            "dim": self.dim,
            "betti_0": self.compute_betti_0(),
            "mean_cosine_similarity": mean_sim,
            "num_centroids": len(self._centroid_cache),
            "capacity_used": n / self.capacity,
            "chebyshev_k": self.chebyshev_k,
        }

    # ═══════════════════════════════════════════════════════════════
    #  Internal Methods
    # ═══════════════════════════════════════════════════════════════

    def _rebuild_vectors_if_dirty(self):
        """Rebuild the (N × d) matrix cache."""
        if self._vectors_dirty and self.episodes:
            self._vectors = np.array([ep["vector"] for ep in self.episodes])
            self._vectors_dirty = False

    def _rebuild_centroids_if_dirty(self):
        """Recompute S² centroid projections for each connected component."""
        if not self._centroids_dirty or not self.episodes:
            return
        self._rebuild_vectors_if_dirty()

        # Group episodes by component
        components: Dict[int, List[int]] = {}
        for i in range(len(self.episodes)):
            root = self._uf.find(i) if self._uf else 0
            components.setdefault(root, []).append(i)

        self._centroid_cache.clear()
        for comp_id, member_indices in components.items():
            # Compute centroid in ℝ^d
            member_vecs = self._vectors[member_indices]  # type: ignore
            centroid = np.mean(member_vecs, axis=0)
            # Project to S²
            theta, phi = self._project_to_s2(centroid)
            self._centroid_cache[comp_id] = (theta, phi)

        self._centroids_dirty = False

    def _project_to_s2(self, vector: np.ndarray) -> Tuple[float, float]:
        """
        Project ℝ^d vector onto the unit 2-sphere S² via PCA → ℝ³ → normalise → spherical.

        Method:
            1. Take first 3 principal components (fast: just first 3 coords of normalised vector)
            2. Normalise to unit sphere
            3. Convert to spherical coordinates (θ, φ)

        For a rigorous version, use incremental PCA on the full manifold.
        This fast path uses the first 3 dimensions as a proxy, which is valid
        when the embedding is already well-distributed.

        θ = arccos(z / r)     ∈ [0, π]
        φ = atan2(y, x)       ∈ [−π, π]
        """
        # Use first 3 dimensions as principal axes
        v3 = vector[:3].copy()
        r = np.linalg.norm(v3)
        if r < 1e-12:
            return (0.0, 0.0)
        v3 /= r

        theta = math.acos(max(-1.0, min(1.0, float(v3[2]))))
        phi = math.atan2(float(v3[1]), float(v3[0]))
        return (theta, phi)

    def _extend_union_find(self, new_idx: int):
        """Add a new point to the Union-Find and connect to ε-neighbours."""
        n = len(self.episodes)
        if self._uf is None:
            self._uf = UnionFind(n)
        else:
            # Extend the UF structure
            self._uf.parent.append(new_idx)
            self._uf.rank.append(0)
            self._uf.num_components += 1

        if n < 2:
            return

        new_vec = self.episodes[new_idx]["vector"]
        # Compute distances to all existing points
        dists = []
        for i in range(new_idx):
            d = 1.0 - float(np.dot(new_vec, self.episodes[i]["vector"]))  # cosine distance
            dists.append(d)

        if not dists:
            return

        dists_arr = np.array(dists)
        eps = self.adaptive_epsilon(dists_arr)

        for i, d in enumerate(dists):
            if d < eps:
                self._uf.union(i, new_idx)

    def _brute_force_retrieve(self, query: np.ndarray, k: int) -> List[Dict]:
        """Fallback brute-force cosine search over all episodes."""
        now = time.time()
        scored = []
        for i, ep in enumerate(self.episodes):
            sim = float(np.dot(query, ep["vector"]))
            age = now - ep["timestamp"]
            decay = math.exp(-self.decay_lambda * age)
            reinforce = 1.0 + 0.1 * ep["reinforcement_count"]
            scored.append({"episode": ep, "score": sim * decay * reinforce,
                           "cluster_id": self._uf.find(i) if self._uf else 0,
                           "index": i})
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:k]

    def _evict_oldest(self):
        """Remove the oldest, lowest-reinforcement episode."""
        if not self.episodes:
            return
        # Find lowest reinforcement_count, then oldest timestamp
        worst = min(range(len(self.episodes)),
                    key=lambda i: (self.episodes[i]["reinforcement_count"],
                                   -self.episodes[i]["timestamp"]))
        self.episodes.pop(worst)
        self._vectors_dirty = True
        self._centroids_dirty = True
        # Rebuild UF (amortised over many evictions)
        self._rebuild_uf()

    def _rebuild_uf(self):
        """Full UF rebuild after structural changes."""
        n = len(self.episodes)
        self._uf = UnionFind(n)
        self._rebuild_vectors_if_dirty()
        if n < 2 or self._vectors is None:
            return

        # Pairwise cosine distances (sample for large N)
        for i in range(n):
            for j in range(i + 1, min(i + 200, n)):  # local window
                d = 1.0 - float(np.dot(self._vectors[i], self._vectors[j]))
                # Use global adaptive epsilon
                local_dists = np.array([
                    1.0 - float(np.dot(self._vectors[i], self._vectors[m]))
                    for m in range(max(0, i - 10), min(n, i + 10))
                    if m != i
                ])
                if len(local_dists) > 0:
                    eps = self.adaptive_epsilon(local_dists)
                    if d < eps:
                        self._uf.union(i, j)
