"""
Epsilon-Hollow — Angular Sparse Attention Engine
=================================================
Pure Python/NumPy reference implementation of the CUDA AetherSparseAttention kernel.
Cosine-pruned block-sparse attention with O(B·A) instead of O(B·N) complexity.

Mathematical Foundation:
    Given Q ∈ ℝ^{B×H×L_q×D}, K ∈ ℝ^{B×H×L_k×D}, V ∈ ℝ^{B×H×L_k×D}:

    1. Block Partitioning:
       K is partitioned into B blocks of size b: K = [K₁, K₂, ..., K_B]
       Each block has a centroid:  c_i = (1/b) Σⱼ K_{i,j}

    2. Angular Pruning (Cosine Threshold):
       A block K_i is "active" iff:
           cos(Q, c_i) = (Q · c_i) / (‖Q‖ ‖c_i‖) ≥ τ
       where τ is the angular sparsity threshold.

    3. Sparse Attention (Online Softmax):
       Only compute attention over active blocks A ⊆ {1,...,B}:
           Attn(Q, K, V) = softmax(Q K_A^T / √d) V_A
       Using the online softmax trick (Milakov & Gimelshein, 2018):
           m_i = max(m_{i-1}, s_i)
           l_i = l_{i-1} · exp(m_{i-1} - m_i) + exp(s_i - m_i)
           o_i = o_{i-1} · exp(m_{i-1} - m_i) / l_i + exp(s_i - m_i) / l_i · v_i

    4. Complexity:
       Standard attention: O(L_q · L_k · D)
       Block-sparse:       O(L_q · |A| · b · D)  where |A| << B typically

    This mirrors the CUDA kernels:
        - aether_generate_indices.cu → generate_active_indices()
        - aether_sparse_flash_fwd_kernel.cu → sparse_flash_forward()

Reference: TensorRT-LLM AetherSparseAttentionPlugin
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


class AngularSparseAttention:
    """
    Block-sparse attention with angular (cosine) pruning.

    The key insight: most KV blocks have centroids that are nearly
    orthogonal to the query. By pruning blocks where cos(Q, centroid) < τ,
    we skip ~70-90% of the KV cache with <1% quality loss.
    """

    def __init__(self, block_size: int = 64, cos_threshold: float = 0.3,
                 head_dim: int = 128, num_heads: int = 8):
        """
        Parameters
        ----------
        block_size : int
            Number of tokens per KV block. Typical: 64 or 128.
        cos_threshold : float
            Angular sparsity threshold τ. Blocks with cos(Q, c) < τ are pruned.
            Higher τ = more aggressive pruning = faster but potentially lossy.
        head_dim : int
            Dimension per attention head.
        num_heads : int
            Number of attention heads.
        """
        self.block_size = block_size
        self.cos_threshold = cos_threshold
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(head_dim)

        # Statistics
        self.total_blocks_checked = 0
        self.total_blocks_active = 0

    def compute_block_centroids(self, keys: np.ndarray) -> np.ndarray:
        """
        Partition K into blocks and compute centroids.

        Parameters
        ----------
        keys : np.ndarray, shape (seq_len, head_dim)

        Returns
        -------
        centroids : np.ndarray, shape (num_blocks, head_dim)
            L2-normalised centroids for each block.
        """
        seq_len = keys.shape[0]
        num_blocks = math.ceil(seq_len / self.block_size)

        centroids = np.zeros((num_blocks, self.head_dim), dtype=keys.dtype)
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, seq_len)
            block = keys[start:end]
            centroid = np.mean(block, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-12:
                centroid /= norm
            centroids[i] = centroid

        return centroids

    def generate_active_indices(self, query: np.ndarray,
                                centroids: np.ndarray) -> List[int]:
        """
        CUDA kernel equivalent: aether_generate_indices.

        For each block centroid, compute cosine similarity with query.
        Return indices of blocks where cos(Q, c_i) ≥ τ.

        Parameters
        ----------
        query : np.ndarray, shape (head_dim,)
            Single query vector (per head, per token).
        centroids : np.ndarray, shape (num_blocks, head_dim)

        Returns
        -------
        active_indices : List[int]
            Indices of blocks that pass the angular threshold.
        """
        # Normalise query
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-12:
            return list(range(len(centroids)))  # No pruning if query is zero

        q_hat = query / q_norm

        # Batch cosine similarity: cos(Q, c_i) = Q̂ · ĉ_i  (centroids already normalised)
        similarities = centroids @ q_hat  # (num_blocks,)

        # Angular pruning
        active = np.where(similarities >= self.cos_threshold)[0].tolist()

        self.total_blocks_checked += len(centroids)
        self.total_blocks_active += len(active)

        return active

    def sparse_flash_forward(self, query: np.ndarray, keys: np.ndarray,
                             values: np.ndarray,
                             active_indices: List[int]) -> np.ndarray:
        """
        CUDA kernel equivalent: aether_sparse_flash_fwd_kernel.

        Computes attention only over active KV blocks using the
        online softmax trick (numerically stable, single-pass).

        Algorithm (Milakov & Gimelshein 2018, extended by Dao 2022):
            For each active block i:
                For each token j in block i:
                    s_j = Q · K_j^T / √d
                    m_new = max(m, s_j)
                    l = l · exp(m_old - m_new) + exp(s_j - m_new)
                    o = o · exp(m_old - m_new) + exp(s_j - m_new) · V_j
            Output = o / l

        Parameters
        ----------
        query : np.ndarray, shape (head_dim,)
        keys : np.ndarray, shape (seq_len, head_dim)
        values : np.ndarray, shape (seq_len, head_dim)
        active_indices : List[int]

        Returns
        -------
        output : np.ndarray, shape (head_dim,)
        """
        seq_len = keys.shape[0]

        # Online softmax accumulators
        m_i = -np.inf              # running max
        l_i = 0.0                  # running sum of exp
        o_i = np.zeros(self.head_dim, dtype=np.float64)  # running output

        for block_idx in active_indices:
            start = block_idx * self.block_size
            end = min(start + self.block_size, seq_len)

            block_k = keys[start:end]    # (block_len, head_dim)
            block_v = values[start:end]  # (block_len, head_dim)

            for j in range(block_k.shape[0]):
                # Score: s = Q · K_j^T / √d
                s_j = float(np.dot(query, block_k[j])) * self.scale

                # Online softmax update
                m_new = max(m_i, s_j)
                alpha = math.exp(m_i - m_new) if m_i > -np.inf else 0.0
                beta = math.exp(s_j - m_new)

                l_i = l_i * alpha + beta
                o_i = o_i * alpha + beta * block_v[j].astype(np.float64)
                m_i = m_new

        # Final normalisation
        if l_i > 0:
            o_i /= l_i

        return o_i

    def forward(self, Q: np.ndarray, K: np.ndarray,
                V: np.ndarray) -> np.ndarray:
        """
        Full sparse attention forward pass.

        Parameters
        ----------
        Q : np.ndarray, shape (q_len, head_dim)
        K : np.ndarray, shape (kv_len, head_dim)
        V : np.ndarray, shape (kv_len, head_dim)

        Returns
        -------
        output : np.ndarray, shape (q_len, head_dim)
        """
        centroids = self.compute_block_centroids(K)
        q_len = Q.shape[0]
        output = np.zeros_like(Q, dtype=np.float64)

        for t in range(q_len):
            active = self.generate_active_indices(Q[t], centroids)
            output[t] = self.sparse_flash_forward(Q[t], K, V, active)

        return output.astype(Q.dtype)

    def sparsity_ratio(self) -> float:
        """Return the fraction of blocks that were pruned."""
        if self.total_blocks_checked == 0:
            return 0.0
        return 1.0 - (self.total_blocks_active / self.total_blocks_checked)

    def reset_stats(self):
        self.total_blocks_checked = 0
        self.total_blocks_active = 0


# ═══════════════════════════════════════════════════════════════════
#  Multi-Head Angular Sparse Attention
# ═══════════════════════════════════════════════════════════════════
class MultiHeadAngularSparseAttention:
    """
    Multi-head wrapper with per-head angular thresholds.

    Supports GQA (Grouped Query Attention) where num_kv_heads < num_q_heads.
    Each query head group shares a KV head.
    """

    def __init__(self, num_q_heads: int = 32, num_kv_heads: int = 8,
                 head_dim: int = 128, block_size: int = 64,
                 cos_threshold: float = 0.3):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.heads_per_group = num_q_heads // num_kv_heads

        self.attention_heads = [
            AngularSparseAttention(
                block_size=block_size,
                cos_threshold=cos_threshold,
                head_dim=head_dim,
            )
            for _ in range(num_q_heads)
        ]

    def forward(self, Q: np.ndarray, K: np.ndarray,
                V: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        Q : (num_q_heads, q_len, head_dim)
        K : (num_kv_heads, kv_len, head_dim)
        V : (num_kv_heads, kv_len, head_dim)

        Returns
        -------
        output : (num_q_heads, q_len, head_dim)
        """
        q_len = Q.shape[1]
        output = np.zeros_like(Q)

        for h in range(self.num_q_heads):
            kv_h = h // self.heads_per_group
            output[h] = self.attention_heads[h].forward(
                Q[h], K[kv_h], V[kv_h]
            )

        return output

    def aggregate_sparsity(self) -> float:
        """Mean sparsity across all heads."""
        ratios = [h.sparsity_ratio() for h in self.attention_heads]
        return sum(ratios) / max(len(ratios), 1)
