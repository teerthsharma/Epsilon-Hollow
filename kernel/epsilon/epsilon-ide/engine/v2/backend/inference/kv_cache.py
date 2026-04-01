"""
backend/memory/kv_cache.py
===========================
Sparse KV cache with top-k attention for memory-efficient inference.

Fixes:
  - Deduplicated class names (SparseKVCache was defined twice, second
    definition silently shadowed the first as a subclass of itself)
  - Renamed subclass to SparseAttentionKVCache to make the hierarchy clear
"""

import numpy as np
import math

try:
    from tinygrad.tensor import Tensor
    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False
    # Stub so the rest of the code doesn't crash on import
    class Tensor:
        def numpy(self): raise NotImplementedError("tinygrad not installed")


class SparseKVCache:
    """
    Ring-buffer KV cache using INT8 quantization.

    Stores key and value tensors for all layers in a compact INT8 format.
    This halves memory usage vs float16 with minimal accuracy loss for
    attention score computation.
    """

    def __init__(self, n_layers=32, n_heads=32, max_tokens=512, d_head=64):
        self.n_layers   = n_layers
        self.n_heads    = n_heads
        self.max_tokens = max_tokens
        self.d_head     = d_head
        self.pos        = 0
        self.n_tokens   = 0

        self.k = np.zeros((n_layers, n_heads, max_tokens, d_head), dtype=np.int8)
        self.v = np.zeros((n_layers, n_heads, max_tokens, d_head), dtype=np.int8)

        total_mb = (self.k.nbytes + self.v.nbytes) / (1024 ** 2)
        print(
            f"[KVCache] Allocated {total_mb:.1f} MB "
            f"({n_layers}L × {n_heads}H × {max_tokens}T × {d_head}D × INT8 × 2)"
        )

    def write(self, layer: int, keys, values) -> None:
        if TINYGRAD_AVAILABLE and isinstance(keys, Tensor):
            keys = keys.numpy()
        if TINYGRAD_AVAILABLE and isinstance(values, Tensor):
            values = values.numpy()
        self.k[layer, :, self.pos] = np.clip(keys,   -127, 127).astype(np.int8)
        self.v[layer, :, self.pos] = np.clip(values, -127, 127).astype(np.int8)

    def advance(self) -> None:
        self.n_tokens = min(self.n_tokens + 1, self.max_tokens)
        self.pos = (self.pos + 1) % self.max_tokens

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        valid  = self.n_tokens
        keys   = self.k[layer, :, :valid].astype(np.float32)
        values = self.v[layer, :, :valid].astype(np.float32)
        return keys, values

    def read_as_tensors(self, layer: int):
        if not TINYGRAD_AVAILABLE:
            raise RuntimeError("tinygrad not installed")
        keys, values = self.read(layer)
        return Tensor(keys), Tensor(values)

    def reset(self) -> None:
        self.pos      = 0
        self.n_tokens = 0

    def memory_used_mb(self) -> float:
        bytes_per_token = self.n_layers * self.n_heads * self.d_head * 2
        return (self.n_tokens * bytes_per_token) / (1024 ** 2)

    def utilisation(self) -> float:
        return self.n_tokens / self.max_tokens


class SparseAttentionKVCache(SparseKVCache):
    """
    Extends SparseKVCache with top-k sparse attention.

    Instead of attending to all cached tokens (O(n)), only attends to
    the top_k most relevant tokens scored by dot-product similarity.
    This reduces attention cost by max_tokens // top_k.
    """

    def __init__(self, top_k: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        print(
            f"[SparseAttention] top_k={top_k} — attends to "
            f"at most {top_k} of {self.max_tokens} cached tokens "
            f"({self.max_tokens // top_k}× savings)"
        )

    def sparse_read(self, layer: int, query) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the top_k most query-relevant key/value pairs.

        Args:
            layer: which transformer layer
            query: query tensor, shape (n_heads, d_head)

        Returns:
            (keys, values), each shape (n_heads, top_k, d_head)
        """
        if TINYGRAD_AVAILABLE and isinstance(query, Tensor):
            query = query.numpy()

        all_keys, all_values = self.read(layer)
        valid = self.n_tokens

        if valid <= self.top_k:
            return all_keys, all_values     # nothing to prune yet

        # Dot-product scores: (n_heads, valid)
        scores     = np.einsum("hd,hnd->hn", query, all_keys) / math.sqrt(self.d_head)
        avg_scores = scores.mean(axis=0)    # average across heads

        # Pick top_k indices (unsorted), then sort by score descending
        top_idx = np.argpartition(avg_scores, -self.top_k)[-self.top_k:]
        top_idx = top_idx[np.argsort(avg_scores[top_idx])[::-1]]

        return all_keys[:, top_idx, :], all_values[:, top_idx, :]

    def get_stats(self) -> dict:
        return {
            "tokens_cached":     self.n_tokens,
            "max_tokens":        self.max_tokens,
            "utilisation_pct":   round(self.utilisation() * 100, 1),
            "memory_used_mb":    round(self.memory_used_mb(), 2),
            "sparse_top_k":      self.top_k,
            "attention_savings": f"{self.max_tokens // self.top_k}×",
        }