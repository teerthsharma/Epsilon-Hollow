import numpy as np
from tinygrad.tensor import Tensor
import math


class INT8KVCache:
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
        print(f"[INT8KVCache] Allocated {total_mb:.1f} MB "
              f"({n_layers}L x {n_heads}H x {max_tokens}T x {d_head}D x INT8 x 2)")

    def write(self, layer, keys, values):
        if isinstance(keys, Tensor):
            keys = keys.numpy()
        if isinstance(values, Tensor):
            values = values.numpy()
        self.k[layer, :, self.pos] = np.clip(keys,   -127, 127).astype(np.int8)
        self.v[layer, :, self.pos] = np.clip(values, -127, 127).astype(np.int8)

    def advance(self):
        self.n_tokens = min(self.n_tokens + 1, self.max_tokens)
        self.pos = (self.pos + 1) % self.max_tokens

    def read(self, layer):
        valid = self.n_tokens
        keys   = self.k[layer, :, :valid].astype(np.float32)
        values = self.v[layer, :, :valid].astype(np.float32)
        return keys, values

    def read_as_tensors(self, layer):
        keys, values = self.read(layer)
        return Tensor(keys), Tensor(values)

    def reset(self):
        self.pos      = 0
        self.n_tokens = 0

    def memory_used_mb(self):
        bytes_per_token = self.n_layers * self.n_heads * self.d_head * 2
        return (self.n_tokens * bytes_per_token) / (1024 ** 2)

    def utilisation(self):
        return self.n_tokens / self.max_tokens


class SparseAttentionKVCache(INT8KVCache):
    def __init__(self, top_k=64, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        print(f"[SparseAttention] top_k={top_k} — attends to "
              f"at most {top_k} of {self.max_tokens} cached tokens")

    def sparse_read(self, layer, query):
        if isinstance(query, Tensor):
            query = query.numpy()
        all_keys, all_values = self.read(layer)
        valid = self.n_tokens
        if valid <= self.top_k:
            return all_keys, all_values
        scores = np.sum(query[:, np.newaxis, :] * all_keys, axis=-1)
        scores = scores / math.sqrt(self.d_head)
        avg_scores = scores.mean(axis=0)
        top_k_idx = np.argpartition(avg_scores, -self.top_k)[-self.top_k:]
        top_k_idx = top_k_idx[np.argsort(avg_scores[top_k_idx])[::-1]]
        return all_keys[:, top_k_idx, :], all_values[:, top_k_idx, :]

    def get_stats(self):
        return {
            "tokens_cached":     self.n_tokens,
            "max_tokens":        self.max_tokens,
            "utilisation_pct":   round(self.utilisation() * 100, 1),
            "memory_used_mb":    round(self.memory_used_mb(), 2),
            "sparse_top_k":      self.top_k,
            "attention_savings": f"{self.max_tokens // self.top_k}x",
        }
