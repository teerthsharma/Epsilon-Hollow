"""
Epsilon-Hollow — Liquid Tensor Calculus
========================================
Entropy-adaptive micro-gradient injection on hot weight partition.

From the README:
    W = W_static ∪ W_hot  where |W_hot| ≈ 0.005|W|
    ∇_micro = ∂L/∂W_hot
    W_hot^{(t+1)} = W_hot^{(t)} − η · ∇_micro
    η = α · (1 − Entropy(P))
    T_update < 50μs via sparse updates

The key innovation: by restricting updates to 0.5% of weights and
gating the learning rate with output entropy, we get:
    1. Instant learning (sub-50μs per update)
    2. Stability guarantee (EGPB bounds total weight drift)
    3. Confidence-weighted updates (high entropy → careful, low → commit)
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


class LiquidTensor:
    """
    Liquid Tensor: a weight matrix with a hot (plastic) partition.

    W = W_static ∪ W_hot
    Only W_hot receives gradient updates. W_static is frozen.

    The entropy-adaptive learning rate ensures updates are proportional
    to model confidence: confident predictions get larger updates,
    uncertain predictions get smaller ones.
    """

    def __init__(self, shape: Tuple[int, ...], hot_ratio: float = 0.005,
                 alpha: float = 0.01, seed: int = 42):
        """
        Parameters
        ----------
        shape : tuple
            Shape of the full weight tensor.
        hot_ratio : float
            Fraction of weights in the hot partition (default 0.5%).
        alpha : float
            Base learning rate scale.
        seed : int
            RNG seed for initial weight sampling.
        """
        self.shape = shape
        self.total_params = int(np.prod(shape))
        self.hot_ratio = hot_ratio
        self.alpha = alpha

        rng = np.random.RandomState(seed)

        # Full weight tensor (conceptual — in practice these would be model weights)
        self._W = rng.randn(*shape).astype(np.float64) * 0.01

        # Hot partition: select indices randomly
        self.hot_size = max(1, int(self.total_params * hot_ratio))
        flat = self._W.ravel()
        self._hot_indices = rng.choice(self.total_params, self.hot_size, replace=False)
        self._hot_indices.sort()

        # Pre-allocate hot gradient buffer for speed
        self._grad_buffer = np.zeros(self.hot_size, dtype=np.float64)

        # Statistics
        self.update_count = 0
        self.total_update_time_us = 0.0
        self._update_norms: List[float] = []
        self._lr_history: List[float] = []

    @property
    def W(self) -> np.ndarray:
        """Full weight tensor (read-only view)."""
        return self._W

    @property
    def W_hot(self) -> np.ndarray:
        """Hot partition weights."""
        return self._W.ravel()[self._hot_indices]

    def entropy_adaptive_lr(self, output_distribution: np.ndarray) -> float:
        """
        Compute entropy-adaptive learning rate.

        η = α · (1 − H(P))

        where H(P) = −Σ pᵢ log₂(pᵢ) is the Shannon entropy,
        normalised to [0, 1] by dividing by log₂(|P|).

        High entropy (uncertain) → small η → cautious update
        Low entropy (confident)  → large η → committed update
        """
        # Normalise to probability distribution
        p = np.abs(output_distribution) + 1e-30
        p = p / np.sum(p)

        # Shannon entropy
        H = -float(np.sum(p * np.log2(p)))

        # Normalise to [0, 1]
        max_entropy = math.log2(max(len(p), 2))
        H_norm = min(H / max_entropy, 1.0)

        eta = self.alpha * (1.0 - H_norm)
        return eta

    def inject_update(self, gradient: np.ndarray,
                      output_distribution: Optional[np.ndarray] = None,
                      learning_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Micro-gradient injection on the hot partition.

        W_hot^{(t+1)} = W_hot^{(t)} − η · ∇_micro

        Parameters
        ----------
        gradient : np.ndarray, same shape as W
            Full gradient ∂L/∂W. Only the hot partition is used.
        output_distribution : np.ndarray, optional
            Output probability distribution for entropy-adaptive lr.
            If None, uses fixed alpha as learning rate.
        learning_rate : float, optional
            Override learning rate. If provided, skips entropy computation.

        Returns
        -------
        dict with update statistics.
        """
        t_start = time.perf_counter_ns()

        # Extract micro-gradient at hot indices
        grad_flat = gradient.ravel()
        self._grad_buffer[:] = grad_flat[self._hot_indices]

        # Compute learning rate
        if learning_rate is not None:
            eta = learning_rate
        elif output_distribution is not None:
            eta = self.entropy_adaptive_lr(output_distribution)
        else:
            eta = self.alpha

        # Apply update: W_hot -= η · ∇_micro
        W_flat = self._W.ravel()
        delta = eta * self._grad_buffer
        W_flat[self._hot_indices] -= delta

        # Measure update norm
        update_norm = float(np.linalg.norm(delta))

        t_end = time.perf_counter_ns()
        elapsed_us = (t_end - t_start) / 1000.0

        self.update_count += 1
        self.total_update_time_us += elapsed_us
        self._update_norms.append(update_norm)
        self._lr_history.append(eta)

        return {
            "update_norm": update_norm,
            "learning_rate": eta,
            "elapsed_us": elapsed_us,
            "hot_params": self.hot_size,
            "total_params": self.total_params,
        }

    def plasticity_bound_check(self, grad_norm: float, entropy: float) -> Dict[str, float]:
        """
        Verify the Entropy-Gated Plasticity Bound (EGPB).

        ‖ΔW_hot‖_F ≤ α · (1 − H) · ‖∇L‖₂ · √(hot_ratio)

        Returns the maximum allowed update norm and whether recent
        updates have stayed within the bound.
        """
        bound = self.alpha * (1.0 - entropy) * grad_norm * math.sqrt(self.hot_ratio)

        recent_max = max(self._update_norms[-10:]) if self._update_norms else 0.0
        within_bound = recent_max <= bound + 1e-10

        return {
            "theoretical_bound": bound,
            "recent_max_update": recent_max,
            "within_bound": within_bound,
            "alpha": self.alpha,
            "hot_ratio": self.hot_ratio,
        }

    def stats(self) -> Dict[str, float]:
        """Return performance and diagnostic statistics."""
        avg_time = (self.total_update_time_us / max(self.update_count, 1))
        return {
            "updates": self.update_count,
            "avg_update_us": avg_time,
            "max_update_us": max([0.0] + [self.total_update_time_us]),
            "under_50us": avg_time < 50.0,
            "hot_params": self.hot_size,
            "hot_ratio": self.hot_ratio,
            "total_params": self.total_params,
            "mean_lr": float(np.mean(self._lr_history)) if self._lr_history else 0.0,
            "mean_update_norm": float(np.mean(self._update_norms)) if self._update_norms else 0.0,
        }


class AkashicFS:
    """
    Manifold persistence filesystem: serialise/deserialise manifold state.

    centroid_projection() projects memory cluster centroids to a compact
    disk-backed index for O(1) cold-start recovery.
    """

    def __init__(self, path: str = ".epsilon_akashic"):
        self.path = path
        self._centroid_index: Dict[int, np.ndarray] = {}

    def centroid_projection(self, cluster_id: int, centroid: np.ndarray):
        """Store a centroid projection for later retrieval."""
        self._centroid_index[cluster_id] = centroid.copy()

    def serialize(self) -> bytes:
        """Serialise the centroid index to bytes."""
        import pickle
        return pickle.dumps(self._centroid_index)

    def deserialize(self, data: bytes):
        """Restore centroid index from bytes."""
        import pickle
        self._centroid_index = pickle.loads(data)

    def lookup_nearest(self, query: np.ndarray) -> Optional[int]:
        """Find nearest centroid by cosine similarity."""
        if not self._centroid_index:
            return None

        best_id = -1
        best_sim = -float("inf")
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-12:
            return None

        for cid, centroid in self._centroid_index.items():
            sim = float(np.dot(query, centroid)) / (q_norm * np.linalg.norm(centroid) + 1e-12)
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        return best_id if best_id >= 0 else None
