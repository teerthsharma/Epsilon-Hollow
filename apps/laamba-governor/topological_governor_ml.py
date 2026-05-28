#!/usr/bin/env python3
"""
TOPOLOGICAL GOVERNOR — ML-PICKS-TOPOLOGY, NO MATHS BOOKS.
Invented by Pickle Rick.  Kill Lagirthms.  Learn instead.

This governor looks at raw data, extracts topological vitals,
then predicts WHICH manifold / topology will crush the task.
No hand-wavy formula from dusty textbook.  Trained end-to-end.

Usage:
    gov = TopologicalGovernor()
    rec = gov.predict_topology(your_dataset)   # → ManifoldConfig
    loss = rec.train_embedding(model, data)    # eval
    gov.update(rec, loss)                      # online learning
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 1.  TOPOLOGY ZOO  —  what governor can pick from
# ---------------------------------------------------------------------------

MANIFOLD_CATALOG = [
    "euclidean",
    "spherical",
    "hyperbolic_poincare",
    "hyperboloid",
    "product_s1_x_r",
    "grassmannian",
    "mixed_curvature",
]

# ---------------------------------------------------------------------------
# 2.  DATA VITALS  —  topological feature extraction (cheap & dirty)
# ---------------------------------------------------------------------------

class VitalsExtractor:
    """Strip data naked.  Measure topological bones.  No ripser here — too slow
    for governor hot-path.  Use cheap spectral + geometric proxies."""

    def __call__(self, X: NDArray) -> NDArray:
        n, d = X.shape
        feats = []

        # 0.  raw shape
        feats += [np.log1p(n), np.log1p(d), n / max(d, 1)]

        # 1.  intrinsic dimension (ML-estimate via nearest-neighbour)
        feats += [self._intrinsic_dim(X)]

        # 2.  pairwise distance stats
        D = self._pairwise_dists(X[: min(n, 2048)])
        feats += [np.mean(D), np.std(D), np.percentile(D, 95) / (np.percentile(D, 5) + 1e-9)]

        # 3.  spectral gap (graph Laplacian on k-NN)
        feats += [self._spectral_gap(X)]

        # 4.  cluster count (quick k-means knee)
        feats += [self._knee_clusters(X)]

        # 5.  local curvature proxy (variance of PCA eigenvalue ratios)
        feats += [self._curvature_proxy(X)]

        # 6.  graph diameter / small-world coeff
        feats += [self._small_world_coeff(D)]

        # 7.  data sparsity
        feats += [np.mean(X == 0.0), np.mean(np.isnan(X))]

        return np.asarray(feats, dtype=np.float32)

    @staticmethod
    def _intrinsic_dim(X: NDArray, k: int = 5) -> float:
        """Levina-Bickel estimator (simplified).  Measures how many dims
        data REALLY lives in, not padded ambient space."""
        n = min(len(X), 4096)
        idx = np.random.choice(len(X), n, replace=False)
        S = X[idx]
        # k-th nearest neighbour distances
        dists = np.sort(((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2), axis=1)
        rk = dists[:, k] + 1e-9
        return float((1.0 / (np.mean(np.log(rk[:, None] / (dists[:, 1:k] + 1e-9))) + 1e-9)))

    @staticmethod
    def _pairwise_dists(X: NDArray) -> NDArray:
        Y = X.reshape(len(X), 1, -1)
        return np.sqrt(np.sum((Y - Y.transpose(1, 0, 2)) ** 2, axis=2))

    @staticmethod
    def _spectral_gap(X: NDArray, k: int = 15) -> float:
        """Build k-NN graph, Laplacian, return λ₂ / λ_max.
        Big gap = well-connected, one cluster.
        Small gap = disconnected, many components."""
        n = min(len(X), 2048)
        idx = np.random.choice(len(X), n, replace=False)
        S = X[idx]
        D2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)
        knn = np.argsort(D2, axis=1)[:, 1 : k + 1]
        W = np.zeros((n, n))
        for i, neigh in enumerate(knn):
            W[i, neigh] = 1.0
            W[neigh, i] = 1.0
        deg = W.sum(axis=1)
        L = np.diag(deg) - W
        vals = np.linalg.eigvalsh(L)
        vals = np.sort(vals)
        return float(vals[1] / (vals[-1] + 1e-9))

    @staticmethod
    def _knee_clusters(X: NDArray, max_k: int = 10) -> float:
        """Elbow method on k-means inertia.  Returns k at knee."""
        from sklearn.cluster import KMeans
        n = min(len(X), 4096)
        idx = np.random.choice(len(X), n, replace=False)
        S = X[idx]
        inertias = []
        for k in range(1, max_k + 1):
            km = KMeans(n_clusters=k, n_init=3, random_state=42)
            km.fit(S)
            inertias.append(km.inertia_)
        # second-derivative knee
        deltas = np.diff(inertias, 2)
        knee = int(np.argmax(deltas)) + 2 if len(deltas) else 1
        return float(knee)

    @staticmethod
    def _curvature_proxy(X: NDArray, k: int = 20) -> float:
        """Local PCA eigenvalue ratio variance.  High = curved / twisted manifold.
        Low = flat."""
        n = len(X)
        ratios = []
        for i in np.random.choice(n, min(n, 512), replace=False):
            dists = np.sum((X - X[i]) ** 2, axis=1)
            neigh = np.argsort(dists)[1 : k + 1]
            loc = X[neigh] - X[i]
            cov = loc.T @ loc / k
            vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            ratios.append(vals[1] / (vals[0] + 1e-9))
        return float(np.std(ratios))

    @staticmethod
    def _small_world_coeff(D: NDArray) -> float:
        """Ratio mean path length / clustering of random graph.
        D = distance matrix.  Crude but fast."""
        n = D.shape[0]
        thresh = np.median(D)
        A = (D < thresh).astype(float)
        np.fill_diagonal(A, 0)
        # clustering coeff (local)
        C = []
        for i in range(min(n, 256)):
            neigh = np.where(A[i])[0]
            if len(neigh) < 2:
                continue
            sub = A[np.ix_(neigh, neigh)]
            C.append(sub.sum() / (len(neigh) * (len(neigh) - 1)))
        clustering = np.mean(C) if C else 0.0
        # avg shortest path (via BFS on threshold graph)
        L = []
        for s in np.random.choice(n, min(n, 128), replace=False):
            dist = VitalsExtractor._bfs(A, s)
            L.append(np.mean([v for v in dist if v > 0]))
        path_len = np.mean(L) if L else 1.0
        return float(path_len / (clustering + 1e-3))

    @staticmethod
    def _bfs(A: NDArray, start: int) -> List[int]:
        n = A.shape[0]
        dist = [-1] * n
        dist[start] = 0
        q = [start]
        while q:
            u = q.pop(0)
            for v in np.where(A[u])[0]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist


# ---------------------------------------------------------------------------
# 3.  POLICY NET  —  tiny MLP brain picks manifold
# ---------------------------------------------------------------------------

class PolicyNet:
    """Two-layer MLP.  No ResNet.  No transformer.  Governor must be FAST."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, out_dim)).astype(np.float32) * 0.1
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self._cache: dict = {}

    def forward(self, x: NDArray) -> NDArray:
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)          # ReLU
        z2 = a1 @ self.W2 + self.b2
        # soft-max
        e = np.exp(z2 - np.max(z2))
        probs = e / e.sum()
        self._cache = {"x": x, "z1": z1, "a1": a1, "probs": probs}
        return probs

    def backward(self, target_idx: int, lr: float = 0.01) -> None:
        """Cross-entropy grad.  Online update."""
        c = self._cache
        probs = c["probs"]
        dz2 = probs.copy()
        dz2[target_idx] -= 1.0
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (c["z1"] > 0).astype(float)

        self.W2 -= lr * np.outer(c["a1"], dz2)
        self.b2 -= lr * dz2
        self.W1 -= lr * np.outer(c["x"], dz1)
        self.b1 -= lr * dz1

    def save(self, path: Path) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: Path, in_dim: int, out_dim: int) -> "PolicyNet":
        data = np.load(path)
        net = cls(in_dim, out_dim)
        net.W1 = data["W1"]
        net.b1 = data["b1"]
        net.W2 = data["W2"]
        net.b2 = data["b2"]
        return net


# ---------------------------------------------------------------------------
# 4.  MANIFOLD CONFIG  —  what governor spits out
# ---------------------------------------------------------------------------

@dataclass
class ManifoldConfig:
    name: str
    dim: int
    curvature: float          # κ: 0 = flat, +1 = sphere, -1 = hyperbolic
    learning_rate: float
    batch_size: int
    epochs: int
    # topology-specific hyperparams
    extra: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 5.  GOVERNOR  —  the brains
# ---------------------------------------------------------------------------

class TopologicalGovernor:
    """
    Learns best topology for dataset.  No formula.  Only gradients + pain.

    State space:
        - observe dataset vitals  (cheap spectral stuff)
        - policy net predicts manifold distribution
        - sample one manifold
        - run downstream task, get loss
        - REINFORCE-like update on policy
    """

    def __init__(
        self,
        manifolds: Sequence[str] | None = None,
        hidden_dim: int = 64,
        lr: float = 0.02,
        entropy_bonus: float = 0.05,
        checkpoint: Path | None = None,
    ):
        self.catalog = list(manifolds or MANIFOLD_CATALOG)
        self.extractor = VitalsExtractor()
        self.feat_dim = None          # set on first predict
        self.lr = lr
        self.entropy_bonus = entropy_bonus
        self.history: List[dict] = []
        self._checkpoint = checkpoint

        # policy init deferred until first datum seen
        self.policy: PolicyNet | None = None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def predict_topology(self, X: NDArray, temperature: float = 1.0, force: str | None = None) -> ManifoldConfig:
        """Look at data.  Pick manifold.  Return config.  O(data) not O(book)."""
        feats = self.extractor(X)
        if self.policy is None:
            self.feat_dim = len(feats)
            self.policy = PolicyNet(self.feat_dim, len(self.catalog))
            if self._checkpoint and self._checkpoint.exists():
                self.policy = PolicyNet.load(self._checkpoint, self.feat_dim, len(self.catalog))

        probs = self.policy.forward(feats)
        # temperature sharpen / soften
        logit = np.log(probs + 1e-9) / temperature
        e = np.exp(logit - np.max(logit))
        tempered = e / e.sum()

        if force and force in self.catalog:
            choice = self.catalog.index(force)
            name = force
        else:
            choice = int(np.random.choice(len(self.catalog), p=tempered))
            name = self.catalog[choice]

        cfg = self._default_config(name, X.shape[1])
        cfg.extra["governor_probs"] = {k: float(v) for k, v in zip(self.catalog, probs)}
        cfg.extra["governor_choice_idx"] = choice
        self._last_choice = choice
        self._last_feats = feats
        return cfg

    def update(self, config: ManifoldConfig, loss: float, reward_scale: float = 1.0) -> None:
        """Tell governor how it did.  Loss ↓ = reward ↑.  Policy learns."""
        if self.policy is None or not hasattr(self, "_last_choice"):
            raise RuntimeError("predict_topology() before update(), dummy.")

        # reward = negative normalized loss  (bigger = better)
        reward = -loss * reward_scale

        # REINFORCE gradient: ∇ log π(a|s) * R
        # We already did forward.  Backward with target = chosen action.
        # But we want to encourage actions with high reward, discourage low.
        # Hack: scale lr by reward.
        scaled_lr = self.lr * np.clip(reward, -1.0, 1.0)
        self.policy.backward(self._last_choice, lr=scaled_lr)

        # entropy regularization (exploration bonus)
        probs = self.policy._cache["probs"]
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        # tiny random nudge toward uniformity if entropy collapses
        if entropy < 0.5:
            self.policy.b2 += self.entropy_bonus * (1.0 / len(self.catalog) - probs)

        self.history.append({
            "manifold": config.name,
            "loss": loss,
            "reward": reward,
            "entropy": float(entropy),
        })

        if self._checkpoint:
            self.policy.save(self._checkpoint)

    def rank_manifolds(self, X: NDArray) -> List[tuple]:
        """Return (manifold, prob) sorted by policy confidence.  No sampling."""
        feats = self.extractor(X)
        if self.policy is None:
            self.feat_dim = len(feats)
            self.policy = PolicyNet(self.feat_dim, len(self.catalog))
        probs = self.policy.forward(feats)
        return sorted(
            zip(self.catalog, probs),
            key=lambda t: t[1],
            reverse=True,
        )

    def report(self) -> str:
        """Governor battle log.  See what topology won most."""
        if not self.history:
            return "No battles fought yet.  Governor virgin."
        from collections import Counter
        wins = Counter(h["manifold"] for h in self.history)
        best = min(self.history, key=lambda h: h["loss"])
        return (
            f"Battles: {len(self.history)}\n"
            f"Most picked: {wins.most_common(1)[0]}\n"
            f"Best loss: {best['loss']:.4f} on {best['manifold']}\n"
            f"Avg reward: {np.mean([h['reward'] for h in self.history]):.4f}"
        )

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _default_config(self, name: str, ambient_dim: int) -> ManifoldConfig:
        """Hand-tuned defaults per topology.  Can override later."""
        dim = max(2, min(ambient_dim, 64))
        if name == "euclidean":
            return ManifoldConfig(name, dim, 0.0, 1e-3, 256, 100, {})
        if name == "spherical":
            return ManifoldConfig(name, dim, +1.0, 5e-4, 128, 200, {"proj": "stereographic"})
        if name == "hyperbolic_poincare":
            return ManifoldConfig(name, dim, -1.0, 1e-3, 128, 200, {"proj": "poincare", "c": 1.0})
        if name == "hyperboloid":
            return ManifoldConfig(name, dim, -1.0, 1e-3, 128, 200, {"proj": "hyperboloid", "c": 1.0})
        if name == "product_s1_x_r":
            return ManifoldConfig(name, dim, 0.0, 1e-3, 256, 150, {"factors": ["circle", "euclidean"]})
        if name == "grassmannian":
            return ManifoldConfig(name, dim, 0.0, 5e-4, 64, 300, {"rank": 2})
        if name == "mixed_curvature":
            return ManifoldConfig(name, dim, 0.0, 1e-3, 128, 250, {"num_components": 3})
        # fallback
        return ManifoldConfig(name, dim, 0.0, 1e-3, 256, 100, {})


# ---------------------------------------------------------------------------
# 6.  MOCK DOWNSTREAM  —  so you can train governor standalone
# ---------------------------------------------------------------------------

def mock_downstream_task(X: NDArray, y: NDArray | None, cfg: ManifoldConfig) -> float:
    """
    Fake embedding + k-NN accuracy.  Replace with YOUR model.
    Returns LOSS (lower = better).
    """
    n = len(X)
    # pretend embedding quality depends on manifold curvature match
    # (this is where YOU plug real manifold embedding code)
    rng = np.random.default_rng(hash(cfg.name) % 2**31)
    noise = rng.standard_normal(X.shape) * 0.1
    embedded = X + noise

    # synthetic accuracy: spherical good for cyclic data, hyperbolic for tree-like
    fake_score = 0.5
    if cfg.name == "spherical" and _is_cyclic(X):
        fake_score = 0.92
    elif cfg.name == "hyperbolic_poincare" and _is_tree_like(X):
        fake_score = 0.90
    elif cfg.name == "euclidean" and not (_is_cyclic(X) or _is_tree_like(X)):
        fake_score = 0.88

    # add variance so governor has signal to learn
    fake_score += rng.normal(0, 0.05)
    return 1.0 - fake_score


def _is_cyclic(X: NDArray) -> bool:
    # very crude: high spectral gap + low knee clusters = cyclic ring
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    knn = np.argsort(D, axis=1)[:, 1:4]
    # if local dimension ~ 1 and connected, guess cyclic
    return float(np.mean(np.std(X, axis=0)) < 0.3)


def _is_tree_like(X: NDArray) -> bool:
    # crude: power-law degree distribution in k-NN graph
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    A = (D < np.median(D)).astype(float)
    np.fill_diagonal(A, 0)
    degrees = A.sum(axis=1)
    # tree-like if many low-degree nodes, few hubs
    return float(np.percentile(degrees, 90) / (np.percentile(degrees, 10) + 1) > 5.0)


# ---------------------------------------------------------------------------
# 7.  DEMO  —  train governor on synthetic tasks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PICKLE RICK'S TOPOLOGICAL GOVERNOR")
    print("Training on synthetic dirt.  No maths books harmed.")
    print("=" * 60)

    governor = TopologicalGovernor(lr=0.05, entropy_bonus=0.02)

    # synthetic dataset zoo
    datasets = []
    for _ in range(30):
        # cyclic ring (spherical should win)
        t = np.linspace(0, 2 * np.pi, 500)
        ring = np.stack([np.cos(t), np.sin(t)], axis=1) + np.random.randn(500, 2) * 0.05
        datasets.append(("ring", ring))

        # tree-like hierarchical (hyperbolic should win)
        tree = np.random.randn(500, 16)
        tree[:, 0] = np.random.exponential(1.0, 500)
        datasets.append(("tree", tree))

        # blob (euclidean should win)
        blob = np.random.randn(500, 8)
        datasets.append(("blob", blob))

    random.shuffle(datasets)

    for tag, X in datasets:
        cfg = governor.predict_topology(X, temperature=1.2)
        loss = mock_downstream_task(X, None, cfg)
        governor.update(cfg, loss)
        print(f"  {tag:6} → {cfg.name:20} loss={loss:.3f}")

    print("\n" + "=" * 60)
    print(governor.report())
    print("\nTop picks for a NEW ring dataset:")
    t = np.linspace(0, 2 * np.pi, 300)
    new_ring = np.stack([np.cos(t), np.sin(t)], axis=1) + np.random.randn(300, 2) * 0.03
    for name, prob in governor.rank_manifolds(new_ring)[:3]:
        print(f"  {name:20}  p={prob:.3f}")

    # save brain
    governor.policy.save(Path("governor_brain.npz"))
    print("\nGovernor brain dumped to governor_brain.npz")
    print("WUBBA LUBBA DUB DUB!")
