#!/usr/bin/env python3
"""
GOVERNOR ORCHESTRATOR
Run multiple topologies on SAME data. Compare outcomes. Pick winner. Learn.
No single topology. Topology BATTLE ROYALE.

Data → Governor → Spawn N topologies → Run parallel → Score → Best wins
                        ↓
                   Transfer layer between runners
                   Basic compute kernel for shared ops
"""

from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 0.  COMPUTE KERNEL  —  shared primitives all topologies use
# ---------------------------------------------------------------------------

class ComputeKernel:
    """Basic ops. No pytorch. No tensorflow. Only numpy. Governor control."""

    @staticmethod
    def matmul(a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    @staticmethod
    def softmax(x: NDArray) -> NDArray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def topk(x: NDArray, k: int) -> Tuple[NDArray, NDArray]:
        idx = np.argpartition(x, -k)[-k:]
        return x[idx], idx

    @staticmethod
    def pairwise_dist(x: NDArray) -> NDArray:
        return np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)

    @staticmethod
    def pca_project(x: NDArray, dim: int) -> NDArray:
        cov = np.cov(x.T)
        vals, vecs = np.linalg.eigh(cov)
        return x @ vecs[:, -dim:]

    @staticmethod
    def kmeans(x: NDArray, k: int, iters: int = 10) -> Tuple[NDArray, NDArray]:
        centroids = x[np.random.choice(len(x), k, replace=False)]
        for _ in range(iters):
            dists = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            for i in range(k):
                mask = labels == i
                if mask.any():
                    centroids[i] = x[mask].mean(axis=0)
        return centroids, labels


# ---------------------------------------------------------------------------
# 1.  TOPOLOGY RUNNER  —  one competing algorithm
# ---------------------------------------------------------------------------

@dataclass
class TopologyResult:
    name: str
    output: NDArray
    score: float
    latency_ms: float
    topology_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TopologyRunner:
    """One topology = one manifold + one inference path."""

    def __init__(self, name: str, topology_type: str, kernel: ComputeKernel):
        self.name = name
        self.topology_type = topology_type
        self.kernel = kernel
        self.history: List[TopologyResult] = []

    def run(self, data: NDArray, task: str) -> TopologyResult:
        t0 = time.perf_counter()
        out, score, meta = self._compute(data, task)
        latency = (time.perf_counter() - t0) * 1000
        result = TopologyResult(self.name, out, score, latency, self.topology_type, meta)
        self.history.append(result)
        return result

    def _compute(self, data: NDArray, task: str) -> Tuple[NDArray, float, Dict]:
        """Dispatch by topology type. Each implements different geometry."""
        if self.topology_type == "euclidean":
            return self._euclidean(data, task)
        if self.topology_type == "spherical":
            return self._spherical(data, task)
        if self.topology_type == "hyperbolic_poincare":
            return self._hyperbolic(data, task)
        if self.topology_type == "grassmannian":
            return self._grassmannian(data, task)
        if self.topology_type == "product":
            return self._product(data, task)
        return self._fallback(data, task)

    def _euclidean(self, data: NDArray, task: str) -> Tuple[NDArray, float, Dict]:
        """Flat space. K-means + linear projection."""
        proj = self.kernel.pca_project(data, dim=min(8, data.shape[1]))
        cents, labels = self.kernel.kmeans(proj, k=min(8, len(proj) // 10 + 2))
        score = self._score(task, proj, labels, data)
        return proj, score, {"centroids": cents, "labels": labels}

    def _spherical(self, data: NDArray, task: str) -> Tuple[NDArray, float, Dict]:
        """L2 normalize to sphere. Angular k-means."""
        normed = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-9)
        # spherical: maximize angular separation
        sim = normed @ normed.T
        score = self._score(task, normed, sim, data)
        return normed, score, {"angular_sim": sim}

    def _hyperbolic(self, data: NDArray, task: str) -> Tuple[NDArray, float, Dict]:
        """Poincare ball. Map to disk via tanh scaling."""
        r = np.linalg.norm(data, axis=1, keepdims=True)
        disk = np.tanh(r) * data / (r + 1e-9)
        # hyperbolic distance proxy
        dists = self.kernel.pairwise_dist(disk)
        score = self._score(task, disk, dists, data)
        return disk, score, {"hyperbolic_dists": dists}

    def _grassmannian(self, data: NDArray, task: str) -> Tuple[NDArray, float, Dict]:
        """Subspace geometry. QR on local patches."""
        q, _ = np.linalg.qr(data.T)
        subspace = data @ q[:, : min(4, data.shape[1])]
        score = self._score(task, subspace, q, data)
        return subspace, score, {"subspace_basis": q}

    def _product(self, data: NDArray, task: str) -> Tuple[NDArray, float, Dict]:
        """S1 x R. Separate cyclic and linear components."""
        cyclic = np.stack([np.sin(data[:, 0]), np.cos(data[:, 0])], axis=1)
        linear = data[:, 1:]
        combined = np.concatenate([cyclic, linear], axis=1)
        score = self._score(task, combined, None, data)
        return combined, score, {"cyclic": cyclic, "linear": linear}

    def _fallback(self, data: NDArray, task: str) -> Tuple[NDArray, float, Dict]:
        return data, 0.5, {}

    def _score(self, task: str, embedding: NDArray, aux: Any, data: NDArray | None = None) -> float:
        """Task-specific scoring with REAL sklearn models. Higher = better."""
        # Prefer real data if available, else use embedding as features
        X = embedding
        # Try to infer target from data shape difference or aux
        y = aux if isinstance(aux, np.ndarray) and aux.ndim == 1 and len(aux) == len(X) else None

        if task == "cluster":
            if len(X) < 10 or X.shape[1] < 1:
                return 0.0
            try:
                from sklearn.metrics import silhouette_score
                from sklearn.cluster import KMeans
                k = min(8, max(2, len(X) // 20))
                km = KMeans(n_clusters=k, n_init=3, random_state=42)
                labels = km.fit_predict(X)
                if len(set(labels)) < 2:
                    return 0.0
                return float(silhouette_score(X, labels))
            except Exception:
                return 0.0

        if task == "regress":
            if y is None or len(set(y)) < 2:
                # No target — use reconstruction error as proxy
                recon = self.kernel.pca_project(X, dim=max(1, X.shape[1] // 2))
                error = np.mean((X - recon) ** 2)
                return float(1.0 / (1.0 + error))
            try:
                from sklearn.linear_model import Ridge
                from sklearn.model_selection import cross_val_score
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                model = Ridge(alpha=1.0)
                scores = cross_val_score(model, Xs, y, cv=min(3, len(X)), scoring="r2")
                return float(np.clip(np.mean(scores), -1.0, 1.0))
            except Exception:
                return 0.0

        if task == "classify":
            if y is None or len(set(y)) < 2:
                return 0.0
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import cross_val_score
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                model = LogisticRegression(max_iter=500, C=1.0)
                scores = cross_val_score(model, Xs, y, cv=min(3, len(X), min(np.bincount(y.astype(int)))), scoring="accuracy")
                return float(np.clip(np.mean(scores), 0.0, 1.0))
            except Exception:
                return 0.0

        if task == "compress":
            recon = self.kernel.pca_project(X, dim=max(1, X.shape[1] // 2))
            error = np.mean((X - recon) ** 2)
            return float(1.0 / (1.0 + error))

        if task == "transfer":
            dists = self.kernel.pairwise_dist(X[: min(len(X), 500)])
            smooth = np.mean(np.exp(-dists))
            return float(smooth)

        if task == "search":
            sample = X[: min(len(X), 300)]
            dists = self.kernel.pairwise_dist(sample)
            knn_local = np.mean(np.sort(dists, axis=1)[:, 1:4])
            return float(1.0 / (1.0 + knn_local))

        # default: variance explained
        return float(np.var(X) / (np.var(X) + 1.0))


# ---------------------------------------------------------------------------
# 2.  DATA TRANSFER BUS  —  move state between topologies
# ---------------------------------------------------------------------------

class TransferBus:
    """Governor moves partial results between runners. No copy waste."""

    def __init__(self):
        self.buffers: Dict[str, NDArray] = {}
        self.routes: List[Tuple[str, str, str]] = []  # (from, to, tag)

    def write(self, runner_name: str, tag: str, tensor: NDArray) -> None:
        self.buffers[f"{runner_name}:{tag}"] = tensor

    def read(self, runner_name: str, tag: str) -> Optional[NDArray]:
        return self.buffers.get(f"{runner_name}:{tag}")

    def route(self, from_runner: str, to_runner: str, tag: str) -> None:
        """Governor command: connect output of A to input of B."""
        self.routes.append((from_runner, to_runner, tag))
        key = f"{from_runner}:{tag}"
        if key in self.buffers:
            self.buffers[f"{to_runner}:{tag}_in"] = self.buffers[key]

    def clear(self) -> None:
        self.buffers.clear()
        self.routes.clear()


# ---------------------------------------------------------------------------
# 3.  OUTCOME COMPARATOR  —  who won?
# ---------------------------------------------------------------------------

class OutcomeComparator:
    """Score multiple TopologyResults. Rank. Pick winner."""

    def __init__(self, score_weight: float = 0.7, latency_weight: float = 0.3):
        self.score_weight = score_weight
        self.latency_weight = latency_weight

    def compare(self, results: List[TopologyResult]) -> List[TopologyResult]:
        """Return sorted best-first."""
        if not results:
            return []
        max_score = max(r.score for r in results)
        max_latency = max(r.latency_ms for r in results) + 1e-6
        def combined(r: TopologyResult) -> float:
            s = r.score / max_score if max_score > 0 else 0
            l = 1.0 - (r.latency_ms / max_latency)
            return self.score_weight * s + self.latency_weight * l
        return sorted(results, key=combined, reverse=True)

    def winner(self, results: List[TopologyResult]) -> TopologyResult:
        return self.compare(results)[0]


# ---------------------------------------------------------------------------
# 4.  GOVERNOR ORCHESTRATOR  —  the brain
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    topologies: List[str]
    task: str
    n_competitors: int = 3
    auto_transfer: bool = True
    learning_rate: float = 0.1
    checkpoint: Optional[Path] = None


class GovernorOrchestrator:
    """
    1. Accept data + task
    2. Spawn N topology runners
    3. Run all on same data
    4. Compare outcomes
    5. Winner takes gradient update
    6. Optional: transfer winner state to losers for warm start
    """

    def __init__(self, config: OrchestratorConfig):
        self.cfg = config
        self.kernel = ComputeKernel()
        self.bus = TransferBus()
        self.comparator = OutcomeComparator()
        self.runners: Dict[str, TopologyRunner] = {}
        self.topology_weights = {t: 1.0 / len(config.topologies) for t in config.topologies}
        self.history: List[Dict] = []
        self._build_runners()

    def _build_runners(self) -> None:
        for t in self.cfg.topologies:
            self.runners[t] = TopologyRunner(name=t, topology_type=t, kernel=self.kernel)

    def run_battle(self, data: NDArray) -> Dict[str, Any]:
        """Main entry. All topologies fight. One wins."""
        self.bus.clear()
        results: List[TopologyResult] = []

        # PHASE 1: RUN ALL
        for name, runner in self.runners.items():
            # if transfer enabled, seed with previous winner state
            if self.cfg.auto_transfer and self.history:
                prev_winner = self.history[-1].get("winner", "")
                if prev_winner:
                    xfer = self.bus.read(prev_winner, "embedding")
                    if xfer is not None and xfer.shape == data.shape:
                        data = 0.9 * data + 0.1 * xfer  # warm start blend
            res = runner.run(data, self.cfg.task)
            self.bus.write(name, "embedding", res.output)
            results.append(res)

        # PHASE 2: COMPARE
        ranked = self.comparator.compare(results)
        winner = ranked[0]
        loser = ranked[-1]

        # PHASE 3: UPDATE WEIGHTS (multiplicative weights update)
        for r in results:
            reward = 1.0 if r.name == winner.name else -0.5
            self.topology_weights[r.name] *= np.exp(self.cfg.learning_rate * reward)
        total = sum(self.topology_weights.values())
        self.topology_weights = {k: v / total for k, v in self.topology_weights.items()}

        # PHASE 4: TRANSFER WINNER TO LOSERS (if governor says so)
        if self.cfg.auto_transfer:
            self.bus.route(winner.name, loser.name, "embedding")

        record = {
            "winner": winner.name,
            "winner_score": winner.score,
            "winner_latency_ms": winner.latency_ms,
            "loser": loser.name,
            "loser_score": loser.score,
            "all_scores": {r.name: r.score for r in results},
            "weights": self.topology_weights.copy(),
        }
        self.history.append(record)
        return record

    def run_ensemble(self, data: NDArray) -> NDArray:
        """Blend outputs by learned weights instead of picking one."""
        outputs: Dict[str, NDArray] = {}
        for name, runner in self.runners.items():
            res = runner.run(data, self.cfg.task)
            outputs[name] = res.output

        # align dimensions via PCA to common space
        common_dim = min(o.shape[1] for o in outputs.values())
        aligned = []
        for name, out in outputs.items():
            a = self.kernel.pca_project(out, common_dim)
            aligned.append(self.topology_weights[name] * a)
        return np.sum(aligned, axis=0)

    def run_cascade(self, data: NDArray, chain: List[str]) -> NDArray:
        """Sequential pipeline: topology A output → topology B input."""
        x = data
        for name in chain:
            if name not in self.runners:
                raise ValueError(f"No runner named {name}")
            res = self.runners[name].run(x, self.cfg.task)
            x = res.output
        return x

    def report(self) -> str:
        if not self.history:
            return "No battles."
        wins = {}
        for h in self.history:
            w = h["winner"]
            wins[w] = wins.get(w, 0) + 1
        best = max(wins, key=wins.get)
        return f"Battles: {len(self.history)} | Best topology: {best} ({wins[best]} wins) | Weights: {self.topology_weights}"

    def save(self, path: Path) -> None:
        payload = {
            "weights": self.topology_weights,
            "history": self.history,
            "config": {
                "topologies": self.cfg.topologies,
                "task": self.cfg.task,
                "n_competitors": self.cfg.n_competitors,
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# 5.  ADVANCED: TOPOLOGY FUSION  —  best of all worlds
# ---------------------------------------------------------------------------

class TopologyFusion:
    """Not ensemble. FUSION. Learn cross-topology attention."""

    def __init__(self, topology_names: List[str], embed_dim: int = 64):
        self.names = topology_names
        self.dim = embed_dim
        rng = np.random.default_rng(42)
        # attention: query = euclidean, keys = all, values = all
        self.Wq = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * 0.1
        self.Wk = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * 0.1
        self.Wv = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * 0.1

    def fuse(self, outputs: Dict[str, NDArray]) -> NDArray:
        # project all to common dim
        common = min(o.shape[1] for o in outputs.values())
        mats = []
        for name in self.names:
            o = outputs[name]
            if o.shape[1] > common:
                o = o[:, :common]
            elif o.shape[1] < common:
                pad = np.zeros((o.shape[0], common - o.shape[1]))
                o = np.concatenate([o, pad], axis=1)
            mats.append(o)
        stack = np.stack(mats, axis=1)  # batch x n_topologies x dim

        # self-attention across topologies
        Q = stack @ self.Wq  # (batch, n_top, dim)
        K = stack @ self.Wk
        V = stack @ self.Wv
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.dim)
        attn = np.exp(scores - np.max(scores, axis=2, keepdims=True))
        attn /= attn.sum(axis=2, keepdims=True)
        fused = (attn @ V).mean(axis=1)  # average over topology heads
        return fused


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("GOVERNOR ORCHESTRATOR — BATTLE ROYALE")
    print("=" * 70)

    cfg = OrchestratorConfig(
        topologies=["euclidean", "spherical", "hyperbolic_poincare", "grassmannian", "product"],
        task="cluster",
        auto_transfer=True,
        learning_rate=0.2,
    )
    gov = GovernorOrchestrator(cfg)

    # synthetic datasets
    datasets = []

    # ring = spherical should win
    t = np.linspace(0, 2 * np.pi, 800)
    ring = np.stack([np.cos(t), np.sin(t)], axis=1) + np.random.randn(800, 2) * 0.03
    datasets.append(("ring", ring))

    # tree = hyperbolic should win
    tree = np.random.randn(800, 16)
    tree[:, 0] = np.random.exponential(1.0, 800)
    datasets.append(("tree", tree))

    # blob = euclidean should win
    blob = np.random.randn(800, 8)
    datasets.append(("blob", blob))

    # product = cyclic + linear
    cyclic_lin = np.column_stack([np.linspace(0, 4 * np.pi, 800), np.random.randn(800) * 2])
    datasets.append(("cyclic_linear", cyclic_lin))

    for tag, data in datasets:
        print(f"\n--- DATASET: {tag} ---")
        for round_num in range(3):
            rec = gov.run_battle(data)
            print(f"  Round {round_num + 1}: WINNER={rec['winner']:18} score={rec['winner_score']:.3f}  "
                  f"loser={rec['loser']:18} score={rec['loser_score']:.3f}")

    print("\n" + "=" * 70)
    print(gov.report())

    # ENSEMBLE demo
    print("\n--- ENSEMBLE BLEND ---")
    blend = gov.run_ensemble(blob)
    print(f"Ensemble output shape: {blend.shape}")

    # CASCADE demo
    print("\n--- CASCADE pipeline ---")
    cascade_out = gov.run_cascade(blob, ["euclidean", "spherical"])
    print(f"Cascade output shape: {cascade_out.shape}")

    # FUSION demo
    print("\n--- FUSION (attention across topologies) ---")
    fusion = TopologyFusion(cfg.topologies, embed_dim=8)
    outs = {name: gov.runners[name].run(blob, cfg.task).output for name in cfg.topologies}
    fused = fusion.fuse(outs)
    print(f"Fused output shape: {fused.shape}")

    gov.save(Path("orchestrator_state.json"))
    print("\nState saved to orchestrator_state.json")
    print("DONE.")
