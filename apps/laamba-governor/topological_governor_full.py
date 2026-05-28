#!/usr/bin/env python3
"""
TOPOLOGICAL GOVERNOR — FULL IMPLEMENTATION, ALL USECASES IN CODE.
No markdown. No stories. Only code.

Usage:
    from topological_governor_full import GovernorUsecases
    gu = GovernorUsecases()
    result = gu.cv_image_manifold(train_images, val_images)
    result = gu.nlp_embedding_space(cooc_matrix, corpus)
    result = gu.graph_neural_net(edge_index, node_features, labels)
    ... etc ...
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 0.  BASE GOVERNOR (from topological_governor_ml.py, inlined so one file)
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


class VitalsExtractor:
    def __call__(self, X: NDArray) -> NDArray:
        n, d = X.shape
        feats = [np.log1p(n), np.log1p(d), n / max(d, 1)]
        feats += [self._intrinsic_dim(X)]
        D = self._pairwise_dists(X[: min(n, 2048)])
        feats += [np.mean(D), np.std(D), np.percentile(D, 95) / (np.percentile(D, 5) + 1e-9)]
        feats += [self._spectral_gap(X)]
        feats += [self._knee_clusters(X)]
        feats += [self._curvature_proxy(X)]
        feats += [self._small_world_coeff(D)]
        feats += [np.mean(X == 0.0), np.mean(np.isnan(X))]
        return np.asarray(feats, dtype=np.float32)

    @staticmethod
    def _intrinsic_dim(X: NDArray, k: int = 5) -> float:
        n = min(len(X), 4096)
        idx = np.random.choice(len(X), n, replace=False)
        S = X[idx]
        dists = np.sort(((S[:, None, :] - S[None, :, :]) ** 2).sum(axis=2), axis=1)
        rk = dists[:, k] + 1e-9
        return float(1.0 / (np.mean(np.log(rk[:, None] / (dists[:, 1:k] + 1e-9))) + 1e-9))

    @staticmethod
    def _pairwise_dists(X: NDArray) -> NDArray:
        Y = X.reshape(len(X), 1, -1)
        return np.sqrt(np.sum((Y - Y.transpose(1, 0, 2)) ** 2, axis=2))

    @staticmethod
    def _spectral_gap(X: NDArray, k: int = 15) -> float:
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
        from sklearn.cluster import KMeans
        n = min(len(X), 4096)
        idx = np.random.choice(len(X), n, replace=False)
        S = X[idx]
        inertias = []
        for k in range(1, max_k + 1):
            km = KMeans(n_clusters=k, n_init=3, random_state=42)
            km.fit(S)
            inertias.append(km.inertia_)
        deltas = np.diff(inertias, 2)
        knee = int(np.argmax(deltas)) + 2 if len(deltas) else 1
        return float(knee)

    @staticmethod
    def _curvature_proxy(X: NDArray, k: int = 20) -> float:
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
        n = D.shape[0]
        thresh = np.median(D)
        A = (D < thresh).astype(float)
        np.fill_diagonal(A, 0)
        C = []
        for i in range(min(n, 256)):
            neigh = np.where(A[i])[0]
            if len(neigh) < 2:
                continue
            sub = A[np.ix_(neigh, neigh)]
            C.append(sub.sum() / (len(neigh) * (len(neigh) - 1)))
        clustering = np.mean(C) if C else 0.0
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


class PolicyNet:
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim, hidden)).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden, out_dim)).astype(np.float32) * 0.1
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self._cache: dict = {}

    def forward(self, x: NDArray) -> NDArray:
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        e = np.exp(z2 - np.max(z2))
        probs = e / e.sum()
        self._cache = {"x": x, "z1": z1, "a1": a1, "probs": probs}
        return probs

    def backward(self, target_idx: int, lr: float = 0.01) -> None:
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


@dataclass
class ManifoldConfig:
    name: str
    dim: int
    curvature: float
    learning_rate: float
    batch_size: int
    epochs: int
    extra: dict

    def to_dict(self) -> dict:
        return asdict(self)


class TopologicalGovernor:
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
        self.lr = lr
        self.entropy_bonus = entropy_bonus
        self.history: List[dict] = []
        self._checkpoint = checkpoint
        self.policy: PolicyNet | None = None

    def predict_topology(self, X: NDArray, temperature: float = 1.0) -> ManifoldConfig:
        feats = self.extractor(X)
        if self.policy is None:
            self.policy = PolicyNet(len(feats), len(self.catalog))
            if self._checkpoint and self._checkpoint.exists():
                self.policy = PolicyNet.load(self._checkpoint, len(feats), len(self.catalog))
        probs = self.policy.forward(feats)
        logit = np.log(probs + 1e-9) / temperature
        e = np.exp(logit - np.max(logit))
        tempered = e / e.sum()
        choice = int(np.random.choice(len(self.catalog), p=tempered))
        name = self.catalog[choice]
        cfg = self._default_config(name, X.shape[1])
        cfg.extra["governor_probs"] = {k: float(v) for k, v in zip(self.catalog, probs)}
        cfg.extra["governor_choice_idx"] = choice
        self._last_choice = choice
        self._last_feats = feats
        return cfg

    def update(self, config: ManifoldConfig, loss: float, reward_scale: float = 1.0) -> None:
        if self.policy is None or not hasattr(self, "_last_choice"):
            raise RuntimeError("predict_topology() before update().")
        reward = -loss * reward_scale
        scaled_lr = self.lr * np.clip(reward, -1.0, 1.0)
        self.policy.backward(self._last_choice, lr=scaled_lr)
        probs = self.policy._cache["probs"]
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        if entropy < 0.5:
            self.policy.b2 += self.entropy_bonus * (1.0 / len(self.catalog) - probs)
        self.history.append({"manifold": config.name, "loss": loss, "reward": reward, "entropy": float(entropy)})
        if self._checkpoint:
            self.policy.save(self._checkpoint)

    def rank_manifolds(self, X: NDArray) -> List[Tuple[str, float]]:
        feats = self.extractor(X)
        if self.policy is None:
            self.policy = PolicyNet(len(feats), len(self.catalog))
        probs = self.policy.forward(feats)
        return sorted(zip(self.catalog, probs), key=lambda t: t[1], reverse=True)

    def report(self) -> str:
        if not self.history:
            return "No battles."
        from collections import Counter
        wins = Counter(h["manifold"] for h in self.history)
        best = min(self.history, key=lambda h: h["loss"])
        return f"Battles: {len(self.history)} | Top: {wins.most_common(1)[0]} | Best: {best['loss']:.4f} on {best['manifold']}"

    def _default_config(self, name: str, ambient_dim: int) -> ManifoldConfig:
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
        return ManifoldConfig(name, dim, 0.0, 1e-3, 256, 100, {})


# ---------------------------------------------------------------------------
# 1.  USECASE ROUTER —  all domains in one class
# ---------------------------------------------------------------------------

class GovernorUsecases:
    """Plug governor into any domain.  Each method returns (config, loss, metadata)."""

    def __init__(self, governor: TopologicalGovernor | None = None):
        self.gov = governor or TopologicalGovernor()

    # ------------------------------------------------------------------
    # 1. CV — IMAGE MANIFOLD
    # ------------------------------------------------------------------
    def cv_image_manifold(self, train_images: NDArray, val_images: NDArray, labels: NDArray | None = None) -> dict:
        """Images → patch vitals → manifold → mock encoder eval."""
        patches = self._extract_patches(train_images, n=5000)
        cfg = self.gov.predict_topology(patches)
        loss = self._mock_cv_eval(cfg, val_images, labels)
        self.gov.update(cfg, loss)
        return {"task": "cv_image", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 2. NLP — EMBEDDING SPACE
    # ------------------------------------------------------------------
    def nlp_embedding_space(self, cooc_matrix: NDArray, vocab_size: int) -> dict:
        """Co-occurrence matrix → manifold for word embeddings."""
        cfg = self.gov.predict_topology(cooc_matrix[: min(len(cooc_matrix), 2000)])
        loss = self._mock_nlp_eval(cfg, vocab_size)
        self.gov.update(cfg, loss)
        return {"task": "nlp_embed", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 3. GNN — GRAPH NEURAL NET
    # ------------------------------------------------------------------
    def graph_neural_net(self, edge_index: NDArray, node_features: NDArray, labels: NDArray) -> dict:
        """Graph Laplacian eigenvectors → manifold → GNN eval."""
        laplacian_feats = self._graph_spectral_features(edge_index, node_features)
        cfg = self.gov.predict_topology(laplacian_feats)
        loss = self._mock_gnn_eval(cfg, node_features, labels)
        self.gov.update(cfg, loss)
        return {"task": "gnn", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 4. RECSYS — RECOMMENDATION
    # ------------------------------------------------------------------
    def recommendation_system(self, user_item_matrix: NDArray) -> dict:
        """User-item interaction matrix → manifold for matrix factorization."""
        cfg = self.gov.predict_topology(user_item_matrix[: min(len(user_item_matrix), 2000)])
        loss = self._mock_recsys_eval(cfg, user_item_matrix)
        self.gov.update(cfg, loss)
        return {"task": "recsys", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 5. TIME SERIES
    # ------------------------------------------------------------------
    def time_series_forecast(self, series: NDArray, forecast_horizon: int = 24) -> dict:
        """Recurrence plot vectors → manifold → forecaster."""
        recurrence_cloud = self._recurrence_vectors(series, delay=5, dim=3)
        cfg = self.gov.predict_topology(recurrence_cloud)
        loss = self._mock_ts_eval(cfg, series, forecast_horizon)
        self.gov.update(cfg, loss)
        return {"task": "time_series", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 6. RL — STATE SPACE
    # ------------------------------------------------------------------
    def rl_state_space(self, replay_states: NDArray, episode_returns: NDArray) -> dict:
        """Replay buffer states → manifold for policy."""
        cfg = self.gov.predict_topology(replay_states)
        loss = self._mock_rl_eval(cfg, episode_returns)
        self.gov.update(cfg, loss)
        return {"task": "rl", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 7. POINT CLOUD / LIDAR
    # ------------------------------------------------------------------
    def point_cloud_segmentation(self, point_cloud: NDArray, num_classes: int = 8) -> dict:
        """Local PCA eigenratios → manifold for point cloud model."""
        patch_features = self._point_cloud_pca_features(point_cloud)
        cfg = self.gov.predict_topology(patch_features)
        loss = self._mock_pointcloud_eval(cfg, num_classes)
        self.gov.update(cfg, loss)
        return {"task": "point_cloud", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 8. ANOMALY DETECTION
    # ------------------------------------------------------------------
    def anomaly_detection(self, normal_data: NDArray, test_data: NDArray) -> dict:
        """Normal data vitals → manifold → autoencoder reconstruction error."""
        cfg = self.gov.predict_topology(normal_data)
        loss = self._mock_anomaly_eval(cfg, normal_data, test_data)
        self.gov.update(cfg, loss)
        return {"task": "anomaly", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 9. MOLECULAR
    # ------------------------------------------------------------------
    def molecular_property(self, molecule_features: NDArray) -> dict:
        """Molecular descriptors → manifold for property prediction."""
        cfg = self.gov.predict_topology(molecule_features)
        loss = self._mock_molecular_eval(cfg)
        self.gov.update(cfg, loss)
        return {"task": "molecular", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 10. AUDIO / SPEECH
    # ------------------------------------------------------------------
    def audio_speech(self, mel_spectrogram: NDArray) -> dict:
        """Mel patches → manifold for ASR/synthesis."""
        mel_patches = self._extract_mel_patches(mel_spectrogram, n=5000)
        cfg = self.gov.predict_topology(mel_patches)
        loss = self._mock_audio_eval(cfg)
        self.gov.update(cfg, loss)
        return {"task": "audio", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 11. OS / KERNEL (EPSILON-HOLLOW STYLE)
    # ------------------------------------------------------------------
    def os_kernel_topology(self, lba_stream: NDArray, throughput: float) -> dict:
        """LBA telemetry → manifold → scheduler theorem boost."""
        cfg = self.gov.predict_topology(lba_stream.reshape(-1, lba_stream.shape[-1]))
        loss = 1.0 / max(throughput, 1e-6)  # inverse throughput = loss
        self.gov.update(cfg, loss)
        return {"task": "os_kernel", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 12. GENERATIVE MODEL
    # ------------------------------------------------------------------
    def generative_model(self, train_data: NDArray) -> dict:
        """Training data → manifold for VAE/GAN latent space."""
        flat = train_data.reshape(len(train_data), -1)[: min(len(train_data), 2000)]
        cfg = self.gov.predict_topology(flat)
        loss = self._mock_generative_eval(cfg, flat)
        self.gov.update(cfg, loss)
        return {"task": "generative", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 13. MULTI-MODAL ALIGNMENT
    # ------------------------------------------------------------------
    def multimodal_alignment(self, image_features: NDArray, text_features: NDArray) -> dict:
        """Both modalities → negotiate shared manifold."""
        img_cfg = self.gov.predict_topology(image_features)
        txt_cfg = self.gov.predict_topology(text_features)
        # pick higher-confidence manifold as shared
        shared = img_cfg if max(img_cfg.extra["governor_probs"].values()) > max(txt_cfg.extra["governor_probs"].values()) else txt_cfg
        loss = self._mock_multimodal_eval(shared, image_features, text_features)
        self.gov.update(shared, loss)
        return {"task": "multimodal", "config": shared.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 14. COMPRESSION / DIM REDUCTION
    # ------------------------------------------------------------------
    def compression_reduction(self, data: NDArray) -> dict:
        """Data → manifold for autoencoder bottleneck."""
        local_dims = self._estimate_local_dims(data)
        cfg = self.gov.predict_topology(local_dims.reshape(-1, 1))
        loss = self._mock_compression_eval(cfg, data)
        self.gov.update(cfg, loss)
        return {"task": "compression", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 15. META-LEARNING
    # ------------------------------------------------------------------
    def meta_learning(self, task_batch: List[NDArray]) -> dict:
        """Task embeddings → manifold for MAML init."""
        task_embeds = np.stack([t.mean(axis=0) for t in task_batch])
        cfg = self.gov.predict_topology(task_embeds)
        losses = [self._mock_meta_eval(cfg, t) for t in task_batch]
        loss = float(np.mean(losses))
        self.gov.update(cfg, loss)
        return {"task": "meta_learning", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 16. CAUSAL INFERENCE
    # ------------------------------------------------------------------
    def causal_inference(self, data: NDArray, graph_features: NDArray) -> dict:
        """Partial correlation graph → manifold for causal embed."""
        cfg = self.gov.predict_topology(graph_features)
        loss = self._mock_causal_eval(cfg, data)
        self.gov.update(cfg, loss)
        return {"task": "causal", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 17. ASTRONOMY / COSMOLOGY
    # ------------------------------------------------------------------
    def cosmology_simulation(self, halo_positions: NDArray) -> dict:
        """Dark matter halos → manifold for N-body."""
        cfg = self.gov.predict_topology(halo_positions)
        loss = self._mock_cosmo_eval(cfg)
        self.gov.update(cfg, loss)
        return {"task": "cosmology", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 18. CACHE / PREFETCH
    # ------------------------------------------------------------------
    def cache_prefetch(self, access_stream: NDArray, miss_rate: float) -> dict:
        """Memory access stream → manifold for prefetch policy."""
        cfg = self.gov.predict_topology(access_stream.reshape(-1, access_stream.shape[-1]))
        loss = miss_rate
        self.gov.update(cfg, loss)
        return {"task": "cache_prefetch", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 19. FINANCE / PORTFOLIO
    # ------------------------------------------------------------------
    def finance_portfolio(self, returns: NDArray) -> dict:
        """Rolling covariance → manifold for portfolio optimization."""
        rolling_cov = self._rolling_covariance_features(returns)
        cfg = self.gov.predict_topology(rolling_cov)
        loss = self._mock_finance_eval(cfg, returns)
        self.gov.update(cfg, loss)
        return {"task": "finance", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # 20. HARDWARE / CHIP PLACEMENT
    # ------------------------------------------------------------------
    def chip_placement(self, netlist_features: NDArray) -> dict:
        """Netlist spectral embedding → manifold for placer."""
        cfg = self.gov.predict_topology(netlist_features)
        loss = self._mock_chip_eval(cfg)
        self.gov.update(cfg, loss)
        return {"task": "chip_placement", "config": cfg.to_dict(), "loss": loss}

    # ------------------------------------------------------------------
    # HELPERS (mock evals + feature extractors)
    # ------------------------------------------------------------------

    def _extract_patches(self, images: NDArray, n: int = 5000) -> NDArray:
        h, w = images.shape[1], images.shape[2]
        patches = []
        for _ in range(n):
            img = images[np.random.randint(len(images))]
            y, x = np.random.randint(0, h - 8), np.random.randint(0, w - 8)
            patch = img[y : y + 8, x : x + 8]
            patches.append(patch.flatten())
        return np.asarray(patches, dtype=np.float32)

    def _mock_cv_eval(self, cfg: ManifoldConfig, val_images: NDArray, labels: NDArray | None) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = 0.6 if cfg.name == "euclidean" else 0.55
        if cfg.name == "spherical" and labels is not None and len(set(labels)) < 20:
            base = 0.85
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_nlp_eval(self, cfg: ManifoldConfig, vocab_size: int) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.78, "hyperboloid": 0.76, "euclidean": 0.70}.get(cfg.name, 0.65)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.02)))

    def _graph_spectral_features(self, edge_index: NDArray, node_features: NDArray) -> NDArray:
        n = len(node_features)
        A = np.zeros((n, n))
        for i, j in edge_index.T:
            A[int(i), int(j)] = 1.0
        deg = A.sum(axis=1)
        L = np.diag(deg) - A
        vals = np.linalg.eigvalsh(L)
        k = min(16, n)
        feats = np.tile(vals[:k], (n, 1)).astype(np.float32)
        return np.concatenate([node_features, feats], axis=1)

    def _mock_gnn_eval(self, cfg: ManifoldConfig, node_features: NDArray, labels: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.82, "euclidean": 0.80, "spherical": 0.75}.get(cfg.name, 0.70)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_recsys_eval(self, cfg: ManifoldConfig, matrix: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.75, "euclidean": 0.72, "product_s1_x_r": 0.73}.get(cfg.name, 0.68)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.02)))

    def _recurrence_vectors(self, series: NDArray, delay: int, dim: int) -> NDArray:
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        N = len(series) - (dim - 1) * delay
        if N <= 0:
            return np.random.randn(100, dim * series.shape[1]).astype(np.float32)
        vecs = np.stack([series[i : i + N] for i in range(0, dim * delay, delay)], axis=1)
        return vecs.reshape(N, -1).astype(np.float32)

    def _mock_ts_eval(self, cfg: ManifoldConfig, series: NDArray, horizon: int) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"spherical": 0.74, "euclidean": 0.72, "product_s1_x_r": 0.76}.get(cfg.name, 0.68)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_rl_eval(self, cfg: ManifoldConfig, returns: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.80, "euclidean": 0.75, "spherical": 0.72}.get(cfg.name, 0.70)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.04)))

    def _point_cloud_pca_features(self, pc: NDArray, k: int = 20) -> NDArray:
        n = len(pc)
        feats = []
        for i in np.random.choice(n, min(n, 512), replace=False):
            dists = np.sum((pc - pc[i]) ** 2, axis=1)
            neigh = np.argsort(dists)[1 : k + 1]
            loc = pc[neigh] - pc[i]
            cov = loc.T @ loc / k
            vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            feats.append([vals[0], vals[1], vals[2], vals[1] / (vals[0] + 1e-9), vals[2] / (vals[1] + 1e-9)])
        return np.asarray(feats, dtype=np.float32)

    def _mock_pointcloud_eval(self, cfg: ManifoldConfig, num_classes: int) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"euclidean": 0.78, "spherical": 0.76, "hyperbolic_poincare": 0.72}.get(cfg.name, 0.70)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_anomaly_eval(self, cfg: ManifoldConfig, normal: NDArray, test: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"mixed_curvature": 0.82, "euclidean": 0.75, "hyperbolic_poincare": 0.74}.get(cfg.name, 0.70)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_molecular_eval(self, cfg: ManifoldConfig) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.79, "spherical": 0.76, "euclidean": 0.73}.get(cfg.name, 0.70)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.02)))

    def _extract_mel_patches(self, mel: NDArray, n: int = 5000) -> NDArray:
        if mel.ndim == 1:
            mel = mel.reshape(1, -1)
        T, F = mel.shape
        patches = []
        for _ in range(n):
            t, f = np.random.randint(0, max(1, T - 16)), np.random.randint(0, max(1, F - 16))
            patch = mel[t : t + 16, f : f + 16]
            patches.append(patch.flatten())
        return np.asarray(patches, dtype=np.float32)

    def _mock_audio_eval(self, cfg: ManifoldConfig) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"spherical": 0.77, "euclidean": 0.75, "hyperbolic_poincare": 0.74}.get(cfg.name, 0.70)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_generative_eval(self, cfg: ManifoldConfig, flat: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"spherical": 0.76, "hyperbolic_poincare": 0.75, "euclidean": 0.73, "mixed_curvature": 0.77}.get(cfg.name, 0.70)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_multimodal_eval(self, cfg: ManifoldConfig, img: NDArray, txt: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        return max(0.0, 1.0 - (0.75 + rng.normal(0, 0.03)))

    def _estimate_local_dims(self, data: NDArray, k: int = 20) -> NDArray:
        n = len(data)
        dims = []
        for i in np.random.choice(n, min(n, 512), replace=False):
            dists = np.sum((data - data[i]) ** 2, axis=1)
            neigh = np.argsort(dists)[1 : k + 1]
            loc = data[neigh] - data[i]
            cov = loc.T @ loc / k
            vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            dims.append([vals[0] / (vals[1] + 1e-9)])
        return np.asarray(dims, dtype=np.float32)

    def _mock_compression_eval(self, cfg: ManifoldConfig, data: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"grassmannian": 0.80, "euclidean": 0.76, "mixed_curvature": 0.78}.get(cfg.name, 0.72)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.02)))

    def _mock_meta_eval(self, cfg: ManifoldConfig, task: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        return max(0.0, 1.0 - (0.73 + rng.normal(0, 0.04)))

    def _mock_causal_eval(self, cfg: ManifoldConfig, data: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.77, "euclidean": 0.74}.get(cfg.name, 0.71)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_cosmo_eval(self, cfg: ManifoldConfig) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.78, "spherical": 0.75, "mixed_curvature": 0.79}.get(cfg.name, 0.72)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _rolling_covariance_features(self, returns: NDArray, window: int = 30) -> NDArray:
        n, d = returns.shape
        feats = []
        for i in range(window, n):
            sub = returns[i - window : i]
            cov = np.cov(sub.T)
            feats.append(cov.flatten())
        return np.asarray(feats, dtype=np.float32)

    def _mock_finance_eval(self, cfg: ManifoldConfig, returns: NDArray) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.76, "euclidean": 0.74}.get(cfg.name, 0.71)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))

    def _mock_chip_eval(self, cfg: ManifoldConfig) -> float:
        rng = np.random.default_rng(hash(cfg.name) % 2**31)
        base = {"hyperbolic_poincare": 0.77, "euclidean": 0.75}.get(cfg.name, 0.72)
        return max(0.0, 1.0 - (base + rng.normal(0, 0.03)))


# ---------------------------------------------------------------------------
# DEMO — train governor across all 20 domains with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("GOVERNOR FULL — ALL USECASES")
    print("=" * 70)

    gu = GovernorUsecases()

    # 1. CV
    images = np.random.randn(1000, 32, 32, 3).astype(np.float32)
    r = gu.cv_image_manifold(images, images[:100], np.random.randint(0, 10, 1000))
    print(f"1.  CV_IMAGE        → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 2. NLP
    cooc = np.random.randn(5000, 5000).astype(np.float32)
    cooc = np.abs(cooc @ cooc.T)
    r = gu.nlp_embedding_space(cooc, vocab_size=5000)
    print(f"2.  NLP_EMBED       → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 3. GNN
    nodes = np.random.randn(1000, 64).astype(np.float32)
    edges = np.stack([np.random.randint(0, 1000, 4000), np.random.randint(0, 1000, 4000)])
    labels = np.random.randint(0, 7, 1000)
    r = gu.graph_neural_net(edges, nodes, labels)
    print(f"3.  GNN             → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 4. RECSYS
    ui = np.random.randn(2000, 5000).astype(np.float32)
    r = gu.recommendation_system(ui)
    print(f"4.  RECSYS          → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 5. TIME SERIES
    ts = np.cumsum(np.random.randn(10000)) + 10 * np.sin(np.linspace(0, 50, 10000))
    r = gu.time_series_forecast(ts)
    print(f"5.  TIME_SERIES     → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 6. RL
    states = np.random.randn(5000, 16).astype(np.float32)
    returns = np.random.randn(100)
    r = gu.rl_state_space(states, returns)
    print(f"6.  RL              → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 7. POINT CLOUD
    pc = np.random.randn(8000, 3).astype(np.float32)
    r = gu.point_cloud_segmentation(pc)
    print(f"7.  POINT_CLOUD     → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 8. ANOMALY
    normal = np.random.randn(2000, 32).astype(np.float32)
    test = np.random.randn(500, 32).astype(np.float32)
    r = gu.anomaly_detection(normal, test)
    print(f"8.  ANOMALY         → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 9. MOLECULAR
    mol = np.random.randn(1000, 200).astype(np.float32)
    r = gu.molecular_property(mol)
    print(f"9.  MOLECULAR       → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 10. AUDIO
    mel = np.random.randn(256, 128).astype(np.float32)
    r = gu.audio_speech(mel)
    print(f"10. AUDIO           → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 11. OS KERNEL
    lba = np.random.randint(0, 1_000_000, (1000, 1)).astype(np.float32)
    r = gu.os_kernel_topology(lba, throughput=1e6)
    print(f"11. OS_KERNEL       → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 12. GENERATIVE
    gen = np.random.randn(5000, 64).astype(np.float32)
    r = gu.generative_model(gen)
    print(f"12. GENERATIVE      → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 13. MULTIMODAL
    img = np.random.randn(1000, 512).astype(np.float32)
    txt = np.random.randn(1000, 512).astype(np.float32)
    r = gu.multimodal_alignment(img, txt)
    print(f"13. MULTIMODAL      → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 14. COMPRESSION
    comp = np.random.randn(3000, 128).astype(np.float32)
    r = gu.compression_reduction(comp)
    print(f"14. COMPRESSION     → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 15. META LEARNING
    tasks = [np.random.randn(100, 32).astype(np.float32) for _ in range(10)]
    r = gu.meta_learning(tasks)
    print(f"15. META_LEARNING   → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 16. CAUSAL
    data = np.random.randn(1000, 16).astype(np.float32)
    graph = np.random.randn(16, 16).astype(np.float32)
    r = gu.causal_inference(data, graph)
    print(f"16. CAUSAL          → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 17. COSMOLOGY
    halos = np.random.randn(5000, 3).astype(np.float32)
    r = gu.cosmology_simulation(halos)
    print(f"17. COSMOLOGY       → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 18. CACHE
    access = np.random.randint(0, 4096, (1000, 4)).astype(np.float32)
    r = gu.cache_prefetch(access, miss_rate=0.15)
    print(f"18. CACHE_PREFETCH  → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 19. FINANCE
    returns = np.random.randn(1000, 50).astype(np.float32) * 0.02
    r = gu.finance_portfolio(returns)
    print(f"19. FINANCE         → {r['config']['name']:20} loss={r['loss']:.3f}")

    # 20. CHIP
    netlist = np.random.randn(2000, 64).astype(np.float32)
    r = gu.chip_placement(netlist)
    print(f"20. CHIP_PLACEMENT  → {r['config']['name']:20} loss={r['loss']:.3f}")

    print("=" * 70)
    print(gu.gov.report())
    print("=" * 70)
    print("ALL USECASES DONE.")
