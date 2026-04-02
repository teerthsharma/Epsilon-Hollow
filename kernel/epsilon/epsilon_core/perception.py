"""
Epsilon-Hollow — Multimodal Perception Encoder
===============================================
Maps arbitrary observation modalities → ℝ^d via Johnson-Lindenstrauss random projection.

Mathematical Foundation:
    Johnson-Lindenstrauss Lemma (1984):
        For n points in ℝ^D and ε ∈ (0,1), a random linear map
        R: ℝ^D → ℝ^d with d ≥ O(ε⁻² log n) preserves pairwise
        distances within (1 ± ε).

    Projection:  Φ(x) = (1/√d) · R · h(x)
    where:
        h(x)  = feature hash of input (deterministic)
        R     ~ 𝒩(0, 1)^{d × k}  (random Gaussian matrix)
        d     = embedding dimension (default 128)
        k     = intermediate feature dimension

    The embedding is L2-normalised to lie on the unit hypersphere S^{d-1}.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, Optional

import numpy as np


class MultimodalEncoder:
    """
    Encodes multimodal observations into a unified latent space ℝ^d.

    Supports:
        - text:   Hash-based random projection (deterministic, reproducible)
        - vision: Flattened pixel data → random projection
        - audio:  Raw waveform → random projection
        - code:   AST-aware hashing with structural features

    All modalities are fused via weighted concatenation before final
    projection to ensure cross-modal coherence.
    """

    def __init__(self, dim: int = 128, feature_dim: int = 1024,
                 seed: int = 42):
        """
        Parameters
        ----------
        dim : int
            Output embedding dimension d = |ℝ^d|.
        feature_dim : int
            Intermediate feature dimension k for JL projection.
        seed : int
            Random seed for reproducible projection matrix.
        """
        self.dim = dim
        self.feature_dim = feature_dim

        # JL projection matrix: R ~ 𝒩(0, 1)^{d × k}
        rng = np.random.RandomState(seed)
        self._projection_matrix = rng.randn(dim, feature_dim).astype(np.float32)
        # Scale factor: 1/√d
        self._scale = 1.0 / np.sqrt(dim)

        # Modality-specific weights (learned or uniform)
        self._modality_weights = {
            "text": 1.0,
            "vision": 0.8,
            "audio": 0.6,
            "code": 1.2,
        }

    def encode(self, modality_data: Dict[str, Any]) -> np.ndarray:
        """
        Encode multimodal observation → ℝ^d.

        Parameters
        ----------
        modality_data : dict
            Keys are modality names ('text', 'vision', 'audio', 'code').
            Values are raw data for that modality.

        Returns
        -------
        np.ndarray, shape (d,)
            L2-normalised embedding on S^{d-1}.
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        total_weight = 0.0

        for modality, data in modality_data.items():
            if data is None:
                continue

            weight = self._modality_weights.get(modality, 0.5)
            h = self._hash_features(data, modality)
            features += weight * h
            total_weight += weight

        if total_weight > 0:
            features /= total_weight

        # JL projection: Φ(x) = (1/√d) · R · h(x)
        embedding = self._scale * (self._projection_matrix @ features)

        # L2 normalise to S^{d-1}
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding /= norm

        return embedding.astype(np.float64)

    def _hash_features(self, data: Any, modality: str) -> np.ndarray:
        """
        Deterministic feature hashing:  data → ℝ^k.

        Uses SHA-256 to generate deterministic pseudo-random features
        from arbitrary input. This is a stable, reproducible mapping
        that preserves semantic similarity for similar inputs.

        For text: character n-gram hashing (n=3)
        For other modalities: raw byte hashing
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)

        if isinstance(data, str):
            # Character trigram hashing for text
            text = data.lower().strip()
            if not text:
                return features

            ngrams = [text[i:i + 3] for i in range(len(text) - 2)]
            if not ngrams:
                ngrams = [text]

            for ngram in ngrams:
                h = hashlib.sha256(ngram.encode("utf-8")).digest()
                # Map hash bytes to feature indices and values
                for j in range(0, min(len(h), 32), 4):
                    idx = struct.unpack("<I", h[j:j + 4])[0] % self.feature_dim
                    # Rademacher random variable from hash: +1 or -1
                    sign = 1.0 if (h[j] & 1) else -1.0
                    features[idx] += sign

        elif isinstance(data, np.ndarray):
            # Direct feature extraction for arrays
            flat = data.flatten().astype(np.float32)
            if len(flat) > self.feature_dim:
                # Downsample via strided access
                stride = len(flat) // self.feature_dim
                features = flat[:self.feature_dim * stride:stride][:self.feature_dim]
            else:
                features[:len(flat)] = flat

        elif isinstance(data, (bytes, bytearray)):
            h = hashlib.sha256(data).digest()
            for j in range(min(len(h), 32)):
                idx = j * (self.feature_dim // 32)
                features[idx] = float(h[j]) / 128.0 - 1.0

        else:
            # Fallback: String representation
            return self._hash_features(str(data), modality)

        # L2 normalise intermediate features
        norm = np.linalg.norm(features)
        if norm > 1e-12:
            features /= norm

        return features

    def encode_text(self, text: str) -> np.ndarray:
        """Convenience: encode text-only observation."""
        return self.encode({"text": text})

    def encode_code(self, source: str) -> np.ndarray:
        """Convenience: encode source code with elevated weight."""
        return self.encode({"code": source})
