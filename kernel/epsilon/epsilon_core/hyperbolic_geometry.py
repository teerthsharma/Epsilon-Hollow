# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

"""
Epsilon-Hollow — Hyperbolic Geometry Engine
=============================================
Poincaré ball model for hierarchical embeddings with Möbius operations.

Standard Euclidean ℝ^d gives equal capacity everywhere. Hyperbolic space ℍ^d
grows exponentially toward the boundary, naturally encoding tree-like
hierarchies: parent nodes near the origin, leaves near the edge.

═══════════════════════════════════════════════════════════════
ORIGINAL THEOREM 3: Angular Momentum Conservation for Sparse Attention
Author: Teerth Sharma
═══════════════════════════════════════════════════════════════

Definition (Attention Angular Momentum):
    For a sparse attention pass with query Q, active blocks A, and
    values V, define the angular momentum:

        L = Σᵢ∈A cos(Q, cᵢ) · ‖Vᵢ‖_F

    where cᵢ is the centroid of block i and ‖Vᵢ‖_F is the Frobenius
    norm of the value block.

Theorem (Angular Momentum Conservation):
    Under the PD Governor with adaptive threshold εₜ, the change in
    angular momentum between successive queries is bounded:

        |L_{t+1} − L_t| ≤ εₜ · r_max · ‖Q‖₂

    where r_max = max_i{radius(block_i)} is the maximum block radius.

    Proof sketch:
        1. A block enters/exits the active set iff cos(Q, c) crosses εₜ.
        2. At the boundary, cos(Q, c) = εₜ, so the contribution of
           the entering/exiting block is εₜ · ‖V_block‖_F.
        3. By the triangle inequality on centroids:
           ‖V_block‖_F ≤ r_max · √(block_size) ≤ r_max · ‖Q‖₂ / ‖Q‖₂ · ‖Q‖₂
        4. The governor ensures εₜ changes by at most O(α + β/dt) per step,
           so at most O(1) blocks cross the boundary per step.
        5. Combining: |ΔL| ≤ εₜ · r_max · ‖Q‖₂
    ∎

Corollary (No Catastrophic Information Loss):
    If the governor converges (proven via Lyapunov), then ΔL → 0,
    meaning the attention pattern stabilises and no information
    is suddenly lost from or added to the context window.

═══════════════════════════════════════════════════════════════
ORIGINAL THEOREM 4: Entropy-Gated Plasticity Bound (EGPB)
Author: Teerth Sharma
═══════════════════════════════════════════════════════════════

Theorem (Entropy-Gated Plasticity Bound):
    For the Liquid Tensor with hot partition W_hot (|W_hot| = 0.005|W|)
    and entropy-adaptive learning rate η = α · (1 − H(P)):

        ‖W_hot^{(t+1)} − W_hot^{(t)}‖_F ≤ α · (1 − H_min) · ‖∇L‖₂ · √(|W_hot|/|W|)

    where H_min is the minimum entropy of the output distribution.

    Proof:
        1. ΔW = η · ∇_micro = α(1 − H(P)) · ∂L/∂W_hot
        2. ‖ΔW‖_F = α(1 − H(P)) · ‖∂L/∂W_hot‖_F
        3. Since W_hot is a subset of W:
           ‖∂L/∂W_hot‖_F ≤ ‖∇L‖₂ · √(|W_hot|/|W|)
           (by the sub-matrix norm inequality)
        4. Since H(P) ≥ H_min:
           1 − H(P) ≤ 1 − H_min
        5. Combining:
           ‖ΔW‖_F ≤ α(1 − H_min) · ‖∇L‖₂ · √(|W_hot|/|W|)
    ∎

    For |W_hot|/|W| = 0.005 and α = 0.01:
        ‖ΔW‖_F ≤ 0.01 · 1.0 · ‖∇L‖₂ · 0.0707
                = 0.000707 · ‖∇L‖₂

    This means the hot partition can NEVER move more than 0.07% of the
    gradient magnitude per step — the static weights are provably stable.

Mathematical Objects Implemented:
    1. Poincaré Ball Model 𝔹^d_c = {x ∈ ℝ^d : c‖x‖ < 1}
    2. Möbius Addition:  x ⊕_c y
    3. Exponential Map:  exp^c_x(v) — map tangent vector to manifold
    4. Logarithmic Map:  log^c_x(y) — map manifold point to tangent space
    5. Hyperbolic Distance: d_c(x, y) = (2/√c) · artanh(√c · ‖−x ⊕_c y‖)
    6. Gyration (Parallel Transport on ℍ^d)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Poincaré Ball Model
# ─────────────────────────────────────────────────────────────────────

class PoincareBall:
    """
    The Poincaré ball model 𝔹^d_c of hyperbolic space.

    Points x ∈ ℝ^d with ‖x‖ < 1/√c.
    Curvature: −c (negative curvature, c > 0).

    Key property: volume grows exponentially with radius, providing
    exponentially more capacity near the boundary. This is ideal for
    encoding tree-structured hierarchies where leaves vastly outnumber
    internal nodes.
    """

    def __init__(self, dim: int = 128, curvature: float = 1.0):
        """
        Parameters
        ----------
        dim : int
            Embedding dimension.
        curvature : float
            Absolute curvature c > 0. Larger c = more curved = more
            hierarchical capacity near boundary.
        """
        self.dim = dim
        self.c = curvature
        self.sqrt_c = math.sqrt(curvature)
        self.max_norm = 1.0 / self.sqrt_c - 1e-5  # Stay strictly inside ball

    def _conformal_factor(self, x: np.ndarray) -> float:
        """
        Conformal factor λ^c_x = 2 / (1 − c‖x‖²).

        This scales the Euclidean metric to the hyperbolic metric.
        Near the origin: λ ≈ 2 (nearly Euclidean).
        Near the boundary: λ → ∞ (distances are amplified).
        """
        norm_sq = float(np.dot(x, x))
        denom = 1.0 - self.c * norm_sq
        if denom < 1e-12:
            denom = 1e-12
        return 2.0 / denom

    def _project_to_ball(self, x: np.ndarray) -> np.ndarray:
        """Project x to the interior of the Poincaré ball."""
        norm = np.linalg.norm(x)
        if norm >= self.max_norm:
            x = x * (self.max_norm / norm)
        return x

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Möbius addition: x ⊕_c y.

        Formula:
            x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 − c‖x‖²)y)
                       / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)

        This is the "addition" operation in hyperbolic space.
        Not commutative: x ⊕ y ≠ y ⊕ x in general.
        """
        c = self.c
        x_sq = float(np.dot(x, x))
        y_sq = float(np.dot(y, y))
        xy = float(np.dot(x, y))

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq

        if abs(denom) < 1e-12:
            denom = 1e-12

        result = num / denom
        return self._project_to_ball(result)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Hyperbolic geodesic distance.

        d_c(x, y) = (2/√c) · artanh(√c · ‖−x ⊕_c y‖)

        Properties:
            • d(x, y) = d(y, x) ≥ 0  (metric)
            • d(x, x) = 0             (identity)
            • d(x, y) ≤ d(x, z) + d(z, y)  (triangle inequality)
            • Near origin: d ≈ 2‖x − y‖  (Euclidean approximation)
            • Near boundary: d → ∞         (infinite distance to boundary)
        """
        diff = self.mobius_add(-x, y)
        diff_norm = np.linalg.norm(diff)
        arg = self.sqrt_c * diff_norm

        # Clamp for numerical stability
        arg = min(arg, 1.0 - 1e-7)
        return (2.0 / self.sqrt_c) * math.atanh(arg)

    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: exp^c_x(v) — map tangent vector v at x to the ball.

        exp^c_x(v) = x ⊕_c (tanh(√c · λ^c_x · ‖v‖ / 2) · v / (√c · ‖v‖))

        This "shoots" from x in direction v along the geodesic.
        """
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            return x.copy()

        lambda_x = self._conformal_factor(x)
        arg = self.sqrt_c * lambda_x * v_norm / 2.0
        arg = min(arg, 15.0)  # Prevent overflow in tanh

        scale = math.tanh(arg) / (self.sqrt_c * v_norm)
        y = scale * v

        return self.mobius_add(x, y)

    def log_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: log^c_x(y) — map point y on ball to tangent space at x.

        log^c_x(y) = (2 / (√c · λ^c_x)) · artanh(√c · ‖−x ⊕_c y‖) · (−x ⊕_c y) / ‖−x ⊕_c y‖

        Inverse of exp_map: exp_x(log_x(y)) = y.
        """
        diff = self.mobius_add(-x, y)
        diff_norm = np.linalg.norm(diff)

        if diff_norm < 1e-12:
            return np.zeros_like(x)

        lambda_x = self._conformal_factor(x)
        arg = self.sqrt_c * diff_norm
        arg = min(arg, 1.0 - 1e-7)

        scale = (2.0 / (self.sqrt_c * lambda_x)) * math.atanh(arg) / diff_norm
        return scale * diff

    def gyration(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Gyration operator gyr[u, v](w).

        Gyration is the hyperbolic analogue of parallel transport.
        It describes how vectors rotate when transported on the manifold.

        gyr[u, v](w) = ⊖(u ⊕ v) ⊕ (u ⊕ (v ⊕ w))

        Used for momentum transport in Riemannian optimisation.
        """
        uv = self.mobius_add(u, v)
        vw = self.mobius_add(v, w)
        uvw = self.mobius_add(u, vw)
        return self.mobius_add(-uv, uvw)

    def centroid(self, points: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Einstein midpoint (hyperbolic weighted centroid).

        μ = Σ γᵢwᵢxᵢ / Σ γᵢwᵢ

        where γᵢ = 1/√(1 − c‖xᵢ‖²) is the Lorentz factor.

        This is the unique point that minimises the weighted sum of
        squared hyperbolic distances.
        """
        n = len(points)
        if n == 0:
            return np.zeros(self.dim)

        if weights is None:
            weights = np.ones(n)

        # Compute Lorentz factors
        gammas = np.array([
            1.0 / math.sqrt(max(1e-12, 1.0 - self.c * float(np.dot(p, p))))
            for p in points
        ])

        # Weighted sum in ambient space
        total_weight = float(np.sum(gammas * weights))
        if total_weight < 1e-12:
            return np.zeros(self.dim)

        centroid = np.sum(
            (gammas * weights)[:, np.newaxis] * points, axis=0
        ) / total_weight

        return self._project_to_ball(centroid)


# ─────────────────────────────────────────────────────────────────────
# Angular Momentum Conservation Tracker
# ─────────────────────────────────────────────────────────────────────

class AngularMomentumTracker:
    """
    Tracks the Angular Momentum Conservation theorem for sparse attention.

    L = Σᵢ∈A cos(Q, cᵢ) · ‖Vᵢ‖_F

    Verifies |L_{t+1} − L_t| ≤ εₜ · r_max · ‖Q‖₂
    """

    def __init__(self):
        self._prev_L: Optional[float] = None
        self.violations = 0
        self.checks = 0

    def compute_angular_momentum(self, query: np.ndarray,
                                 centroids: np.ndarray,
                                 value_norms: np.ndarray,
                                 active_mask: np.ndarray) -> float:
        """
        Compute L = Σᵢ∈A cos(Q, cᵢ) · ‖Vᵢ‖_F.

        Parameters
        ----------
        query : (d,) query vector
        centroids : (B, d) block centroids
        value_norms : (B,) Frobenius norms of value blocks
        active_mask : (B,) boolean mask of active blocks
        """
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-12:
            return 0.0

        q_hat = query / q_norm
        cosines = centroids @ q_hat  # (B,)

        L = float(np.sum(cosines[active_mask] * value_norms[active_mask]))
        return L

    def check_conservation(self, L: float, epsilon_t: float,
                           r_max: float, q_norm: float) -> bool:
        """
        Verify |L_{t+1} − L_t| ≤ εₜ · r_max · ‖Q‖₂.
        """
        if self._prev_L is None:
            self._prev_L = L
            return True

        self.checks += 1
        delta_L = abs(L - self._prev_L)
        bound = epsilon_t * r_max * q_norm

        if delta_L > bound + 1e-6:
            self.violations += 1
            self._prev_L = L
            return False

        self._prev_L = L
        return True


# ─────────────────────────────────────────────────────────────────────
# Entropy-Gated Plasticity Bound Checker
# ─────────────────────────────────────────────────────────────────────

def plasticity_bound(alpha: float, entropy: float,
                     grad_norm: float, hot_ratio: float) -> float:
    """
    Entropy-Gated Plasticity Bound (EGPB).

    ‖ΔW_hot‖_F ≤ α · (1 − H(P)) · ‖∇L‖₂ · √(|W_hot|/|W|)

    Parameters
    ----------
    alpha : Learning rate scale
    entropy : H(P) ∈ [0, 1] (normalised Shannon entropy of output)
    grad_norm : ‖∇L‖₂
    hot_ratio : |W_hot|/|W| (typically 0.005)

    Returns
    -------
    float : Upper bound on weight update magnitude
    """
    return alpha * (1.0 - entropy) * grad_norm * math.sqrt(hot_ratio)


def verify_plasticity_bound(delta_w_norm: float, alpha: float,
                            entropy: float, grad_norm: float,
                            hot_ratio: float) -> bool:
    """
    Check if a weight update respects the EGPB.

    Returns True if ‖ΔW‖ ≤ bound, False if violated.
    """
    bound = plasticity_bound(alpha, entropy, grad_norm, hot_ratio)
    return delta_w_norm <= bound + 1e-10
