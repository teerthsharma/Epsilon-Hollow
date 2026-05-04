# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

"""
Epsilon-Hollow — Riemannian Gradient Flow Optimizer
====================================================
Manifold-aware gradient descent on curved spaces.

Standard SGD operates in flat Euclidean space ℝ^n. When weights live on
a manifold (orthogonality constraints, unit-norm embeddings, subspace tracking),
Euclidean gradients can violate manifold constraints. This optimizer
performs updates along geodesics using the exponential map.

Mathematical Foundation:
    1. Riemannian Gradient:
       ∇_M f(x) = Π_{T_x M}(∇f(x))
       where Π projects the Euclidean gradient onto the tangent space T_x M.

    2. Exponential Map (geodesic step):
       x_{t+1} = Exp_x(-η · ∇_M f(x))
       On Stiefel manifold St(n, p) = {X ∈ ℝ^{n×p} : X^T X = I_p}:
           Exp_X(Z) = (X + Z)(I + Z^T Z)^{-1/2}  [Cayley retraction, fast]

    3. Parallel Transport:
       Momentum must be transported along the manifold:
       Γ_{x→y}(v) = (I - ½ δδ^T / ‖δ‖²) v   where δ = y - x
       [Vector transport by projection, Absil et al. 2008]

    4. Grassmann Manifold Gr(n, p):
       Space of p-dimensional subspaces of ℝ^n.
       Used for tracking low-rank subspaces in weight matrices.
       Projection: Π_X(Z) = Z - X(X^T Z)

Supported Manifolds:
    - Stiefel(n, p):   Orthogonality-constrained weight matrices
    - Sphere(n):       Unit-norm embedding vectors (S^{n-1})
    - Grassmann(n, p): Low-rank subspace tracking
    - Euclidean(n):    Standard flat space (fallback)

Reference: Absil, Mahony, Sepulchre — "Optimization Algorithms on Matrix Manifolds" (2008)
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ManifoldType(Enum):
    EUCLIDEAN = "euclidean"
    SPHERE = "sphere"
    STIEFEL = "stiefel"
    GRASSMANN = "grassmann"


# ─────────────────────────────────────────────────────────────────────
# Manifold Operations
# ─────────────────────────────────────────────────────────────────────

def _stiefel_project_tangent(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Project Z onto tangent space T_X St(n, p).

    For Stiefel manifold St(n, p) = {X : X^T X = I_p}:
        Π_X(Z) = Z - X · sym(X^T Z)
    where sym(A) = (A + A^T) / 2.
    """
    XtZ = X.T @ Z
    sym = 0.5 * (XtZ + XtZ.T)
    return Z - X @ sym


def _stiefel_retract(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Cayley retraction on Stiefel manifold.

    R_X(Z) = (I - Z/2)^{-1} (I + Z/2) X
    This is a first-order approximation to the exponential map,
    cheaper than QR or SVD-based retractions.

    Actually for numerical stability, use QR:
        X_new = qr(X + Z).Q
    """
    Y = X + Z
    Q, R = np.linalg.qr(Y)
    # Fix sign ambiguity: ensure diagonal of R is positive
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q *= signs[np.newaxis, :]
    return Q


def _sphere_project_tangent(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Project z onto tangent space T_x S^{n-1}.

    Π_x(z) = z - (x · z) x
    """
    return z - np.dot(x, z) * x


def _sphere_retract(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Retraction on the sphere: normalise after step.

    R_x(z) = (x + z) / ‖x + z‖

    This is the cheapest retraction and first-order accurate.
    """
    y = x + z
    norm = np.linalg.norm(y)
    if norm < 1e-12:
        return x
    return y / norm


def _grassmann_project_tangent(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Project Z onto tangent space T_X Gr(n, p).

    Π_X(Z) = (I - XX^T) Z = Z - X(X^T Z)
    """
    return Z - X @ (X.T @ Z)


def _grassmann_retract(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    QR-based retraction on Grassmann manifold.
    """
    Y = X + Z
    Q, _ = np.linalg.qr(Y)
    return Q


def _parallel_transport(x: np.ndarray, y: np.ndarray,
                        v: np.ndarray) -> np.ndarray:
    """
    Vector transport by projection (generic).

    Γ_{x→y}(v) = Π_{T_y}(v)

    For the sphere, this is:
        v' = v - (y · v) y

    For general manifolds, subtract the normal component at y.
    """
    delta = y - x
    norm_sq = np.dot(delta, delta) if delta.ndim == 1 else np.sum(delta ** 2)
    if norm_sq < 1e-24:
        return v.copy()
    # Householder-type transport
    return v - (np.dot(delta, v) / norm_sq) * delta if v.ndim == 1 else \
        v - (delta @ (delta.T @ v)) / norm_sq


# ─────────────────────────────────────────────────────────────────────
# Riemannian Optimizer
# ─────────────────────────────────────────────────────────────────────

class RiemannianSGD:
    """
    Stochastic Gradient Descent on Riemannian manifolds.

    Supports momentum with parallel transport:
        v_t = β · Γ(v_{t-1}) + ∇_M f(x_t)
        x_{t+1} = Ret_x(-η · v_t)

    Parameters
    ----------
    lr : float
        Learning rate η.
    momentum : float
        Momentum coefficient β ∈ [0, 1).
    manifold : ManifoldType
        Which manifold the parameters live on.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9,
                 manifold: ManifoldType = ManifoldType.STIEFEL):
        self.lr = lr
        self.momentum = momentum
        self.manifold = manifold
        self._velocity: Optional[np.ndarray] = None
        self._prev_point: Optional[np.ndarray] = None

        # Statistics
        self.steps = 0
        self.total_grad_norm = 0.0

    def step(self, X: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        One Riemannian gradient step.

        Parameters
        ----------
        X : np.ndarray
            Current point on the manifold.
        grad : np.ndarray
            Euclidean gradient ∇f(X) (will be projected to tangent space).

        Returns
        -------
        X_new : np.ndarray
            Updated point on the manifold.
        """
        # 1. Project gradient to tangent space: ∇_M f = Π_{T_X}(∇f)
        rgrad = self._project_tangent(X, grad)

        # Track statistics
        self.total_grad_norm += float(np.linalg.norm(rgrad))
        self.steps += 1

        # 2. Momentum with parallel transport
        if self._velocity is not None and self._prev_point is not None:
            transported_v = _parallel_transport(self._prev_point, X, self._velocity)
            # Re-project to ensure we stay in tangent space
            transported_v = self._project_tangent(X, transported_v)
            velocity = self.momentum * transported_v + rgrad
        else:
            velocity = rgrad.copy()

        self._velocity = velocity.copy()
        self._prev_point = X.copy()

        # 3. Retraction: x_{t+1} = Ret_X(-η · v)
        X_new = self._retract(X, -self.lr * velocity)

        return X_new

    def _project_tangent(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.manifold == ManifoldType.STIEFEL:
            return _stiefel_project_tangent(X, Z)
        elif self.manifold == ManifoldType.SPHERE:
            return _sphere_project_tangent(X, Z)
        elif self.manifold == ManifoldType.GRASSMANN:
            return _grassmann_project_tangent(X, Z)
        else:
            return Z  # Euclidean: no projection needed

    def _retract(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.manifold == ManifoldType.STIEFEL:
            return _stiefel_retract(X, Z)
        elif self.manifold == ManifoldType.SPHERE:
            return _sphere_retract(X, Z)
        elif self.manifold == ManifoldType.GRASSMANN:
            return _grassmann_retract(X, Z)
        else:
            return X + Z  # Euclidean

    def avg_grad_norm(self) -> float:
        return self.total_grad_norm / max(self.steps, 1)


# ─────────────────────────────────────────────────────────────────────
# Riemannian Adam (RAdam)
# ─────────────────────────────────────────────────────────────────────

class RiemannianAdam:
    """
    Adam optimizer on Riemannian manifolds.

    Extends Adam with:
        1. Riemannian gradient (tangent projection)
        2. Exponential moving averages transported along geodesics
        3. Bias-corrected moment estimates on the manifold

    Update rule:
        g_t = Π_{T_x}(∇f(x_t))
        m_t = β₁ · Γ(m_{t-1}) + (1 - β₁) · g_t
        v_t = β₂ · v_{t-1} + (1 - β₂) · ‖g_t‖²   (scalar, no transport needed)
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        x_{t+1} = Ret_x(-η · m̂_t / (√v̂_t + ε))
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 manifold: ManifoldType = ManifoldType.STIEFEL):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.manifold = manifold

        self._m: Optional[np.ndarray] = None  # First moment
        self._v: float = 0.0                  # Second moment (scalar)
        self._prev_point: Optional[np.ndarray] = None
        self._t: int = 0

    def step(self, X: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """One Riemannian Adam step."""
        self._t += 1

        # Project to tangent space
        rgrad = self._project_tangent(X, grad)
        grad_norm_sq = float(np.sum(rgrad ** 2))

        # Update moments
        if self._m is not None and self._prev_point is not None:
            transported_m = _parallel_transport(self._prev_point, X, self._m)
            transported_m = self._project_tangent(X, transported_m)
            self._m = self.beta1 * transported_m + (1 - self.beta1) * rgrad
        else:
            self._m = (1 - self.beta1) * rgrad

        self._v = self.beta2 * self._v + (1 - self.beta2) * grad_norm_sq

        # Bias correction
        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)

        # Step
        step_size = self.lr / (math.sqrt(v_hat) + self.eps)
        X_new = self._retract(X, -step_size * m_hat)

        self._prev_point = X.copy()
        return X_new

    def _project_tangent(self, X, Z):
        if self.manifold == ManifoldType.STIEFEL:
            return _stiefel_project_tangent(X, Z)
        elif self.manifold == ManifoldType.SPHERE:
            return _sphere_project_tangent(X, Z)
        elif self.manifold == ManifoldType.GRASSMANN:
            return _grassmann_project_tangent(X, Z)
        return Z

    def _retract(self, X, Z):
        if self.manifold == ManifoldType.STIEFEL:
            return _stiefel_retract(X, Z)
        elif self.manifold == ManifoldType.SPHERE:
            return _sphere_retract(X, Z)
        elif self.manifold == ManifoldType.GRASSMANN:
            return _grassmann_retract(X, Z)
        return X + Z
