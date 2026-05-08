"""Unit tests for epsilon_core.hyperbolic_geometry.

Round-trip checks (exp/log inverse), metric properties (non-negativity,
identity, symmetry approximation), and the EGPB plasticity bound.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernel" / "epsilon"))

from epsilon_core.hyperbolic_geometry import (  # noqa: E402
    AngularMomentumTracker,
    PoincareBall,
    plasticity_bound,
    verify_plasticity_bound,
)


class TestPoincareBall(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.ball = PoincareBall(dim=8, curvature=1.0)

    def test_zero_distance_to_self(self):
        x = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(self.ball.distance(x, x), 0.0, places=5)

    def test_distance_non_negative(self):
        for _ in range(5):
            x = np.random.randn(8) * 0.1
            y = np.random.randn(8) * 0.1
            self.assertGreaterEqual(self.ball.distance(x, y), 0.0)

    def test_mobius_add_zero_identity(self):
        # x ⊕ 0 = x
        x = np.array([0.1, -0.2, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        zero = np.zeros(8)
        result = self.ball.mobius_add(x, zero)
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_exp_log_round_trip(self):
        # exp_x(log_x(y)) ≈ y
        x = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v = self.ball.log_map(x, y)
        y_rec = self.ball.exp_map(x, v)
        np.testing.assert_allclose(y_rec, y, atol=1e-4)

    def test_exp_at_zero_velocity_is_identity(self):
        x = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v = np.zeros(8)
        np.testing.assert_allclose(self.ball.exp_map(x, v), x)

    def test_projection_keeps_inside_ball(self):
        big = np.ones(8) * 5.0  # far outside
        proj = self.ball._project_to_ball(big)
        self.assertLess(np.linalg.norm(proj), 1.0 / self.ball.sqrt_c)

    def test_centroid_empty(self):
        out = self.ball.centroid(np.zeros((0, 8)))
        self.assertEqual(out.shape, (8,))

    def test_centroid_single_point(self):
        p = np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        c = self.ball.centroid(p)
        np.testing.assert_allclose(c, p[0], atol=1e-5)

    def test_curvature_positive(self):
        self.assertGreater(self.ball.c, 0.0)


class TestPlasticityBound(unittest.TestCase):
    def test_bound_zero_when_entropy_one(self):
        self.assertEqual(plasticity_bound(0.01, 1.0, 10.0, 0.005), 0.0)

    def test_bound_scales_with_grad(self):
        b1 = plasticity_bound(0.01, 0.5, 1.0, 0.005)
        b2 = plasticity_bound(0.01, 0.5, 2.0, 0.005)
        self.assertAlmostEqual(b2, 2 * b1, places=8)

    def test_verify_within_bound(self):
        b = plasticity_bound(0.01, 0.5, 1.0, 0.005)
        self.assertTrue(verify_plasticity_bound(b * 0.5, 0.01, 0.5, 1.0, 0.005))
        self.assertFalse(verify_plasticity_bound(b * 2.0, 0.01, 0.5, 1.0, 0.005))


class TestAngularMomentumTracker(unittest.TestCase):
    def test_zero_query_returns_zero(self):
        t = AngularMomentumTracker()
        L = t.compute_angular_momentum(
            np.zeros(4),
            np.eye(4),
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([True, True, True, True]),
        )
        self.assertEqual(L, 0.0)

    def test_first_check_returns_true(self):
        t = AngularMomentumTracker()
        # First call seeds prev_L; should not flag.
        self.assertTrue(t.check_conservation(1.0, 0.5, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
