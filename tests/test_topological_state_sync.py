"""Unit tests for epsilon_core.topological_state_sync (pure-numpy parts)."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernel" / "epsilon"))

from epsilon_core.topological_state_sync import (  # noqa: E402
    SphericalGridHash,
    compute_p_max,
    compute_theta_min,
    epsilon_from_chebyshev,
)


class TestComputeBounds(unittest.TestCase):
    def test_p_max_zero_theta_is_inf(self):
        self.assertEqual(compute_p_max(0.0), float("inf"))

    def test_p_max_decreases_with_theta(self):
        small = compute_p_max(0.1)
        large = compute_p_max(1.0)
        self.assertGreater(small, large)

    def test_theta_min_monotonic(self):
        self.assertLess(compute_theta_min(0.1), compute_theta_min(0.5))

    def test_epsilon_from_chebyshev(self):
        self.assertAlmostEqual(epsilon_from_chebyshev(1.0, 0.5, k=2.0), 2.0)


class TestSphericalGridHash(unittest.TestCase):
    def test_auto_sized_dimensions(self):
        g = SphericalGridHash.auto_sized(100)
        self.assertGreaterEqual(g.n_theta, 4)
        self.assertGreaterEqual(g.n_phi, 8)

    def test_locate_empty_returns_minus_one(self):
        g = SphericalGridHash(n_theta=4, n_phi=8)
        self.assertEqual(g.locate(0.0, 0.0), -1)

    def test_build_and_locate_returns_valid_index(self):
        g = SphericalGridHash(n_theta=4, n_phi=8)
        centroids = [(0.5, 0.0), (math.pi / 2, 1.0), (math.pi - 0.5, -1.0)]
        g.build(centroids)
        idx = g.locate(0.5, 0.0)
        self.assertEqual(idx, 0)

    def test_locate_picks_nearest(self):
        g = SphericalGridHash(n_theta=8, n_phi=16)
        centroids = [(0.1, 0.0), (math.pi - 0.1, 0.0)]
        g.build(centroids)
        # Query near north pole should hit centroid 0.
        self.assertEqual(g.locate(0.1, 0.0), 0)
        # Query near south pole should hit centroid 1.
        self.assertEqual(g.locate(math.pi - 0.1, 0.0), 1)

    def test_stats_reports_counts(self):
        g = SphericalGridHash(n_theta=4, n_phi=8)
        g.build([(0.5, 0.0), (1.0, 1.0)])
        s = g.stats()
        self.assertEqual(s["total_centroids"], 2)
        self.assertEqual(s["total_cells"], 32)

    def test_o1_hit_rate_after_lookup(self):
        g = SphericalGridHash(n_theta=4, n_phi=8)
        g.build([(0.5, 0.0)])
        g.locate(0.5, 0.0)
        self.assertGreaterEqual(g.o1_hit_rate(), 0.0)
        self.assertLessEqual(g.o1_hit_rate(), 1.0)


if __name__ == "__main__":
    unittest.main()
