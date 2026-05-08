"""Unit tests for epsilon_core.memory (TopologicalManifoldMemory + UnionFind)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernel" / "epsilon"))

from epsilon_core.memory import TopologicalManifoldMemory, UnionFind  # noqa: E402


class TestUnionFind(unittest.TestCase):
    def test_initial_components(self):
        uf = UnionFind(5)
        self.assertEqual(uf.num_components, 5)

    def test_union_reduces_components(self):
        uf = UnionFind(5)
        self.assertTrue(uf.union(0, 1))
        self.assertEqual(uf.num_components, 4)
        self.assertEqual(uf.find(0), uf.find(1))

    def test_redundant_union_no_op(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        self.assertFalse(uf.union(0, 1))
        self.assertEqual(uf.num_components, 2)


class TestTopologicalManifoldMemory(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.mem = TopologicalManifoldMemory(dim=8, capacity=100)

    def test_empty_retrieve(self):
        self.assertEqual(self.mem.retrieve(np.ones(8)), [])

    def test_empty_betti_zero(self):
        self.assertEqual(self.mem.compute_betti_0(), 0)

    def test_store_returns_index(self):
        idx0 = self.mem.store(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))
        idx1 = self.mem.store(np.array([0.0, 1.0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 1)
        self.assertEqual(len(self.mem.episodes), 2)

    def test_store_wrong_shape_raises(self):
        with self.assertRaises(AssertionError):
            self.mem.store(np.zeros(3))

    def test_retrieve_returns_results(self):
        for i in range(5):
            v = np.random.randn(8)
            self.mem.store(v, metadata={"i": i})
        out = self.mem.retrieve(np.random.randn(8), k=3)
        self.assertGreater(len(out), 0)
        self.assertLessEqual(len(out), 5)

    def test_reinforce_increments_count(self):
        idx = self.mem.store(np.ones(8))
        self.mem.reinforce(idx)
        self.assertEqual(self.mem.episodes[idx]["reinforcement_count"], 1)

    def test_reinforce_out_of_range_silent(self):
        # Should not raise.
        self.mem.reinforce(999)

    def test_adaptive_epsilon_default_when_too_few(self):
        eps = self.mem.adaptive_epsilon(np.array([0.5]))
        self.assertEqual(eps, 1.0)

    def test_adaptive_epsilon_uses_chebyshev(self):
        d = np.array([0.0, 1.0, 2.0, 3.0])
        eps = self.mem.adaptive_epsilon(d)
        # μ + k σ should be > μ.
        self.assertGreater(eps, float(d.mean()))

    def test_manifold_stats_after_inserts(self):
        for _ in range(3):
            self.mem.store(np.random.randn(8))
        stats = self.mem.manifold_stats()
        self.assertEqual(stats["size"], 3)
        self.assertEqual(stats["dim"], 8)
        self.assertIn("betti_0", stats)


if __name__ == "__main__":
    unittest.main()
