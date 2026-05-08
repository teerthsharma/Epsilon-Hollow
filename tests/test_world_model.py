"""Smoke tests for epsilon_core.world_model.LiquidMemoryWorldModel."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernel" / "epsilon"))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from epsilon_core.world_model import LiquidMemoryWorldModel  # noqa: E402


class TestLiquidMemoryWorldModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.wm = LiquidMemoryWorldModel(dim=16, action_dim=4, seed=0,
                                         memory_capacity=64)

    def test_construct(self):
        self.assertEqual(self.wm.dim, 16)
        self.assertEqual(self.wm.action_dim, 4)
        self.assertIsNotNone(self.wm.encoder)
        self.assertIsNotNone(self.wm.memory)

    def test_encode_observation_shape(self):
        latent = self.wm.encode_observation({"text": "hello"})
        self.assertEqual(latent.shape, (16,))

    def test_ingest_returns_metadata(self):
        out = self.wm.ingest({"text": "hi"}, metadata={"tag": "x"})
        self.assertIn("index", out)
        self.assertIn("memory_size", out)
        self.assertEqual(out["memory_size"], 1)

    def test_query_after_ingest(self):
        self.wm.ingest({"text": "alpha"})
        self.wm.ingest({"text": "beta"})
        result = self.wm.query({"text": "alpha"}, k=2)
        self.assertIn("top_k", result)

    def test_reset_memory(self):
        self.wm.ingest({"text": "x"})
        self.wm.reset_memory()
        self.assertEqual(len(self.wm.memory.episodes), 0)


if __name__ == "__main__":
    unittest.main()
