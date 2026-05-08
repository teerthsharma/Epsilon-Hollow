"""Smoke tests for epsilon_core.perception.MultimodalEncoder."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernel" / "epsilon"))

from epsilon_core.perception import MultimodalEncoder  # noqa: E402


class TestMultimodalEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.enc = MultimodalEncoder(dim=16, feature_dim=64, seed=0)

    def test_construct(self):
        self.assertEqual(self.enc.dim, 16)
        self.assertEqual(self.enc.feature_dim, 64)

    def test_encode_text_shape(self):
        v = self.enc.encode({"text": "hello"})
        self.assertEqual(v.shape, (16,))

    def test_encode_l2_normalised(self):
        v = self.enc.encode({"text": "hello world"})
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_encode_deterministic(self):
        a = self.enc.encode({"text": "foo"})
        b = self.enc.encode({"text": "foo"})
        np.testing.assert_allclose(a, b)

    def test_encode_different_text_different_vector(self):
        a = self.enc.encode({"text": "foo"})
        b = self.enc.encode({"text": "bar"})
        self.assertGreater(float(np.linalg.norm(a - b)), 1e-6)

    def test_encode_empty_dict(self):
        v = self.enc.encode({})
        self.assertEqual(v.shape, (16,))


if __name__ == "__main__":
    unittest.main()
