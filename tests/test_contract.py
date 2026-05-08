"""Round-trip tests for the Python ↔ Rust ManifoldPayload contract.

The Rust struct is the source of truth; this test exists to catch Python-side
drift before it reaches the wire.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "kernel" / "epsilon"))

from epsilon_core.contract import (  # noqa: E402
    SCHEMA_ID,
    ManifoldPayload,
    decode,
    encode,
    validate,
)


class TestManifoldPayloadContract(unittest.TestCase):
    def test_round_trip(self):
        payload = ManifoldPayload.from_points(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dim=3, b0=1, b1=0, b2=1
        )
        wire = encode(payload)
        restored = decode(wire)
        self.assertEqual(payload, restored)

    def test_dim_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ManifoldPayload.from_points([[0.1, 0.2]], dim=3)

    def test_point_count_exceeds_array(self):
        bad = encode(ManifoldPayload.from_points([[1.0]], dim=1))
        bad["point_count"] = 99
        with self.assertRaises(ValueError):
            validate(bad)

    def test_extra_field_rejected(self):
        wire = encode(ManifoldPayload.from_points([[1.0]], dim=1))
        wire["bogus"] = True
        with self.assertRaises(ValueError):
            validate(wire)

    def test_missing_field_rejected(self):
        wire = encode(ManifoldPayload.from_points([[1.0]], dim=1))
        del wire["signature_b1"]
        with self.assertRaises(ValueError):
            validate(wire)

    def test_negative_liveness_rejected(self):
        wire = encode(ManifoldPayload.from_points([[1.0]], dim=1))
        wire["liveness_anchor"] = -0.5
        with self.assertRaises(ValueError):
            validate(wire)

    def test_schema_file_exists_and_parses(self):
        schema_path = REPO_ROOT / "schemas" / "manifold_payload.schema.json"
        self.assertTrue(schema_path.exists(), schema_path)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], SCHEMA_ID)


if __name__ == "__main__":
    unittest.main()
