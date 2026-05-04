# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

"""
Smoke tests for the Epsilon-Hollow agent core loop.

These tests verify that the basic agent pipeline works end-to-end:
  1. A single step() call succeeds
  2. Two consecutive step() calls don't crash (catches string-vs-ndarray bug)
  3. perception.encode() returns an ndarray, not a string
"""

import sys
import os
import numpy as np

# Ensure epsilon_core and tools are importable
# tests/ -> epsilon_core/ -> kernel/epsilon/ (for epsilon_core.*)
# tests/ -> epsilon_core/ -> kernel/epsilon/ -> kernel/ -> repo_root/ -> infrastructure/ (for tools.*)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_EPSILON_CORE_DIR = os.path.dirname(_TESTS_DIR)
_KERNEL_EPSILON_DIR = os.path.dirname(_EPSILON_CORE_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_KERNEL_EPSILON_DIR))
_INFRASTRUCTURE_DIR = os.path.join(_REPO_ROOT, "infrastructure")

sys.path.insert(0, _KERNEL_EPSILON_DIR)
sys.path.insert(0, _INFRASTRUCTURE_DIR)

from epsilon_core.agent import EpsilonHollowCore
from epsilon_core.perception import MultimodalEncoder


def test_single_step():
    """One step() call should succeed and return a dict."""
    agent = EpsilonHollowCore()
    observation = {"text": "Hello world", "vision": None}
    result = agent.step(observation)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    print("[PASS] test_single_step")


def test_two_consecutive_steps():
    """Two consecutive step() calls must not crash.

    This is the regression test for the original bug where perception.encode()
    returned a string, causing memory.retrieve() to crash on np.dot() during
    the second call (when the episodic store contained the string "embedding").
    """
    agent = EpsilonHollowCore()
    obs1 = {"text": "First observation", "vision": None}
    obs2 = {"text": "Second observation", "vision": None}

    result1 = agent.step(obs1)
    assert isinstance(result1, dict), f"Step 1 failed: expected dict, got {type(result1)}"

    result2 = agent.step(obs2)
    assert isinstance(result2, dict), f"Step 2 failed: expected dict, got {type(result2)}"
    print("[PASS] test_two_consecutive_steps")


def test_latent_is_ndarray():
    """perception.encode() must return a numpy ndarray, not a string."""
    encoder = MultimodalEncoder(dim=512)
    observation = {"text": "Test input", "vision": None}
    latent = encoder.encode(observation)

    assert isinstance(latent, np.ndarray), (
        f"Expected np.ndarray, got {type(latent)}. "
        f"Value: {repr(latent)[:100]}"
    )
    assert latent.shape == (512,), f"Expected shape (512,), got {latent.shape}"

    # Verify it's numeric and can be used in np.dot
    dot_result = np.dot(latent, latent)
    assert isinstance(dot_result, (float, np.floating)), (
        f"np.dot(latent, latent) should return float, got {type(dot_result)}"
    )
    print("[PASS] test_latent_is_ndarray")


def test_deterministic_encoding():
    """Same input should produce same embedding (reproducibility)."""
    encoder = MultimodalEncoder(dim=512)
    obs = {"text": "Determinism test", "vision": None}

    v1 = encoder.encode(obs)
    v2 = encoder.encode(obs)

    assert np.allclose(v1, v2), "Encoding is not deterministic"
    print("[PASS] test_deterministic_encoding")


if __name__ == "__main__":
    test_single_step()
    test_two_consecutive_steps()
    test_latent_is_ndarray()
    test_deterministic_encoding()
    print("\n[ALL PASS] Agent smoke tests completed successfully.")
