from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


class MetaController:
    """Decides between execution, reasoning, exploration, and refusal.

    Policy:
        π(a|s) = softmax(Ws + b)

    The controller keeps a lightweight linear policy head that can be
    trained by PPO (`infrastructure/training/train.py`) while remaining
    deterministic enough for local development loops.
    """

    ACTIONS = ("execute", "reason", "explore", "refusal")

    def __init__(self, tools_registry, safety_filter,
                 state_dim: int = 128, temperature: float = 0.8,
                 seed: int = 42):
        self.tools = tools_registry or {}
        self.safety = safety_filter
        self.state_dim = state_dim
        self.temperature = max(temperature, 1e-3)

        self._rng = np.random.RandomState(seed)
        scale = math.sqrt(2.0 / (state_dim + len(self.ACTIONS)))
        self.W = self._rng.randn(state_dim, len(self.ACTIONS)).astype(np.float64) * scale
        self.b = np.zeros(len(self.ACTIONS), dtype=np.float64)

        # Keep refusal conservative by default.
        self.b[self.ACTIONS.index("execute")] = 0.20
        self.b[self.ACTIONS.index("reason")] = 0.35
        self.b[self.ACTIONS.index("refusal")] = -0.70

    def _extract_state_vector(self, latent_state: Any) -> np.ndarray:
        """Normalize latent input into a fixed-size policy state vector."""
        context_len = 0
        step = 0.0

        if isinstance(latent_state, dict):
            raw = latent_state.get("latent")
            context = latent_state.get("context", [])
            if isinstance(context, list):
                context_len = len(context)
            step = float(latent_state.get("step", 0) or 0)
        else:
            raw = latent_state

        if raw is None:
            vec = np.zeros(0, dtype=np.float64)
        else:
            vec = np.asarray(raw, dtype=np.float64).reshape(-1)

        features = np.zeros(self.state_dim, dtype=np.float64)
        copy_len = min(self.state_dim, vec.size)
        if copy_len > 0:
            features[:copy_len] = vec[:copy_len]

        norm = np.linalg.norm(features)
        if norm > 1e-12:
            features /= norm

        # Inject small control signals so routing reacts to memory/context pressure.
        if self.state_dim > 0:
            features[0] += min(context_len, 20) / 20.0
        if self.state_dim > 1:
            features[1] += math.tanh(step / 1000.0)

        return features

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.temperature
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def _policy_probs(self, state_vec: np.ndarray, latent_state: Any) -> np.ndarray:
        logits = state_vec @ self.W + self.b

        # If no tools are available, downweight execution.
        if not self.tools:
            logits[self.ACTIONS.index("execute")] -= 1.5

        # If context is empty, encourage exploration.
        if isinstance(latent_state, dict) and not latent_state.get("context"):
            logits[self.ACTIONS.index("explore")] += 0.2

        return self._softmax(logits)

    def _execution_payload(self, latent_state: Any) -> Dict[str, Any]:
        if isinstance(latent_state, dict):
            step = latent_state.get("step", 0)
            return {"code": f"print('Epsilon-Hollow step {step}: policy execution path')"}
        return {"code": "print('Epsilon-Hollow policy execution path')"}

    def decide_action(self, latent_state) -> dict:
        state_vec = self._extract_state_vector(latent_state)
        probs = self._policy_probs(state_vec, latent_state)
        action_idx = int(self._rng.choice(len(self.ACTIONS), p=probs))
        policy_decision = self.ACTIONS[action_idx]

        if not self.safety.check_compliance(state_vec, policy_decision):
            return {
                "action": "refusal",
                "reason": "Constitutional alignment violation",
                "policy_probs": probs.tolist(),
            }

        if policy_decision == "execute":
            if "PythonExecutor" in self.tools:
                tool = "PythonExecutor"
            elif self.tools:
                tool = sorted(self.tools.keys())[0]
            else:
                tool = ""

            if tool:
                return {
                    "action": "execute",
                    "tool": tool,
                    "payload": self._execution_payload(latent_state),
                    "policy_probs": probs.tolist(),
                }

            policy_decision = "reason"

        if policy_decision == "explore":
            return {
                "action": "explore",
                "target": "memory_frontier",
                "policy_probs": probs.tolist(),
            }

        if policy_decision == "refusal":
            return {
                "action": "refusal",
                "reason": "Policy-selected refusal",
                "policy_probs": probs.tolist(),
            }

        return {
            "action": "reason",
            "target": "internal_model",
            "policy_probs": probs.tolist(),
        }
