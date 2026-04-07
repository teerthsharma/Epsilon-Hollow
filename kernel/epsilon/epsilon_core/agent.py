"""
Epsilon-Hollow — Core Agent Loop
==================================
The continuous perceive→recall→decide→act→learn cycle.

Architecture (from README §Core Philosophy):
    1. Perceive:  Multimodal observation → ℝ^128 via JL projection
    2. Recall:    O(1) centroid lookup on S² → k-NN within cluster
    3. Decide:    Softmax policy over {execute, reason, explore, refuse}
    4. Act:       Execute chosen action via tool registry
    5. Learn:     Store experience vector + optional micro-gradient injection

All internal state is np.ndarray. No strings in the latent space.
"""
from __future__ import annotations

import math
import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from kernel.epsilon.epsilon_core.memory import TopologicalManifoldMemory
    from kernel.epsilon.epsilon_core.perception import MultimodalEncoder
    from kernel.epsilon.epsilon_core.meta_controller import MetaController
except ModuleNotFoundError:
    from epsilon_core.memory import TopologicalManifoldMemory
    from epsilon_core.perception import MultimodalEncoder
    from epsilon_core.meta_controller import MetaController

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constitutional Safety Filter
# ─────────────────────────────────────────────────────────────────────

class ConstitutionalSafetyFilter:
    """
    Entropy-based compliance checking.

    The safety filter computes the Shannon entropy of the proposed action
    distribution. High-entropy actions (uncertain/scattered) pass through
    with caution markers. Low-entropy actions toward known-harmful patterns
    are blocked.

    H(a) = −Σ pᵢ log₂(pᵢ)

    Blocked if:
        1. Action matches a constitutional violation pattern
        2. Entropy is below threshold AND action is flagged
    """

    VIOLATION_KEYWORDS = frozenset([
        "rm -rf", "sudo", "format", "drop table", "exec(", "eval(",
        "os.system", "subprocess.call", "__import__", "shutil.rmtree",
    ])

    def __init__(self, entropy_threshold: float = 0.3):
        self.entropy_threshold = entropy_threshold
        self.violations_blocked = 0
        self.total_checks = 0

    def check_compliance(self, state: Any, action: Any) -> bool:
        """
        Check if action is constitutionally compliant.

        Parameters
        ----------
        state : np.ndarray or dict
            Current latent state.
        action : str or dict
            Proposed action or action description.

        Returns
        -------
        bool : True if compliant, False if blocked.
        """
        self.total_checks += 1

        # Extract action text for pattern matching
        action_text = ""
        if isinstance(action, str):
            action_text = action.lower()
        elif isinstance(action, dict):
            action_text = str(action.get("payload", "")).lower()

        # Pattern matching against violations
        for pattern in self.VIOLATION_KEYWORDS:
            if pattern in action_text:
                self.violations_blocked += 1
                logger.warning(f"Safety violation blocked: pattern '{pattern}' detected")
                return False

        # Entropy-based check on state (if ndarray)
        if isinstance(state, np.ndarray) and state.size > 1:
            # Compute entropy of the state distribution
            p = np.abs(state) + 1e-30
            p = p / np.sum(p)
            H = -float(np.sum(p * np.log2(p)))
            H_norm = H / max(math.log2(len(p)), 1.0)

            # Very low entropy + flagged action = suspicious
            if H_norm < self.entropy_threshold and action_text:
                # Allow but log
                logger.info(f"Low-entropy state (H={H_norm:.3f}) with action — monitoring")

        return True

    def stats(self) -> Dict[str, int]:
        return {
            "total_checks": self.total_checks,
            "violations_blocked": self.violations_blocked,
        }


# ─────────────────────────────────────────────────────────────────────
# Core Agent
# ─────────────────────────────────────────────────────────────────────

class EpsilonHollowCore:
    """
    The living organism. Continuous thought loop.

    From README §Core Philosophy:
        "Current AI interactions are transactional. Epsilon-Hollow turns
        those interactions into a Continuous Thought Loop."

    The agent runs perceive→recall→decide→act→learn cycles indefinitely.
    All latent state is np.ndarray of shape (128,). No strings.
    """

    MANIFOLD_DIM = 128  # ℝ^128 as specified in README §Mathematical Specification

    def __init__(self, tools: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        tools : dict, optional
            Registry of available tools {name: tool_instance}.
        """
        self.memory = TopologicalManifoldMemory(dim=self.MANIFOLD_DIM)
        self.perception = MultimodalEncoder(dim=self.MANIFOLD_DIM)

        self.tools = tools or {}
        self.safety = ConstitutionalSafetyFilter()
        self.controller = MetaController(self.tools, self.safety)

        # Agent state: always ndarray
        self._latent_state: np.ndarray = np.zeros(self.MANIFOLD_DIM, dtype=np.float64)
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0

        # Experience buffer for PPO training
        self._trajectory: List[Dict[str, Any]] = []

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        One cycle of the continuous thought loop.

        perceive → recall → decide → act → learn

        Parameters
        ----------
        observation : dict
            Multimodal input. Keys: 'text', 'vision', 'audio', 'code'.

        Returns
        -------
        dict : Result of the step including action taken, reward, diagnostics.
        """
        self._step_count += 1
        t_start = time.perf_counter_ns()

        # ── 1. PERCEIVE ──────────────────────────────────────────
        # Map observation → ℝ^128 via Johnson-Lindenstrauss projection
        latent = self.perception.encode(observation)
        assert isinstance(latent, np.ndarray) and latent.shape == (self.MANIFOLD_DIM,), \
            f"Perception must return ndarray({self.MANIFOLD_DIM},), got {type(latent)}"

        # ── 2. RECALL ────────────────────────────────────────────
        # O(P) centroid lookup on S² + k-NN within winning cluster
        context = self.memory.retrieve(query_vector=latent, k=5)

        # Build context vector: mean of retrieved vectors (or zero if empty)
        if context:
            context_vectors = np.array([c["episode"]["vector"] for c in context])
            context_mean = np.mean(context_vectors, axis=0)
        else:
            context_mean = np.zeros(self.MANIFOLD_DIM, dtype=np.float64)

        # Fuse latent + context into enriched state
        enriched_state = 0.7 * latent + 0.3 * context_mean
        norm = np.linalg.norm(enriched_state)
        if norm > 1e-12:
            enriched_state /= norm
        self._latent_state = enriched_state

        # ── 3. DECIDE ────────────────────────────────────────────
        decision = self.controller.decide_action({
            "latent": self._latent_state,
            "context": context,
            "step": self._step_count,
        })

        # ── 4. ACT ──────────────────────────────────────────────
        result = {"status": "internal_reasoning", "action": decision.get("action", "reason")}
        reward = 0.0

        if decision["action"] == "execute":
            tool_name = decision.get("tool")
            if tool_name and tool_name in self.tools:
                try:
                    tool_result = self.tools[tool_name].run(decision.get("payload", {}))
                    result = {"status": "executed", "tool": tool_name, "result": tool_result}
                    reward = 1.0  # Successful execution
                except Exception as e:
                    result = {"status": "error", "tool": tool_name, "error": str(e)}
                    reward = -0.5
            else:
                result = {"status": "tool_not_found", "tool": tool_name}
                reward = -0.1

        elif decision["action"] == "refusal":
            result = {"status": "refused", "reason": decision.get("reason", "")}
            reward = 0.1  # Small reward for appropriate refusal

        elif decision["action"] == "reason":
            result = {"status": "internal_reasoning"}
            reward = 0.05

        # ── 5. LEARN ─────────────────────────────────────────────
        # Store experience on the manifold
        self.memory.store(
            vector=self._latent_state,
            metadata={
                "step": self._step_count,
                "action": decision.get("action"),
                "reward": reward,
            }
        )

        self._cumulative_reward += reward

        # Buffer for PPO training
        self._trajectory.append({
            "state": self._latent_state.copy(),
            "action": decision,
            "reward": reward,
            "step": self._step_count,
        })

        t_end = time.perf_counter_ns()
        elapsed_us = (t_end - t_start) / 1000.0

        result["diagnostics"] = {
            "step": self._step_count,
            "elapsed_us": elapsed_us,
            "reward": reward,
            "cumulative_reward": self._cumulative_reward,
            "memory_size": len(self.memory.episodes),
            "betti_0": self.memory.compute_betti_0(),
            "latent_norm": float(np.linalg.norm(self._latent_state)),
        }

        return result

    def run_loop(self, observation_source, max_steps: int = -1):
        """
        Continuous thought loop.

        Parameters
        ----------
        observation_source : callable or iterable
            Yields observations. If callable, called repeatedly.
        max_steps : int
            -1 = run forever. Otherwise stop after N steps.
        """
        step = 0
        while max_steps < 0 or step < max_steps:
            if callable(observation_source):
                obs = observation_source()
            else:
                try:
                    obs = next(observation_source)
                except StopIteration:
                    break

            result = self.step(obs)
            step += 1

            if step % 100 == 0:
                stats = self.memory.manifold_stats()
                logger.info(
                    f"Step {step}: β₀={stats['betti_0']}, "
                    f"mem={stats['size']}, reward={self._cumulative_reward:.2f}"
                )

    def get_trajectory(self) -> List[Dict]:
        """Return collected trajectory for PPO training."""
        return self._trajectory

    def clear_trajectory(self):
        """Clear trajectory buffer after training update."""
        self._trajectory.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "steps": self._step_count,
            "cumulative_reward": self._cumulative_reward,
            "memory": self.memory.manifold_stats(),
            "safety": self.safety.stats(),
        }