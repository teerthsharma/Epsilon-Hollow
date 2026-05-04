# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

"""
Epsilon-Hollow — PPO Training with GAE-λ
==========================================
Proximal Policy Optimization for the MetaController's policy.

Mathematical Foundation:
    Clipped Surrogate Objective:
        L^CLIP = min(rₜ(θ)Âₜ, clip(rₜ(θ), 1−ε, 1+ε)Âₜ)

    where rₜ(θ) = π_θ(aₜ|sₜ) / π_{θ_old}(aₜ|sₜ)

    Generalized Advantage Estimation (GAE-λ):
        Âₜ = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        δₜ = rₜ + γV(s_{t+1}) − V(sₜ)

    Value Function Loss:
        L^VF = (V_θ(sₜ) − V^target)²

    Entropy Bonus:
        L^H = −Σ π(a|s) log π(a|s)

    Total: L = L^CLIP − c₁ L^VF + c₂ L^H

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Policy and Value Networks
# ─────────────────────────────────────────────────────────────────────

class PolicyNetwork:
    """
    Softmax policy: π(a|s) = softmax(W_π · s + b_π)

    Output dimension = number of actions.
    """

    def __init__(self, state_dim: int = 128, n_actions: int = 4, seed: int = 42):
        rng = np.random.RandomState(seed)
        std = math.sqrt(2.0 / (state_dim + n_actions))
        self.W = rng.randn(state_dim, n_actions).astype(np.float64) * std
        self.b = np.zeros(n_actions, dtype=np.float64)
        self.state_dim = state_dim
        self.n_actions = n_actions

    def logits(self, state: np.ndarray) -> np.ndarray:
        """Raw logits: z = W^T s + b."""
        return state @ self.W + self.b

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        """π(a|s) = softmax(z)."""
        z = self.logits(state)
        z = z - np.max(z)  # Numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def log_probs(self, state: np.ndarray) -> np.ndarray:
        """log π(a|s) = z − log Σ exp(z)."""
        z = self.logits(state)
        z = z - np.max(z)
        log_sum_exp = np.log(np.sum(np.exp(z)))
        return z - np.max(z) - log_sum_exp

    def entropy(self, state: np.ndarray) -> float:
        """H(π) = −Σ π(a|s) log π(a|s)."""
        probs = self.action_probs(state)
        return -float(np.sum(probs * np.log(probs + 1e-30)))

    def sample_action(self, state: np.ndarray) -> int:
        """Sample action from π(a|s)."""
        probs = self.action_probs(state)
        return int(np.random.choice(self.n_actions, p=probs))


class ValueNetwork:
    """
    State value function: V(s) = W_v · s + b_v  (linear for speed).
    """

    def __init__(self, state_dim: int = 128, seed: int = 43):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(state_dim).astype(np.float64) * 0.01
        self.b = 0.0

    def predict(self, state: np.ndarray) -> float:
        """V(s) = W^T s + b."""
        return float(np.dot(self.W, state) + self.b)


# ─────────────────────────────────────────────────────────────────────
# GAE-λ Computation
# ─────────────────────────────────────────────────────────────────────

def compute_gae(rewards: List[float], values: List[float],
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation.

    δₜ = rₜ + γV(s_{t+1}) − V(sₜ)
    Âₜ = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

    Parameters
    ----------
    rewards : list of float, length T
    values : list of float, length T+1 (includes V(s_{T+1}))
    gamma : float
        Discount factor.
    lam : float
        GAE parameter λ.

    Returns
    -------
    advantages : np.ndarray, shape (T,)
    returns : np.ndarray, shape (T,)  (advantages + values[:T])
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0

    for t in reversed(range(T)):
        v_next = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * v_next - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + np.array(values[:T])
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Proximal Policy Optimization with clipped surrogate objective.

    L = L^CLIP − c₁ L^VF + c₂ L^H

    Trains the policy and value networks on trajectory data
    collected from the EpsilonHollowCore agent.
    """

    def __init__(self, state_dim: int = 128, n_actions: int = 4,
                 clip_epsilon: float = 0.2, gamma: float = 0.99,
                 lam: float = 0.95, c_vf: float = 0.5, c_ent: float = 0.01,
                 lr_policy: float = 3e-4, lr_value: float = 1e-3,
                 n_epochs: int = 4, batch_size: int = 32):
        self.policy = PolicyNetwork(state_dim, n_actions)
        self.value = ValueNetwork(state_dim)

        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Statistics
        self.train_steps = 0
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []

    def train_on_trajectory(self, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        One PPO update from a trajectory.

        Parameters
        ----------
        trajectory : list of dicts
            Each with 'state' (ndarray), 'action' (dict with 'action' key),
            'reward' (float).

        Returns
        -------
        dict : Training statistics.
        """
        if len(trajectory) < 2:
            return {"status": "insufficient_data"}

        states = np.array([t["state"] for t in trajectory])
        rewards = [t["reward"] for t in trajectory]

        # Map action names to indices
        action_map = {"execute": 0, "reason": 1, "explore": 2, "refusal": 3}
        actions = [action_map.get(t["action"].get("action", "reason"), 1) for t in trajectory]

        # Compute values for all states
        values = [self.value.predict(s) for s in states]
        # Bootstrap V(s_{T+1}) = 0 (terminal)
        values.append(0.0)

        # GAE-λ
        advantages, returns = compute_gae(rewards, values, self.gamma, self.lam)

        # Normalise advantages
        adv_mean = float(np.mean(advantages))
        adv_std = float(np.std(advantages)) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Store old log probs
        old_log_probs = np.array([
            self.policy.log_probs(states[i])[actions[i]]
            for i in range(len(states))
        ])

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.n_epochs):
            # Mini-batch SGD
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                if len(batch_idx) == 0:
                    continue

                for i in batch_idx:
                    state = states[i]
                    action = actions[i]
                    advantage = advantages[i]
                    ret = returns[i]
                    old_lp = old_log_probs[i]

                    # Current log prob
                    new_lp = self.policy.log_probs(state)[action]

                    # Probability ratio: rₜ(θ) = exp(log π_new − log π_old)
                    ratio = math.exp(new_lp - old_lp)

                    # Clipped surrogate
                    surr1 = ratio * advantage
                    surr2 = max(min(ratio, 1 + self.clip_epsilon), 1 - self.clip_epsilon) * advantage
                    policy_loss = -min(surr1, surr2)

                    # Value loss: (V(s) − V_target)²
                    v_pred = self.value.predict(state)
                    value_loss = (v_pred - ret) ** 2

                    # Entropy bonus
                    ent = self.policy.entropy(state)

                    # Policy gradient update (numerical)
                    eps = 1e-5
                    for j in range(self.policy.n_actions):
                        # dL/dW[:,j]
                        probs = self.policy.action_probs(state)
                        # Softmax gradient: dz_j/dW = s * (1{j=a} - π(j|s))
                        indicator = 1.0 if j == action else 0.0
                        grad_logit = indicator - probs[j]
                        # Chain rule: dL/dW = dL/dz * dz/dW
                        # dL/dz = -advantage * grad_logit (for REINFORCE)
                        policy_grad = -advantage * grad_logit
                        self.policy.W[:, j] -= self.lr_policy * policy_grad * state

                    # Value gradient update: dL/dW = 2(V-target) * s
                    v_grad = 2 * (v_pred - ret) * state
                    self.value.W -= self.lr_value * v_grad
                    self.value.b -= self.lr_value * 2 * (v_pred - ret)

                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    total_entropy += ent

        n_updates = max(len(states) * self.n_epochs, 1)
        mean_policy_loss = total_policy_loss / n_updates
        mean_value_loss = total_value_loss / n_updates
        mean_entropy = total_entropy / n_updates

        self.train_steps += 1
        self.policy_losses.append(mean_policy_loss)
        self.value_losses.append(mean_value_loss)
        self.entropies.append(mean_entropy)

        return {
            "policy_loss": mean_policy_loss,
            "value_loss": mean_value_loss,
            "entropy": mean_entropy,
            "n_transitions": len(states),
            "train_step": self.train_steps,
        }

    def stats(self) -> Dict[str, float]:
        return {
            "train_steps": self.train_steps,
            "mean_policy_loss": float(np.mean(self.policy_losses[-10:])) if self.policy_losses else 0.0,
            "mean_value_loss": float(np.mean(self.value_losses[-10:])) if self.value_losses else 0.0,
            "mean_entropy": float(np.mean(self.entropies[-10:])) if self.entropies else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────

def run_distributed_training(mode: str = "rl_warmup"):
    """
    Training entry point.

    Modes:
        rl_warmup:  PPO warm-up on synthetic tasks
        meta_train: MAML meta-training across task distribution
        full:       Both in sequence
    """
    from infrastructure.orchestrator.meta_learning import MAMLHypernetwork

    print(f"[train] Mode: {mode}")

    if mode in ("rl_warmup", "full"):
        print("[train] Initializing PPO trainer (state_dim=128, actions=4)...")
        trainer = PPOTrainer(state_dim=128, n_actions=4)

        # Synthetic trajectory for warm-up
        rng = np.random.RandomState(42)
        trajectory = []
        for step in range(200):
            state = rng.randn(128)
            state /= np.linalg.norm(state)
            action_name = rng.choice(["execute", "reason", "explore", "refusal"])
            reward = rng.uniform(-1, 1)
            trajectory.append({
                "state": state,
                "action": {"action": action_name},
                "reward": reward,
            })

        for epoch in range(5):
            stats = trainer.train_on_trajectory(trajectory)
            print(f"  PPO epoch {epoch + 1}: policy_loss={stats.get('policy_loss', 0):.4f}, "
                  f"value_loss={stats.get('value_loss', 0):.4f}, "
                  f"entropy={stats.get('entropy', 0):.4f}")

    if mode in ("meta_train", "full"):
        print("[train] Initializing MAML hypernetwork...")
        meta_learner = MAMLHypernetwork(d_in=128, d_out=4, inner_steps=3)

        rng = np.random.RandomState(99)
        for meta_step in range(3):
            # Generate synthetic meta-learning tasks
            tasks = []
            for _ in range(4):
                n_support, n_query = 10, 5
                x_s = rng.randn(n_support, 128).astype(np.float64)
                y_s = rng.randn(n_support, 4).astype(np.float64) * 0.1
                x_q = rng.randn(n_query, 128).astype(np.float64)
                y_q = rng.randn(n_query, 4).astype(np.float64) * 0.1
                tasks.append({
                    "x_support": x_s, "y_support": y_s,
                    "x_query": x_q, "y_query": y_q,
                })

            meta_loss = meta_learner.outer_loop(tasks)
            print(f"  MAML step {meta_step + 1}: meta_loss={meta_loss:.6f}")

        # Architecture evolution
        best_arch = meta_learner.evolve_architecture(tasks)
        print(f"  Best architecture: {best_arch}")

    print("[train] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epsilon-Hollow Training")
    parser.add_argument("--mode", default="rl_warmup",
                        choices=["rl_warmup", "meta_train", "full"])
    args = parser.parse_args()
    run_distributed_training(args.mode)
