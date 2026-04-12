"""
Epsilon-Hollow Liquid Memory simulation engine.

Usage:
    python kernel/epsilon/epsilon_core/simulation_engine.py
    python kernel/epsilon/epsilon_core/simulation_engine.py --full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_EPSILON_DIR = os.path.dirname(SCRIPT_DIR)
KERNEL_DIR = os.path.dirname(KERNEL_EPSILON_DIR)
REPO_ROOT = os.path.dirname(KERNEL_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KERNEL_EPSILON_DIR not in sys.path:
    sys.path.insert(0, KERNEL_EPSILON_DIR)

try:
    from kernel.epsilon.epsilon_core.world_model import LiquidMemoryWorldModel
except ModuleNotFoundError:
    from epsilon_core.world_model import LiquidMemoryWorldModel


def _print_section(title: str, payload: Dict[str, Any]):
    print(f"\n[{title}]")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _make_action_sequence(n: int, seed: int = 7) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    actions: List[Dict[str, Any]] = []
    for i in range(n):
        actions.append({
            "type": "hypothetical_transition",
            "step": i,
            "vector": rng.randn(32).tolist(),
            "temperature": float(rng.uniform(0.1, 1.0)),
        })
    return actions


def run_engine(fast: bool = True) -> Dict[str, Any]:
    model = LiquidMemoryWorldModel(dim=128, action_dim=32, entropy_rate=10.0, seed=42)

    sim = model.simulate(
        n_memories=5000 if fast else 12000,
        n_queries=1000 if fast else 3000,
        k=5,
        n_clusters=64 if fast else 128,
        reset_before=True,
    )

    seed_obs = {
        "text": "liquid memory manifold planning state",
        "code": "def transition(s, a): return s + a",
    }
    seed_latent = model.encode_observation(seed_obs)

    dream = model.dream(
        latent_state=seed_latent,
        action_sequence=_make_action_sequence(24 if fast else 64),
        k=5,
        store_imagined=False,
    )

    theorem = model.run_theorem_suite(fast=fast)

    betti_trajectory = [
        step["context_hits"] for step in dream.get("trace", [])
    ]

    summary = {
        "simulation": {
            "mean_latency_ms": sim.get("mean_latency_ms"),
            "p95_latency_ms": sim.get("p95_latency_ms"),
            "o1_hit_rate": sim.get("o1_hit_rate"),
            "top1_cluster_accuracy": sim.get("top1_cluster_accuracy"),
        },
        "scm": sim.get("scm_contraction", {}),
        "dream": {
            "horizon": dream.get("horizon"),
            "total_reward": dream.get("total_reward"),
            "final_state_norm": dream.get("final_state_norm"),
            "mean_context_hits": float(np.mean(betti_trajectory)) if betti_trajectory else 0.0,
        },
        "theorems": theorem.get("checks", {}),
        "all_theorems_passed": theorem.get("all_passed", False),
        "wphb_horizon_tokens": theorem.get("results", {}).get("T10", {}).get("h100_analysis", {}).get("combined_horizon_millions"),
        "hcs_hyperbolic_better": theorem.get("results", {}).get("T5", {}).get("hyperbolic_better"),
        "teb_conclusion": theorem.get("results", {}).get("T8", {}).get("conclusion"),
    }

    return {
        "summary": summary,
        "details": {
            "simulation": sim,
            "dream": dream,
            "theorem": theorem,
            "status": model.status(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Liquid Memory World Model simulation engine")
    parser.add_argument("--full", action="store_true", help="Run larger simulation/theorem settings")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    args = parser.parse_args()

    result = run_engine(fast=not args.full)

    _print_section("SimulationSummary", result["summary"])

    if args.json:
        _print_section("SimulationDetails", result["details"])


if __name__ == "__main__":
    main()
