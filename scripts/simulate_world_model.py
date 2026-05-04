# Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
# SPDX-License-Identifier: Epsilon-Hollow

"""
Single-model simulation entrypoint for Epsilon-Hollow unified world model.

Usage:
    python scripts/simulate_world_model.py
    python scripts/simulate_world_model.py --memories 20000 --queries 5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kernel.epsilon.epsilon_core.world_model import UnifiedWorldModel


def _print_json(title: str, payload: Dict[str, Any]):
    print(f"\n[{title}]")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Run unified world-model simulation")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--memories", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--entropy-rate", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-theorems", action="store_true")
    parser.add_argument("--full-theorems", action="store_true")
    args = parser.parse_args()

    model = UnifiedWorldModel(
        dim=args.dim,
        entropy_rate=args.entropy_rate,
        seed=args.seed,
    )

    print("=" * 72)
    print("EPSILON-HOLLOW UNIFIED WORLD MODEL SIMULATION")
    print("=" * 72)

    sim = model.simulate(
        n_memories=args.memories,
        n_queries=args.queries,
        k=args.k,
        n_clusters=args.clusters,
        reset_before=True,
    )

    _print_json("Simulation", sim)

    if not args.skip_theorems:
        theorem = model.run_theorem_suite(fast=not args.full_theorems)
        checks = theorem.get("checks", {})
        passed = [k for k, v in checks.items() if v]
        failed = [k for k, v in checks.items() if not v]

        print("\n[Theorem Summary]")
        print(f"Passed: {len(passed)}/10")
        print(f"Failed: {', '.join(failed) if failed else 'None'}")

        _print_json("TheoremChecks", checks)

    _print_json("Status", model.status())


if __name__ == "__main__":
    main()
