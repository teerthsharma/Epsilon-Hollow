"""
Epsilon-Hollow Core — Agent Entry Point

Run from repo root:
    python kernel/epsilon/epsilon_core/main.py

Or via the orchestrator (README command):
    python infrastructure/orchestrator/main.py
"""

import sys
import os
import time

# Ensure epsilon_core is importable regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from epsilon_core.agent import EpsilonHollowCore


def main():
    print("Initializing Epsilon-Hollow Orchestrator...")
    omega_agent = EpsilonHollowCore()
    
    print("Epsilon-Hollow Online. Beginning continuous learning loop.")
    env_observation = {"text": "Solve the Riemann hypothesis via numerical simulation code.", "vision": None}
    
    try:
        while True:
            result = omega_agent.step(env_observation)
            print(f"Action Result: {result}")
            # Simulated environment step
            time.sleep(2)
            break  # Break for safety in standalone mode
    except KeyboardInterrupt:
        print("Shutting down Epsilon-Hollow cleanly.")

if __name__ == "__main__":
    main()