"""
Epsilon-Hollow Orchestrator — Main Entry Point

This is the authoritative entrypoint referenced by the README:
    python infrastructure/orchestrator/main.py

It bootstraps the Epsilon-Hollow core agent and launches the IDE backend
on http://localhost:8742.
"""

import sys
import os
import argparse

# Resolve paths relative to repo root so this works regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # infrastructure/orchestrator -> repo root
KERNEL_DIR = os.path.join(REPO_ROOT, "kernel", "epsilon")

# Add kernel to path so epsilon_core imports work
if KERNEL_DIR not in sys.path:
    sys.path.insert(0, KERNEL_DIR)


def launch_agent():
    """Boot the Epsilon-Hollow core agent loop."""
    from epsilon_core.agent import EpsilonHollowCore

    print("[Epsilon-Hollow] Initializing Orchestrator...")
    agent = EpsilonHollowCore()
    print("[Epsilon-Hollow] Agent core online.")

    observation = {
        "text": "System ready. Awaiting instructions.",
        "vision": None,
    }

    try:
        result = agent.step(observation)
        print(f"[Epsilon-Hollow] Initial step result: {result}")
    except Exception as e:
        print(f"[Epsilon-Hollow] Agent step failed: {e}")
        return False

    return True


def launch_ide_backend():
    """Launch the sealMega IDE backend on port 8742."""
    ide_backend_dir = os.path.join(
        REPO_ROOT, "kernel", "epsilon", "epsilon-ide", "pentesting", "backend"
    )

    if not os.path.isdir(ide_backend_dir):
        print(f"[Epsilon-Hollow] IDE backend not found at {ide_backend_dir}")
        return False

    # Add backend to path so its imports resolve
    if ide_backend_dir not in sys.path:
        sys.path.insert(0, ide_backend_dir)

    try:
        import uvicorn
        print("[Epsilon-Hollow] Starting IDE backend on http://127.0.0.1:8742")
        # Import the FastAPI app from the backend
        sys.path.insert(0, ide_backend_dir)
        from main import app
        uvicorn.run(app, host="127.0.0.1", port=8742)
    except ImportError as e:
        print(f"[Epsilon-Hollow] Cannot start IDE backend: {e}")
        print("[Epsilon-Hollow] Install dependencies: pip install fastapi uvicorn")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Epsilon-Hollow Orchestrator — Unified Agent Entry Point"
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "ide", "full"],
        default="ide",
        help="Launch mode: agent (core loop only), ide (backend server), full (both)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8742,
        help="Port for IDE backend (default: 8742)",
    )
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Launch in developer mode",
    )
    args = parser.parse_args()

    print("=" * 52)
    print("  EPSILON-HOLLOW ORCHESTRATOR")
    print(f"  Mode: {args.mode}")
    if args.dev_mode:
        print("  Developer Mode: Enabled")
    print("=" * 52)

    if args.mode in ("agent", "full"):
        launch_agent()

    if args.mode in ("ide", "full"):
        launch_ide_backend()


if __name__ == "__main__":
    main()
