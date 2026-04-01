"""
backend/main.py
================
Epsilon IDE Engine v2 — entry point.

Fixes:
  - memory.flush() now exists (was crashing on shutdown)
  - llama_server_bin passed through config to model_manager
  - Cleaner boot sequence with error recovery
"""

import asyncio
import builtins
import sys
import argparse
import yaml
import psutil

# ── Redirect print() to stderr to keep stdout clean for JSON ─────────────────
_real_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    _real_print(*args, **kwargs)
builtins.print = _stderr_print

sys.path.insert(0, "/mnt/d/epsilon/v2")

from backend.tiers.model_manager  import TieredModelManager
from backend.memory.conversation  import ConversationMemory
from backend.clara.oracle         import ClaraOracle
from backend.agents.orchestrator  import Orchestrator
from backend.aether.link          import AetherLink


def log(msg: str):
    _real_print(msg, file=sys.stderr, flush=True)

def ram() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)

def load_config(path: str = "/mnt/d/epsilon/v2/config.yaml") -> dict:
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                log("ERROR: config.yaml is empty or malformed")
                sys.exit(1)
            return cfg
    except FileNotFoundError:
        log(f"ERROR: config.yaml not found at {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        log(f"ERROR: config.yaml parse error: {e}")
        sys.exit(1)


async def boot(config: dict):
    log("=" * 52)
    log("  EPSILON IDE ENGINE v2.2")
    log("  Three-Tier Async Model Routing")
    log("=" * 52)
    log(f"  RAM at start: {ram():.0f} MB")

    # ── 1. Model manager ──────────────────────────────────────────────────────
    log("\n[1/4] Starting async model manager...")
    model_manager = TieredModelManager(config)
    await model_manager.startup()
    log(f"  Active tier: {model_manager.current_tier}")
    log(f"  RAM after model load: {ram():.0f} MB")

    # ── 2. Conversation memory ────────────────────────────────────────────────
    log("\n[2/4] Loading conversation memory...")
    memory = ConversationMemory(
        memory_path=config.get("memory_path", "/mnt/d/epsilon/v2/conversation.json"),
        max_turns=config.get("memory_turns", 10),
    )
    s = memory.stats()
    log(f"  {s['turns']} turns in history ({s['total_messages']} messages)")

    # ── 3. Clara code search ──────────────────────────────────────────────────
    log("\n[3/4] Starting Clara code search...")
    try:
        clara = ClaraOracle(config.get("db_path", "/mnt/d/epsilon/v2/clara.db"))
        if config.get("crawl_on_start") and config.get("project_dir"):
            clara.crawl(config["project_dir"])
    except Exception as e:
        log(f"  WARNING: Clara failed to start: {e}")
        clara = None

    # ── 4. Orchestrator ───────────────────────────────────────────────────────
    log("\n[4/4] Building orchestrator...")
    orchestrator = Orchestrator(
        model_manager=model_manager,
        kv_cache=None,
        memory=memory,
        clara=clara,
        config=config,
    )

    log(f"\n  RAM ready: {ram():.0f} MB")
    log("\n" + "=" * 52)
    log("  ENGINE READY — awaiting requests")
    log("=" * 52 + "\n")

    return orchestrator, model_manager, memory


async def main_async():
    parser = argparse.ArgumentParser(description="Epsilon IDE Engine")
    parser.add_argument("--telegram", action="store_true",
                        help="Run Telegram bot interface")
    parser.add_argument("--all", action="store_true",
                        help="Run both Telegram bot and stdin link")
    parser.add_argument("--oneshot", action="store_true",
                        help="Process one request and exit (for testing)")
    parser.add_argument("--config", default="/mnt/d/epsilon/v2/config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    orchestrator, model_manager, memory = await boot(config)

    try:
        if args.telegram or args.all:
            token = config.get("telegram_token", "")
            if not token or token == "YOUR_BOT_TOKEN_HERE":
                log("ERROR: Set telegram_token in config.yaml")
                sys.exit(1)

            from backend.telegram.bot import EpsilonTelegramBot
            bot = EpsilonTelegramBot(
                token=token,
                orchestrator=orchestrator,
                memory=memory,
                model=model_manager,
                allowed_users=config.get("telegram_allowed_users", []),
            )

            if args.all:
                import threading
                t = threading.Thread(target=bot.run, daemon=True)
                t.start()
                link = AetherLink(orchestrator, oneshot=args.oneshot)
                await link.run()
            else:
                bot.run()
        else:
            link = AetherLink(orchestrator, oneshot=args.oneshot)
            await link.run()

    except KeyboardInterrupt:
        log("\nInterrupted — shutting down...")
    finally:
        await model_manager.shutdown()
        memory.flush()
        log("Engine stopped cleanly.")


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()