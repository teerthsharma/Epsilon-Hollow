import asyncio
import sys
import builtins
import yaml
import psutil

# ── Redirect ALL print() calls to stderr ─────────────────────────────────────
# stdout is reserved exclusively for JSON responses to VS Code.
# Every component (bitnet_model, clara, orchestrator etc.) uses plain print().
# This one override ensures none of those lines pollute the JSON output stream.
_real_print = builtins.print
def _stderr_print(*args, **kwargs):
    kwargs.setdefault('file', sys.stderr)
    _real_print(*args, **kwargs)
builtins.print = _stderr_print

# Now import everything — their print() calls will go to stderr automatically
sys.path.insert(0, '/home/lovekesh/epsilon')

from backend.tiers.bitnet_model           import BitNetModel
from backend.inference.tinygrad_kv        import SparseAttentionKVCache
from backend.clara.potato_oracle          import PotatoClaraOracle
from backend.picoclaw.potato_orchestrator import PotatoOrchestrator
from backend.aether.aether_link           import AetherLink


def log(msg: str) -> None:
    _real_print(msg, file=sys.stderr, flush=True)


def print_ram(label: str) -> None:
    ram_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    log(f"  [{label}] RAM: {ram_mb:.0f} MB")


def load_config() -> dict:
    config_path = '/home/lovekesh/epsilon/config.yaml'
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log(f"ERROR: config.yaml not found at {config_path}")
        sys.exit(1)


def main():
    config = load_config()

    log("=" * 45)
    log("  EPSILON ENGINE STARTING")
    log("=" * 45)
    print_ram("start")

    log("\n[1/4] Loading BitNet model...")
    model = BitNetModel(
        model_path = config['model_path'],
        context    = config.get('context_len', 512),
        threads    = config.get('cpu_threads', 4),
    )
    print_ram("model loaded")

    log("\n[2/4] Allocating KV cache...")
    kv = SparseAttentionKVCache(
        top_k      = 64,
        n_layers   = 32,
        n_heads    = 32,
        max_tokens = config.get('context_len', 512),
        d_head     = 64,
    )
    print_ram("kv cache ready")

    log("\n[3/4] Starting Clara...")
    clara = PotatoClaraOracle(config.get('db_path', 'clara.db'))
    if config.get('crawl_on_start') and config.get('project_dir'):
        clara.crawl(config['project_dir'])
    print_ram("clara ready")

    log("\n[4/4] Starting orchestrator...")
    orchestrator = PotatoOrchestrator(model, clara)
    link         = AetherLink(orchestrator)
    print_ram("orchestrator ready")

    log("\n" + "=" * 45)
    log("  ENGINE READY — waiting for requests")
    log("=" * 45 + "\n")

    try:
        asyncio.run(link.run())
    except KeyboardInterrupt:
        log("\nCtrl+C received — shutting down...")
    finally:
        model.shutdown()
        log("Engine stopped.")


if __name__ == '__main__':
    main()
