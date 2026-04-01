"""
backend/tiers/router.py
========================
The ModelRouter manages all three model tiers.

Core responsibilities:
  1. Score request complexity (1-10) to pick the right tier
  2. Load the correct model server for that tier
  3. Unload the previous model if switching tiers (VRAM is limited)
  4. Expose a single generate() method that callers do not need to change

Why complexity scoring instead of just using the biggest model always:
  - The 33B model takes 30-120 seconds to respond
  - 90% of requests are simple completions that the 1.5B handles perfectly
  - Routing correctly means fast responses for simple tasks, deep
    reasoning only when actually needed

Complexity scoring logic:
  - Short completions, single functions → score 1-3 → fast (1.5B)
  - Full file generation, bug fixing, explanations → score 4-7 → balanced (7B)
  - System design, architecture, multi-file projects → score 8-10 → deep (33B)

The user can always override with explicit keywords:
  "use fast: ..."      → forces fast tier
  "use balanced: ..."  → forces balanced tier
  "use deep: ..."      → forces deep tier
"""

import re
import time
from backend.tiers.model import ModelServer


# ── Complexity scoring keywords ───────────────────────────────────────────────
# Each keyword adds to the complexity score of a request.
# The final score determines which model tier handles the request.

COMPLEXITY_SIGNALS = {
    # High complexity — architectural thinking needed (+3 each)
    "design a system":      3,
    "architect":            3,
    "microservices":        3,
    "design pattern":       3,
    "entire project":       3,
    "from scratch":         3,
    "production ready":     3,
    "scalable":             2,
    "full application":     3,
    "complete project":     3,

    # Medium complexity — full file or module generation (+2 each)
    "create a file":        2,
    "write a module":       2,
    "implement a class":    2,
    "entire file":          2,
    "authentication":       2,
    "database schema":      2,
    "rest api":             2,
    "api endpoint":         2,
    "unit tests":           2,
    "test suite":           2,

    # Low-medium complexity — function or explanation (+1 each)
    "explain":              1,
    "write a function":     1,
    "implement":            1,
    "fix the bug":          1,
    "refactor":             1,
    "optimize":             1,
    "how does":             1,
}

# Explicit tier override keywords
TIER_OVERRIDES = {
    "use fast:":      "fast",
    "use balanced:":  "balanced",
    "use deep:":      "deep",
    "quick:":         "fast",
    "deep:":          "deep",
}


class ModelRouter:
    """
    Manages three model tiers and routes requests to the right one.

    Usage:
        router = ModelRouter(config)
        result = router.generate("write a binary search function")
        router.shutdown()

    The router automatically:
      - Scores the request complexity
      - Picks the appropriate tier
      - Loads that tier's model if not already loaded
      - Unloads the previous model if switching (VRAM management)
      - Generates the response
    """

    def __init__(self, config: dict):
        self.config          = config
        self.models_config   = config.get("models", {})
        self.routing_config  = config.get("routing", {
            "fast_max": 3, "balanced_max": 7, "deep_min": 8
        })

        # Currently loaded model server and its tier name
        self._active_server: ModelServer = None
        self._active_tier:   str         = None

        # Track how long each tier has been idle
        # If a tier has been idle for 5+ minutes, unload it to free VRAM
        self._last_used:     float       = time.time()
        self._idle_timeout:  int         = 300  # 5 minutes

        # Which tiers are available (model file exists on disk)
        self._available_tiers = self._detect_available_tiers()

        print(f"[ModelRouter] Available tiers: {self._available_tiers}")
        print(f"[ModelRouter] Routing mode: {config.get('model_tier', 'auto')}")

        # Pre-load the fast tier on startup — it should always be ready
        if "fast" in self._available_tiers:
            self._load_tier("fast")

    def _detect_available_tiers(self) -> list:
        """
        Check which model files actually exist on disk.
        Only available tiers can be used — missing files are skipped gracefully.
        """
        from pathlib import Path
        available = []
        for tier_name, tier_config in self.models_config.items():
            path = tier_config.get("path", "")
            if Path(path).exists():
                available.append(tier_name)
                print(f"[ModelRouter] Found {tier_name} model: {path}")
            else:
                print(f"[ModelRouter] {tier_name} model not found at {path} — tier unavailable")
        return available

    def score_complexity(self, prompt: str) -> int:
        """
        Score the complexity of a request from 1-10.

        Uses keyword matching to estimate how much reasoning the request needs.
        Simple one-line completions score 1-2.
        Full file generation scores 5-7.
        System design scores 8-10.

        Returns:
            int between 1 and 10
        """
        lower = prompt.lower()
        score = 1  # base score — everything starts at 1

        # Add complexity points from signal keywords
        for keyword, points in COMPLEXITY_SIGNALS.items():
            if keyword in lower:
                score += points

        # Extra points for long prompts (detailed requests are usually complex)
        word_count = len(prompt.split())
        if word_count > 50:
            score += 1
        if word_count > 100:
            score += 1
        if word_count > 200:
            score += 2

        # Extra points for multi-file requests
        if prompt.count("\n") > 10:
            score += 1

        return min(score, 10)  # cap at 10

    def pick_tier(self, prompt: str) -> str:
        """
        Decide which tier should handle this request.

        Checks for explicit tier overrides first (e.g. "use deep: ...").
        Falls back to complexity scoring with config thresholds.
        If the ideal tier is not available (model not downloaded),
        falls back to the best available tier.

        Returns:
            tier name: "fast", "balanced", or "deep"
        """
        lower = prompt.lower()

        # Check for explicit user override first
        for prefix, tier in TIER_OVERRIDES.items():
            if lower.startswith(prefix):
                if tier in self._available_tiers:
                    print(f"[ModelRouter] Explicit override: using {tier} tier")
                    return tier
                else:
                    print(f"[ModelRouter] Override {tier} requested but not available — falling back")

        # Check global tier config
        global_tier = self.config.get("model_tier", "auto")
        if global_tier != "auto":
            if global_tier in self._available_tiers:
                return global_tier

        # Auto routing — score complexity and pick tier
        score = self.score_complexity(prompt)
        print(f"[ModelRouter] Complexity score: {score}/10")

        fast_max     = self.routing_config.get("fast_max", 3)
        balanced_max = self.routing_config.get("balanced_max", 7)

        if score <= fast_max:
            ideal = "fast"
        elif score <= balanced_max:
            ideal = "balanced"
        else:
            ideal = "deep"

        # Fall back if ideal tier not available
        if ideal in self._available_tiers:
            return ideal

        # Fallback cascade: deep → balanced → fast
        for fallback in ["balanced", "fast", "deep"]:
            if fallback in self._available_tiers:
                print(f"[ModelRouter] {ideal} not available — using {fallback}")
                return fallback

        raise RuntimeError("No model tiers available. Check model paths in config.yaml")

    def _load_tier(self, tier_name: str) -> None:
        """
        Load a model tier, unloading the current one first if needed.

        VRAM management:
          - fast (1.5B) uses ~1 GB VRAM
          - balanced (7B) uses ~4 GB VRAM
          - deep (33B) uses 0 VRAM (CPU streaming)

        When switching from fast to balanced:
          1. Shutdown fast server (frees 1 GB VRAM)
          2. Start balanced server (needs 4 GB VRAM)

        When switching from balanced to fast:
          1. Shutdown balanced server (frees 4 GB VRAM)
          2. Start fast server (needs 1 GB VRAM)
        """
        if self._active_tier == tier_name:
            return  # already loaded — nothing to do

        # Unload current model to free VRAM before loading new one
        if self._active_server is not None:
            print(f"[ModelRouter] Unloading {self._active_tier} tier to free VRAM...")
            self._active_server.shutdown()
            self._active_server = None
            self._active_tier   = None
            time.sleep(1)  # brief pause to let VRAM fully release

        # Get config for the target tier
        tier_config = self.models_config.get(tier_name, {})
        if not tier_config:
            raise ValueError(f"No config found for tier: {tier_name}")

        print(f"[ModelRouter] Loading {tier_name} tier...")
        print(f"[ModelRouter] {tier_config.get('description', tier_name)}")

        # Build a server config from the tier config
        server_config = {
            "model_path":   tier_config["path"],
            "context_len":  tier_config.get("context_len", 2048),
            "cpu_threads":  self.config.get("cpu_threads", 4),
            "server_port":  self.config.get("server_port", 8088),
            "server_host":  self.config.get("server_host", "localhost"),
            "gpu_layers":   tier_config.get("gpu_layers", 28),
        }

        self._active_server = ModelServer(server_config)
        self._active_tier   = tier_name
        self._last_used     = time.time()

        print(f"[ModelRouter] {tier_name} tier ready")

    def generate(self, prompt: str,
                 max_tokens: int = None,
                 temperature: float = None,
                 force_tier: str = None) -> str:
        """
        Generate a response, automatically routing to the right tier.

        Args:
            prompt:      The request text
            max_tokens:  Override the tier's default max_tokens
            temperature: Override the tier's default temperature
            force_tier:  Skip scoring and use this tier directly

        Returns:
            Generated text string
        """

        # Pick tier
        tier_name = force_tier if force_tier else self.pick_tier(prompt)
        print(f"[ModelRouter] Using {tier_name} tier")

        # Load the tier (unloads current if different)
        self._load_tier(tier_name)
        self._last_used = time.time()

        # Use tier defaults if not overridden
        tier_config  = self.models_config.get(tier_name, {})
        actual_max   = max_tokens   or tier_config.get("max_tokens", 256)
        actual_temp  = temperature  or tier_config.get("temperature", 0.2)

        # Generate
        result = self._active_server.generate(
            prompt,
            max_tokens  = actual_max,
            temperature = actual_temp,
            raw         = True,
        )

        return result

    def get_active_tier(self) -> str:
        """Return the name of the currently loaded tier."""
        return self._active_tier or "none"

    def get_status(self) -> dict:
        """Return router status for monitoring and Telegram /status command."""
        return {
            "active_tier":       self._active_tier,
            "available_tiers":   self._available_tiers,
            "routing_mode":      self.config.get("model_tier", "auto"),
            "server_alive":      self._active_server.is_alive() if self._active_server else False,
        }

    def shutdown(self) -> None:
        """Shutdown the active model and free VRAM."""
        if self._active_server:
            self._active_server.shutdown()
            self._active_server = None
            self._active_tier   = None
