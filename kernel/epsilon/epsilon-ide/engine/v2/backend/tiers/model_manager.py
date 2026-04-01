"""
backend/tiers/model_manager.py
================================
Async 3-tier model manager for Epsilon IDE Engine v2.

Fixes in this version:
  - llama-server binary path comes from config (not hardcoded)
  - Deep tier has its own process management (was silently broken)
  - _swap_to avoids restart if tier is already loaded
  - Idle watchdog only kills if a server is actually running
  - generate() logs token stream progress to stderr
  - Graceful fallback: if balanced/deep not found, falls back to fast
"""

import asyncio
import subprocess
import time
import httpx
import json
import sys
from enum import Enum
from pathlib import Path


class Tier(str, Enum):
    FAST     = "fast"
    BALANCED = "balanced"
    DEEP     = "deep"


TIER_PORTS = {
    Tier.FAST:     8088,
    Tier.BALANCED: 8089,
    Tier.DEEP:     8090,
}

TIER_TIMEOUTS = {
    Tier.FAST:     30,
    Tier.BALANCED: 120,
    Tier.DEEP:     300,
}

# How long to wait for llama-server to become healthy
STARTUP_TIMEOUT = {
    Tier.FAST:     45,
    Tier.BALANCED: 90,
    Tier.DEEP:     180,
}


class TieredModelManager:
    def __init__(self, config: dict):
        self.config       = config
        self.models_cfg   = config.get("models", {})

        # One process slot per tier (fast/balanced share a slot; deep is separate)
        self._proc_main:  subprocess.Popen | None = None   # fast or balanced
        self._proc_deep:  subprocess.Popen | None = None   # deep only
        self.current_tier: Tier | None = None

        self._last_request: float | None = None
        self._idle_timeout: int = config.get("idle_timeout", 300)

        # Binary path — must be in config
        self._llama_bin = config.get(
            "llama_server_bin",
            "./BitNet/build/bin/llama-server"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def startup(self):
        """Start FAST tier on boot + launch idle watchdog."""
        if not await self._tier_available(Tier.FAST):
            print("[Model] WARNING: fast tier model file not found — startup skipped")
            return
        await self._start_server(Tier.FAST)
        asyncio.create_task(self._idle_watchdog())

    async def shutdown(self):
        """Kill all running servers and free VRAM."""
        self._kill(self._proc_main)
        self._kill(self._proc_deep)
        self._proc_main   = None
        self._proc_deep   = None
        self.current_tier = None
        print("[Model] All servers stopped — VRAM freed")

    async def generate(
        self,
        prompt:         str,
        tier:           str  = "fast",
        max_tokens:     int  = 512,
        temperature:    float = 0.1,
        repeat_penalty: float = 1.1,
        stop:           list | None = None,
    ) -> str:
        """
        Streaming text generation with automatic tier management.
        Falls back to fast tier if the requested tier is unavailable.
        """
        self._last_request = time.time()
        t = Tier(tier)

        # ── Availability check with graceful fallback ─────────────────────────
        if not await self._tier_available(t):
            print(f"[Model] ⚠ {t.value} tier unavailable — falling back to fast")
            t = Tier.FAST
            if not await self._tier_available(t):
                raise RuntimeError("No model tiers available — check model paths in config")

        # ── Lazy restart if cold ───────────────────────────────────────────────
        if t != Tier.DEEP:
            if self._proc_main is None or self._proc_main.poll() is not None:
                print(f"[Model] Cold start — launching {t.value}")
                await self._start_server(t)
            elif t != self.current_tier:
                await self._swap_to(t)
        else:
            # Deep tier has its own process slot
            if self._proc_deep is None or self._proc_deep.poll() is not None:
                print("[Model] Cold start — launching deep tier")
                await self._start_server(Tier.DEEP)

        port    = TIER_PORTS[t]
        timeout = TIER_TIMEOUTS[t]
        stop    = stop or ["<|im_end|>", "<|endoftext|>"]

        payload = {
            "prompt":         prompt,
            "n_predict":      max_tokens,
            "temperature":    temperature,
            "repeat_penalty": repeat_penalty,
            "stop":           stop,
            "stream":         True,
        }

        result = []
        char_count = 0

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"http://127.0.0.1:{port}/completion",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            token = json.loads(chunk).get("content", "")
                            print(token, end="", flush=True, file=sys.stderr)
                            result.append(token)
                            char_count += len(token)
                        except Exception:
                            pass

            print(f"\n[Model] Done — {char_count} chars generated", file=sys.stderr)

        except httpx.TimeoutException:
            raise RuntimeError(f"{t.value} tier timed out after {timeout}s")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"{t.value} tier HTTP error: {e.response.status_code}")

        return "".join(result).strip()

    # ── Server lifecycle ──────────────────────────────────────────────────────

    async def _start_server(self, tier: Tier):
        cfg  = self.models_cfg.get(tier.value, {})
        path = cfg.get("path", "")
        ngl  = cfg.get("gpu_layers", 0)
        ctx  = cfg.get("context_len", 2048)
        port = TIER_PORTS[tier]

        print(f"[Model] Starting {tier.value} on port {port} — {Path(path).name}")
        print(f"[Model] GPU layers: {ngl} | context: {ctx}")

        cmd = [
            self._llama_bin,
            "-m",     path,
            "-c",     str(ctx),
            "-t",     str(self.config.get("cpu_threads", 4)),
            "--port", str(port),
            "-ngl",   str(ngl),
            "--log-disable",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if tier == Tier.DEEP:
            self._proc_deep = proc
        else:
            self._proc_main = proc
            self.current_tier = tier

        start_timeout = STARTUP_TIMEOUT[tier]
        await self._wait_ready(port, timeout=start_timeout)
        print(f"[Model] {tier.value} ready on port {port}")

    async def _swap_to(self, tier: Tier):
        """Unload current main tier, load new one."""
        if self.current_tier == tier:
            return
        print(f"[Model] Swapping {self.current_tier.value if self.current_tier else '?'} → {tier.value}")
        self._kill(self._proc_main)
        self._proc_main   = None
        self.current_tier = None
        await asyncio.sleep(0.5)          # let VRAM release
        await self._start_server(tier)

    async def _wait_ready(self, port: int, timeout: int = 60):
        deadline = time.time() + timeout
        dots = 0

        async with httpx.AsyncClient(timeout=2) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(f"http://127.0.0.1:{port}/health")
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("status") in ("ok", "loading model"):
                            # "loading model" → keep waiting
                            if data.get("status") == "ok":
                                return
                except Exception:
                    pass
                await asyncio.sleep(1)
                dots += 1
                if dots % 5 == 0:
                    elapsed = int(time.time() - (deadline - timeout))
                    print(f"[Model] Still loading... ({elapsed}s)", file=sys.stderr)

        raise RuntimeError(
            f"llama-server on port {port} did not become ready in {timeout}s.\n"
            f"Check that the model file exists and {self._llama_bin!r} is executable."
        )

    async def _tier_available(self, tier: Tier) -> bool:
        """Return True if the model file for this tier exists on disk."""
        cfg  = self.models_cfg.get(tier.value, {})
        path = cfg.get("path", "")
        return bool(path) and Path(path).exists()

    def _kill(self, proc: subprocess.Popen | None):
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ── Idle watchdog ─────────────────────────────────────────────────────────

    async def _idle_watchdog(self, check_interval: int = 30):
        """Unload the main-slot model after idle_timeout seconds of inactivity."""
        while True:
            await asyncio.sleep(check_interval)
            if (
                self._proc_main is not None
                and self._last_request is not None
                and (time.time() - self._last_request) > self._idle_timeout
            ):
                print(f"[Model] Idle for {self._idle_timeout}s — unloading {self.current_tier.value if self.current_tier else 'model'}")
                self._kill(self._proc_main)
                self._proc_main   = None
                self.current_tier = None
                self._last_request = None

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "current_tier":  self.current_tier.value if self.current_tier else None,
            "main_alive":    self._proc_main is not None and self._proc_main.poll() is None,
            "deep_alive":    self._proc_deep is not None and self._proc_deep.poll() is None,
            "idle_timeout":  self._idle_timeout,
            "last_request":  self._last_request,
        }