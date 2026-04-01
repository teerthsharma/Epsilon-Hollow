"""
sealMega IDE — VRAMGuard Python Wrapper
// Wraps the C++ vram_guard extension because python's GIL and asyncio are actual garbage.
// Falls back to threading.Lock if C++ not compiled.

// Built this because I am hardware constrained and wanted to help my friends out.
// Spent 3 days debugging why the lock didn't work. Python locks are useless on GPUs. Fuck.
"""

import threading
import time

# Try to load the compiled C++ extension, pray to God it works
try:
    import vram_guard as _native
    NATIVE_AVAILABLE = True
    print("[VRAMGuard] C++ native extension loaded. Finally something that isn't Python.")
except ImportError:
    NATIVE_AVAILABLE = False
    print("[VRAMGuard] FUCK. C++ extension not compiled. Using OS-level threading.Lock fallback. We are going to OOM.")

# Tier constants - Don't touch these
TIER_FOREMAN = 0
TIER_LOGICGATE = 1
TIER_ARCHITECT = 2

# Fallback: OS-level lock (still better than asyncio.Lock but not by much)
_fallback_lock = threading.Lock()
_fallback_holder = -1
_fallback_fenced = False


def fence_vram(tier_id: int) -> bool:
    """
    Acquire the VRAM fence for a tier.
    BLOCKS until the fence is available.
    The 7B model MUST NOT generate while this is held by the Architect.
    """
    global _fallback_holder, _fallback_fenced
    
    if NATIVE_AVAILABLE:
        return _native.fence_vram(tier_id)
    
    _fallback_lock.acquire()
    _fallback_fenced = True
    _fallback_holder = tier_id
    return True


def release_vram(tier_id: int) -> bool:
    """
    Release the VRAM fence.
    """
    global _fallback_holder, _fallback_fenced
    
    if NATIVE_AVAILABLE:
        return _native.release_vram(tier_id)
    
    if _fallback_holder != tier_id:
        return False
    _fallback_fenced = False
    _fallback_holder = -1
    _fallback_lock.release()
    return True


def is_vram_fenced() -> bool:
    """
    Non-blocking check: is the VRAM currently fenced?
    The 7B model's token loop MUST poll this before every forward pass.
    If True → SLEEP. Do not generate. Show "Thinking..." in the UI.
    """
    if NATIVE_AVAILABLE:
        return _native.is_vram_fenced()
    return _fallback_fenced


def get_fence_holder() -> int:
    """Which tier holds the fence? -1 = nobody."""
    if NATIVE_AVAILABLE:
        return _native.get_fence_holder()
    return _fallback_holder


def wait_for_vram(tier_id: int, timeout_ms: int = 30000) -> bool:
    """
    Block until the VRAM is free, then acquire it.
    Returns False if timeout exceeded.
    """
    start = time.monotonic()
    while is_vram_fenced() and get_fence_holder() != tier_id:
        if (time.monotonic() - start) * 1000 > timeout_ms:
            return False
        time.sleep(0.005)  # 5ms poll interval — not busy-wait
    return fence_vram(tier_id)
