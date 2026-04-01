"""
backend/tiers/model.py
======================
Wrapper around the llama-server HTTP inference server.

This file is the ONLY place in the entire codebase that talks to the AI model.
Everything else goes through the generate() method here.

Architecture decision: we run the model as a separate subprocess (llama-server)
rather than loading it in Python directly. Reasons:
  1. No PyTorch needed — saves 800 MB RAM
  2. If the model crashes, Python keeps running
  3. llama-server handles GPU memory management natively
  4. We talk to it via HTTP — simple, debuggable, language-agnostic

Model: Qwen2.5-Coder-1.5B-Instruct
  - 1.5 billion parameters, trained specifically on code
  - Q4_K_M quantization: ~1 GB on disk, fits in 4 GB VRAM with room to spare
  - Context window: 2048 tokens (about 1500 words)
  - Chat template: ChatML format (<|im_start|>user...<|im_end|>)
"""

import subprocess
import time
import requests
from threading import Lock


# The ChatML template used by Qwen models.
# Every prompt must be wrapped in these tokens for the model to understand
# that it is in a conversation and should respond as an assistant.
CHATML_SYSTEM = "<|im_start|>system\nYou are an expert Python developer. Write clean, well-commented, production-ready code. Always complete the full implementation.<|im_end|>\n"
CHATML_USER   = "<|im_start|>user\n{prompt}<|im_end|>\n"
CHATML_ASST   = "<|im_start|>assistant\n"

# Stop tokens — generation halts when any of these appear
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


class ModelServer:
    """
    Manages the llama-server subprocess lifecycle.

    Usage:
        server = ModelServer(config)
        result = server.generate("write a binary search function")
        server.shutdown()
    """

    def __init__(self, config: dict):
        """
        config: the loaded config.yaml as a dict
        """
        self.config   = config
        self._process = None
        self._lock    = Lock()  # one request at a time — model is not thread-safe
        self._start()

    def _start(self) -> None:
        """
        Launch llama-server as a background process.

        -ngl 28 means "put 28 layers on the GPU" — all of them.
        With Qwen2.5-Coder-1.5B this uses about 1 GB of VRAM,
        leaving 3 GB free on your RTX A2000.
        """
        cmd = [
            "/mnt/d/epsilon/v2/BitNet/build/bin/llama-server",
            "-m",      self.config["model_path"],
            "-c",      str(self.config.get("context_len", 2048)),
            "-t",      str(self.config.get("cpu_threads", 4)),
            "--port",  str(self.config.get("server_port", 8088)),
            "-ngl",    str(self.config.get("gpu_layers", 28)),  # GPU layers
            "--log-disable",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"[Model] Server started (PID {self._process.pid})")
        print(f"[Model] GPU layers: {self.config.get('gpu_layers', 28)} — running on RTX A2000")
        print("[Model] Loading model into VRAM...")
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: int = 60) -> None:
        """Poll /health until the model is fully loaded into VRAM."""
        url   = f"http://{self.config.get('server_host','localhost')}:{self.config.get('server_port',8088)}/health"
        start = time.time()

        while True:
            if self._process.poll() is not None:
                raise RuntimeError("[Model] llama-server crashed during startup. Check model path.")

            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    elapsed = time.time() - start
                    print(f"[Model] Ready in {elapsed:.1f}s — model loaded into VRAM")
                    return
            except requests.exceptions.ConnectionError:
                pass  # still loading — this is normal

            if time.time() - start > timeout:
                self._process.terminate()
                raise RuntimeError("[Model] Startup timeout — took longer than 60 seconds")

            time.sleep(1)
            print("[Model] Still loading...", flush=True)

    def generate(self,
                 prompt: str,
                 max_tokens: int = 256,
                 temperature: float = 0.2,
                 raw: bool = False) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt:      The user's request (plain text)
            max_tokens:  Maximum tokens to generate
            temperature: 0.0 = deterministic, 0.2 = slightly random (good for code)
            raw:         If True, send prompt as-is (already formatted with ChatML)
                         If False, wrap in ChatML template automatically

        Returns:
            Generated text as a plain Python string.
            Returns an error comment string on failure.
        """

        # Wrap in ChatML template unless already formatted
        if raw:
            full_prompt = prompt
        else:
            full_prompt = CHATML_SYSTEM + CHATML_USER.format(prompt=prompt) + CHATML_ASST

        url = f"http://{self.config.get('server_host','localhost')}:{self.config.get('server_port',8088)}/completion"

        with self._lock:
            try:
                response = requests.post(
                    url,
                    json={
                        "prompt":      full_prompt,
                        "n_predict":   max_tokens,
                        "temperature": temperature,
                        "stop":        STOP_TOKENS,
                        "stream":      False,
                    },
                    timeout=120,
                )
                data = response.json()
                return data.get("content", "").strip()

            except requests.Timeout:
                return "# Error: model timed out after 120 seconds"
            except requests.exceptions.ConnectionError:
                return "# Error: cannot connect to model server"
            except Exception as e:
                return f"# Error: {e}"

    def is_alive(self) -> bool:
        """True if the server process is still running."""
        return self._process is not None and self._process.poll() is None

    def get_speed_stats(self) -> dict:
        """
        Run a quick benchmark and return tokens/second.
        Useful for health checks and monitoring.
        """
        start  = time.time()
        result = self.generate("def hello():", max_tokens=20, temperature=0.0)
        tokens = len(result.split())
        elapsed = time.time() - start
        return {
            "tokens_generated": tokens,
            "elapsed_seconds":  round(elapsed, 2),
            "tokens_per_second": round(tokens / elapsed, 1) if elapsed > 0 else 0,
        }

    def shutdown(self) -> None:
        """Terminate the server and free VRAM."""
        if self._process and self._process.poll() is None:
            print("[Model] Shutting down server...")
            self._process.terminate()
            self._process.wait()
            print("[Model] Server stopped — VRAM freed")
