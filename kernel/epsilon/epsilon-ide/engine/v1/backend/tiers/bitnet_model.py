# backend/tiers/bitnet_model.py
#
# The only file in the entire project that touches the AI model.
# Starts bitnet.cpp as a background process, sends prompts via HTTP,
# returns generated text as a plain Python string.

import subprocess
import time
import requests
from threading import Lock


class BitNetModel:
    """
    Wrapper around the bitnet.cpp HTTP inference server.

    Lifecycle:
        1. __init__ → _start_server() launches llama-server subprocess
        2. generate() → HTTP POST to localhost:8088/completion
        3. shutdown() → SIGTERM to the subprocess
    """

    SERVER_URL = "http://localhost:8088/completion"
    HEALTH_URL = "http://localhost:8088/health"

    def __init__(self, model_path: str, context: int = 512, threads: int = 4):
        self._lock        = Lock()
        self._process     = None
        self._model_path  = model_path
        self._context     = context
        self._threads     = threads
        self._start_server(model_path, context, threads)

    def _start_server(self, model_path: str, ctx: int, threads: int) -> None:
        """
        Launch llama-server as a background subprocess.

        What llama-server does at startup:
            1. Reads the .gguf file from disk
            2. Parses the header (layers, heads, vocab size)
            3. Unpacks ternary weights from 2-bit packed format into RAM
            4. Starts HTTP server on port 8088
            5. Sits idle, waiting for POST requests to /completion

        subprocess.Popen() returns IMMEDIATELY — the server is not
        ready yet. _wait_for_ready() polls /health until it responds.
        """
        command = [
            "./BitNet/build/bin/llama-server",
            "-m",   model_path,
            "-c",   str(ctx),
            "-t",   str(threads),
            "--port", "8088",
            "-ngl", "0",          # GPU layers = 0 → CPU only
            "--log-disable",
        ]

        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/home/lovekesh/epsilon"
        )

        print(f"[BitNetModel] Server process started (PID {self._process.pid})")
        print(f"[BitNetModel] Loading model from {model_path} ...")
        print(f"[BitNetModel] This takes 10-20 seconds on first load ...")
        self._wait_for_ready()

    def _wait_for_ready(self, timeout_seconds: int = 90) -> None:
        """
        Poll /health every second until the server responds with status:ok.

        Why we need this:
            Popen() returns immediately but llama-server takes 10-20
            seconds to read and unpack the 1.19 GB model into RAM.
            Sending a completion request before it is ready returns
            ConnectionRefused. This loop waits until it is ready.
        """
        start = time.time()

        while True:
            # Check if process crashed during startup
            if self._process.poll() is not None:
                raise RuntimeError(
                    "[BitNetModel] llama-server died during startup. "
                    "Check model path and binary."
                )

            try:
                r = requests.get(self.HEALTH_URL, timeout=2)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    elapsed = time.time() - start
                    print(f"[BitNetModel] Server ready! (took {elapsed:.1f}s to load model)")
                    return
            except requests.exceptions.ConnectionError:
                pass   # not ready yet — normal during startup
            except Exception:
                pass

            if time.time() - start > timeout_seconds:
                self._process.terminate()
                raise RuntimeError(
                    f"[BitNetModel] Server did not start within {timeout_seconds}s"
                )

            time.sleep(1)
            print("[BitNetModel] Still loading model...", flush=True)

    def generate(self, prompt: str,
                 max_tokens: int = 128,
                 temperature: float = 0.2) -> str:
        """
        Send a prompt to the model and return the generated text.

        Parameters:
            prompt      : text to complete
            max_tokens  : max tokens to generate (128 ≈ 10 lines of code)
            temperature : 0.0 = deterministic, 0.2 = mostly deterministic,
                          1.0 = creative/random

        The Lock ensures only one generate() call runs at a time.
        If two agents call simultaneously, the second waits for the first.

        Stop tokens explained:
            "\n\n\n"        — stop at triple blank line (end of code block)
            "```\n\n"       — stop after closing markdown fence
            "<|eot_id|>"    — Llama 3 end-of-turn token (chat format)
            "<|end_of_text|>" — Llama 3 end-of-text token
        """
        with self._lock:
            try:
                response = requests.post(
                    self.SERVER_URL,
                    json={
                        "prompt":      prompt,
                        "n_predict":   max_tokens,
                        "temperature": temperature,
                        "stop": [
                            "\n\n\n",
                            "```\n\n",
                            "<|eot_id|>",
                            "<|end_of_text|>",
                        ],
                    },
                    timeout=120,
                )
                return response.json().get("content", "")

            except requests.Timeout:
                return "# Error: model timed out"
            except requests.exceptions.ConnectionError:
                return "# Error: cannot connect to bitnet.cpp server"
            except Exception as e:
                return f"# Error: {e}"

    def is_alive(self) -> bool:
        """Return True if the server process is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def shutdown(self) -> None:
        """Stop llama-server cleanly — frees all RAM it was using."""
        if self._process and self._process.poll() is None:
            print("[BitNetModel] Shutting down server...")
            self._process.terminate()
            self._process.wait()
            print("[BitNetModel] Server stopped.")
