"""
backend/aether/link.py
=======================
The communication bridge between VS Code and the engine.
FIXED VERSION — adds oneshot mode to prevent test hangs.

Reads JSON requests from stdin line by line.
Writes JSON responses to stdout.
All other output (logs, debug info) goes to stderr.

Protocol:
  Input  (stdin):  {"prompt": "your request", "file_path": "/optional/path.py", "prefix": "...", "suffix": "..."}
  Output (stdout): {"ok": true, "result": "...", "task_type": "CODE_GEN", "tier_used": "fast", "complexity": 2}
  Error  (stdout): {"ok": false, "error": "description"}
"""

import asyncio
import json
import sys


class AetherLink:
    def __init__(self, orchestrator, oneshot=False):
        """
        Args:
            orchestrator: The Orchestrator instance to dispatch requests to
            oneshot: If True, exit after processing one request (for testing)
        """
        self.orchestrator = orchestrator
        self.oneshot = oneshot

    async def run(self) -> None:
        """Main loop — reads stdin, dispatches, writes stdout."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_event_loop()

        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        
        mode = "oneshot" if self.oneshot else "interactive"
        print(f"[AetherLink] Ready ({mode}) — listening on stdin", file=sys.stderr, flush=True)

        request_count = 0

        async for raw in reader:
            line = raw.decode("utf-8").strip()
            if not line:
                continue

            request_count += 1

            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                self._respond({"ok": False, "error": f"Invalid JSON: {e}"})
                if self.oneshot:
                    return
                continue

            prompt = req.get("prompt", "").strip()
            if not prompt:
                self._respond({"ok": False, "error": "Empty prompt"})
                if self.oneshot:
                    return
                continue

            # If VS Code sends cursor context, prepend it to the prompt
            prefix = req.get("prefix", "")
            suffix = req.get("suffix", "")
            if prefix or suffix:
                prompt = f"{prefix}\n{prompt}" if prefix else prompt

            try:
                # ✅ FIXED: Properly await async orchestrator call
                output = await self.orchestrator.run(prompt)
                
                self._respond({
                    "ok": True,
                    "result": output.get("result", ""),
                    "task_type": output.get("task_type", "UNKNOWN"),
                    "tier_used": output.get("tier_used", "unknown"),
                    "complexity": output.get("complexity", 0),
                    "syntax_errors": output.get("syntax_errors", []),
                    "files_written": output.get("files_written", []),
                })
            except Exception as e:
                self._respond({
                    "ok": False,
                    "error": str(e),
                    "traceback": str(e),  # For debugging
                })

            # ✅ NEW: Exit after one request in oneshot mode (for testing)
            if self.oneshot:
                print(f"[AetherLink] Oneshot mode — processed {request_count} request(s), exiting", 
                      file=sys.stderr, flush=True)
                return

    def _respond(self, data: dict) -> None:
        """Write JSON to stdout directly — bypasses any print() overrides."""
        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()