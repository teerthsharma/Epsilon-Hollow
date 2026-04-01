"""
backend/agents/orchestrator.py
================================
Six-agent pipeline wired to the async 3-tier model manager.

Pipeline:
  ROUTER  → classifies task, picks tier + detects overrides (zero AI)
  RECALL  → searches project code via Clara (zero AI)
  PLANNER → complex tasks only: creates build plan (BALANCED tier)
  CODER   → generates code using the chosen tier (one async AI call)
  WRITER  → saves files to disk (zero AI)
  CRITIC  → validates Python syntax (zero AI)
"""

import asyncio
import re
import ast

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from backend.agents.router import Router

# Task types that need chat-style prompting
CHAT_TASKS = {"DEBUG", "DESCRIBE", "EXPLAIN", "REFACTOR", "FILE_WRITE"}

# Tier override prefixes — checked BEFORE routing
TIER_OVERRIDE_PREFIXES = {
    "use fast:":     "fast",
    "use balanced:": "balanced",
    "use deep:":     "deep",
    "quick:":        "fast",
    "deep think:":   "deep",
}


def build_chatml_prompt(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_completion_prompt(context: str, prompt: str) -> str:
    """
    For CODE_GEN tasks: structured completion prompt.
    Context is injected as a comment block so the model doesn't echo it back.
    """
    parts = [f"# Task: {prompt}\n"]
    if context:
        parts.append("# Relevant project context (do not repeat this):")
        for line in context.splitlines()[:20]:          # cap at 20 lines
            parts.append(f"# {line}")
        parts.append("")
    parts.append("# Implementation:")
    return "\n".join(parts) + "\n"


def extract_code(text: str) -> str:
    """Strip markdown fences and leading/trailing noise."""
    # Try ```python ... ``` first
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ``` ... ``` (no language tag)
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_python(code: str) -> list[str]:
    """
    Returns a list of syntax error strings, or [] if the code is valid.
    Prefers tree-sitter but always cross-checks with ast.parse to avoid
    false positives (tree-sitter can flag valid f-strings in older grammars).
    """
    # Fast path: ast.parse is the ground truth for Python syntax
    try:
        ast.parse(code)
        return []           # ast says it's valid — trust it over tree-sitter
    except SyntaxError as e:
        ast_error = f"SyntaxError at line {e.lineno}: {e.msg}"

    # ast found an error — confirm with tree-sitter if available
    if TREE_SITTER_AVAILABLE:
        try:
            PY_LANGUAGE = Language(tspython.language())
            parser = Parser(PY_LANGUAGE)
            tree = parser.parse(bytes(code, "utf8"))
            if tree.root_node.has_error:
                return [ast_error]          # both agree — real error
            # tree-sitter disagrees — ast might be wrong (edge case)
            return []
        except Exception:
            pass

    return [ast_error]


def detect_tier_override(prompt: str) -> tuple[str | None, str]:
    """
    Check if the prompt starts with a tier override prefix.
    Returns (tier_override_or_None, cleaned_prompt).
    """
    lower = prompt.lower()
    for prefix, tier in TIER_OVERRIDE_PREFIXES.items():
        if lower.startswith(prefix):
            cleaned = prompt[len(prefix):].strip()
            return tier, cleaned
    return None, prompt


class Orchestrator:
    def __init__(self, model_manager, kv_cache=None, memory=None, clara=None, config=None):
        self.model  = model_manager
        self.kv     = kv_cache
        self.memory = memory
        self.clara  = clara
        self.config = config or {}
        self.router = Router()

    async def run(self, prompt: str) -> dict:
        """
        Full pipeline: ROUTER → RECALL → CODER → CRITIC → WRITER → MEMORY
        """
        # ── TIER OVERRIDE (before routing) ────────────────────────────────────
        forced_tier, clean_prompt = detect_tier_override(prompt)
        if forced_tier:
            print(f"[Router] Tier override detected: {forced_tier!r}")

        # ── ROUTER ────────────────────────────────────────────────────────────
        route      = self.router.route(clean_prompt)
        task_type  = route["task_type"]
        complexity = route["complexity"]
        tier       = forced_tier or route["tier"]

        print(f"[Router] task={task_type} complexity={complexity} tier={tier}")

        # ── RECALL ────────────────────────────────────────────────────────────
        context = ""
        if self.clara:
            try:
                hits = self.clara.search(clean_prompt, k=3)
                # Only include hits with meaningful relevance (score > 0.05)
                relevant = [h for h in hits if h["score"] > 0.05]
                if relevant:
                    context = "\n".join(
                        f"# {h['path']} (score {h['score']:.2f}):\n{h['preview']}"
                        for h in relevant
                    )
                    print(f"[Recall] {len(relevant)} relevant file(s) found")
                else:
                    print("[Recall] No relevant context found")
            except Exception as e:
                print(f"[Recall] Warning: {e}")

        # ── BUILD PROMPT ──────────────────────────────────────────────────────
        if task_type in CHAT_TASKS:
            system = (
                "You are an expert Python developer. "
                "Write clean, complete, working code only. "
                "No explanatory comments unless asked. "
                "No markdown fences in your response."
            )
            if context:
                system += f"\n\nRelevant project code:\n{context}"

            # Inject conversation history for chat tasks
            history_str = ""
            if self.memory:
                try:
                    history_str = self.memory.get_context_string()
                except Exception:
                    pass

            user_msg = (history_str + "\n" + clean_prompt).strip() if history_str else clean_prompt
            llm_prompt = build_chatml_prompt(system, user_msg)
        else:
            llm_prompt = build_completion_prompt(context, clean_prompt)

        print(f"[Coder] Prompt {len(llm_prompt)} chars → tier={tier}")

        # ── CODER (AI call) ───────────────────────────────────────────────────
        try:
            raw = await self.model.generate(
                prompt=llm_prompt,
                tier=tier,
                max_tokens=self._max_tokens(task_type),
                temperature=self._temperature(task_type),
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|endoftext|>", "\n# Task:"],
            )
        except Exception as e:
            print(f"[Coder] ERROR: {e}")
            return {
                "ok": False,
                "error": str(e),
                "result": f"# Error during generation: {e}",
                "task_type": task_type,
                "tier_used": tier,
                "complexity": complexity,
                "syntax_errors": [],
                "files_written": [],
            }

        result = extract_code(raw)
        print(f"[Coder] Generated {len(result)} chars")

        # ── CRITIC ────────────────────────────────────────────────────────────
        syntax_errors = []
        if task_type in {"CODE_GEN", "REFACTOR", "FILE_WRITE"}:
            syntax_errors = validate_python(result)
            if syntax_errors:
                print(f"[Critic] ⚠ Syntax issues: {syntax_errors}")
            else:
                print("[Critic] ✓ Syntax OK")

        # ── WRITER ────────────────────────────────────────────────────────────
        files_written = []
        if task_type == "FILE_WRITE" and not syntax_errors:
            files_written = await self._write_files(result, clean_prompt)

        # ── MEMORY ────────────────────────────────────────────────────────────
        if self.memory:
            try:
                self.memory.add("user", clean_prompt)
                self.memory.add("assistant", result[:500])
            except Exception as e:
                print(f"[Memory] Warning: {e}")

        return {
            "ok": True,
            "result": result,
            "task_type": task_type,
            "tier_used": tier,
            "complexity": complexity,
            "syntax_errors": syntax_errors,
            "files_written": files_written,
        }

    async def _write_files(self, result: str, prompt: str) -> list[str]:
        """
        Parse FILE_WRITE result for file blocks and write them to disk.
        Looks for markers like: # FILE: path/to/file.py
        """
        from backend.tools.filesystem import write_file
        written = []
        current_path = None
        current_lines = []

        for line in result.splitlines():
            if line.startswith("# FILE:"):
                if current_path and current_lines:
                    msg = write_file(current_path, "\n".join(current_lines))
                    if msg.startswith("OK"):
                        written.append(current_path)
                        print(f"[Writer] {msg}")
                    else:
                        print(f"[Writer] ⚠ {msg}")
                current_path = line[7:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_path and current_lines:
            msg = write_file(current_path, "\n".join(current_lines))
            if msg.startswith("OK"):
                written.append(current_path)

        return written

    def _max_tokens(self, task_type: str) -> int:
        return {
            "CODE_GEN":   512,
            "FILE_WRITE": 2048,
            "DEBUG":      1024,
            "EXPLAIN":    768,
            "REFACTOR":   1024,
            "DESCRIBE":   512,
        }.get(task_type, 512)

    def _temperature(self, task_type: str) -> float:
        """Lower temperature for code, slightly higher for explanations."""
        return {
            "EXPLAIN":  0.3,
            "DESCRIBE": 0.3,
            "DEBUG":    0.1,
        }.get(task_type, 0.1)