"""
backend/agents/router.py
=========================
The ROUTER agent — classifies task type and scores complexity for tier selection.

Tier selection:
  0-2  → FAST     (1.5B)  — tab completions, trivial snippets
  3-6  → BALANCED (7B)    — functions, file gen, debugging, explanations
  7-10 → DEEP     (33B)   — architecture, system design, multi-file projects

Note: Tier override prefixes (use fast: / use deep: / etc.) are handled
upstream in the Orchestrator, not here. The Router only does classification
and complexity scoring on the cleaned prompt.
"""

import re

# ── Task classification ──────────────────────────────────────────────────────
# Checked in order — first match wins. Put more specific patterns first.

TASK_ROUTES: list[tuple[str, list[str]]] = [
    ("FILE_WRITE", [
        "create a file", "write a file", "build a project",
        "create a project", "scaffold", "create a module",
        "generate a complete", "build a complete", "create the following files",
        "write the following files", "write the full", "create the full",
    ]),
    ("DEBUG", [
        "debug", "fix this error", "fix this bug", "traceback", "exception",
        "why is this failing", "why does this fail", "not working", "broken",
        "error:", "attributeerror", "typeerror", "valueerror", "importerror",
    ]),
    ("EXPLAIN", [
        "explain", "what does", "how does", "walk me through",
        "describe this", "what is", "why does",
    ]),
    ("REFACTOR", [
        "refactor", "clean up", "improve this", "optimise", "optimize",
        "rewrite this", "make this better", "simplify",
    ]),
    # CODE_GEN is the fallback — matches anything with a code creation intent
    ("CODE_GEN", [
        "write", "implement", "create", "build", "generate",
        "make", "add", "function", "class", "api", "script",
    ]),
]

# ── Complexity scoring ───────────────────────────────────────────────────────
# Each rule is (score_delta, condition_fn).
# Score is clamped to [0, 10]. Tier thresholds: 0-2=fast, 3-6=balanced, 7+=deep.

COMPLEXITY_RULES: list[tuple[int, object]] = [
    # Architecture / system design → deep
    (+4, lambda p: any(w in p for w in [
        "complete application", "complete app", "entire service",
        "production ready", "full stack", "microservices",
        "system design", "design a system", "architect",
    ])),
    # Multi-file / project generation → balanced/deep
    (+3, lambda p: any(w in p for w in [
        "multiple files", "project structure", "create a project",
        "build a project", "scaffold", "from scratch",
        "complete rest api", "full application",
    ])),
    # Database / auth / infrastructure → balanced
    (+2, lambda p: any(w in p for w in [
        "authentication", "jwt", "oauth", "database", "sqlite",
        "postgresql", "redis", "docker", "design pattern",
    ])),
    # Has an error traceback attached → balanced (needs reasoning)
    (+2, lambda p: "traceback" in p or "error:" in p.lower()),
    # Standard code generation verbs → slight push toward balanced
    (+1, lambda p: any(w in p for w in [
        "write", "implement", "create", "build", "generate",
    ])),
    # Long prompt usually means more context/complexity
    (+1, lambda p: len(p) > 100),
    (+1, lambda p: len(p) > 250),
    # Explanations and refactors benefit from a bigger model
    (+1, lambda p: any(w in p for w in [
        "explain", "refactor", "analyse", "analyze", "how does",
    ])),
    # Short one-liners → definitely fast
    (-2, lambda p: len(p) < 30),
]


def classify_task(prompt: str) -> str:
    pl = prompt.lower()
    for task, keywords in TASK_ROUTES:
        for kw in keywords:
            if kw in pl:
                return task
    return "CODE_GEN"


def score_complexity(prompt: str) -> int:
    pl = prompt.lower()
    score = 0
    for delta, condition in COMPLEXITY_RULES:
        try:
            if condition(pl):
                score += delta
        except Exception:
            pass
    return max(0, min(10, score))


def tier_from_score(score: int) -> str:
    if score <= 2:
        return "fast"
    elif score <= 6:
        return "balanced"
    else:
        return "deep"


class Router:
    def route(self, prompt: str) -> dict:
        task_type  = classify_task(prompt)
        complexity = score_complexity(prompt)
        tier       = tier_from_score(complexity)

        return {
            "task_type":  task_type,
            "complexity": complexity,
            "tier":       tier,
        }