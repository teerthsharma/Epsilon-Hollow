import asyncio
import gc
import re

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    TREE_SITTER_OK = True
except ImportError:
    TREE_SITTER_OK = False
    print("[Orchestrator] WARNING: tree-sitter not available — CRITIC disabled")


# ── ROUTER keyword table ──────────────────────────────────────────────────────
# Dict order matters — first match wins.
# DEBUG must come before CODE_GEN so "fix the bug in def add()" → DEBUG not CODE_GEN.
# DESCRIBE must come before CODE_GEN so "write a condition that..." → DESCRIBE not CODE_GEN.
# DESCRIBE routes to the chat template (full natural language response).
# CODE_GEN routes to completion style (model continues a function signature).

ROUTES = {
    'DEBUG': [
        'bug', 'error', 'crash', 'exception', 'traceback',
        'fix', 'broken', 'wrong', 'fail', 'issue', 'not working',
    ],
    'DESCRIBE': [
        # Natural language requests that describe what they want
        # but do NOT contain a function/class signature.
        # These need a full chat response, not a code completion.
        'that checks', 'that matches', 'that finds', 'that counts',
        'that prints', 'that returns', 'that reads', 'that writes',
        'that loops', 'that iterates', 'that converts', 'that sorts',
        'which checks', 'which matches', 'which prints', 'which returns',
        'a condition', 'a loop', 'a program', 'a script',
        'manually', 'step by step', 'using a loop', 'using recursion',
    ],
    'CODE_GEN': [
        'write', 'create', 'implement', 'generate',
        'function', 'class', 'def', 'make', 'build', 'add',
    ],
    'EXPLAIN': [
        'explain', 'what', 'how', 'why', 'describe',
        'tell me', 'what does', 'understand',
    ],
    'REFACTOR': [
        'refactor', 'clean', 'improve', 'optimise', 'optimize',
        'simplify', 'rewrite', 'restructure',
    ],
    'SEARCH': [
        'find', 'search', 'where', 'locate', 'show me', 'list',
    ],
}

URGENCY_WORDS = ['urgent', 'critical', 'broken', 'asap', 'immediately']

# Task types that use the Llama 3 chat template (full natural language response).
# Everything else uses completion style (model continues a code signature).
CHAT_TEMPLATE_TASKS = {'DEBUG', 'DESCRIBE', 'EXPLAIN', 'REFACTOR'}


class PotatoOrchestrator:
    """
    Routes each request through ROUTER -> RECALL -> CODER -> CRITIC.
    The model is called exactly once per request.

    ROUTER  : keyword matching         — zero AI calls
    RECALL  : TF-IDF search via Clara  — zero AI calls
    CODER   : HTTP POST to bitnet.cpp  — ONE AI call
    CRITIC  : Tree-sitter parse check  — zero AI calls
    """

    def __init__(self, model, clara):
        self.model = model
        self.clara = clara
        self._setup_critic()

    def _setup_critic(self):
        """
        Initialise the Tree-sitter Python parser.

        Tree-sitter parses source code into an Abstract Syntax Tree.
        ERROR nodes in the AST = syntax errors in generated code.
        Runs in microseconds — no model call needed.
        """
        if not TREE_SITTER_OK:
            self.parser = None
            return
        try:
            PY_LANGUAGE = Language(tspython.language())
            self.parser = Parser(PY_LANGUAGE)
            print("[Orchestrator] Tree-sitter Python parser ready")
        except Exception as e:
            print(f"[Orchestrator] Tree-sitter setup failed: {e}")
            self.parser = None

    # ── AGENT 1: ROUTER ──────────────────────────────────────────────────────

    def route(self, intent: str) -> tuple:
        """
        Classify intent using keyword matching. Zero AI calls.

        Routing logic:
            1. Check for urgency words → set priority 1
            2. Scan ROUTES dict in order — first match wins
            3. Default to CODE_GEN if nothing matches

        Key ordering decisions:
            DEBUG before CODE_GEN  — "fix the bug in def foo()" → DEBUG
            DESCRIBE before CODE_GEN — "write a loop that prints 1-50" → DESCRIBE
            Both use chat template so the model gives a full answer,
            not just a code continuation.

        Returns:
            (task_type, priority)
            priority 1 = urgent (process first), 3 = normal
        """
        lower    = intent.lower()
        priority = 1 if any(w in lower for w in URGENCY_WORDS) else 3

        for task_type, keywords in ROUTES.items():
            if any(kw in lower for kw in keywords):
                return task_type, priority

        return 'CODE_GEN', priority

    # ── AGENT 2: RECALL ──────────────────────────────────────────────────────

    def recall(self, intent: str) -> str:
        """
        Search indexed project files for relevant code via Clara.

        TF-IDF cosine similarity — zero AI calls, runs in 2-5 ms.
        Returns empty string if Clara has no indexed documents yet.
        Better context = better and more project-aware suggestions.
        """
        try:
            return self.clara.get_context_for_prompt(
                intent, k=3, max_chars=400
            )
        except Exception:
            return ""

    # ── AGENT 3: CODER ───────────────────────────────────────────────────────

    def _build_prompt(self, intent: str, task_type: str, context: str) -> str:
        """
        Build the prompt in the correct format for BitNet-b1.58-2B-4T.

        This model is Llama 3 based and understands two formats:

        COMPLETION STYLE — used for CODE_GEN only:
            We give a partial function signature and the model fills
            in the body. No special tokens needed.
            Example input:  "def bubble_sort("
            Model output:   "arr: list) -> list:\n    n = len(arr)..."
            Best for: the user has a clear function name in mind.

        CHAT TEMPLATE STYLE — used for DEBUG, DESCRIBE, EXPLAIN, REFACTOR:
            Uses Llama 3 special tokens so the model treats the input
            as a user message and responds as an assistant.
            Best for: natural language requests, bug fixes, explanations.
            Example input:  "write a loop that prints even numbers to 50"
            Model output:   "Here is a Python loop that prints even numbers..."

        The CHAT_TEMPLATE_TASKS set at the top of this file controls
        which task types use which format. Adding a task type to that
        set is all you need to switch it to chat template style.
        """

        if task_type not in CHAT_TEMPLATE_TASKS:
            # ── Completion style ─────────────────────────────────────────
            parts = []

            # Add relevant project code as comments above the function.
            # The model sees these and tends to match the project's style.
            if context.strip():
                ctx_lines = context.strip().split('\n')[:6]
                for line in ctx_lines:
                    if line.strip():
                        parts.append(f"# {line.strip()[:80]}")
                parts.append("")

            # If the intent already contains a def/class signature, use it.
            # Example: "implement def quicksort(arr: list) -> list:"
            code_match = re.search(
                r'(def\s+\w+\s*\([^)]*\)[^:]*:|class\s+\w+[^:]*:)',
                intent
            )

            if code_match:
                # User gave us the signature directly — use it as-is
                parts.append(code_match.group(1))

            else:
                # Extract a function name hint from natural language.
                # "write a bubble sort function" → fn_hint = "bubble"
                # We open "def bubble(" and let the model complete it.
                skip = {
                    'write', 'create', 'implement', 'make', 'build',
                    'a', 'an', 'the', 'function', 'method', 'class',
                    'in', 'python', 'that', 'which', 'for', 'to',
                    'simple', 'basic', 'new', 'my', 'generate', 'add',
                }
                words   = intent.lower().split()
                fn_hint = ""
                for w in words:
                    clean_w = re.sub(r'[^a-z0-9_]', '', w)
                    if clean_w and clean_w not in skip and len(clean_w) > 2:
                        fn_hint = clean_w
                        break

                parts.append(f"def {fn_hint}(" if fn_hint else "def solution(")

            return "\n".join(parts)

        else:
            # ── Llama 3 chat template ────────────────────────────────────
            # Build the user message body
            user_parts = []

            # Include relevant project code as context if Clara found any
            if context.strip():
                user_parts.append(
                    f"Here is some relevant code from the project:\n"
                    f"{context.strip()}\n"
                )

            user_parts.append(intent[:300])
            user_message = "\n".join(user_parts)

            # Llama 3 special tokens:
            #   <|begin_of_text|>                         start of prompt
            #   <|start_header_id|>role<|end_header_id|>  role declaration
            #   <|eot_id|>                                end of turn
            # The model generates from the assistant header onwards.
            return (
                f"<|begin_of_text|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_message}"
                f"<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

    async def _call_model(self, prompt: str,
                          max_tokens: int = 128,
                          temperature: float = 0.2) -> str:
        """
        Call model.generate() in a thread pool so it does not block
        the asyncio event loop during the 1-30 second generation time.

        run_in_executor(None, fn) runs fn in a background thread.
        The event loop stays free to handle other work while waiting.
        When the thread finishes the coroutine resumes with the result.
        """
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return result

    # ── AGENT 4: CRITIC ──────────────────────────────────────────────────────

    def is_valid_syntax(self, code: str) -> bool:
        """
        Parse generated code with Tree-sitter, check for ERROR nodes.

        An ERROR node means the parser could not understand part of
        the code — i.e. there is a syntax error. We only run this
        check on CODE_GEN output, not chat template responses.

        Returns True  if syntax is valid (or parser is unavailable).
        Returns False if an ERROR node is found.
        """
        if self.parser is None:
            return True

        # Strip markdown code fences the model sometimes wraps output in
        clean = code.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean

        if not clean:
            return True

        try:
            tree = self.parser.parse(clean.encode())
            return "ERROR" not in tree.root_node.sexp()
        except Exception:
            return True

    # ── MAIN DISPATCH ────────────────────────────────────────────────────────

    async def dispatch(self, intent: str) -> dict:
        """
        Full pipeline: ROUTER -> RECALL -> CODER -> CRITIC.

        Called once per user request.

        Returns dict:
            result       : generated text string
            task_type    : DEBUG / DESCRIBE / CODE_GEN / EXPLAIN / REFACTOR / SEARCH
            priority     : 1 (urgent) or 3 (normal)
            valid_syntax : True if CRITIC approved (or task is not CODE_GEN)
            retried      : True if CRITIC triggered a re-generation
        """

        # Step 1 — ROUTER
        task_type, priority = self.route(intent)
        print(f"[Router] task={task_type}, priority={priority}")

        # Step 2 — RECALL
        context = self.recall(intent)
        if context:
            print(f"[Recall] Found context ({len(context)} chars)")
        else:
            print(f"[Recall] No context found")

        # Step 3 — CODER (the only model call in the entire pipeline)
        prompt = self._build_prompt(intent, task_type, context)
        print(f"[Coder] Prompt ({len(prompt)} chars) "
              f"[{'chat' if task_type in CHAT_TEMPLATE_TASKS else 'completion'}]"
              f" — generating...")

        # Chat responses need more tokens — explanations are longer than code
        max_tokens = 256 if task_type in CHAT_TEMPLATE_TASKS else 128

        result = await self._call_model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        print(f"[Coder] Generated {len(result)} chars")

        # Step 4 — CRITIC
        # Only run syntax check on CODE_GEN — chat responses are prose,
        # not Python code, so Tree-sitter would always flag them as errors.
        valid   = True
        retried = False

        if task_type == 'CODE_GEN':
            valid = self.is_valid_syntax(result)

            if not valid:
                print("[Critic] Syntax error detected — retrying once...")
                fix_prompt = (
                    prompt + result +
                    "\n# Fix the syntax error above. "
                    "Output only the corrected Python code:\n"
                )
                result  = await self._call_model(
                    fix_prompt,
                    max_tokens=64,
                    temperature=0.1,
                )
                valid   = self.is_valid_syntax(result)
                retried = True
                print(f"[Critic] After retry — valid: {valid}")
            else:
                print(f"[Critic] Syntax OK")

        gc.collect()

        return {
            'result':       result,
            'task_type':    task_type,
            'priority':     priority,
            'valid_syntax': valid,
            'retried':      retried,
        }
