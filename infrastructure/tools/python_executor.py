import subprocess
import logging
from tools.tool_interface import ToolInterface
from typing import Any, Dict

logger = logging.getLogger(__name__)


class UnsafePythonExecutor(ToolInterface):
    """Executes arbitrary Python code with NO isolation.

    WARNING: This executor provides NO sandboxing whatsoever.
    Setting cwd=sandbox_path is NOT a sandbox boundary — the executed code
    has full access to the host filesystem, network, and environment.

    Use only for trusted code in development. For production use, replace
    with a container-based, namespace-isolated, or seccomp-filtered executor.

    See also: infrastructure/configs/global.yaml `allowed_filesystem_scope`
    which advertises "/sandbox/" but is NOT enforced by this executor.
    """

    def __init__(self, sandbox_path: str):
        self.sandbox_path = sandbox_path
        self.state = {"executions": 0}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        code = inputs.get("code", "")
        logger.warning(
            "UnsafePythonExecutor: running arbitrary code WITHOUT sandboxing. "
            "cwd=%s is NOT an isolation boundary.",
            self.sandbox_path,
        )
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.sandbox_path
            )
            return {"stdout": result.stdout, "stderr": result.stderr, "status": result.returncode}
        except subprocess.TimeoutExpired:
            return {"error": "Execution timed out", "status": 124}

    def update(self, state: Dict[str, Any]) -> None:
        self.state.update(state)
        self.state["executions"] += 1


# Backward-compatible alias — import sites that reference the old name still work,
# but the real name makes the trust boundary explicit.
PythonExecutorTool = UnsafePythonExecutor
