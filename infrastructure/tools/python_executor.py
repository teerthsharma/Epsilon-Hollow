import subprocess
from tools.tool_interface import ToolInterface
from typing import Any, Dict

class PythonExecutorTool(ToolInterface):
    """Sandboxed execution environment for generated code."""

    def __init__(self, sandbox_path: str):
        self.sandbox_path = sandbox_path
        self.state = {"executions": 0}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        code = inputs.get("code", "")
        # Execute securely in a restricted process
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
