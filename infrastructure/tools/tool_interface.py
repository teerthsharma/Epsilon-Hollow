from abc import ABC, abstractmethod
from typing import Any, Dict

class ToolInterface(ABC):
    """Uniform interface adapter for all Epsilon-Hollow tools."""
    
    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given bounds."""
        pass

    @abstractmethod
    def update(self, state: Dict[str, Any]) -> None:
        """Allow the tool to update its internal state tracking."""
        pass
