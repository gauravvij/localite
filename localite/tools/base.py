"""Base tool protocol and ToolResult dataclass for localite tools."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: int = 0
    data: Any = None


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The tool name used in model tool calls."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON schema dict describing this tool's parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given keyword arguments.

        Returns:
            ToolResult with success flag and output.
        """
        ...


def measure_duration(func):
    """Decorator that sets duration_ms on ToolResult."""
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = int((time.perf_counter() - start) * 1000)
        result.duration_ms = elapsed
        return result
    return wrapper