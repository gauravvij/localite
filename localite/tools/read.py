"""Read file tool - reads file content from the filesystem."""

import os
from localite.tools.base import BaseTool, ToolResult, measure_duration


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: all)",
                    "default": None,
                },
            },
            "required": ["path"],
        }

    @measure_duration
    async def execute(self, path: str, max_lines: int | None = None) -> ToolResult:
        """Read file content from the given path."""
        try:
            if not os.path.exists(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}",
                )
            with open(path, "r") as f:
                if max_lines is not None:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()
            return ToolResult(
                success=True,
                output=content,
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied reading: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error reading file: {e}",
            )