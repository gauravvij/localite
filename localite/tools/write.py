"""Write file tool - writes content to the filesystem."""

import os
from localite.tools.base import BaseTool, ToolResult, measure_duration


class WriteFileTool(BaseTool):
    """Tool for writing content to files."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if they don't exist."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    @measure_duration
    async def execute(self, path: str, content: str) -> ToolResult:
        """Write content to the given path, creating dirs if needed."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {path}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied writing: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error writing file: {e}",
            )