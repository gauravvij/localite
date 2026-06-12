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
        return (
            "Write content to a file. Creates parent directories if they don't exist. "
            "WHEN TO USE: Creating new files, completely overwriting existing files that need "
            "substantial changes, writing test files, writing config files. "
            "WHEN NOT TO USE: For surgical edits to existing files (use edit_file with search_text/"
            "replace_text instead), for reading files (use read_file). "
            "PARAMETERS: 'path' (required, absolute path to the file), "
            "'content' (required, complete content to write to the file). "
            "EXAMPLE: {\"path\": \"/home/user/project/src/utils.py\", "
            "\"content\": \"import os\\n\\ndef helper():\\n    return os.getcwd()\\n\"} "
            "COMMON MISTAKES: Using write_file when a surgical edit_file would be safer; "
            "not escaping newlines in the content string (use \\n); forgetting this OVERWRITES "
            "the entire file — always read_file first to avoid data loss."
        )

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
            # Resolve relative paths against workdir
            if not path.startswith('/') and hasattr(self, 'workdir') and self.workdir:
                resolved_path = os.path.join(self.workdir, path)
            else:
                resolved_path = path

            dirpath = os.path.dirname(resolved_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(resolved_path, "w") as f:
                f.write(content)
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {resolved_path}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied writing: {resolved_path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error writing file: {e}",
            )