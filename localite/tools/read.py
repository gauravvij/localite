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
        return (
            "Read the contents of a file at the given path. "
            "WHEN TO USE: Before editing any file (always read first), to inspect source code, "
            "config files, logs, or any text file. This is your PRIMARY exploration tool. "
            "WHEN NOT TO USE: For binary files (images, pickles), for listing directories "
            "(use list_files instead), for very large files (use max_lines to cap output). "
            "PARAMETERS: 'path' (required, absolute path to the file), "
            "'max_lines' (optional, integer, max lines to return). "
            "EXAMPLE: {\"path\": \"/home/user/project/src/main.py\", \"max_lines\": 100} "
            "COMMON MISTAKES: Using a relative path that doesn't resolve correctly; "
            "not using max_lines for known-large files and flooding the context."
        )

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
    async def execute(self, path: str = None, max_lines: int | None = None, **kwargs) -> ToolResult:
        """Read file content from the given path."""
        try:
            # Guard: model may omit path or pass it under an unexpected key
            if path is None:
                path = kwargs.get('filepath') or kwargs.get('file') or kwargs.get('file_path') or kwargs.get('filename')
            if not path:
                return ToolResult(
                    success=False,
                    output="",
                    error="Missing required argument: 'path'. Usage: read_file(path=\"/absolute/path/to/file\")",
                )
            # Resolve relative paths against workdir
            if not path.startswith('/') and hasattr(self, 'workdir') and self.workdir:
                resolved_path = os.path.join(self.workdir, path)
            else:
                resolved_path = path

            if not os.path.exists(resolved_path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {resolved_path}",
                )
            with open(resolved_path, "r") as f:
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