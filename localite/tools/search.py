"""Search tool - grep for patterns in files."""

import os
import re
import fnmatch
from localite.tools.base import BaseTool, ToolResult, measure_duration


class GrepSearchTool(BaseTool):
    """Tool for searching text patterns in files."""

    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return (
            "Search for a text pattern in files using regex or literal matching. "
            "WHEN TO USE: Finding where functions are defined or called, searching for import "
            "statements, locating configuration values, discovering code patterns across the codebase. "
            "Essential when you don't know which file contains a particular identifier. "
            "WHEN NOT TO USE: For reading file contents (use read_file), for listing directories "
            "(use list_files), for running shell commands (use run_shell). "
            "PARAMETERS: 'pattern' (required, regex pattern string), 'path' (required, file or "
            "directory to search), 'glob_pattern' (optional, file filter like '*.py'), "
            "'max_results' (optional, int, default 50). "
            "EXAMPLE: {\"pattern\": \"def train_\", \"path\": \"/home/user/project/src\", "
            "\"glob_pattern\": \"*.py\", \"max_results\": 20} "
            "COMMON MISTAKES: Using glob syntax in 'pattern' (it's regex, not glob); "
            "forgetting the pattern is case-insensitive; not using 'glob_pattern' to filter "
            "relevant file types; setting max_results too low and missing matches."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search in",
                },
                "glob_pattern": {
                    "type": "string",
                    "description": "Optional file glob pattern (e.g. '*.py')",
                    "default": None,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 50)",
                    "default": 50,
                },
            },
            "required": ["pattern", "path"],
        }

    @measure_duration
    async def execute(
        self,
        pattern: str,
        path: str,
        glob_pattern: str | None = None,
        max_results: int = 50,
    ) -> ToolResult:
        """Search for pattern in files under the given path."""
        try:
            if not os.path.exists(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Path not found: {path}",
                )

            results = []
            compiled = re.compile(pattern, re.IGNORECASE)

            if os.path.isfile(path):
                files_to_check = [path]
            else:
                files_to_check = []
                for root, dirs, files in os.walk(path):
                    # Skip hidden dirs
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for f in sorted(files):
                        if glob_pattern and not fnmatch.fnmatch(f, glob_pattern):
                            continue
                        files_to_check.append(os.path.join(root, f))

            for filepath in files_to_check:
                if len(results) >= max_results:
                    break
                try:
                    with open(filepath, "r", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if len(results) >= max_results:
                                break
                            if compiled.search(line):
                                if os.path.isfile(path):
                                    relpath = os.path.basename(filepath)
                                else:
                                    relpath = os.path.relpath(filepath, os.path.commonpath([path]))
                                results.append(f"{relpath}:{i}: {line.rstrip()}")
                except (PermissionError, IsADirectoryError):
                    continue

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No matches found for pattern: {pattern}",
                )

            output = f"Found {len(results)} match(es) for '{pattern}':\n"
            output += "\n".join(results)
            return ToolResult(success=True, output=output)

        except re.error as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid regex pattern: {e}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error searching: {e}",
            )