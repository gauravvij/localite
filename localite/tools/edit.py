"""Edit file tool - search-and-replace in existing files."""

import os
from localite.tools.base import BaseTool, ToolResult, measure_duration


class EditFileTool(BaseTool):
    """Tool for search-and-replace editing of files."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by finding and replacing text. "
            "WHEN TO USE: To make surgical changes to existing files — add docstrings, fix bugs, "
            "refactor code, change parameters, fix imports. Two modes: (1) search-and-replace with "
            "search_text+replace_text for targeted edits, (2) full content replacement with 'content'. "
            "WHEN NOT TO USE: For creating new files (use write_file), for rewriting entire existing "
            "files where you want to replace most content (use write_file for clarity). "
            "PARAMETERS: 'path' (required, absolute path), 'search_text' (optional, exact text to find), "
            "'replace_text' (optional, text to replace with), 'content' (optional, full new file content). "
            "EXAMPLE: {\"path\": \"/home/user/project/train.py\", "
            "\"search_text\": \"learning_rate = 0.01\", "
            "\"replace_text\": \"learning_rate = 0.001\"} "
            "COMMON MISTAKES: search_text not matching whitespace exactly (indentation matters); "
            "using 'old_value' or 'new_value' as parameter names instead of the correct "
            "'search_text' and 'replace_text'; expecting it to replace ALL occurrences "
            "(it only replaces the FIRST)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit",
                },
                "search_text": {
                    "type": "string",
                    "description": "Exact text to find in the file (omit for full content replacement)",
                },
                "replace_text": {
                    "type": "string",
                    "description": "Text to replace the found text with (omit for full content replacement)",
                },
                "content": {
                    "type": "string",
                    "description": "Full new file content (alternative to search_text+replace_text for writing the entire file)",
                },
            },
            "required": ["path"],
        }

    @measure_duration
    async def execute(
        self,
        path: str,
        search_text: str | None = None,
        replace_text: str | None = None,
        content: str | None = None,
    ) -> ToolResult:
        """Edit a file. Two modes:
        1. search-and-replace: provide search_text + replace_text
        2. full replacement: provide content (replaces entire file content)
        """
        try:
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

            # Mode 2: full content replacement
            if content is not None and search_text is None:
                with open(resolved_path, "w") as f:
                    f.write(content)
                return ToolResult(
                    success=True,
                    output=f"Successfully wrote new content to {resolved_path} ({len(content)} chars)",
                )

            # Mode 1: search-and-replace
            if search_text is None or replace_text is None:
                return ToolResult(
                    success=False,
                    output="",
                    error="edit_file requires either (search_text + replace_text) or content",
                )

            with open(resolved_path, "r") as f:
                current = f.read()

            if search_text not in current:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Search text not found in {resolved_path}",
                )

            new_content = current.replace(search_text, replace_text, 1)
            with open(resolved_path, "w") as f:
                f.write(new_content)

            return ToolResult(
                success=True,
                output=f"Successfully edited {resolved_path}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied editing: {resolved_path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error editing file: {e}",
            )