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
        return "Edit a file by finding and replacing text. Only replaces the first occurrence."

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
            if not os.path.exists(path):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File not found: {path}",
                )

            # Mode 2: full content replacement
            if content is not None and search_text is None:
                with open(path, "w") as f:
                    f.write(content)
                return ToolResult(
                    success=True,
                    output=f"Successfully wrote new content to {path} ({len(content)} chars)",
                )

            # Mode 1: search-and-replace
            if search_text is None or replace_text is None:
                return ToolResult(
                    success=False,
                    output="",
                    error="edit_file requires either (search_text + replace_text) or content",
                )

            with open(path, "r") as f:
                current = f.read()

            if search_text not in current:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Search text not found in {path}",
                )

            new_content = current.replace(search_text, replace_text, 1)
            with open(path, "w") as f:
                f.write(new_content)

            return ToolResult(
                success=True,
                output=f"Successfully edited {path}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied editing: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error editing file: {e}",
            )