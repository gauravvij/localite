"""Diff view tool - generates unified diffs with Rich rendering."""

import difflib
from localite.tools.base import BaseTool, ToolResult, measure_duration


class DiffViewTool(BaseTool):
    """Tool for generating and displaying diffs between file versions."""

    @property
    def name(self) -> str:
        return "diff_view"

    @property
    def description(self) -> str:
        return (
            "Generate a unified diff between original and new content for a file. "
            "WHEN TO USE: To preview changes before applying them, to show what changed between "
            "two versions of a file, for code review and documentation of modifications. "
            "WHEN NOT TO USE: For making actual edits to files (use edit_file or write_file), "
            "for reading file contents (use read_file), for running shell diff commands (use run_shell). "
            "PARAMETERS: 'original_content' (required, string, the original file content), "
            "'new_content' (required, string, the modified file content), "
            "'filepath' (required, string, file path for display in diff header). "
            "EXAMPLE: {\"original_content\": \"def old():\", "
            "\"new_content\": \"def new():\", "
            "\"filepath\": \"src/utils.py\"} "
            "COMMON MISTAKES: Passing file paths instead of actual content strings; "
            "forgetting to include the 'filepath' for proper diff header display."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "original_content": {
                    "type": "string",
                    "description": "Original file content",
                },
                "new_content": {
                    "type": "string",
                    "description": "New/modified file content",
                },
                "filepath": {
                    "type": "string",
                    "description": "File path for display in the diff header",
                },
            },
            "required": ["original_content", "new_content", "filepath"],
        }

    @measure_duration
    async def execute(
        self,
        original_content: str,
        new_content: str,
        filepath: str,
    ) -> ToolResult:
        """Generate a unified diff between two versions of a file."""
        try:
            original_lines = original_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)

            diff = difflib.unified_diff(
                original_lines,
                new_lines,
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
            )

            diff_text = "".join(diff)
            if not diff_text:
                return ToolResult(
                    success=True,
                    output="No differences found.",
                )

            # Count changes
            added = diff_text.count("\n+") - 1  # subtract the header line
            removed = diff_text.count("\n-") - 1  # subtract the header line

            summary = f"Diff for {filepath}: {added} additions, {removed} removals\n"
            stats = {"additions": max(0, added), "removals": max(0, removed)}

            return ToolResult(
                success=True,
                output=summary + diff_text,
                data=stats,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error generating diff: {e}",
            )

    @staticmethod
    def render_diff(diff_text: str) -> str:
        """Render diff text with ANSI color codes for terminal display.

        Returns the diff with color markers: green for additions, red for removals.
        """
        lines = []
        for line in diff_text.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                lines.append(f"\033[32m{line}\033[0m")  # green
            elif line.startswith("-") and not line.startswith("---"):
                lines.append(f"\033[31m{line}\033[0m")  # red
            elif line.startswith("@@"):
                lines.append(f"\033[36m{line}\033[0m")  # cyan
            else:
                lines.append(line)
        return "\n".join(lines)