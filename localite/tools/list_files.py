"""List files tool - lists directory contents intelligently, excluding noise directories."""

import os
import subprocess
from localite.tools.base import BaseTool, ToolResult, measure_duration


# Directories to exclude from recursive listing to avoid flooding context
EXCLUDED_DIRS = {
    "venv", ".git", "__pycache__", "node_modules", ".tox", ".egg-info",
    ".mypy_cache", ".pytest_cache", ".sass-cache", ".ruff_cache",
    "site-packages", "dist", "build", ".direnv", ".nox", ".hypothesis",
    ".coverage",
}


class ListFilesTool(BaseTool):
    """Tool for listing directory contents intelligently."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return (
            "List files and directories. Shows one level by default. "
            "Use depth=2 or 3 for recursive listing. Automatically excludes noise "
            "directories like venv, .git, __pycache__, node_modules, .tox, .egg-info, "
            ".mypy_cache, .pytest_cache. Use this instead of run_shell (ls -R) for directory exploration. "
            "WHEN TO USE: In EXPLORE phase to discover project structure before reading files; "
            "finding what files exist in a directory without flooding context. "
            "WHEN NOT TO USE: For reading file contents (use read_file), for searching text "
            "patterns in files (use grep_search), for running shell commands (use run_shell). "
            "PARAMETERS: 'path' (optional, string, default '.'), 'depth' (optional, int 1-3, "
            "default 1), 'show_hidden' (optional, bool, default false), "
            "'max_results' (optional, int, default 200), 'show_file_sizes' (optional, bool, default false). "
            "EXAMPLE: {\"path\": \"/home/user/project/src\", \"depth\": 2, \"show_file_sizes\": true} "
            "COMMON MISTAKES: Using 'directory' as parameter name instead of the correct 'path'; "
            "using run_shell with 'ls -R' instead of list_files which auto-excludes noise dirs; "
            "setting depth > 3 which is silently capped."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                    "default": ".",
                },
                "depth": {
                    "type": "integer",
                    "description": "Recursion depth (max 3)",
                    "default": 1,
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Show dotfiles",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max entries to return",
                    "default": 200,
                },
                "show_file_sizes": {
                    "type": "boolean",
                    "description": "Show file sizes",
                    "default": False,
                },
            },
            "required": [],
        }

    @measure_duration
    async def execute(
        self,
        path: str = ".",
        depth: int = 1,
        show_hidden: bool = False,
        max_results: int = 200,
        show_file_sizes: bool = False,
    ) -> ToolResult:
        """List directory contents intelligently.

        Args:
            path: Directory path to list.
            depth: Recursion depth (default 1, max 3).
            show_hidden: Whether to show dotfiles.
            max_results: Maximum number of entries to return.
            show_file_sizes: Whether to show file sizes.

        Returns:
            ToolResult with formatted directory listing.
        """
        try:
            # Type coercion: handle string params the model often passes
            # (e.g. depth="2" instead of depth=2)
            if isinstance(depth, str):
                try:
                    depth = int(depth)
                except (ValueError, TypeError):
                    pass
            if isinstance(max_results, str):
                try:
                    max_results = int(max_results)
                except (ValueError, TypeError):
                    pass

            # Resolve relative paths against workdir
            if not path.startswith("/"):
                resolved = os.path.join(self.workdir, path) if hasattr(self, 'workdir') and self.workdir else path
            else:
                resolved = path

            # Validate path
            if not os.path.exists(resolved):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Path does not exist: {resolved}",
                )
            if not os.path.isdir(resolved):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Path is not a directory: {resolved}",
                )
            path = resolved  # use resolved path for listing

            # Clamp depth
            depth = max(1, min(depth, 3))

            # Walk the directory tree
            lines = []
            count_ref = [0]

            self._list_dir(
                path=path,
                current_depth=0,
                max_depth=depth,
                show_hidden=show_hidden,
                max_results=max_results,
                show_file_sizes=show_file_sizes,
                lines=lines,
                count_ref=count_ref,
            )

            output = "\n".join(lines)
            if not output:
                output = "(empty directory)"
            # Label the output clearly so the model doesn't misinterpret the tree format
            output = (
                "[DIRECTORY TREE - Source code files below. "
                "Use read_file to read their CONTENTS.]\n"
                f"{output}"
            )

            return ToolResult(
                success=True,
                output=output,
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output="",
                error=f"Permission denied listing directory: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error listing directory: {e}",
            )

    def _list_dir(
        self,
        path: str,
        current_depth: int,
        max_depth: int,
        show_hidden: bool,
        max_results: int,
        show_file_sizes: bool,
        lines: list,
        count_ref: list,
        prefix: str = "",
    ):
        """Recursively list directory contents.

        Args:
            path: Directory path to list.
            current_depth: Current recursion depth.
            max_depth: Maximum recursion depth.
            show_hidden: Whether to show dotfiles.
            max_results: Maximum entries to return.
            show_file_sizes: Whether to show file sizes.
            lines: Output lines accumulator.
            count_ref: Mutable counter [count].
            prefix: Indentation prefix for tree display.
        """
        try:
            entries = list(os.scandir(path))
        except PermissionError:
            lines.append(f"{prefix}[permission denied: {os.path.basename(path)}]")
            return

        # Sort: directories first, then files, alphabetically within each group
        dirs = []
        files = []
        for entry in entries:
            if not show_hidden and entry.name.startswith("."):
                continue
            # Skip excluded directories entirely
            if entry.is_dir(follow_symlinks=False) and entry.name in EXCLUDED_DIRS:
                continue
            if entry.is_dir(follow_symlinks=False):
                dirs.append(entry)
            else:
                files.append(entry)

        dirs.sort(key=lambda e: e.name.lower())
        files.sort(key=lambda e: e.name.lower())

        # Determine display prefix
        is_root = current_depth == 0
        indent = "" if is_root else prefix

        for entry in dirs:
            if count_ref[0] >= max_results:
                remaining = len(dirs) + len(files) - (dirs.index(entry) + len(files) if entry in dirs else files.index(entry))
                lines.append(f"{indent} ... and {remaining} more entries")
                return

            # Get size for directories if requested
            size_str = ""
            if show_file_sizes:
                try:
                    result = subprocess.run(
                        ["stat", "--format=%s", entry.path],
                        capture_output=True, text=True, timeout=5,
                    )
                    raw_size = result.stdout.strip()
                    if raw_size:
                        size_str = f" ({raw_size} bytes)"
                except Exception:
                    size_str = ""

            count_ref[0] += 1
            lines.append(f"{indent}/{entry.name} (dir){size_str}")

            # Recurse into subdirectory
            if current_depth + 1 < max_depth:
                self._list_dir(
                    path=entry.path,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    show_hidden=show_hidden,
                    max_results=max_results,
                    show_file_sizes=show_file_sizes,
                    lines=lines,
                    count_ref=count_ref,
                    prefix=indent + "  ",
                )

        for entry in files:
            if count_ref[0] >= max_results:
                remaining = len(files) - files.index(entry)
                lines.append(f"{indent} ... and {remaining} more entries")
                return

            # Get file size if requested
            size_str = ""
            if show_file_sizes:
                try:
                    result = subprocess.run(
                        ["stat", "--format=%s", entry.path],
                        capture_output=True, text=True, timeout=5,
                    )
                    raw_size = result.stdout.strip()
                    if raw_size:
                        size_str = f" ({raw_size} bytes)"
                except Exception:
                    size_str = ""

            count_ref[0] += 1
            lines.append(f"{indent}{entry.name} (file){size_str}")