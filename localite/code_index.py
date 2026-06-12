"""CodeIndex — a ctags-based symbol-to-file mapping for code navigation.

Provides fast, case-insensitive lookup of where identifiers (classes, functions,
types) are defined in a repository, enabling precise file recommendations over
keyword-based filename guessing.
"""

import logging
import subprocess
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class CodeIndex:
    """Maps symbol names (classes, functions, types) to their file locations
    using universal-ctags.

    The index is built by running ``ctags -R`` on the repository and parsing
    the tab-separated output.  Lookup is case-insensitive and results are
    sorted with Python files and class definitions preferred.

    If ctags is not installed the index is gracefully disabled (logged warning)
    and all lookups return empty results.
    """

    def __init__(self, repo_path: str):
        """Initialise the code index by running ctags on *repo_path*.

        Args:
            repo_path: Absolute path to the repository root to index.
        """
        self._repo_path = repo_path
        self._index: Dict[str, List[Tuple[str, int, str]]] = {}
        self._disabled = False

        try:
            self._run_ctags()
            logger.info(
                "CodeIndex initialised — %d unique symbols indexed from %s",
                len(self._index),
                repo_path,
            )
        except FileNotFoundError:
            logger.warning(
                "ctags not found — CodeIndex disabled.  "
                "Install universal-ctags: sudo apt install universal-ctags"
            )
            self._disabled = True
        except Exception as exc:
            logger.warning("ctags failed (%s) — CodeIndex disabled.", exc)
            self._disabled = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> List[Tuple[str, int, str]]:
        """Look up a symbol name in the index (case-insensitive).

        Tries, in order:
        1. Exact match (e.g. ``PersonName``)
        2. Trailing-digit-stripped match (e.g. ``PersonName3`` → ``PersonName``)
        3. Prefix match using common-case prefixes (e.g. ``my_new_class`` → entries starting with ``my_new_``)

        Args:
            name: Symbol name to search for.

        Returns:
            List of ``(file_path, line_number, kind)`` tuples sorted so that
            Python files appear first, class definitions before function/
            variable entries, and shallower paths first.
        """
        if self._disabled or not self._index:
            return []

        name_lower = name.lower()

        # 1. Exact match
        results = self._index.get(name_lower, [])
        if results:
            return self._sort_results(results)

        # 2. Strip trailing digits (PersonName3 → PersonName)
        stripped = name_lower.rstrip("0123456789")
        if stripped and stripped != name_lower:
            results = self._index.get(stripped, [])
            if results:
                return self._sort_results(results)

        # 3. Prefix match — find keys that start with the name or vice versa
        for key, entries in self._index.items():
            if key.startswith(name_lower) or name_lower.startswith(key):
                results.extend(entries)
        if results:
            return self._sort_results(results)

        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_results(results: List[Tuple[str, int, str]]) -> List[Tuple[str, int, str]]:
        """Sort lookup results: Python files first, class before func/variable."""

        def _sort_key(item: Tuple[str, int, str]) -> tuple:
            file_path, _line_number, kind = item
            # Python files (0) before non-Python (1)
            is_python = 0 if file_path.endswith(".py") else 1
            # class (0) before function/method (1), then everything else (2)
            kind_priority = (
                0 if kind in ("c", "class")
                else 1 if kind in ("f", "function", "m", "method")
                else 2
            )
            return (is_python, kind_priority, len(file_path))

        return sorted(results, key=_sort_key)

    def reindex(self) -> None:
        """Rebuild the entire index from scratch."""
        if self._disabled:
            return
        try:
            self._run_ctags()
            logger.info("Full re-index complete — %d unique symbols", len(self._index))
        except Exception as exc:
            logger.warning("Full re-index failed (%s) — index may be stale.", exc)

    def reindex_file(self, filepath: str) -> None:
        """Re-index a single file after modification.

        For simplicity this triggers a full re-index (``ctags -R`` is fast
        enough — typically < 200 ms for SWE-bench repos).

        Args:
            filepath: Path to the file that was changed (used for logging).
        """
        if self._disabled:
            return
        logger.debug("Re-indexing after change to %s", filepath)
        self.reindex()

    def __repr__(self) -> str:
        status = "disabled" if self._disabled else f"active ({len(self._index)} symbols)"
        return f"CodeIndex(repo_path={self._repo_path!r}, {status})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_ctags(self) -> None:
        """Run ctags recursively on the repository and parse the output."""
        result = subprocess.run(
            ["ctags", "-R", "-f", "-", "--fields=+n", self._repo_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ctags exited with code {result.returncode}: {result.stderr}"
            )
        self._parse_output(result.stdout)

    def _parse_output(self, text: str) -> None:
        """Parse ctags tab-separated output into the internal index.

        ctags output format (tab-separated)::

            tag_name\\tfile_path\\tpattern;"\\tkind\\tline:number

        Example::

            PersonName3\tvaluerep.py\t/^class PersonName3: pass$/;"\tc\tline:1
        """
        self._index.clear()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("!_"):
                # Skip meta lines (ctags header metadata)
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue

            tag_name = parts[0]
            file_path = parts[1]

            # Parse kind (single-letter or long form) and line number from the
            # remaining fields.
            kind = ""
            line_number = 0
            for field in parts[3:]:
                if field.startswith("line:"):
                    try:
                        line_number = int(field[5:])
                    except ValueError:
                        pass
                elif field in (
                    # Universal Ctags single-letter kind codes for common
                    # languages — c, f, m, v cover most Python cases.
                    "c",    # class
                    "f",    # function / file
                    "m",    # member / method
                    "v",    # variable
                    "t",    # typedef
                    "d",    # macro
                    "e",    # enum
                    "i",    # interface
                    "p",    # prototype / package
                    "x",    # externvar
                    "z",    # zone
                    "n",    # namespace
                    "g",    # enum constant
                    "s",    # struct / namespace
                    "u",    # union
                    "l",    # local
                    "k",    # kind
                    "w",    # wildcard
                    "C",    # constant
                    "F",    # file
                    "M",    # module
                    "V",    # variable
                    "T",    # typedef
                ):
                    kind = field
                elif field in (
                    "class", "function", "method", "variable", "member",
                    "struct", "enum", "interface", "typedef", "macro",
                    "constant", "namespace", "module",
                ):
                    kind = field

            key = tag_name.lower()
            if key not in self._index:
                self._index[key] = []
            self._index[key].append((file_path, line_number, kind))