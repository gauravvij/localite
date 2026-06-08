"""Session fact buffer — maintains key facts about the current session."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionFacts:
    """Key facts about the current working session.

    These are injected into context to help the model maintain awareness
    of the current task state across turns.
    """
    current_objective: str = ""
    current_file: Optional[str] = None
    key_constraints: list[str] = field(default_factory=list)
    last_tool_used: Optional[str] = None
    last_tool_result: Optional[str] = None
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    test_results: Optional[str] = None

    def to_context_block(self) -> str:
        """Format session facts as a context block for the model."""
        lines = ["## Session Facts"]
        if self.current_objective:
            lines.append(f"Current Objective: {self.current_objective}")
        if self.current_file:
            lines.append(f"Current File: {self.current_file}")
        if self.key_constraints:
            lines.append("Constraints:")
            for c in self.key_constraints:
                lines.append(f"  - {c}")
        if self.files_created:
            lines.append(f"Files Created: {', '.join(self.files_created)}")
        if self.files_modified:
            lines.append(f"Files Modified: {', '.join(self.files_modified)}")
        if self.last_tool_used:
            lines.append(f"Last Tool: {self.last_tool_used}")
        if self.last_tool_result:
            truncated = self.last_tool_result[:200]
            lines.append(f"Last Result: {truncated}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Short summary for episode compression."""
        parts = []
        if self.current_objective:
            parts.append(f"obj={self.current_objective[:60]}")
        if self.files_created:
            parts.append(f"created={len(self.files_created)}")
        if self.files_modified:
            parts.append(f"modified={len(self.files_modified)}")
        return "; ".join(parts) if parts else "no session facts"