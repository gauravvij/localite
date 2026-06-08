"""Episode and Turn dataclasses for localite session persistence."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional


@dataclass
class Turn:
    """A single turn in an episode."""
    turn_number: int
    phase: str  # EXPLORE, PLAN, EXECUTE, VERIFY, ITERATE
    tool_call: Optional[dict] = None
    tool_result: Optional[dict] = None  # serialized ToolResult
    user_approval: Optional[str] = None  # "approved", "skipped", "rejected", "edited"
    model_output: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "phase": self.phase,
            "tool_call": self.tool_call,
            "tool_result": self.tool_result,
            "user_approval": self.user_approval,
            "model_output": self.model_output,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        return cls(**data)


@dataclass
class Episode:
    """A single episode (one user request cycle)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    objective: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    turns: list[Turn] = field(default_factory=list)
    plan: Optional[str] = None
    files_changed: list[str] = field(default_factory=list)
    summary: Optional[str] = None

    def compress(self) -> str:
        """Extract a compressed summary of this episode.

        Returns a concise text summary with key decisions, files changed,
        and important facts.
        """
        parts = [
            f"Objective: {self.objective}",
            f"Turns: {len(self.turns)}",
        ]
        if self.plan:
            parts.append(f"Plan: {self.plan[:200]}")
        if self.files_changed:
            parts.append(f"Files changed: {', '.join(self.files_changed)}")
        if self.summary:
            parts.append(f"Summary: {self.summary}")

        # Add key decisions from turns
        tool_calls = [t for t in self.turns if t.tool_call]
        if tool_calls:
            tools_used = set()
            for t in tool_calls:
                name = t.tool_call.get("name", "?")
                tools_used.add(name)
            parts.append(f"Tools used: {', '.join(sorted(tools_used))}")

        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "objective": self.objective,
            "created_at": self.created_at,
            "turns": [t.to_dict() for t in self.turns],
            "plan": self.plan,
            "files_changed": self.files_changed,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        turns = [Turn.from_dict(t) for t in data.get("turns", [])]
        return cls(
            id=data.get("id", ""),
            session_id=data.get("session_id", ""),
            objective=data.get("objective", ""),
            created_at=data.get("created_at", ""),
            turns=turns,
            plan=data.get("plan"),
            files_changed=data.get("files_changed", []),
            summary=data.get("summary"),
        )


@dataclass
class SessionOverview:
    """Summary of a session for listing purposes."""
    session_id: str
    project_name: str = ""
    episode_count: int = 0
    last_updated: str = ""
    last_objective: str = ""