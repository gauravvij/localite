"""Episodic memory store — tool-based, on-demand access, no auto-injection."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """Persistent episodic memory for agent sessions.

    Memory is NEVER auto-injected into context. It is stored on disk
    and accessed via memory_read/memory_write tools on demand.

    Storage structure:
      <base_dir>/sessions/index.json — compact session summaries (latest 10)
      <base_dir>/topics/<topic>.md — detailed topic files (written on demand)

    Context budget: memory_read results are capped at 500 tokens.
    Session start injection: single <50-token line in system prompt for most recent session only.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.sessions_dir = os.path.join(base_dir, "sessions")
        self.topics_dir = os.path.join(base_dir, "topics")
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.topics_dir, exist_ok=True)
        self._index: list[dict] = []
        self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self.sessions_dir, "index.json")

    def _load_index(self):
        """Load session index from disk."""
        path = self._index_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load session index: {e}")
                self._index = []
        if not self._index:
            self._index = []

    def _save_index(self):
        """Save session index to disk (keep latest 10)."""
        # Keep only latest 10
        trimmed = self._index[-10:]
        path = self._index_path()
        try:
            with open(path, "w") as f:
                json.dump(trimmed, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save session index: {e}")

    def save_session_summary(self, session_id: str, task: str, outcome: str, status: str):
        """Save a compact session summary to the index."""
        entry = {
            "session_id": session_id,
            "task": task[:100],
            "outcome": outcome[:200],
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._index.append(entry)
        self._save_index()

    def get_recent_session(self) -> Optional[dict]:
        """Get the most recent session summary (for system prompt injection)."""
        if not self._index:
            return None
        return self._index[-1]

    def get_recent_summary_line(self) -> str:
        """Get a single <50-token line for system prompt injection."""
        recent = self.get_recent_session()
        if not recent:
            return ""
        # Target: ~40 tokens
        task_short = recent["task"][:60]
        outcome_short = recent["outcome"][:80]
        return f"[PREVIOUS SESSION: {task_short} → {outcome_short}]"

    def read_topic(self, topic: str, max_tokens: int = 500) -> str:
        """Read a topic file (capped at max_tokens)."""
        # Sanitize topic name to avoid path traversal
        safe_topic = os.path.basename(topic.replace(" ", "_").replace("/", "_"))
        path = os.path.join(self.topics_dir, f"{safe_topic}.md")
        if not os.path.exists(path):
            return f"No saved memory found for topic: {topic}"
        try:
            with open(path) as f:
                content = f.read()
            # Rough token cap: 4 chars ≈ 1 token for English text
            max_chars = max_tokens * 4
            if len(content) > max_chars:
                content = content[:max_chars] + "\n...[truncated]"
            return content
        except OSError as e:
            return f"Error reading topic '{topic}': {e}"

    def write_topic(self, topic: str, content: str):
        """Write content to a topic file."""
        safe_topic = os.path.basename(topic.replace(" ", "_").replace("/", "_"))
        path = os.path.join(self.topics_dir, f"{safe_topic}.md")
        try:
            with open(path, "w") as f:
                f.write(content)
            logger.info(f"Saved memory topic: {topic} ({len(content)} chars)")
        except OSError as e:
            logger.error(f"Failed to write topic '{topic}': {e}")

    def get_session_count(self) -> int:
        """Get total number of stored sessions."""
        return len(self._index)