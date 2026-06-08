"""EpisodeStore — JSON-based file persistence for episodes and sessions."""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from localite.episodes.model import Episode, SessionOverview, Turn


DEFAULT_SESSION_DIR = os.path.expanduser("~/.local-code-agent/sessions")


class EpisodeStore:
    """Persists episodes to JSON files on disk.

    Each session gets a directory under session_dir.
    Episodes are stored as individual JSON files within the session directory.
    """

    def __init__(self, session_dir: str = DEFAULT_SESSION_DIR):
        self.session_dir = session_dir

    def _ensure_session_dir(self, session_id: str) -> str:
        """Create the directory for a session if it doesn't exist."""
        path = os.path.join(self.session_dir, session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _episode_path(self, session_id: str, episode_id: str) -> str:
        return os.path.join(self.session_dir, session_id, f"{episode_id}.json")

    def new_episode(self, objective: str, session_id: Optional[str] = None) -> Episode:
        """Create a new episode with a fresh ID."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        return Episode(
            id=str(uuid.uuid4()),
            session_id=session_id,
            objective=objective,
            created_at=datetime.now().isoformat(),
        )

    def save_episode(self, episode: Episode) -> str:
        """Save an episode to disk. Returns the episode ID."""
        session_path = self._ensure_session_dir(episode.session_id)
        path = os.path.join(session_path, f"{episode.id}.json")
        with open(path, "w") as f:
            json.dump(episode.to_dict(), f, indent=2, default=str)
        return episode.id

    def load_episode(self, session_id: str, episode_id: str) -> Optional[Episode]:
        """Load an episode from disk."""
        path = self._episode_path(session_id, episode_id)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return Episode.from_dict(data)

    def list_sessions(self) -> list[SessionOverview]:
        """List all available sessions."""
        if not os.path.isdir(self.session_dir):
            return []
        sessions = []
        for name in sorted(os.listdir(self.session_dir), reverse=True):
            session_path = os.path.join(self.session_dir, name)
            if not os.path.isdir(session_path):
                continue
            episodes = [
                f for f in os.listdir(session_path)
                if f.endswith(".json")
            ]
            last_updated = ""
            last_objective = ""
            if episodes:
                # Find the most recent episode
                latest = max(episodes)
                ep_path = os.path.join(session_path, latest)
                try:
                    with open(ep_path, "r") as f:
                        data = json.load(f)
                    last_updated = data.get("created_at", "")
                    last_objective = data.get("objective", "")
                except Exception:
                    pass
            sessions.append(SessionOverview(
                session_id=name,
                episode_count=len(episodes),
                last_updated=last_updated,
                last_objective=last_objective,
            ))
        return sessions

    def load_latest_session(self) -> Optional[tuple[str, list[Episode]]]:
        """Load the most recent session and all its episodes."""
        sessions = self.list_sessions()
        if not sessions:
            return None
        latest = sessions[0]  # sorted reverse by name
        episodes = self.load_session_episodes(latest.session_id)
        return (latest.session_id, episodes)

    def load_session_episodes(self, session_id: str) -> list[Episode]:
        """Load all episodes for a given session."""
        session_path = os.path.join(self.session_dir, session_id)
        if not os.path.isdir(session_path):
            return []
        episodes = []
        for fname in sorted(os.listdir(session_path)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(session_path, fname)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                episodes.append(Episode.from_dict(data))
            except Exception:
                continue
        return episodes

    def close_episode(self, episode: Episode, summary: str) -> Episode:
        """Finalize an episode with a summary and compress it.

        Saves the closed episode to disk and returns it.
        """
        episode.summary = summary
        self.save_episode(episode)
        return episode

    def compress_episode(self, episode: Episode) -> str:
        """Generate a compressed string summary from an episode.

        Delegates to Episode.compress() method.
        """
        return episode.compress()