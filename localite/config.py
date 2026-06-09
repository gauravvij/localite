"""Configuration management for localite.

Provides ModelProfile dataclass and ConfigLoader for TOML-based profiles.
"""

import os
import tomli
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelProfile:
    """Configuration profile for a model.

    Attributes:
        name: Full model name/identifier used in Ollama.
        provider: Provider name (e.g. "ollama").
        max_turns: Maximum consecutive turns before forced refresh (default 4).
        memory_horizon: Maximum turns before memory retrieval degrades (default 5).
        format_guard: Whether to enforce JSON output format checks (default True).
        recency_protection: Whether to inject standing instructions block (default True).
        has_thinking_tags: Whether model uses <thinking>/<response> XML tags (default True).
        base_url: Ollama API base URL (default http://localhost:11434).
        timeout: API timeout in seconds (default 30).
    """
    name: str
    provider: str = "ollama"
    max_turns: int = 4
    memory_horizon: int = 5
    format_guard: bool = True
    recency_protection: bool = True
    has_thinking_tags: bool = True
    iad_horizon: int = 30
    base_url: str = "http://localhost:11434"
    timeout: int = 30


class ConfigLoader:
    """Loads model profiles from TOML files."""

    def __init__(self, profiles_dir: Optional[str] = None):
        self.profiles_dir = profiles_dir or self._default_profiles_dir()

    @staticmethod
    def _default_profiles_dir() -> str:
        """Resolve the default profiles directory relative to this file."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")

    def load_profile(self, profile_name: str) -> ModelProfile:
        """Load a model profile from a TOML file.

        Args:
            profile_name: Name of the profile file (with or without .toml extension).

        Returns:
            ModelProfile instance populated from the TOML file.

        Raises:
            FileNotFoundError: If the profile file does not exist.
            tomli.TOMLDecodeError: If the TOML file is malformed.
        """
        if not profile_name.endswith(".toml"):
            profile_name += ".toml"
        path = os.path.join(self.profiles_dir, profile_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile not found: {path}")
        with open(path, "rb") as f:
            data = tomli.load(f)
        return ModelProfile(**data["model"])

    def list_profiles(self) -> list[str]:
        """List available profile names."""
        if not os.path.isdir(self.profiles_dir):
            return []
        return [
            f.replace(".toml", "")
            for f in os.listdir(self.profiles_dir)
            if f.endswith(".toml")
        ]