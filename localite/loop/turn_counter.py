"""Turn counter — tracks consecutive turns and enforces hard limit."""

from dataclasses import dataclass


@dataclass
class TurnCounter:
    """Tracks turns in the current segment and enforces hard limits.

    The hard limit prevents the model from running too many consecutive
    turns without a context refresh or user interaction.
    """
    hard_limit: int = 4
    _count: int = 0

    def increment(self):
        """Increment the turn counter."""
        self._count += 1

    def remaining(self) -> int:
        """Return remaining turns before limit is reached."""
        return max(0, self.hard_limit - self._count)

    def is_limit_reached(self) -> bool:
        """Check if the turn limit has been reached."""
        return self._count >= self.hard_limit

    def reset(self):
        """Reset the turn counter to zero."""
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def __str__(self) -> str:
        return f"{self._count}/{self.hard_limit}"