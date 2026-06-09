"""FormatMonitor — tracks tool call JSON format quality over recent turns.

Used to detect when a model's output format degrades (stops producing valid
JSON tool calls) and triggers an early context refresh.
"""

import json
import logging
from collections import deque

logger = logging.getLogger(__name__)

# Default window size: track last N tool calls
DEFAULT_WINDOW = 10
# Threshold below which should_refresh() returns True
DEFAULT_THRESHOLD = 0.3


class FormatMonitor:
    """Monitors tool call format quality and signals when degradation occurs.

    Attributes:
        window_size: Number of recent tool calls to track.
        threshold: Running average below this triggers should_refresh() -> True.
        _scores: Deque of recent scores (each 0.0-1.0).
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW, threshold: float = DEFAULT_THRESHOLD):
        self.window_size = window_size
        self.threshold = threshold
        self._scores: deque[float] = deque(maxlen=window_size)

    def record_tool_call(self, tool_call: dict | None, response_text: str) -> float:
        """Score a tool call output for JSON adherence and record the score.

        Args:
            tool_call: Parsed tool call dict (None if parsing failed).
            response_text: Raw model output text.

        Returns:
            Score 0.0-1.0 indicating JSON adherence quality.
        """
        score = self._score_response(response_text, tool_call)
        self._scores.append(score)
        logger.debug(f"FormatMonitor recorded score={score:.2f}, avg={self.average():.2f}")
        return score

    def _score_response(self, response_text: str, tool_call: dict | None) -> float:
        """Score a response for JSON tool call adherence.

        Scoring rules:
        - 1.0: Valid JSON with proper tool/name and arguments fields
        - 0.7: Valid JSON structure but missing required tool call keys
        - 0.3: Partial JSON (starts with { but can't be fully parsed)
        - 0.0: No JSON structure at all (plain text/prose)

        Args:
            response_text: Raw model output.
            tool_call: Parsed tool call dict (None if parsing failed).

        Returns:
            Score 0.0-1.0.
        """
        text = response_text.strip()

        if not text:
            return 0.0

        # Check if output contains a JSON block anywhere (text-embedded JSON
        # — e.g. "I'll write the function. {\"tool\": ...}" — which is the
        # default output style for models like Gemma 4 E4B).
        brace_start = text.find('{')
        if brace_start == -1:
            # No JSON structure found anywhere — plain text/prose
            return 0.0

        # Find matching closing brace
        depth = 0
        json_end = -1
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        if json_end == -1:
            # Unmatched braces — partial JSON
            return 0.3

        try:
            data = json.loads(text[brace_start:json_end])
        except (json.JSONDecodeError, ValueError):
            # Contains braces but not valid JSON
            return 0.3

        if not isinstance(data, dict):
            return 0.3

        # Check for tool call keys
        has_tool_key = bool(data.get("tool") or data.get("name") or data.get("function"))
        has_arguments = bool(data.get("arguments") or data.get("args") or data.get("parameters"))
        has_thought = "thought" in data

        if has_tool_key and has_arguments:
            return 1.0
        elif has_tool_key or has_arguments:
            return 0.7
        elif has_thought:
            # Valid JSON but missing tool call shape — might be a message-only response
            return 0.7
        else:
            return 0.3

    def should_refresh(self) -> bool:
        """Check if format quality has degraded below threshold.

        Returns:
            True if the running average is below threshold (needs refresh).
            False if no data yet or quality is acceptable.
        """
        if not self._scores:
            return False
        return self.average() < self.threshold

    def average(self) -> float:
        """Compute running average of recent scores.

        Returns:
            Average score (0.0-1.0), or 0.0 if no data.
        """
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    def reset(self):
        """Clear all recorded scores."""
        self._scores.clear()
        logger.debug("FormatMonitor reset")