"""Context refresh logic — re-injects system prompt and resets turn counter."""

import logging

logger = logging.getLogger(__name__)


class ContextRefresher:
    """Manages context refreshes to prevent degradation in long sessions.

    Handles re-injection of system prompt, standing instructions, and
    session facts when the turn limit is reached or quality degrades.
    """

    def __init__(
        self,
        system_prompt_template: str,
        standing_instructions: str,
    ):
        self.system_prompt_template = system_prompt_template
        self.standing_instructions = standing_instructions
        self.refresh_count = 0

    def build_refreshed_context(
        self,
        session_facts_block: str,
        episode_history_block: str = "",
        conversation_turns: list[dict] | None = None,
    ) -> list[dict]:
        """Build a complete refreshed context block.

        Returns a list of message dicts suitable for the model API:
        [system, user with standing instructions + facts, conversation turns...]
        """
        self.refresh_count += 1
        logger.info(f"Context refresh #{self.refresh_count}")

        # Build the system message
        system_msg = {
            "role": "system",
            "content": self.system_prompt_template,
        }

        # Build the context injection message
        context_parts = [
            "## Context Refresh",
            f"Refresh #{self.refresh_count}",
            "",
            self.standing_instructions,
            "",
            session_facts_block,
        ]
        if episode_history_block:
            context_parts.extend(["", episode_history_block])

        context_content = "\n".join(context_parts)

        # System message carries the core identity
        messages = [system_msg]

        # Inject standing instructions + session facts as a user message
        # (some models handle this better than system messages)
        messages.append({"role": "user", "content": context_content})

        # Append recent conversation turns
        if conversation_turns:
            messages.extend(conversation_turns)

        return messages

    def get_refresh_count(self) -> int:
        """Return number of refreshes performed."""
        return self.refresh_count