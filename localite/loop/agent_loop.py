"""AgentLoop — the core 5-phase agent loop."""

import json
import logging
import re
from typing import Any, Optional

from localite.loop.phases import Phase, next_phase
from localite.loop.turn_counter import TurnCounter
from localite.context.buffer import SessionFacts
from localite.context.refresh import ContextRefresher
from localite.context.standing_instructions import StandingInstructions
from localite.model.client import AsyncOllamaClient
from localite.permissions.gate import PermissionGate, PermissionResult
from localite.episodes.model import Episode, Turn
from localite.episodes.store import EpisodeStore

logger = logging.getLogger(__name__)

# Default system prompt template
SYSTEM_PROMPT = """You are localite, a fully local AI coding agent. You help users with codebase
understanding, modification, testing, and debugging.

You have access to the following tools:
{tool_descriptions}

## Output Format
When you want to perform an action, respond with JSON:
{{"thought": "Your reasoning", "tool": "tool_name", "arguments": {{"key": "value"}}}}

When you want to communicate with the user (no tool call needed), respond with:
{{"thought": "Your reasoning", "message": "Your message to the user"}}

## Rules
1. Always explore before acting. Read files before modifying them.
2. Propose a plan before executing.
3. Run tests after making changes.
4. Stay within the turn limit. Let the user know when you need more turns.
"""


class AgentLoop:
    """Core agent loop orchestrating the 5-phase cycle.

    Manages model interaction, tool execution, permission gating,
    context construction, degradation checks, and episode recording.
    """

    def __init__(
        self,
        model_client: AsyncOllamaClient,
        tools: dict[str, Any],
        permission_gate: PermissionGate,
        episode_store: EpisodeStore,
        standing_instructions: Optional[StandingInstructions] = None,
        max_iterations: int = 3,
    ):
        self.model = model_client
        self.tools = tools
        self.gate = permission_gate
        self.store = episode_store
        self.standing_instructions = standing_instructions or StandingInstructions()
        self.max_iterations = max_iterations

        self.episode: Optional[Episode] = None
        self.session_facts = SessionFacts()
        self.turn_counter = TurnCounter(hard_limit=4)
        self.current_phase = Phase.EXPLORE
        self.iteration_count = 0
        self.conversation_history: list[dict] = []

        # Init refresher
        tool_descs = "\n".join(
            f"  - {t.name}: {t.description}" for t in tools.values()
        )
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs)
        self.refresher = ContextRefresher(
            system_prompt_template=system_prompt,
            standing_instructions=self.standing_instructions.get_text(),
        )

    async def run(self, user_request: str) -> dict:
        """Main entry point — run the agent loop for a user request.

        Args:
            user_request: The user's initial request/objective.

        Returns:
            A dict with final state: phase, episode_id, files_changed, summary.
        """
        # Create episode
        self.episode = self.store.new_episode(objective=user_request)
        self.session_facts.current_objective = user_request
        self.current_phase = Phase.EXPLORE
        self.turn_counter.reset()
        self.iteration_count = 0

        logger.info(f"Starting episode: {user_request}")

        # Add user request to conversation
        self.conversation_history.append({"role": "user", "content": user_request})

        while self.current_phase != Phase.COMPLETE:
            logger.info(f"Phase: {self.current_phase.value}, "
                       f"Turn: {self.turn_counter.count}/{self.turn_counter.hard_limit}")

            # Check degradation
            if self._check_degradation():
                self._refresh_context()

            # Execute the current phase
            phase_complete = await self._execute_phase()

            if not phase_complete:
                # Phase didn't complete normally (e.g., user cancelled)
                break

            # Determine next phase
            tests_passed = self._get_tests_passed()
            reached = self.iteration_count >= self.max_iterations

            if self.current_phase == Phase.VERIFY:
                self.current_phase = next_phase(
                    self.current_phase,
                    tests_passed=tests_passed,
                    max_iterations_reached=reached,
                )
                if self.current_phase == Phase.ITERATE:
                    self.iteration_count += 1
            elif self.current_phase == Phase.ITERATE:
                self.current_phase = next_phase(
                    self.current_phase,
                    max_iterations_reached=reached,
                )
            else:
                self.current_phase = next_phase(self.current_phase)

        # Close episode
        summary = self.session_facts.summary()
        self.store.close_episode(self.episode, summary)

        return {
            "phase": "COMPLETE",
            "episode_id": self.episode.id,
            "files_changed": self.episode.files_changed,
            "summary": summary,
        }

    async def _execute_phase(self) -> bool:
        """Execute the current phase by running the model.

        Returns:
            True if the phase completed normally, False if cancelled.
        """
        # Build context
        context = self._build_context()
        context.extend(self.conversation_history)

        # Get model response
        try:
            response_text = await self.model.chat(context, stream=False)
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Model error: {e}")
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Error: {e}",
            })
            return False

        # Record turn
        turn = Turn(
            turn_number=self.turn_counter.count + 1,
            phase=self.current_phase.value,
            model_output=response_text,
        )

        # Parse and handle tool calls
        tool_call = self._parse_tool_call(response_text)
        if tool_call:
            turn.tool_call = tool_call

            # Get permission
            action_desc = f"Execute {tool_call.get('name', '?')} in phase {self.current_phase.value}"
            permission = self.gate.propose(action_desc, tool_call)

            turn.user_approval = permission.decision

            if permission.decision == "approved":
                modified_call = permission.modified_tool_call or tool_call
                result = await self._handle_tool_call(modified_call)
                turn.tool_result = {
                    "success": result.success,
                    "output": result.output[:500] if result.output else "",
                    "error": result.error,
                }
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                })
                self.conversation_history.append({
                    "role": "tool",
                    "content": result.output or result.error or "",
                    "name": modified_call.get("name", "tool"),
                })
            elif permission.decision == "skipped":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                })
            elif permission.decision == "rejected":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                })
            elif permission.decision == "edited":
                modified_call = permission.modified_tool_call or tool_call
                result = await self._handle_tool_call(modified_call)
                turn.tool_result = {
                    "success": result.success,
                    "output": result.output[:500] if result.output else "",
                    "error": result.error,
                }
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                })
                self.conversation_history.append({
                    "role": "tool",
                    "content": result.output or result.error or "",
                    "name": modified_call.get("name", "tool"),
                })
        else:
            # No tool call — just a message
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text,
            })

        # Record turn in episode and increment counter
        self.episode.turns.append(turn)
        self.turn_counter.increment()

        return True

    def _build_context(self) -> list[dict]:
        """Construct the full context for the model.

        Includes: system prompt, standing instructions, session facts,
        episode history, and conversation history (trimmed to fit).
        """
        # Build the core system message
        tool_descs = "\n".join(
            f"  - {t.name}: {t.description}" for t in self.tools.values()
        )
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs)

        messages = [{"role": "system", "content": system_prompt}]

        # Add session facts as a user context message
        facts_block = self.session_facts.to_context_block()
        messages.append({"role": "user", "content": facts_block})

        # Add episode history if available
        # Post-MVP: load selective reference episodes here

        return messages

    def _check_degradation(self) -> bool:
        """Check for degradation signals that should trigger a refresh.

        Checks:
        1. Turn limit reached
        2. Format drift (if format_guard enabled)
        3. Memory horizon exceeded

        Returns:
            True if a context refresh is needed.
        """
        if self.turn_counter.is_limit_reached():
            logger.info(f"Turn limit reached ({self.turn_counter.count}/{self.turn_counter.hard_limit})")
            return True
        return False

    def _refresh_context(self):
        """Re-inject system prompt, standing instructions, and reset turn counter."""
        facts_block = self.session_facts.to_context_block()

        refreshed = self.refresher.build_refreshed_context(
            session_facts_block=facts_block,
            conversation_turns=self.conversation_history[-4:] if len(self.conversation_history) > 4 else self.conversation_history,
        )

        # Replace conversation history with refreshed context
        self.conversation_history = refreshed
        self.turn_counter.reset()

        logger.info(f"Context refreshed (refresh #{self.refresher.get_refresh_count()})")

    @staticmethod
    def _normalize_args(args: dict, tool: Any) -> dict:
        """Normalize parameter names from LLM output to match tool schema.

        LLMs often invent parameter names (e.g. ``file_path`` instead of ``path``,
        or ``key`` instead of ``command``). This uses multiple strategies:
        1. Direct match against tool's declared parameter names
        2. Known alias map (file_path→path, cmd→command, etc.)
        3. Fuzzy match: if a single required param is missing and exactly one
           unrecognized arg exists, map it as the missing required param
        """
        # Global alias map: LLM-invented names → canonical parameter names
        _PARAM_ALIASES: dict[str, str] = {
            "file_path": "path",
            "filepath": "path",
            "filename": "path",
            "file_content": "content",
            "filecontent": "content",
            "content_text": "content",
            "target_text": "search_text",
            "search_for": "search_text",
            "search_pattern": "pattern",
            "cmd": "command",
            "shell_command": "command",
            "command_line": "command",
            "timeout_seconds": "timeout",
            "timeout_sec": "timeout",
        }

        tool_params = tool.parameters.get("properties", {})
        required_params = tool.parameters.get("required", [])

        normalized: dict[str, Any] = {}
        unknown: dict[str, Any] = {}

        for key, val in args.items():
            if key in tool_params:
                normalized[key] = val
            elif key in _PARAM_ALIASES and _PARAM_ALIASES[key] in tool_params:
                normalized[_PARAM_ALIASES[key]] = val
            else:
                unknown[key] = val

        # Fuzzy fallback: if there's exactly 1 unknown arg and exactly 1 required
        # param that is still missing, map the unknown arg to that required param
        if unknown and len(unknown) == 1:
            unknown_key, unknown_val = next(iter(unknown.items()))
            missing_required = [r for r in required_params if r not in normalized]
            if len(missing_required) == 1:
                logger.debug(
                    "Fuzzy-mapped unknown param '%s' → required param '%s'",
                    unknown_key, missing_required[0],
                )
                normalized[missing_required[0]] = unknown_val
                return normalized

        # If fuzzy fallback didn't apply, drop unknown params silently
        # (LLMs often hallucinate extra params like "key" — passing them
        #  through causes TypeError in the tool methods)
        if unknown:
            logger.debug(
                "Dropped %d unknown param(s): %s — not in tool schema",
                len(unknown), list(unknown.keys()),
            )
        return normalized

    async def _handle_tool_call(self, tool_call: dict) -> Any:
        """Parse a tool call, execute it, and return the result.

        Args:
            tool_call: Dict with "name" and "arguments"/"args" keys.

        Returns:
            ToolResult from the executed tool.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", tool_call.get("args", {}))

        if not name:
            from localite.tools.base import ToolResult
            return ToolResult(success=False, output="", error="No tool name in tool call")

        if name not in self.tools:
            from localite.tools.base import ToolResult
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {name}. Available: {list(self.tools.keys())}",
            )

        tool = self.tools[name]
        # Normalize LLM-invented parameter names to match the tool's schema
        args = self._normalize_args(args, tool)
        result = await tool.execute(**args)

        # Update session facts
        self.session_facts.last_tool_used = name
        if result.success:
            self.session_facts.last_tool_result = result.output[:500]
        else:
            self.session_facts.last_tool_result = f"Error: {result.error}"

        return result

    def _parse_tool_call(self, response_text: str) -> Optional[dict]:
        """Parse a tool call from model response text.

        Supports multiple formats:
        1. JSON: {"tool": "name", "arguments": {...}}
        2. LFM2.5 native: <|tool_call_start|>[tool_name(arg1='val1')]<|tool_call_end|>

        Returns:
            Dict with "name" and "args" if found, None otherwise.
        """
        # Strip thinking tags first
        from localite.model.client import strip_thinking
        cleaned = strip_thinking(response_text)

        text = cleaned.strip()
        if not text:
            return None

        # --- Format 1: <|tool_call_start|>[tool_name(key='val')]<|tool_call_end|> ---
        import ast
        tool_call_match = re.search(
            r'<\|tool_call_start\|>\s*\[(\w+)\s*\(([^)]*)\)\]\s*<\|tool_call_end\|>',
            text,
        )
        if tool_call_match:
            tool_name = tool_call_match.group(1)
            args_str = tool_call_match.group(2)
            args = {}
            if args_str.strip():
                # Parse key=value, key='value', key="value" pairs
                for pair in re.findall(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\s)]+))", args_str):
                    key = pair[0]
                    val = pair[1] or pair[2] or pair[3]
                    args[key] = val
            return {"name": tool_name, "args": args}

        # --- Format 2: JSON with "tool" (or "name") and "arguments" keys ---
        brace_start = text.find("{")
        if brace_start == -1:
            return None

        # Try to parse from the first brace
        for start in range(brace_start, min(brace_start + 200, len(text))):
            if text[start] != "{":
                continue
            try:
                # Try progressively larger slices
                for end in range(start + 1, min(len(text) + 1, start + 5000)):
                    if text[end - 1] == "}":
                        try:
                            data = json.loads(text[start:end])
                            # Check if this looks like a tool call
                            if isinstance(data, dict):
                                tool_name = data.get("tool") or data.get("name") or data.get("function")
                                if tool_name:
                                    args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
                                    if isinstance(args, dict):
                                        return {"name": tool_name, "args": args}
                                    elif isinstance(args, str):
                                        try:
                                            parsed_args = json.loads(args)
                                            if isinstance(parsed_args, dict):
                                                return {"name": tool_name, "args": parsed_args}
                                        except (json.JSONDecodeError, ValueError):
                                            pass
                                        return {"name": tool_name, "args": {"input": args}}
                            # No tool call key found, continue scanning
                        except (json.JSONDecodeError, ValueError):
                            continue
            except (ValueError, IndexError):
                continue
            break

        return None

    def _get_tests_passed(self) -> bool | None:
        """Check if the last test execution passed.

        Returns True if tests passed, False if failed, None if no tests run.
        """
        for turn in reversed(self.episode.turns):
            if turn.tool_call and turn.tool_call.get("name") == "test_executor":
                if turn.tool_result:
                    return turn.tool_result.get("success", False)
        return None

    def get_status(self) -> dict:
        """Return current agent loop status."""
        return {
            "phase": self.current_phase.value,
            "turn": str(self.turn_counter),
            "episode_id": self.episode.id if self.episode else None,
            "iteration_count": self.iteration_count,
            "files_changed": self.episode.files_changed if self.episode else [],
        }