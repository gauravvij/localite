"""AgentLoop — the core 5-phase agent loop."""

import json
import logging
import re
from typing import Any, Optional

from localite.loop.phases import Phase, next_phase
from localite.loop.turn_counter import TurnCounter
from localite.config import ModelProfile
from localite.context.buffer import SessionFacts
from localite.context.refresh import ContextRefresher
from localite.context.format_monitor import FormatMonitor
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
3. Use **edit_file** to make code changes (add docstrings, fix bugs, refactor).
4. Run tests after making changes using **test_executor**.
5. Stay within the turn limit. Let the user know when you need more turns.
6. When you are done with ALL tasks (code changes + tests passing), transition to COMPLETE and call **task_complete**.
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
        model_profile: Optional[ModelProfile] = None,
        standing_instructions: Optional[StandingInstructions] = None,
        max_iterations: int = 3,
        memory_store: Optional['EpisodicMemoryStore'] = None,
    ):
        self.model = model_client
        self.tools = tools
        self.gate = permission_gate
        self.store = episode_store
        self.standing_instructions = standing_instructions or StandingInstructions()
        self.max_iterations = max_iterations

        # Profile-driven configuration
        self.profile = model_profile
        self.format_guard = model_profile.format_guard if model_profile else True
        self.memory_horizon = model_profile.memory_horizon if model_profile else 5
        self.recency_protection = model_profile.recency_protection if model_profile else True

        self.episode: Optional[Episode] = None
        self.session_facts = SessionFacts()
        self.turn_counter = TurnCounter(hard_limit=model_profile.max_turns if model_profile else 4)
        self.current_phase = Phase.EXPLORE
        self.iteration_count = 0
        self.conversation_history: list[dict] = []
        self.active_plan: Optional[str] = None
        self.format_monitor = FormatMonitor()
        self.stall_count = 0
        self.stall_threshold = getattr(model_profile, 'stall_threshold', 3)
        self.task_complete_called = False

        # Delegation telemetry
        self.tool_stats: dict[str, dict] = {}  # tool_name -> {calls, successes, failures, total_duration_ms, trust_score}

        # Episodic memory store (optional, tool-accessed only)
        self.memory_store = memory_store

        # Wire memory store into memory tools
        if self.memory_store:
            for t_name in ("memory_read", "memory_write"):
                if t_name in self.tools:
                    self.tools[t_name]._memory_store = self.memory_store

        # Init refresher using _get_tool_descriptions for the initial system prompt
        tool_descs = self._get_tool_descriptions()
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs)
        self.refresher = ContextRefresher(
            system_prompt_template=system_prompt,
            standing_instructions=self.standing_instructions.get_text(),
        )

    def _get_tool_descriptions(self) -> str:
        """Build tool descriptions for system prompt, demoting low-trust tools."""
        lines = []
        for t in self.tools.values():
            stats = self.tool_stats.get(t.name, {})
            trust = stats.get("trust_score", 1.0)
            if trust < 0.3 and stats.get("calls", 0) >= 3:
                # Drop the tool entirely — it keeps failing
                logger.info(f"Dropping tool '{t.name}' from descriptions (trust={trust:.2f})")
                continue
            elif trust < 0.7 and stats.get("calls", 0) >= 3:
                # Demote the tool — add warning
                lines.append(f"  - {t.name}: [⚠️ LOW RELIABILITY] {t.description}")
            else:
                lines.append(f"  - {t.name}: {t.description}")
        return "\n".join(lines)

    def _should_skip_phase(self, phase: Phase) -> bool:
        """Check if the current phase should be skipped based on conditions."""
        if phase == Phase.EXPLORE:
            # Skip if no files to explore (e.g., user provided all context)
            # Skip if user's request is trivial (no file operations needed)
            # Placeholder — full explore skip requires memory_store context
            return False

        elif phase == Phase.PLAN:
            # Skip if user provided an explicit plan in their request
            request_lower = self.session_facts.current_objective.lower()
            plan_keywords = ["plan:", "1.", "first", "step 1", "step1"]
            if any(kw in request_lower for kw in plan_keywords):
                logger.info(f"Skipping PLAN phase — user provided explicit plan in request")
                # If skipping plan, we still want the model to acknowledge the plan
                # Set a minimal active_plan from the user's request
                if not self.active_plan:
                    self.active_plan = self.session_facts.current_objective[:500]
                return True
            return False

        elif phase == Phase.VERIFY:
            # Skip if no files were changed — nothing to verify
            files_changed_count = len(self.episode.files_changed) if self.episode else 0
            if files_changed_count == 0:
                logger.info("Skipping VERIFY phase — no files changed")
                return True
            return False

        elif phase == Phase.EXECUTE:
            # Never skip EXECUTE — always let the model act
            return False

        elif phase == Phase.ITERATE:
            # Skip if no files changed (nothing to iterate on)
            files_changed_count = len(self.episode.files_changed) if self.episode else 0
            if files_changed_count == 0 and self.iteration_count > 0:
                logger.info("Skipping ITERATE phase — no files changed")
                return True
            return False

        return False

    def _update_tool_stats(self, name: str, success: bool, duration_ms: int):
        """Update tool telemetry stats."""
        if name not in self.tool_stats:
            self.tool_stats[name] = {"calls": 0, "successes": 0, "failures": 0, "total_duration_ms": 0, "trust_score": 1.0}
        stats = self.tool_stats[name]
        stats["calls"] += 1
        stats["total_duration_ms"] += duration_ms
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        # Trust score: exponential moving average of success rate
        # After 1 call: trust = 0.5 if success else 0.0
        # After 10+ calls: trust converges to actual success rate
        if stats["calls"] >= 3:
            stats["trust_score"] = stats["successes"] / stats["calls"]
        elif stats["calls"] == 1:
            stats["trust_score"] = 0.5 if success else 0.0
        # else: keep 1.0 for 2 calls (too early to judge)

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
        self._complete_turn_given = False

        logger.info(f"Starting episode: {user_request}")

        # Add user request to conversation
        self.conversation_history.append({"role": "user", "content": user_request})

        while True:
            # If we're in COMPLETE and already gave the model a turn, exit
            if self.current_phase == Phase.COMPLETE and self._complete_turn_given:
                # If model never called task_complete across the whole run, log it
                if not self.task_complete_called:
                    logger.warning("Episode completed without model calling task_complete")
                break

            logger.info(f"Phase: {self.current_phase.value}, "
                       f"Turn: {self.turn_counter.count}/{self.turn_counter.hard_limit}")

            # Check degradation
            if self._check_degradation():
                self._refresh_context()

            # Check if this phase should be skipped (never skip COMPLETE — model always gets a closing turn)
            if self.current_phase != Phase.COMPLETE and self._should_skip_phase(self.current_phase):
                logger.info(f"Skipping phase: {self.current_phase.value}")
                # Transition to next phase
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
                continue

            # Execute the current phase (model gets a turn)
            phase_complete = await self._execute_phase()

            if not phase_complete:
                # Phase didn't complete normally (e.g., model error)
                break

            # If we just completed the COMPLETE turn, mark it so we exit on next iteration
            if self.current_phase == Phase.COMPLETE:
                self._complete_turn_given = True
                continue

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

        # Save session summary to episodic memory store (if available)
        if self.memory_store and self.episode:
            outcome = f"Files changed: {len(self.episode.files_changed)}, Turns: {len(self.episode.turns)}"
            status = "success" if self._get_tests_passed() else "complete"
            self.memory_store.save_session_summary(
                session_id=self.episode.id,
                task=user_request,
                outcome=outcome,
                status=status,
            )

        # Close episode
        summary = self.session_facts.summary()
        self.store.close_episode(self.episode, summary)

        return {
            "phase": "COMPLETE",
            "episode_id": self.episode.id,
            "files_changed": self.episode.files_changed,
            "summary": summary,
            "task_complete_called": self.task_complete_called,
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

        # Plan anchoring: capture plan text when in PLAN phase
        if self.current_phase == Phase.PLAN:
            # Extract the plan from model output
            plan_text = response_text.strip()
            if plan_text:
                self.active_plan = plan_text
                logger.info(f"Plan captured ({len(plan_text)} chars)")

        # Parse and handle tool calls
        tool_call = self._parse_tool_call(response_text)
        if tool_call:
            turn.tool_call = tool_call

            # === Task complete intercept ===
            # If the model calls task_complete, record completion, transition
            # to COMPLETE phase, and give the model a COMPLETE turn to wrap up.
            # If already in COMPLETE phase, the turn is recorded and the loop
            # exits naturally after _complete_turn_given flag is set.
            if tool_call.get("name") == "task_complete":
                turn.user_approval = "approved"
                task_args = tool_call.get("args", tool_call.get("arguments", {}))
                self.task_complete_called = True
                self.task_complete_args = task_args
                self.session_facts.last_tool_used = "task_complete"
                self.session_facts.last_tool_result = f"Task completed: {task_args.get('summary', '')}"
                # Transition to COMPLETE if called early (e.g., during EXECUTE)
                if self.current_phase != Phase.COMPLETE:
                    self.current_phase = Phase.COMPLETE
                self.conversation_history.append({"role": "assistant", "content": response_text})
                self.episode.turns.append(turn)
                self.turn_counter.increment()
                return True
            # === END Task complete intercept ===

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
            # No tool call — just a message (or "none"/"null" was filtered out)
            self.stall_count += 1
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

        Includes: system prompt, standing instructions (if recency_protection),
        session facts, active plan (if any), and conversation history (trimmed).
        """
        # Build the core system message using _get_tool_descriptions
        tool_descs = self._get_tool_descriptions()
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs)

        # Append recent session line if memory store available
        if self.memory_store:
            recent_line = self.memory_store.get_recent_summary_line()
            if recent_line:
                system_prompt += f"\n\n{recent_line}"

        messages = [{"role": "system", "content": system_prompt}]

        # Inject standing instructions as a user message (counters recency bias)
        if self.recency_protection:
            messages.append({
                "role": "user",
                "content": f"[STANDING INSTRUCTIONS]\n{self.standing_instructions.get_text()}",
            })

        # Add session facts as a user context message
        facts_block = self.session_facts.to_context_block()
        messages.append({"role": "user", "content": facts_block})

        # Inject active plan block if one exists
        if self.active_plan:
            messages.append({
                "role": "user",
                "content": f"[ACTIVE PLAN]\n{self.active_plan}",
            })

        # Inject current phase indicator — gives the model concrete guidance per phase
        phase = self.current_phase.value.upper()
        if phase == "EXPLORE":
            guidance = "Read files with read_file or search with grep_search to understand the code."
        elif phase == "PLAN":
            guidance = "Formulate a clear step-by-step plan for what you need to change."
        elif phase == "EXECUTE":
            guidance = (
                "Use **edit_file** to make code changes (add docstrings, fix bugs, refactor). "
                "Do NOT skip this phase — you MUST modify the file(s) before moving on."
            )
        elif phase == "VERIFY":
            guidance = "Run tests with **test_executor** to confirm your changes work."
        elif phase == "ITERATE":
            guidance = "Fix any issues found during verification, then re-run tests."
        elif phase == "COMPLETE":
            guidance = (
                "Call the **task_complete** tool with status, reason_code, and summary "
                "to signal that all work is done. This is your FINAL turn — use it."
            )
        else:
            guidance = ""
        messages.append({
            "role": "user",
            "content": f"[CURRENT PHASE: {phase}]\n"
                       f"You are in the {phase} phase. {guidance}",
        })

        # Add episode history if available
        # Post-MVP: load selective reference episodes here

        return messages

    def _check_degradation(self) -> bool:
        """Check for degradation signals that should trigger a refresh.

        Waterfall — checks the fastest/cheapest signals first:
        1. Stall detection — model emitting "none"/"null"/noop tool names
        2. Format decay — tool call format quality drops below threshold
        3. Turn limit — hard cap on consecutive turns

        Returns:
            True if a context refresh is needed.
        """
        # 1. Stall detection (fastest: just a counter check)
        if self.stall_count >= self.stall_threshold:
            logger.info(
                f"Stall detected ({self.stall_count} consecutive invalid tool names), "
                f"triggering refresh"
            )
            self.stall_count = 0
            self.format_monitor.reset()
            return True

        # 2. Format decay (always monitored for a coding agent)
        if self.format_monitor.should_refresh():
            logger.info(
                f"Format degradation detected (avg={self.format_monitor.average():.2f}), "
                f"triggering refresh"
            )
            self.format_monitor.reset()
            return True

        # 3. Turn limit (failsafe)
        if self.turn_counter.is_limit_reached():
            logger.info(f"Turn limit reached ({self.turn_counter.count}/{self.turn_counter.hard_limit})")
            return True

        return False

    def _refresh_context(self):
        """Re-inject system prompt, standing instructions, session facts, and
        active plan. Trims conversation to last N turns (N from memory_horizon).
        """
        # Build context blocks for re-injection
        facts_block = self.session_facts.to_context_block()

        # Keep recent conversation turns (trimmed to memory_horizon)
        keep = self.memory_horizon if self.memory_horizon > 0 else 4
        trimmed_turns = (
            self.conversation_history[-keep:]
            if len(self.conversation_history) > keep
            else self.conversation_history
        )

        # Build the refreshed context using the refresher
        # Re-build the system prompt with current tool descriptions (may have demoted tools)
        tool_descs = self._get_tool_descriptions()
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs)
        if self.memory_store:
            recent_line = self.memory_store.get_recent_summary_line()
            if recent_line:
                system_prompt += f"\n\n{recent_line}"

        refreshed = self.refresher.build_refreshed_context(
            session_facts_block=facts_block,
            conversation_turns=trimmed_turns,
        )

        # Build additional injection blocks that the refresher doesn't handle
        extra_blocks: list[dict] = []

        # 1. Standing instructions block
        if self.recency_protection:
            extra_blocks.append({
                "role": "user",
                "content": f"[STANDING INSTRUCTIONS]\n{self.standing_instructions.get_text()}",
            })

        # 2. Active plan block (if one exists)
        if self.active_plan:
            extra_blocks.append({
                "role": "user",
                "content": f"[ACTIVE PLAN]\n{self.active_plan}",
            })

        # 3. Last tool result as a tool-role message (if available)
        last_call = self.session_facts.last_tool_used
        last_result = self.session_facts.last_tool_result
        if last_call and last_result:
            extra_blocks.append({
                "role": "tool",
                "content": last_result[:500],
                "name": last_call,
            })

        # Build: [system, user(standing+facts), extra_blocks..., conversation_turns]
        # Start with the system prompt (update it with current tool descriptions)
        combined = [
            {"role": "system", "content": system_prompt},
        ]
        if len(refreshed) > 1:
            combined.append(refreshed[1])  # user standing+fats
        combined.extend(extra_blocks)
        if trimmed_turns:
            combined.extend(trimmed_turns)

        # Replace conversation history with refreshed context
        self.conversation_history = combined
        self.turn_counter.reset()
        # NOTE: active_plan is intentionally preserved across refresh.
        # A context refresh fixes context quality (degradation), it should not
        # erase the plan the model was working on. The plan is re-injected
        # via extra_blocks above.

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
            "old_value": "search_text",
            "new_value": "replace_text",
            "old_string": "search_text",
            "new_string": "replace_text",
            "old_content": "search_text",
            "new_content": "replace_text",
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

        # Update delegation telemetry
        self._update_tool_stats(name=name, success=result.success, duration_ms=result.duration_ms)

        # Update session facts
        self.session_facts.last_tool_used = name
        if result.success:
            self.session_facts.last_tool_result = result.output[:500]
        else:
            self.session_facts.last_tool_result = f"Error: {result.error}"

        # Record tool call in format monitor
        self.format_monitor.record_tool_call({"name": name, "args": args}, "")

        # Reset stall counter — any valid tool call breaks the stall cycle
        self.stall_count = 0

        # Track files changed for write_file/edit_file tools
        if result.success and name in ("write_file", "edit_file"):
            file_path = args.get("filepath", args.get("path", ""))
            if file_path:
                if file_path not in self.episode.files_changed:
                    self.episode.files_changed.append(file_path)
                if file_path not in self.session_facts.files_created:
                    self.session_facts.files_created.append(file_path)

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
                                if tool_name and tool_name.lower() not in ("none", "null", "noop", ""):
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

        Checks both `test_executor` (formal test runner) and `run_shell`
        (code execution as verification) tool calls. A successful `run_shell`
        with exit code 0 counts as tests passed for coding tasks.

        Returns True if tests passed, False if failed, None if no tests run.
        """
        for turn in reversed(self.episode.turns):
            if turn.tool_call and turn.tool_result:
                name = turn.tool_call.get("name", "")
                if name == "test_executor":
                    return turn.tool_result.get("success", False)
                if name == "run_shell":
                    # run_shell success means the code executed without errors
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
            "tool_stats": self.tool_stats,
        }