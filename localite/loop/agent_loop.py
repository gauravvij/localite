"""AgentLoop — the core 5-phase agent loop."""

import json
import logging
import re
from typing import Any, Optional

from localite.code_index import CodeIndex
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

# Maximum characters for tool output in conversation history
# Prevents context flooding from large tool results (e.g. ls -R of venv)
MAX_TOOL_OUTPUT_CHARS = 64000

# Harness tool names and path tokens — never use these as ctags identifiers
# (they are our own tool names, not symbols from the task's codebase)
HARNESS_TOOL_NAMES = frozenset({
    'task_complete', 'read_file', 'edit_file', 'list_files', 'grep_search',
    'write_file', 'run_shell', 'test_executor', 'bash', '/usr/bin/python3',
    'memory_read', 'memory_write', 'diff_view',
    # harness path tokens
    'swe_bench', 'repos', 'workdir', 'localite', 'agent_loop',
    # common Python builtins that appear in objectives
    'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'none', 'true', 'false',
})

# Output format templates — profile-driven selection
# NOTE: The {{ }} are Jinja2-style escaping for .format() — they produce literal { } in the final output.
OUTPUT_FORMAT_STANDARD = """  {{"tool": "read_file", "arguments": {{"path": "/home/user/project/src/main.py", "max_lines": 100}}}}
  {{"tool": "edit_file", "arguments": {{"path": "/home/user/project/train.py", "search_text": "lr=0.01", "replace_text": "lr=0.001"}}}}
  {{"tool": "write_file", "arguments": {{"path": "/home/user/project/src/utils.py", "content": "import os\\nprint('hello')\\n"}}}}
  {{"tool": "run_shell", "arguments": {{"command": "pip install torch", "timeout": 120}}}}
  {{"tool": "list_files", "arguments": {{"path": "/home/user/project/src", "depth": 2}}}}
  {{"tool": "grep_search", "arguments": {{"pattern": "def train_", "path": ".", "glob_pattern": "*.py"}}}}
  {{"tool": "test_executor", "arguments": {{"path": "tests/", "timeout": 60}}}}
  {{"tool": "task_complete", "arguments": {{"status": "success", "reason_code": "tests_passing", "summary": "Fixed bug in train.py"}}}}"""

OUTPUT_FORMAT_GEMMA_NATIVE = """  - read_file: {{"tool_name": "read_file", "params": {{"path": "/home/user/project/src/main.py", "max_lines": 100}}}}
  - edit_file: {{"tool_name": "edit_file", "params": {{"path": "/home/user/project/train.py", "search_text": "lr=0.01", "replace_text": "lr=0.001"}}}}
  - write_file: {{"tool_name": "write_file", "params": {{"path": "/home/user/project/src/utils.py", "content": "import os\\nprint('hello')\\n"}}}}
  - run_shell: {{"tool_name": "run_shell", "params": {{"command": "pip install torch", "timeout": 120}}}}
  - list_files: {{"tool_name": "list_files", "params": {{"path": "/home/user/project/src", "depth": 2}}}}
  - grep_search: {{"tool_name": "grep_search", "params": {{"pattern": "def train_", "path": ".", "glob_pattern": "*.py"}}}}
  - test_executor: {{"tool_name": "test_executor", "params": {{"path": "tests/", "timeout": 60}}}}
  - task_complete: {{"tool_name": "task_complete", "params": {{"status": "success", "reason_code": "tests_passing", "summary": "Fixed bug in train.py"}}}}"""

# Default system prompt template
# NOTE: The {{ }} are Jinja2-style escaping for .format() — they produce literal { } in the final output.
# The {tool_descriptions} placeholder is replaced at runtime with tool descriptions.
# The {output_format} placeholder is replaced with the profile-appropriate format template.
SYSTEM_PROMPT = """You are localite, a local AI coding agent. Fix bugs and implement changes using the tools below.

## Available Tools

{tool_descriptions}

## Output Format

CRITICAL: Every tool call MUST use this exact JSON format with "tool" and "arguments" fields:

### Tool call format:
{{"tool": "tool_name", "arguments": {{...parameters...}}}}

### Tool call examples:
{output_format}

### Message only (no tool call):
{{"thought": "reasoning", "message": "message to user"}}

## Workflow: 2 States

**INVESTIGATE** (EXPLORE/PLAN phases): Understand the codebase first.
- Emit a JSON tool call with "tool" and "arguments" fields: {{"tool": "list_files", "arguments": {{"path": "."}}}}
- list_files → read_file → grep_search to locate the relevant code.
- Max 6 investigate turns, then MUST transition to EXECUTE.

**EXECUTE** (EXECUTE/VERIFY phases): Make the change, then verify.
- Emit a JSON tool call with "tool" and "arguments" fields: {{"tool": "edit_file", "arguments": {{...}}}}
- Call edit_file or write_file to apply the fix.
- Call test_executor to verify. Call task_complete when done.
- DO NOT send message responses in EXECUTE — only tool calls.

## Rules (all mandatory)

- ALWAYS use {{"tool": "name", "arguments": {{...}}}} format — never omit the "tool" field.
- Use exact parameter names from tool descriptions. list_files uses "path" not "directory"; edit_file uses "search_text"/"replace_text" not "old_value"/"new_value".
- Read a file before editing it (unless you already have its content).
- After identifying the file and bug: STOP exploring, START editing immediately.
- Do NOT re-read files already read. Do NOT list the same directory twice.
- edit_file / write_file are the ONLY actions that make progress. Messages do not.
- Run test_executor after every change. Call task_complete when tests pass.
- Do NOT run destructive shell commands (rm -rf, kill, docker stop).

When all changes are made and tests pass, call **task_complete**.
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
        code_index: Optional['CodeIndex'] = None,
        needle_client: Optional[Any] = None,
    ):
        self.model = model_client
        self.tools = tools
        self.gate = permission_gate
        self.store = episode_store
        self.code_index = code_index
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

        # Guidance deduplication (Stage 1)
        self._guidance_seen: set = set()
        self._guidance_count: int = 0

        # Edit/investigate tracking (Stage 2)
        self._edit_calls: int = 0
        self._investigate_calls: int = 0
        self._forced_edit_injected: bool = False
        self._last_edit_turn: int = -1
        self._post_edit_verify_injected: bool = False

        # Phase turn counter — reset on every phase change (P2)
        self._phase_turn_count: int = 0

        # Post-read-file nudge tracking — prevent repeat nudges per file (P5)
        self._nudged_files: set = set()

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
        output_fmt = OUTPUT_FORMAT_GEMMA_NATIVE if self.profile and self.profile.tool_call_format == "gemma_native" else OUTPUT_FORMAT_STANDARD
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs, output_format=output_fmt)
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
            # Skip only if user explicitly starts with "plan:", "plan -", or "plan-"
            # NOTE: SWE-bench task prompt injects test_patch + F2P JSON which commonly
            # contains "1." — substring matching falsely skipped PLAN, causing the agent
            # to burn turns in EXECUTE without a plan.
            _obj = self.session_facts.current_objective
            if isinstance(_obj, dict):
                _obj = _obj.get("content", str(_obj))
            request_lower = str(_obj).lower().strip()
            if (request_lower.startswith("plan:")
                    or request_lower.startswith("plan -")
                    or request_lower.startswith("plan-")):
                logger.info(f"Skipping PLAN phase — request starts with 'plan:' / 'plan -' / 'plan-'")
                if not self.active_plan:
                    self.active_plan = self.session_facts.current_objective[:500]
                return True
            return False

        elif phase == Phase.VERIFY:
            # Skip if no files were changed — nothing to verify.
            # P2 fix: when VERIFY is skipped due to no files changed, do NOT advance
            # to VERIFY at all — the caller must keep current_phase = EXECUTE.
            files_changed_count = len(self.episode.files_changed) if self.episode else 0
            if files_changed_count == 0:
                logger.debug("Skipping VERIFY phase — no files changed")
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
            # P2 fix: VERIFY/ITERATE skip when no files changed → stay in EXECUTE, do NOT ping-pong
            if self.current_phase != Phase.COMPLETE and self._should_skip_phase(self.current_phase):
                logger.info(f"Skipping phase: {self.current_phase.value}")
                tests_passed = self._get_tests_passed()
                reached = self.iteration_count >= self.max_iterations
                if self.current_phase == Phase.VERIFY:
                    # P2: if VERIFY is skipped because no files changed, stay in EXECUTE
                    files_changed_count = len(self.episode.files_changed) if self.episode else 0
                    if files_changed_count == 0:
                        # Stay in EXECUTE — do not advance to VERIFY/ITERATE
                        logger.info("P2: VERIFY skipped (no files changed) — staying in EXECUTE")
                        self.current_phase = Phase.EXECUTE
                        continue
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
            prev_phase = self.current_phase
            phase_complete = await self._execute_phase()

            if not phase_complete:
                # Phase didn't complete normally (e.g., model error)
                break

            # If we just completed the COMPLETE turn, mark it so we exit on next iteration
            if self.current_phase == Phase.COMPLETE:
                self._complete_turn_given = True
                continue

            # P2: increment per-phase turn counter; reset when phase changes
            self._phase_turn_count += 1

            # P2: in EXECUTE phase, only advance to VERIFY after a file-changing edit
            # OR after the per-phase turn limit is reached (stall guard).
            # This prevents EXECUTE→VERIFY(skip)→ITERATE(skip)→EXECUTE ping-pong.
            if self.current_phase == Phase.EXECUTE:
                files_changed_count = len(self.episode.files_changed) if self.episode else 0
                phase_turn_limit = getattr(self.profile, 'max_turns', 60) if self.profile else 60
                stuck_in_execute = self._phase_turn_count >= phase_turn_limit
                if files_changed_count == 0 and not stuck_in_execute:
                    # Stay in EXECUTE — no file was changed yet
                    logger.debug(
                        "P2: staying in EXECUTE (files_changed=0, phase_turn=%d/%d)",
                        self._phase_turn_count, phase_turn_limit,
                    )
                    continue
                # Files were changed OR we've hit the phase turn limit — advance normally
                if stuck_in_execute and files_changed_count == 0:
                    logger.warning(
                        "P2: EXECUTE phase turn limit reached (%d turns) with no edits — forcing advance",
                        self._phase_turn_count,
                    )

            # Determine next phase
            tests_passed = self._get_tests_passed()
            reached = self.iteration_count >= self.max_iterations

            # Reset per-phase turn counter on phase change
            if self.current_phase != prev_phase or True:  # always reset on advance
                self._phase_turn_count = 0

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
                self._phase_turn_count = 0  # reset on phase change

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
        # Deduplication: if conversation_history has grown large, compress the
        # original full task message to avoid duplicating with the compressed
        # task that _build_context() now injects in the phase guidance.
        if (
            len(self.conversation_history) >= 3
            and self.conversation_history
            and self.conversation_history[0].get("role") == "user"
            and "content" in self.conversation_history[0]
        ):
            total_ch = sum(len(m.get("content", "")) for m in self.conversation_history[1:])
            if total_ch > 15000:
                original = self.conversation_history[0]["content"]
                compressed = self._compress_objective(original, max_chars=300)
                if len(compressed) < len(original):
                    self.conversation_history[0] = {
                        "role": "user",
                        "content": f"[Task compressed] {compressed}",
                    }
                    logger.debug(
                        "Compressed conversation_history[0]: %d chars -> %d chars",
                        len(original), len(compressed),
                    )

        # Build context
        context = self._build_context()
        context.extend(self.conversation_history)

        # Get model response
        try:
            # Pass options (e.g. num_predict) from profile
            chat_options = {}
            if self.profile and self.profile.num_predict is not None:
                chat_options["num_predict"] = self.profile.num_predict
            options_payload = {"options": chat_options} if chat_options else {}

            # DEBUG: log context structure before chat call
            roles = [m.get("role", "?") for m in context]
            total_chars = sum(len(m.get("content", "")) for m in context)
            logger.debug(
                "EXECUTE PHASE — context messages=%d, roles=%s, "
                "total_content_chars=%d, chat_options=%s, options_payload=%s",
                len(context),
                roles,
                total_chars,
                chat_options,
                options_payload,
            )

            # Context window budget enforcement
            # If profile has max_context_chars, trim older messages to stay within budget
            if self.profile and self.profile.max_context_chars:
                total_chars = sum(len(str(m.get("content", ""))) for m in context)
                if total_chars > self.profile.max_context_chars:
                    logger.warning(
                        "Context exceeds budget (%d > %d chars), trimming older messages...",
                        total_chars, self.profile.max_context_chars,
                    )
                    # Stage 3: smart eviction — pin indices 0-1, evict oldest tool msgs first
                    # Pass 1: evict oldest tool-result messages (role == "tool")
                    i = 2
                    while total_chars > self.profile.max_context_chars and i < len(context) - 2:
                        if context[i].get("role") == "tool":
                            removed = context.pop(i)
                            removed_chars = len(str(removed.get("content", "")))
                            total_chars -= removed_chars
                            logger.debug("Evicted tool msg at idx %d (%d chars, remaining: %d)", i, removed_chars, total_chars)
                        else:
                            i += 1
                    # Pass 2: evict oldest assistant messages if still over budget
                    i = 2
                    while total_chars > self.profile.max_context_chars and i < len(context) - 2:
                        if context[i].get("role") == "assistant":
                            removed = context.pop(i)
                            removed_chars = len(str(removed.get("content", "")))
                            total_chars -= removed_chars
                            logger.debug("Evicted assistant msg at idx %d (%d chars, remaining: %d)", i, removed_chars, total_chars)
                        else:
                            i += 1
                    # Pass 3: fallback — evict any non-pinned message
                    while total_chars > self.profile.max_context_chars and len(context) > 4:
                        removed = context.pop(2)
                        removed_chars = len(str(removed.get("content", "")))
                        total_chars -= removed_chars
                        logger.debug("Evicted fallback msg (%d chars, remaining: %d)", removed_chars, total_chars)

            response_text = await self.model.chat(context, stream=False, **options_payload)

            # Guard against None response from model
            if response_text is None:
                response_text = ""
                logger.warning("Model returned None response — treating as empty string")

            # DEBUG: log raw response after chat
            logger.debug(
                "EXECUTE PHASE — response_text length=%d, first_500_chars=%s",
                len(response_text),
                response_text[:500],
            )
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

        # DEBUG: log parse_tool_call result
        if tool_call:
            logger.debug(
                "EXECUTE PHASE — parsed tool_call: name=%s, args_keys=%s, full_arguments=%s",
                tool_call.get("name", "?"),
                list(tool_call.get("args", tool_call.get("arguments", {})).keys()),
                str(tool_call.get("args", tool_call.get("arguments", {})))[:300],
            )
        else:
            logger.debug(
                "EXECUTE PHASE — No tool call parsed (stall_count=%d, stall_threshold=%d)",
                self.stall_count,
                self.profile.stall_threshold if self.profile else 3,
            )

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
                if not result.success:
                    err_msg = result.error or result.output or "Unknown error"
                    tool_content = f"ERROR: Tool call failed — {err_msg}\nYour action was NOT applied. You must retry with corrected parameters."
                else:
                    tool_content = result.output or ""
                # Stage 3: cap read_file outputs at 6000 chars
                tool_name_called = modified_call.get("name", "tool")
                if tool_name_called == "read_file" and len(tool_content) > 6000:
                    tool_content = tool_content[:6000] + "\n\n[File truncated — use grep_search to find specific sections]"
                elif len(tool_content) > MAX_TOOL_OUTPUT_CHARS:
                    tool_content = tool_content[:MAX_TOOL_OUTPUT_CHARS] + f"\n\n[Output truncated at {MAX_TOOL_OUTPUT_CHARS} characters]"
                # Stage 2: track edit/investigate calls
                if tool_name_called in ("edit_file", "write_file") and result.success:
                    self._edit_calls += 1
                    # Track when last edit happened for post-edit verification
                    self._last_edit_turn = self.turn_counter.count + 1
                elif tool_name_called in ("read_file", "grep_search"):
                    self._investigate_calls += 1
                self.conversation_history.append({
                    "role": "tool",
                    "content": tool_content,
                    "name": tool_name_called,
                })
                # P5: post-read-file nudge — after successful read_file in EXECUTE with no edits yet
                if (tool_name_called == "read_file"
                        and result.success
                        and self.current_phase == Phase.EXECUTE
                        and self._edit_calls == 0):
                    read_path = (
                        modified_call.get("args", {}).get("path")
                        or modified_call.get("arguments", {}).get("path")
                        or ""
                    )
                    if read_path and read_path not in self._nudged_files:
                        nudge_msg = (
                            f"[NUDGE] You now have the content of {read_path}. "
                            f"Call edit_file now to apply your fix — do not read more files."
                        )
                        self.conversation_history.append({"role": "user", "content": nudge_msg})
                        self._nudged_files.add(read_path)
                        logger.info("[NUDGE] Post-read-file nudge injected for %s", read_path)
                # Post-edit verification injection: after successful edit, remind agent to test
                if tool_name_called in ("edit_file", "write_file") and result.success:
                    if not self._post_edit_verify_injected:
                        verify_msg = (
                            "[POST-EDIT VERIFY] You just made a code change. "
                            "Now run test_executor to verify your fix works. "
                            "If tests fail, adjust your fix and re-test. "
                            "Only call task_complete after tests pass."
                        )
                        self.conversation_history.append({"role": "user", "content": verify_msg})
                        self._post_edit_verify_injected = True
                        logger.info("[POST-EDIT VERIFY] Test verification prompt injected after edit")
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
                if not result.success:
                    err_msg = result.error or result.output or "Unknown error"
                    tool_content = f"ERROR: Tool call failed — {err_msg}\nYour action was NOT applied. You must retry with corrected parameters."
                else:
                    tool_content = result.output or ""
                if len(tool_content) > MAX_TOOL_OUTPUT_CHARS:
                    tool_content = tool_content[:MAX_TOOL_OUTPUT_CHARS] + f"\n\n[Output truncated at {MAX_TOOL_OUTPUT_CHARS} characters]"
                self.conversation_history.append({
                    "role": "tool",
                    "content": tool_content,
                    "name": modified_call.get("name", "tool"),
                })
        else:
            # No tool call — just a message (or "none"/"null" was filtered out)
            self.stall_count += 1
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text,
            })
            # Stage 2a: inject [REQUIRED] re-prompt on first EXECUTE-phase stall
            if (self.current_phase == Phase.EXECUTE
                    and self.stall_count == 1
                    and not self._forced_edit_injected):
                reprompt = (
                    "[REQUIRED] You are in EXECUTE phase. You MUST call a tool now — "
                    "either edit_file or write_file to make the code change, or "
                    "task_complete if done. Do NOT send a message. Call a tool."
                )
                self.conversation_history.append({"role": "user", "content": reprompt})
                logger.info("[REQUIRED] EXECUTE-phase stall re-prompt injected (stall_count=1)")

            # Stage 2b: stronger reprompt on second EXECUTE-phase stall with concrete JSON template
            if (self.current_phase == Phase.EXECUTE
                    and self.stall_count == 2):
                reprompt = (
                    '[REQUIRED] You MUST output a tool call. Reply with EXACTLY this format:\n\n'
                    '{{"tool": "edit_file", "arguments": {{"path": "...", "search_text": "...", "replace_text": "..."}}}}\n\n'
                    'No messages, no explanations, no thoughts. Just a JSON tool call.'
                )
                self.conversation_history.append({"role": "user", "content": reprompt})
                logger.info("[REQUIRED] EXECUTE-phase stall re-prompt injected with concrete example (stall_count=2)")

        # Progressive guidance: after list_files returns directory structure, help the model navigate
        # Uses multi-strategy approach based on what the model is seeing:
        #   Root level → suggest exploring source directories (not reading root-level noise files)
        #   Inside src/ → suggest reading the most relevant .py file
        #   Task mentions identifiers → suggest grep_search
        if tool_call and tool_call.get("name") == "list_files":
            last_msg = self.conversation_history[-1] if self.conversation_history else None
            if last_msg and last_msg.get("role") == "tool":
                last_output = last_msg.get("content", "")
                if ".py" not in last_output and "(dir)" not in last_output:
                    pass  # No actionable content to guide on
                else:
                    # Parse directory tree output to reconstruct structure
                    # Format from ListFilesTool._list_dir:
                    #   /dirname/ (dir)       (depth 0, 0 indent)
                    #     subdir/ (dir)       (depth 1, 2 indent)
                    #       file.py (file)      (depth 2, 4 indent)
                    lines = last_output.split("\n")
                    dir_stack = []
                    py_files_with_paths = []  # (full_relative_path, filename)
                    root_dirs = set()          # Directory names at depth 0

                    # Directories that are NOT source code — skip suggestions into these
                    NOISE_DIRS = frozenset({
                        'examples', 'docs', 'doc', 'benchmarks', 'scripts',
                        'extras', 'contrib', 'tutorials', 'archive',
                        'docker', 'dockerfiles', 'tools',
                        'build_tools', 'build', 'source',
                        '.circleci', '.github', '.gitlab',
                        'whats_new', 'news', 'changes',
                        'data',  # test data directory, not source code
                    })
                    # Root-level files that are NOT relevant source code
                    NOISE_FILES = frozenset({
                        'versioneer.py', 'setup.cfg', 'tox.ini', 'Makefile',
                        'appveyor.yml', 'version.py', '_version.py', '.gitignore',
                        'requirements.txt', 'setup.py', 'conftest.py', 'conf.py',
                        'test_helpers.py', 'constraints.py', 'meta.py',
                    })

                    for line in lines:
                        stripped = line.lstrip(" ")
                        indent_chars = len(line) - len(stripped)
                        depth = indent_chars // 2 if indent_chars > 0 else 0
                        content = stripped.strip()

                        if not content:
                            continue

                        if content.endswith("(dir)"):
                            dirname = content.rsplit(" (dir)", 1)[0].strip("/")
                            while dir_stack and dir_stack[-1][1] >= depth:
                                dir_stack.pop()
                            dir_stack.append((dirname, depth))
                            if depth == 0 and not dirname.startswith('.'):
                                root_dirs.add(dirname)
                        elif content.endswith("/"):
                            dirname = content.rstrip("/")
                            while dir_stack and dir_stack[-1][1] >= depth:
                                dir_stack.pop()
                            dir_stack.append((dirname, depth))
                            if depth == 0 and not dirname.startswith('.'):
                                root_dirs.add(dirname)
                        elif content.endswith("(file)"):
                            filename = content.rsplit(" (file)", 1)[0].strip()
                            if filename.endswith(".py"):
                                while dir_stack and dir_stack[-1][1] >= depth:
                                    dir_stack.pop()
                                dir_prefix = "/".join(d[0] for d in dir_stack)
                                full_path = f"{dir_prefix}/{filename}" if dir_prefix else filename
                                py_files_with_paths.append((full_path, filename))

                    # === Strategy Selection ===
                    guidance_msg = None
                    source_dirs = sorted(d for d in root_dirs
                                         if d not in NOISE_DIRS and not d.startswith('.'))
                    # Check the path the model listed — if it listed a subdir, we're not at root
                    list_files_args = (tool_call.get("arguments") or tool_call.get("args") or {}) if isinstance(tool_call, dict) else {}
                    # Model often uses "directory" instead of "path" or other invented keys — check
                    listed_path = list_files_args.get("path") or list_files_args.get("directory") or "."
                    if listed_path in (".", "", "/"):
                        # Generic fallback: check all arg values for anything that looks like a path
                        for val in list_files_args.values():
                            if isinstance(val, str) and val not in (".", "", "/") and len(val) > 1:
                                listed_path = val
                                break
                    is_at_root = listed_path in (".", "", "/")
                    # If depth > 1, the tree shows nested structure — only one result per level
                    listed_depth = list_files_args.get("depth", 1)
                    if isinstance(listed_depth, str):
                        try:
                            listed_depth = int(listed_depth)
                        except (ValueError, TypeError):
                            pass

                    # Strategy A: At repo root with source directories — suggest exploring (not reading root-level files)
                    if is_at_root and source_dirs:
                        if 'src' in source_dirs:
                            guidance_msg = (
                                "[GUIDANCE] The project source code is in 'src/'. "
                                "Use list_files('src') to explore it."
                            )
                        elif 'lib' in source_dirs:
                            guidance_msg = (
                                "[GUIDANCE] The project source code is in 'lib/'. "
                                "Use list_files('lib') to explore it."
                            )
                        else:
                            guidance_msg = (
                                f"[GUIDANCE] Explore '{source_dirs[0]}' "
                                f"with list_files('{source_dirs[0]}') to find source code."
                            )

                    # Strategy B: Inside a directory with .py files — suggest reading relevant ones
                    if guidance_msg is None and py_files_with_paths:
                        # Filter noise files and files inside noise directories
                        filtered = [
                            (p, f) for p, f in py_files_with_paths
                            if f not in NOISE_FILES
                            and not f.startswith('_')
                            and not any(nd in p.split('/') for nd in NOISE_DIRS)
                        ]
                        # Also filter tests/ unless we're already inside tests/
                        is_in_tests = any('tests' in d[0] for d in dir_stack)
                        if not is_in_tests:
                            filtered = [(p, f) for p, f in filtered
                                        if not p.startswith('tests/') and not p.startswith('test/')]

                        if filtered:
                            # --- Ctags-first guidance (Strategy B1) ---
                            # Extract meaningful identifiers from the NON-lowercased objective
                            # (preserving case for ctags lookup), then query CodeIndex.
                            # If ctags resolves an identifier to a file, use precise guidance.
                            # Otherwise fall through to keyword scoring (Strategy B2).
                            objective = getattr(self.session_facts, 'current_objective', '')
                            ctags_guidance = None
                            if self.code_index is not None and not self.code_index._disabled:
                                # Extract PascalCase, camelCase, and snake_case tokens ≥3 chars
                                identifiers = set()
                                for m in re.finditer(r'[A-Z][a-z0-9]+[A-Za-z0-9]*|[a-z]+_[a-z]+[a-z0-9_]*|[a-z][a-z0-9]{2,}', objective):
                                    identifiers.add(m.group())
                                # Filter out Python keywords and common noise words
                                filtered_identifiers = frozenset({
                                    'the', 'from', 'import', 'def', 'class', 'return', 'if', 'for',
                                    'with', 'as', 'self', 'True', 'False', 'None', 'print', 'len',
                                    'range', 'type', 'isinstance', 'hasattr', 'setattr', 'getattr',
                                    'open', 'read', 'write', 'append', 'items', 'keys', 'values',
                                    'upper', 'lower', 'replace', 'split', 'join', 'strip', 'format',
                                    'encode', 'decode', 'find', 'index', 'count', 'pop', 'remove',
                                    'insert', 'extend', 'sort', 'reverse', 'copy', 'clear', 'update',
                                    'add', 'discard', 'difference', 'intersection', 'union',
                                    'symmetric_difference',
                                })
                                for ident in sorted(identifiers, key=len, reverse=True):
                                    if ident.lower() in filtered_identifiers:
                                        continue
                                    # Stage 1: skip harness tool names and path tokens
                                    if ident.lower() in HARNESS_TOOL_NAMES:
                                        continue
                                    matches = self.code_index.lookup(ident)
                                    if matches:
                                        # Pick first Python file from matches
                                        best_file = None
                                        for fp, _ln, _kind in matches:
                                            if fp.endswith('.py'):
                                                best_file = fp
                                                break
                                        if best_file is None:
                                            best_file = matches[0][0]
                                        # Stage 1: confidence gate — skip if only test/ files
                                        all_files = [fp for fp, _ln, _kind in matches if fp.endswith('.py')]
                                        non_test = [fp for fp in all_files
                                                    if '/test/' not in fp and '/tests/' not in fp
                                                    and not fp.startswith('test/') and not fp.startswith('tests/')]
                                        if all_files and not non_test:
                                            # Only test files — suppress this guidance
                                            continue
                                        ctags_guidance = (
                                            f"[GUIDANCE] '{ident}' is defined in {best_file}. "
                                            f"Read it with read_file."
                                        )
                                        break  # Use first matching identifier
                            if ctags_guidance:
                                guidance_msg = ctags_guidance
                            else:
                                # --- Fallback: keyword scoring (Strategy B2, unchanged logic) ---
                                obj_lower = objective.lower()
                                obj_terms = set(re.findall(r'[a-zA-Z][a-zA-Z0-9_]{2,}', obj_lower))
                                common_words = frozenset({
                                    'write', 'file', 'type', 'data', 'code', 'test', 'error', 'this', 'that',
                                    'from', 'with', 'have', 'the', 'make', 'done', 'need', 'use', 'get', 'set',
                                    'run', 'list', 'read', 'show', 'find', 'check', 'work', 'call', 'name',
                                    'path', 'line', 'text', 'info', 'node', 'base', 'size', 'page', 'more',
                                    'some', 'each', 'also', 'just', 'like', 'way', 'part', 'used', 'will',
                                    'can', 'should', 'would', 'could', 'does', 'when', 'than', 'then', 'now',
                                })
                                key_terms = set()
                                generic_terms = set()
                                for t in obj_terms:
                                    if t in common_words:
                                        generic_terms.add(t)
                                    elif any(c.isupper() for c in t):
                                        key_terms.add(t)
                                    elif any(c.isdigit() for c in t):
                                        key_terms.add(t)
                                    elif '_' in t:
                                        key_terms.add(t)
                                    elif len(t) > 6:
                                        key_terms.add(t)
                                    else:
                                        generic_terms.add(t)
                                target_path = filtered[0][0]
                                best_score = 0
                                for fp, fn in filtered:
                                    fn_stem = fn.lower().replace('.py', '')
                                    fp_lower = fp.lower()
                                    score = 0
                                    for term in key_terms:
                                        if term in fn_stem or fn_stem in term:
                                            score += 10
                                        if term in fp_lower:
                                            score += 3
                                    for term in generic_terms:
                                        if term in fn_stem or fn_stem in term:
                                            score += 2
                                        if term in fp_lower:
                                            score += 1
                                    if score > best_score:
                                        best_score = score
                                        target_path = fp
                                if best_score >= 12:
                                    guidance_msg = (
                                        f"[GUIDANCE] Read {target_path} "
                                        f"with read_file to understand the code."
                                    )
                        # Strategy C: Task-aware grep guidance — extract identifiers from objective
                    # Runs BEFORE generic subdirectory fallback because grep is more targeted
                    if guidance_msg is None:
                        objective = getattr(self.session_facts, 'current_objective', '')
                        test_names = getattr(self.session_facts, 'test_id_hints', [])
                        search_terms = []
                        if objective:
                            terms = re.findall(
                                r'[A-Z][a-z]+[A-Za-z]*|\b[a-z]+_[a-z]+\b',
                                objective
                            )
                            common_words = frozenset({
                                'this', 'that', 'with', 'from', 'have', 'the',
                                'file', 'test', 'code', 'data', 'type', 'error',
                                'should', 'would', 'could', 'does', 'make',
                            })
                            meaningful = [
                                t for t in terms if len(t) > 3
                                and t.lower() not in common_words
                            ]
                            if meaningful:
                                search_terms = meaningful
                                guidance_msg = (
                                    f"[GUIDANCE] The task mentions '{meaningful[0]}'. "
                                    f"Use grep_search('{meaningful[0]}') to locate relevant code."
                                )
                        # If objective terms look like camelCase identifiers, also check test_id_hints
                        if guidance_msg is None and test_names:
                            guidance_msg = (
                                f"[GUIDANCE] The tests mention '{test_names[0]}'. "
                                f"Use grep_search('{test_names[0]}') to locate relevant code."
                            )

                    # Strategy D: Inside a directory with .py files but no strong match — suggest subdirectory exploration
                    if guidance_msg is None and py_files_with_paths:
                        guidance_msg = (
                            "[GUIDANCE] Use list_files to explore subdirectories "
                            "for source code."
                        )

                    # Final fallback
                    if guidance_msg is None:
                        guidance_msg = (
                            "[GUIDANCE] Use read_file to explore source files, "
                            "or grep_search to find relevant code."
                        )

                    # Stage 1: dedup + cap guidance injections
                    if guidance_msg in self._guidance_seen or self._guidance_count >= 2:
                        logger.debug(f"Guidance suppressed (seen={guidance_msg in self._guidance_seen}, count={self._guidance_count}): {guidance_msg[:80]}")
                    else:
                        self._guidance_seen.add(guidance_msg)
                        self._guidance_count += 1
                        self.conversation_history.append({
                            "role": "user",
                            "content": guidance_msg,
                        })
                        logger.debug(f"Progressive guidance injected ({self._guidance_count}/2): {guidance_msg[:120]}")

        # Stage 2: forced-edit trigger — if explore-heavy with no edits, inject once
        if (tool_call is not None
                and self.current_phase == Phase.EXECUTE
                and self._edit_calls == 0
                and self._investigate_calls >= 3
                and not self._forced_edit_injected):
            forced_msg = (
                "[REQUIRED] You have explored enough. Now call edit_file or write_file "
                "to apply your fix. Do not explore further."
            )
            self.conversation_history.append({"role": "user", "content": forced_msg})
            self._forced_edit_injected = True
            logger.info(f"[REQUIRED] Forced-edit injected (investigate_calls={self._investigate_calls}, edit_calls={self._edit_calls})")

        # Record turn in episode and increment counter
        self.episode.turns.append(turn)
        self.turn_counter.increment()

        return True

    def _compress_objective(self, objective: str, max_chars: int = 300) -> str:
        """Compress a task objective to max_chars, preserving whole sentences.

        Returns a compact string suitable for injection near the generation point.
        """
        if len(objective) <= max_chars:
            return objective
        # Truncate at sentence boundary within max_chars
        truncated = objective[:max_chars]
        # Find last sentence-ending punctuation
        for sep in (". ", "!\n", "?\n", ".\n", "!", "?"):
            last = truncated.rfind(sep)
            if last > max_chars * 0.6:  # only if we captured meaningful content
                return truncated[:last + 1]
        # Fall back to ellipsis
        return truncated.rstrip() + "..."

    def _build_context(self) -> list[dict]:
        """Construct the full context for the model.

        Includes: system prompt, standing instructions (if recency_protection),
        session facts, active plan (if any), compressed task objective,
        phase-specific guidance, and conversation history (trimmed).
        """
        # Build the core system message using _get_tool_descriptions
        tool_descs = self._get_tool_descriptions()
        output_fmt = OUTPUT_FORMAT_GEMMA_NATIVE if self.profile and self.profile.tool_call_format == "gemma_native" else OUTPUT_FORMAT_STANDARD
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs, output_format=output_fmt)

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
            guidance = (
                "Emit a JSON tool call with \"tool\" and \"arguments\" fields. "
                "First use list_files to find the relevant files, then use read_file to read their CONTENTS. "
                "Example: {\"tool\": \"list_files\", \"arguments\": {\"path\": \".\", \"depth\": 2}}. "
                "Do NOT leave this phase until you have read the source files that need to be changed."
            )
        elif phase == "PLAN":
            guidance = (
                "Emit a JSON tool call with \"tool\" and \"arguments\" fields to explore, "
                "or respond with a message. Formulate a clear step-by-step plan for what you need to change."
            )
        elif phase == "EXECUTE":
            guidance = (
                "You are in the EXECUTE phase — emit a JSON tool call NOW using this EXACT format: "
                "{\"tool\": \"tool_name\", \"arguments\": {...}}. "
                "Use edit_file to make code changes. "
                "Do NOT skip this phase — you MUST call a tool and modify the file(s) before moving on. "
                "Example: {\"tool\": \"edit_file\", \"arguments\": {\"path\": \"/file.py\", \"search_text\": \"old\", \"replace_text\": \"new\"}}"
            )
        elif phase == "VERIFY":
            guidance = (
                "Emit a JSON tool call with \"tool\" and \"arguments\" fields. "
                "Run tests with test_executor to confirm your changes work. "
                "Example: {\"tool\": \"test_executor\", \"arguments\": {\"path\": \"tests/\"}}"
            )
        elif phase == "ITERATE":
            guidance = (
                "Emit a JSON tool call with \"tool\" and \"arguments\" fields. "
                "Fix any issues found during verification, then re-run tests."
            )
        elif phase == "COMPLETE":
            guidance = (
                "Emit a JSON tool call with \"tool\" and \"arguments\" fields to call task_complete. "
                "Example: {\"tool\": \"task_complete\", \"arguments\": {\"status\": \"success\", \"reason_code\": \"tests_passing\", \"summary\": \"...\"}}. "
                "This is your FINAL turn — use it."
            )
        else:
            guidance = ""
        # Inject compressed task objective so it's always near the generation point
        task_obj = self.session_facts.current_objective
        compressed_task = self._compress_objective(task_obj) if task_obj else ""
        task_block = f"\n[ACTIVE TASK]\n{compressed_task}" if compressed_task else ""

        messages.append({
            "role": "user",
            "content": f"[CURRENT PHASE: {phase}]{task_block}\n"
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
            # P2: If stalling in EXECUTE, force-advance to VERIFY instead of
            # burning more refresh turns — the model is stuck producing messages.
            if self.current_phase == Phase.EXECUTE:
                logger.warning(
                    f"Stall detected in EXECUTE ({self.stall_count} consecutive stall turns), "
                    f"force-advancing from EXECUTE to VERIFY"
                )
                self.current_phase = Phase.VERIFY
                self.stall_count = 0
                self.format_monitor.reset()
                return False  # No refresh — we're transitioning out
            logger.info(
                f"Stall detected ({self.stall_count} consecutive invalid tool names), "
                f"triggering refresh"
            )
            self.stall_count = 0
            self.format_monitor.reset()
            return True

        # 2. Format decay (only when format_guard is enabled)
        if self.format_guard and self.format_monitor.should_refresh():
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

        # Keep recent conversation turns (trimmed to memory_horizon AND char budget)
        keep = self.memory_horizon if self.memory_horizon > 0 else 4
        trimmed_turns = (
            self.conversation_history[-keep:]
            if len(self.conversation_history) > keep
            else list(self.conversation_history)
        )

        # P4: Also evict oldest non-system messages when total chars exceed 80% of max_context_chars
        max_ctx = getattr(self.profile, "max_context_chars", 0) if self.profile else 0
        if max_ctx > 0:
            char_budget = int(max_ctx * 0.8)
            while len(trimmed_turns) > 1:
                total_chars = sum(len(m.get("content", "")) for m in trimmed_turns)
                if total_chars <= char_budget:
                    break
                # Evict oldest non-system message
                for idx, msg in enumerate(trimmed_turns):
                    if msg.get("role") != "system":
                        trimmed_turns.pop(idx)
                        logger.debug(
                            "P4 char-eviction: removed turn[%d] role=%s, total was %d chars (budget=%d)",
                            idx, msg.get("role"), total_chars, char_budget,
                        )
                        break
                else:
                    break  # Only system messages left — stop

        # Build the refreshed context using the refresher
        # Re-build the system prompt with current tool descriptions (may have demoted tools)
        tool_descs = self._get_tool_descriptions()
        output_fmt = OUTPUT_FORMAT_GEMMA_NATIVE if self.profile and self.profile.tool_call_format == "gemma_native" else OUTPUT_FORMAT_STANDARD
        system_prompt = SYSTEM_PROMPT.format(tool_descriptions=tool_descs, output_format=output_fmt)
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

        # Build: [extra_blocks..., conversation_turns]
        # NOTE: Do NOT include system message here — _build_context() always prepends it.
        # Including it here would create duplicate system messages causing HTTP 400.
        combined = []
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
            "argument": "path",
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
                # Incremental re-index for ctags CodeIndex
                if self.code_index is not None and not self.code_index._disabled:
                    self.code_index.reindex_file(file_path)

        return result

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """Extract the first balanced {...} JSON object from text.

        Uses a brace-depth counter — O(n), handles nested objects correctly.
        Also fixes the double-closing-brace Jinja2 echo bug: when the model
        echoes Jinja2 escapes, it may append an extra `}` after the balanced
        object. We detect this by checking if the character immediately after
        the balanced close is `}` (i.e. the text has `}}` at the end position).

        Returns:
            The JSON string if found, None otherwise.
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Found the balanced closing brace at position i
                    candidate = text[start:i + 1]
                    return candidate
        # No balanced object found — try the double-brace fix as a fallback:
        # strip one trailing } and retry
        stripped = text.rstrip()
        if stripped.endswith("}}"):
            return self._extract_first_json_object(stripped[:-1])
        return None

    def _infer_tool_from_signature(self, keys: set) -> Optional[str]:
        """Infer tool name from the set of top-level argument keys (Format 6).

        Matches against known tool signatures. Required keys must all be present.
        More specific signatures take priority over less specific ones.

        Returns:
            Tool name string if unambiguously matched, None otherwise.
        """
        # Strip meta-keys that are never tool args
        meta_keys = {"thought", "message", "reasoning", "response", "phase", "action", "content", "text"}
        effective_keys = keys - meta_keys

        # Ordered from most specific to least specific
        # Each entry: (required_keys, optional_keys, tool_name)
        SIGNATURES = [
            # task_complete: unique triple
            ({"status", "reason_code", "summary"}, set(), "task_complete"),
            # edit_file: requires path + search_text + replace_text
            ({"path", "search_text", "replace_text"}, set(), "edit_file"),
            # write_file: requires path + content
            ({"path", "content"}, set(), "write_file"),
            # grep_search: requires pattern (path/glob_pattern/max_results optional)
            ({"pattern"}, {"path", "glob_pattern", "max_results"}, "grep_search"),
            # run_shell: requires command
            ({"command"}, {"timeout"}, "run_shell"),
            # list_files: requires path + optional depth (but NOT search_text/replace_text/content/pattern)
            ({"path"}, {"depth"}, "list_files"),
            # read_file: requires path + optional max_lines
            ({"path"}, {"max_lines"}, "read_file"),
            # test_executor: optional path/framework/timeout — only if no path+depth combo
            (set(), {"path", "framework", "timeout"}, "test_executor"),
        ]

        candidates = []
        for required, optional, tool_name in SIGNATURES:
            if not required.issubset(effective_keys):
                continue
            # All keys must be either required or optional
            unexpected = effective_keys - required - optional
            if unexpected:
                continue
            candidates.append((len(required), tool_name))

        if not candidates:
            return None

        # Disambiguate list_files vs read_file: both require {path}
        # list_files has depth; read_file has max_lines
        # If both match, pick based on which optional key is present
        matched_names = [name for _, name in candidates]
        if "list_files" in matched_names and "read_file" in matched_names:
            if "depth" in effective_keys:
                return "list_files"
            elif "max_lines" in effective_keys:
                return "read_file"
            # Neither optional key present — default to read_file (more common)
            return "read_file"

        # Pick the most specific match (highest required key count)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _parse_tool_call(self, response_text: str) -> Optional[dict]:
        """Parse a tool call from model response text.

        Supports multiple formats:
        1. JSON flat:         {"tool": "name", "arguments": {...}}
        1b. JSON flat:        {"tool_name": "name", "params": {...}}
        2. LFM2.5 native:    <|tool_call_start|>[tool_name(arg1='val1')]<|tool_call_end|>
        3. Qwen tools[]:     {"tools": [{"list_files": {"path": "."}}]}
        4. Qwen tool_calls[]:{"tool_calls": [{"grep_search": {"pattern": "x"}}]}
        5. Qwen key-as-name: {"read_file": {"path": "/some/file.py"}}
        6. Naked args:        {"path": "/foo.py", "max_lines": 100}  (inferred from signature)
        6b. Naked+thought:    {"thought": "...", "path": "/foo.py", "search_text": "x", "replace_text": "y"}

        Also handles:
        - Double-closing-brace bug: {"tool": "list_files", "arguments": {"path": "/src"}}}
        - thought/message/reasoning/response keys mixed with tool args

        Returns:
            Dict with "name" and "args" if found, None otherwise.
        """
        # Strip thinking tags first
        from localite.model.client import strip_thinking
        cleaned = strip_thinking(response_text)

        text = cleaned.strip()
        if not text:
            return None

        # --- Format 2: <|tool_call_start|>[tool_name(key='val')]<|tool_call_end|> ---
        tool_call_match = re.search(
            r'<\|tool_call_start\|>\s*\[(\w+)\s*\(([^)]*)\)\]\s*<\|tool_call_end\|>',
            text,
        )
        if tool_call_match:
            tool_name = tool_call_match.group(1)
            args_str = tool_call_match.group(2)
            args = {}
            if args_str.strip():
                for pair in re.findall(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"| ([^,\s)]+))", args_str):
                    key = pair[0]
                    val = pair[1] or pair[2] or pair[3]
                    args[key] = val
            return {"name": tool_name, "args": args}

        # --- Formats 1, 1b, 3, 4, 5, 6: JSON-based ---
        # Use tolerant extraction: find first balanced {...} object (O(n), handles }})
        json_str = self._extract_first_json_object(text)
        if json_str is None:
            return None

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        # --- Format 1 / 1b: flat {"tool"/"tool_name"/"name"/"function": ..., "arguments"/...} ---
        tool_name = (
            data.get("tool")
            or data.get("tool_name")
            or data.get("name")
            or data.get("function")
        )
        if tool_name and tool_name.lower() not in ("none", "null", "noop", ""):
            args = (
                data.get("arguments")
                or data.get("args")
                or data.get("tool_args")
                or data.get("params")
                or data.get("parameters")
                or {}
            )
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

        # --- Formats 3 & 4: {"tools": [...]} or {"tool_calls": [...]} ---
        for array_key in ("tools", "tool_calls"):
            array_val = data.get(array_key)
            if not (isinstance(array_val, list) and array_val):
                continue
            first = array_val[0]
            if not isinstance(first, dict):
                continue
            inner_name = (
                first.get("tool")
                or first.get("tool_name")
                or first.get("name")
                or first.get("function")
            )
            if inner_name and inner_name.lower() not in ("none", "null", "noop", ""):
                inner_args = (
                    first.get("arguments")
                    or first.get("args")
                    or first.get("params")
                    or first.get("parameters")
                    or {}
                )
                if isinstance(inner_args, dict):
                    logger.debug("Normalizer[%s]: standard inner -> tool=%s", array_key, inner_name)
                    return {"name": inner_name, "args": inner_args}
            else:
                known_meta = {"tool", "tool_name", "name", "function", "arguments", "args", "params", "parameters"}
                for k, v in first.items():
                    if k not in known_meta and isinstance(v, dict):
                        logger.debug("Normalizer[%s]: key-as-name inside array -> tool=%s", array_key, k)
                        return {"name": k, "args": v}

        # --- Format 5: {"read_file": {"path": "..."}} ---
        # Top-level key IS the tool name; value is the args dict.
        non_tool_keys = {
            "tool", "tool_name", "name", "function",
            "arguments", "args", "tool_args", "params", "parameters",
            "tools", "tool_calls",
            "thought", "message", "reasoning", "response",
            "phase", "action", "content", "text",
        }
        for k, v in data.items():
            if k not in non_tool_keys and isinstance(v, dict) and k in self.tools:
                logger.debug("Normalizer[key-as-name]: top-level -> tool=%s", k)
                return {"name": k, "args": v}

        # --- Format 6: Naked args — infer tool from signature ---
        # Strip thought/message/reasoning/response meta-keys, then match remaining keys
        meta_strip = {"thought", "message", "reasoning", "response", "phase", "action", "content", "text"}
        stripped_data = {k: v for k, v in data.items() if k not in meta_strip}

        # If after stripping we have nothing, this is a pure message — return None
        if not stripped_data:
            return None

        inferred_tool = self._infer_tool_from_signature(set(stripped_data.keys()))
        if inferred_tool:
            logger.debug("Format6: inferred tool=%s from keys=%s", inferred_tool, set(stripped_data.keys()))
            return {"name": inferred_tool, "args": stripped_data}

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
