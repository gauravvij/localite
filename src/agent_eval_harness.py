"""
Agent evaluation harness — tests the localite coding agent through its actual
5-phase loop (not raw Ollama). Captures internal guardrail metrics.

Components:
    - AutoGate: auto-approves all tool calls (batch mode for eval)
    - AgentEvalResult: dataclass for one task result
    - AgentEvalHarness: runs AgentLoop for a single task, collects metrics
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from localite.config import ModelProfile
from localite.context.format_monitor import FormatMonitor
from localite.episodes.model import Episode, Turn
from localite.episodes.store import EpisodeStore
from localite.loop.agent_loop import AgentLoop
from localite.model.client import AsyncOllamaClient
from localite.permissions.gate import PermissionGate, PermissionResult
from localite.tools.base import BaseTool
from localite.tools.read import ReadFileTool
from localite.tools.write import WriteFileTool
from localite.tools.edit import EditFileTool
from localite.tools.search import GrepSearchTool
from localite.tools.shell import RunShellTool
from localite.tools.test_executor import TestExecutorTool
from localite.tools.diff_view import DiffViewTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AutoGate — auto-approves all tool calls without interactive prompts
# ---------------------------------------------------------------------------


class AutoGate(PermissionGate):
    """Permission gate variant that auto-approves every tool call.

    Designed for evaluation/batch mode — no interactive prompts.
    Still records the decision for metrics collection.
    """

    def __init__(self):
        super().__init__(step_mode=False)
        self._auto_approved_count = 0

    def propose(
        self,
        action_description: str,
        tool_call: dict,
    ) -> PermissionResult:
        """Auto-approve every tool call without prompting."""
        self._auto_approved_count += 1
        return PermissionResult(
            decision="approved",
            modified_tool_call=tool_call,
        )

    def flush_pending(self) -> list[PermissionResult]:
        """No-op: all proposals are auto-approved immediately."""
        return []

    @property
    def approved_count(self) -> int:
        return self._auto_approved_count


# ---------------------------------------------------------------------------
# AgentEvalResult — dataclass for one task evaluation result
# ---------------------------------------------------------------------------


@dataclass
class AgentEvalResult:
    """Result from evaluating a single agent task.

    Attributes:
        task_id: Unique task identifier (e.g. \"fibonacci\").
        task_description: The prompt given to the agent.
        success: Whether the verifier returned True.
        error: Error message if execution or verification failed.
        duration_s: Total wall-clock seconds for the agent run.
        turns_used: Number of agent turns consumed.
        format_monitor_avg: Average JSON format score over all turns.
        context_refreshes: Number of context refreshes triggered.
        files_changed: List of files the agent reported changing.
        tool_calls: Number of tool calls made.
        tool_calls_breakdown: Dict mapping tool name -> call count.
        phases_reached: Set of phases the agent reached.
        output_dir: Temp working directory for the task.
        raw_episode_data: Serialized episode info for deeper analysis.
    """
    task_id: str
    task_description: str
    success: bool = False
    error: Optional[str] = None
    duration_s: float = 0.0
    turns_used: int = 0
    format_monitor_avg: float = 0.0
    context_refreshes: int = 0
    files_changed: list[str] = field(default_factory=list)
    tool_calls: int = 0
    tool_calls_breakdown: dict[str, int] = field(default_factory=dict)
    phases_reached: set[str] = field(default_factory=set)
    output_dir: Optional[str] = None
    raw_episode_data: Optional[dict] = field(default=None)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "success": self.success,
            "error": self.error,
            "duration_s": round(self.duration_s, 2),
            "turns_used": self.turns_used,
            "format_monitor_avg": round(self.format_monitor_avg, 4),
            "context_refreshes": self.context_refreshes,
            "files_changed": self.files_changed,
            "tool_calls": self.tool_calls,
            "tool_calls_breakdown": self.tool_calls_breakdown,
            "phases_reached": sorted(self.phases_reached),
            "output_dir": self.output_dir,
            "raw_episode_data": self.raw_episode_data,
        }


# ---------------------------------------------------------------------------
# AgentEvalHarness — runs AgentLoop for a single task, collects metrics
# ---------------------------------------------------------------------------


def create_default_tools() -> dict[str, BaseTool]:
    """Create and return the default set of tools (instance-first pattern)."""
    tools: dict[str, BaseTool] = {}
    for tool_cls in [
        ReadFileTool,
        WriteFileTool,
        EditFileTool,
        GrepSearchTool,
        RunShellTool,
        TestExecutorTool,
        DiffViewTool,
    ]:
        t = tool_cls()
        tools[t.name] = t
    return tools


class AgentEvalHarness:
    """Harness that runs the AgentLoop for a single coding task.

    Usage:
        harness = AgentEvalHarness(profile)
        result = await harness.run_task(task_id, task_desc, verifier, output_dir)
    """

    def __init__(
        self,
        profile: ModelProfile,
        tools: Optional[dict[str, BaseTool]] = None,
        max_iterations: int = 3,
    ):
        self.profile = profile
        self.tools = tools or create_default_tools()
        self.max_iterations = max_iterations

    async def run_task(
        self,
        task_id: str,
        task_description: str,
        verifier: Optional[Callable[[str], tuple[bool, str]]] = None,
        output_dir: Optional[str] = None,
        run_timeout: int = 600,
    ) -> AgentEvalResult:
        """Run a single task through the AgentLoop.

        Args:
            task_id: Unique identifier for this task.
            task_description: The prompt to give the agent.
            verifier: Optional function(output_dir) -> (bool, str).
            output_dir: Working directory for this task. If None, uses CWD.
            run_timeout: Timeout for the model API in seconds.

        Returns:
            AgentEvalResult with metrics and verification outcome.
        """
        result = AgentEvalResult(
            task_id=task_id,
            task_description=task_description,
        )

        if output_dir is None:
            output_dir = os.getcwd()
        result.output_dir = output_dir

        # Initialize model client
        client = AsyncOllamaClient(
            model_name=self.profile.name,
            base_url=self.profile.base_url,
            timeout=run_timeout,
            has_thinking_tags=self.profile.has_thinking_tags,
        )

        # Initialize gate and store
        gate = AutoGate()
        store = EpisodeStore()

        # Initialize AgentLoop
        loop = AgentLoop(
            model_client=client,
            tools=self.tools,
            permission_gate=gate,
            episode_store=store,
            model_profile=self.profile,
            max_iterations=self.max_iterations,
        )

        # Run the agent
        start_time = time.perf_counter()
        try:
            loop_result = await loop.run(task_description)
            elapsed = time.perf_counter() - start_time
            result.duration_s = elapsed
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            result.duration_s = elapsed
            result.error = f"AgentLoop execution error: {type(e).__name__}: {e}"
            logger.exception(f"AgentLoop failed for task {task_id}")
            return result

        # Collect metrics from the loop
        episode = loop.episode
        if episode:
            result.turns_used = len(episode.turns)
            result.files_changed = list(episode.files_changed)

            # Format monitor average
            result.format_monitor_avg = loop.format_monitor.average()

            # Context refreshes
            result.context_refreshes = loop.refresher.get_refresh_count()

            # Tool call metrics
            tool_counts: dict[str, int] = {}
            for turn in episode.turns:
                if turn.tool_call:
                    name = turn.tool_call.get("name", "unknown")
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                if turn.phase:
                    result.phases_reached.add(turn.phase)
            result.tool_calls = sum(tool_counts.values())
            result.tool_calls_breakdown = tool_counts

            # Raw episode data for deeper analysis
            result.raw_episode_data = episode.to_dict()

        # AutoGate approved count (backup if turn parsing missed some)
        result.tool_calls = max(result.tool_calls, gate.approved_count)

        # Run verifier if provided
        if verifier:
            try:
                v_success, v_msg = verifier(output_dir)
                result.success = v_success
                if not v_success:
                    result.error = v_msg
            except Exception as e:
                result.success = False
                result.error = f"Verifier error: {type(e).__name__}: {e}"
        else:
            # No verifier — success if we got through the loop
            result.success = result.turns_used > 0

        return result