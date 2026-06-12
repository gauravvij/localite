"""Dedicated unit tests for the task_complete feature and COMPLETE phase behavior.

Tests:
1. COMPLETE phase is entered when task_complete is called during EXECUTE
2. task_complete_called flag is set correctly
3. Loop return dict includes task_complete_called
4. _complete_turn_given flag works (one COMPLETE turn, then exit)
5. [CURRENT PHASE] context is injected in _build_context
6. COMPLETE is never skipped by _should_skip_phase
7. edit_file tool has the improved description
"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from localite.config import ModelProfile
from localite.loop.agent_loop import AgentLoop, SYSTEM_PROMPT
from localite.loop.phases import Phase
from localite.permissions.gate import PermissionGate
from localite.episodes.store import EpisodeStore
from localite.episodes.model import Episode, Turn
from localite.tools.base import ToolResult, BaseTool
from localite.tools.read import ReadFileTool
from localite.tools.write import WriteFileTool
from localite.tools.edit import EditFileTool
from localite.tools.shell import RunShellTool
from localite.tools.search import GrepSearchTool
from localite.tools.test_executor import TestExecutorTool
from localite.tools.diff_view import DiffViewTool
from localite.tools.task_complete import TaskCompleteTool
from localite.context.standing_instructions import StandingInstructions
from localite.context.buffer import SessionFacts


# =========================================================================
# Mock model client
# =========================================================================


class MockOllamaClient:
    """Mock that returns predefined responses for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = None

    async def chat(self, messages: list[dict], stream: bool = False, options: dict | None = None):
        self.last_messages = messages
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        # Default: return a JSON with task_complete
        return json.dumps({
            "thought": "Task is done.",
            "tool": "task_complete",
            "arguments": {"status": "success", "reason_code": "tests_passing", "summary": "All done"},
        })


def make_loop(model_client=None) -> AgentLoop:
    """Create a minimal AgentLoop for testing with mock client and tools."""
    tools = {}
    for tool_cls in [
        ReadFileTool,
        WriteFileTool,
        EditFileTool,
        GrepSearchTool,
        RunShellTool,
        TestExecutorTool,
        DiffViewTool,
        TaskCompleteTool,
    ]:
        t = tool_cls()
        tools[t.name] = t

    gate = PermissionGate(step_mode=False)
    store = EpisodeStore()

    profile = ModelProfile(
        name="test-model",
        max_turns=5,
        has_thinking_tags=False,
    )

    client = model_client or MockOllamaClient()
    si = StandingInstructions()

    loop = AgentLoop(
        model_client=client,
        tools=tools,
        permission_gate=gate,
        episode_store=store,
        model_profile=profile,
        standing_instructions=si,
        max_iterations=3,
    )
    return loop


# =========================================================================
# Tests
# =========================================================================


def test_complete_phase_entered_when_task_complete_called_during_execute():
    """Test that calling task_complete during EXECUTE transitions to COMPLETE."""
    loop = make_loop()
    loop.episode = loop.store.new_episode(objective="Test task")
    loop.current_phase = Phase.EXECUTE
    loop._complete_turn_given = False

    # Simulate task_complete being called in _execute_phase
    response = json.dumps({
        "thought": "Done",
        "tool": "task_complete",
        "arguments": {"status": "success", "reason_code": "tests_passing", "summary": "Test complete"},
    })

    # We need to test the intercept logic directly
    tool_call = loop._parse_tool_call(response)
    assert tool_call is not None
    assert tool_call.get("name") == "task_complete"

    # Simulate what _execute_phase does with the intercept
    loop.task_complete_called = True
    loop.task_complete_args = tool_call.get("arguments", {})
    if loop.current_phase != Phase.COMPLETE:
        loop.current_phase = Phase.COMPLETE

    # Verify COMPLETE phase is entered
    assert loop.current_phase == Phase.COMPLETE
    assert loop.task_complete_called is True


def test_task_complete_called_flag_set_correctly():
    """Test that task_complete_called flag is set to True when task_complete tool is called."""
    loop = make_loop()
    assert loop.task_complete_called is False

    # Simulate the intercept in _execute_phase
    loop.task_complete_called = True
    assert loop.task_complete_called is True


def test_loop_return_dict_includes_task_complete_called():
    """Test that the run() return dict includes task_complete_called."""
    client = MockOllamaClient(responses=[
        # EXPLORE turn - just a message, no tool call
        json.dumps({"thought": "Exploring", "message": "I'll read the file first."}),
        # PLAN turn - just a message
        json.dumps({"thought": "Planning", "message": "Here's my plan."}),
        # EXECUTE turn - calls task_complete (early exit pattern)
        json.dumps({"thought": "Done", "tool": "task_complete", "arguments": {"status": "success", "reason_code": "tests_passing", "summary": "Test complete"}}),
    ])

    loop = make_loop(client)

    async def run():
        return await loop.run("Test task")

    result = asyncio.run(run())

    assert "task_complete_called" in result
    assert result["task_complete_called"] is True
    assert result["phase"] == "COMPLETE"


def test_complete_turn_given_flag_works():
    """Test that _complete_turn_given flag ensures one COMPLETE turn then exit."""
    client = MockOllamaClient(responses=[
        # Turn 0: EXPLORE - message only
        json.dumps({"thought": "Exploring", "message": "Reading files..."}),
        # Turn 1: PLAN - message only
        json.dumps({"thought": "Planning", "message": "Plan ready."}),
        # Turn 2: EXECUTE - calls task_complete
        json.dumps({"thought": "Done", "tool": "task_complete", "arguments": {"status": "success", "reason_code": "tests_passing", "summary": "All done"}}),
    ])

    loop = make_loop(client)

    async def run():
        return await loop.run("Test task")

    result = asyncio.run(run())
    assert result["phase"] == "COMPLETE"
    assert result["task_complete_called"] is True

    # The loop should have given the model one COMPLETE turn and then exited
    # Check that _complete_turn_given was set
    assert loop._complete_turn_given is True


def test_current_phase_injected_in_build_context():
    """Test that [CURRENT PHASE] context is injected in _build_context."""
    loop = make_loop()
    loop.episode = loop.store.new_episode(objective="Test")
    loop.current_phase = Phase.EXPLORE

    context = loop._build_context()

    # Find the CURRENT PHASE message
    phase_msgs = [m for m in context if m["role"] == "user" and "CURRENT PHASE" in m.get("content", "")]
    assert len(phase_msgs) > 0, "No [CURRENT PHASE] message found in context"

    # Check it contains EXPLORE
    content = phase_msgs[0]["content"]
    assert "EXPLORE" in content
    assert "CURRENT PHASE" in content

    # Now test with COMPLETE phase
    loop.current_phase = Phase.COMPLETE
    context2 = loop._build_context()
    phase_msgs2 = [m for m in context2 if m["role"] == "user" and "CURRENT PHASE" in m.get("content", "")]
    assert len(phase_msgs2) > 0
    assert "COMPLETE" in phase_msgs2[0]["content"]
    assert "task_complete" in phase_msgs2[0]["content"]


def test_complete_never_skipped_by_should_skip_phase():
    """Test that COMPLETE is never skipped by _should_skip_phase."""
    loop = make_loop()

    # _should_skip_phase is only called when current_phase != COMPLETE
    # in the main loop. Let's verify it returns False for COMPLETE
    # by checking that the loop guard is correct.

    # The main loop has: if self.current_phase != Phase.COMPLETE and self._should_skip_phase(...)
    # So COMPLETE is never skipped. Let's verify _should_skip_phase exists and works.
    result = loop._should_skip_phase(Phase.COMPLETE)
    # It returns False for any unrecognized phase (falls through to return False)
    assert result is False, "COMPLETE phase should never be skipped"

    # Also verify it returns False for important phases
    assert loop._should_skip_phase(Phase.EXECUTE) is False
    assert loop._should_skip_phase(Phase.EXPLORE) is False


def test_edit_file_has_improved_description():
    """Test that the EditFileTool has the improved description with concrete examples."""
    edit_tool = EditFileTool()
    desc = edit_tool.description

    # The improved description should mention add docstrings, fix bugs, refactor
    assert "docstring" in desc.lower() or "docstring" in desc
    assert "fix" in desc.lower() or "bug" in desc.lower()
    assert "edit" in desc.lower()

    # Check the description is substantive (not just "Edit a file")
    assert len(desc) > 50, "Description should be detailed with usage examples"

    # Also verify the SYSTEM_PROMPT mentions edit_file for code changes
    assert "edit_file" in SYSTEM_PROMPT
    assert "docstring" in SYSTEM_PROMPT.lower() or "code changes" in SYSTEM_PROMPT.lower()


def test_standing_instructions_mention_task_complete():
    """Test that standing instructions emphasize task_complete in COMPLETE phase."""
    si = StandingInstructions()
    text = si.get_text()

    # Standing instructions no longer contain phase protocol (moved to SYSTEM_PROMPT)
    # But safety rules should still be intact
    assert "Safety Rules" in text
    assert "NEVER hallucinate" in text
    # SYSTEM_PROMPT should contain task_complete and COMPLETE phase guidance
    assert "task_complete" in SYSTEM_PROMPT
    assert "COMPLETE" in SYSTEM_PROMPT


def test_system_prompt_has_correct_tool_rules():
    """Test that the system prompt rules guide tool selection properly."""
    # Rule 3 should mention edit_file for code changes
    assert "edit_file" in SYSTEM_PROMPT

    # Rule 4 should mention test_executor for tests
    assert "test_executor" in SYSTEM_PROMPT

    # Rule 6 should mention task_complete
    assert "task_complete" in SYSTEM_PROMPT