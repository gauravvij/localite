#!/usr/bin/env python3
"""Real-model end-to-end test of AgentLoop with all 5 new features.

Tests:
- TaskCompleteTool intercept
- Adaptive phase transitions (_should_skip_phase)
- iad_horizon removal (stall_threshold replacement)
- Delegation telemetry (tool_stats)
- Episodic memory (session persistence)

Uses gemma4:e4b via Ollama. Creates a temp project, runs the agent, and
reports comprehensive results.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set logging to INFO — capture all phase skip messages, tool stats updates, etc.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s:%(lineno)d] %(message)s",
    stream=sys.stderr,
)

# Suppress httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from localite.config import ConfigLoader
from localite.model.client import AsyncOllamaClient
from localite.permissions.gate import PermissionGate
from localite.episodes.store import EpisodeStore
from localite.loop.agent_loop import AgentLoop
from localite.tools.base import BaseTool
from localite.tools.read import ReadFileTool
from localite.tools.write import WriteFileTool
from localite.tools.edit import EditFileTool
from localite.tools.search import GrepSearchTool
from localite.tools.shell import RunShellTool
from localite.tools.test_executor import TestExecutorTool
from localite.tools.diff_view import DiffViewTool
from localite.tools.task_complete import TaskCompleteTool
from localite.tools.memory_tools import MemoryReadTool, MemoryWriteTool
from localite.memory.memory_store import EpisodicMemoryStore
from localite.context.standing_instructions import StandingInstructions

logger = logging.getLogger("E2E")


def setup_temp_project():
    """Create a temp project at /tmp/localite_e2e_test/ with greet.py and test_greet.py."""
    base = Path("/tmp/localite_e2e_test")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    greet_py = base / "greet.py"
    greet_py.write_text(
        '"""Greeting module."""\n\n'
        'def greet(name):\n'
        '    return f"Hello, {name}!"\n\n'
        'def add(a, b):\n'
        '    return a + b\n'
    )

    test_py = base / "test_greet.py"
    test_py.write_text(
        '"""Tests for greet module."""\n\n'
        'import pytest\n'
        'from greet import greet, add\n\n'
        'def test_greet():\n'
        '    assert greet("World") == "Hello, World!"\n'
        '    assert greet("Alice") == "Hello, Alice!"\n\n'
        'def test_add():\n'
        '    assert add(2, 3) == 5\n'
        '    assert add(-1, 1) == 0\n'
        '    assert add(0, 0) == 0\n'
    )

    logger.info(f"Temp project created at {base}")
    return str(base)


def create_standalone_loop(temp_dir: str) -> AgentLoop:
    """Assemble a full AgentLoop with all tools, memory, standing instructions, etc."""
    # 1. Tools (all 10 including new ones)
    tools: dict[str, BaseTool] = {}
    for tool_cls in [
        ReadFileTool,
        WriteFileTool,
        EditFileTool,
        GrepSearchTool,
        RunShellTool,
        TestExecutorTool,
        DiffViewTool,
        TaskCompleteTool,
        MemoryReadTool,
        MemoryWriteTool,
    ]:
        t = tool_cls()
        tools[t.name] = t

    # 2. Permission gate: non-interactive (step_mode=False = batch/auto-approve)
    gate = PermissionGate(step_mode=False)

    # 3. Memory store at /tmp/localite_e2e_memory/
    memory_dir = "/tmp/localite_e2e_memory"
    if os.path.exists(memory_dir):
        shutil.rmtree(memory_dir)
    memory_store = EpisodicMemoryStore(base_dir=memory_dir)

    # 4. Wire memory store into memory tools
    if "memory_read" in tools:
        tools["memory_read"]._memory_store = memory_store
    if "memory_write" in tools:
        tools["memory_write"]._memory_store = memory_store

    # 5. Episode store (in-memory is fine)
    store = EpisodeStore()

    # 6. Load profile for gemma4_e4b
    config_loader = ConfigLoader()
    profile = config_loader.load_profile("gemma4_e4b")

    # 7. Model client: gemma4:e4b via Ollama
    model = AsyncOllamaClient(
        model_name="gemma4:e4b",
        base_url=profile.base_url,
        timeout=profile.timeout,
        has_thinking_tags=profile.has_thinking_tags,
    )

    # 8. Standing instructions
    standing_instructions = StandingInstructions()

    # 9. Build the loop
    loop = AgentLoop(
        model_client=model,
        tools=tools,
        permission_gate=gate,
        episode_store=store,
        model_profile=profile,
        standing_instructions=standing_instructions,
        max_iterations=3,
        memory_store=memory_store,
    )

    logger.info(f"AgentLoop created: {len(tools)} tools, gemma4:e4b, memory at {memory_dir}")
    return loop


def print_separator(title: str, char: str = "="):
    print(f"\n{char * 20} {title} {char * 20}")


def print_turn_phases(turns: list) -> list[str]:
    """Extract phase labels from each turn."""
    return [getattr(t, "phase", "?") for t in turns]


def print_turn_tools(turns: list) -> list[dict]:
    """Extract tool calls from each turn."""
    tools_called = []
    for t in turns:
        tc = getattr(t, "tool_call", None)
        if tc:
            tools_called.append({
                "turn": t.turn_number,
                "phase": t.phase,
                "tool": tc.get("name", tc.get("tool", "?")),
                "args": tc.get("arguments", tc.get("args", {})),
            })
    return tools_called


def check_tests_pass(temp_dir: str) -> bool:
    """Run pytest on the test file in the temp dir."""
    result = subprocess.run(
        ["python3", "-m", "pytest", "test_greet.py", "-v", "-x"],
        cwd=temp_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    passed = result.returncode == 0
    print(f"  pytest exit code: {result.returncode}")
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            print(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-3:]:
            print(f"  stderr: {line}")
    return passed


async def main():
    print_separator("E2E REAL-MODEL TEST")
    print(f"Model: gemma4:e4b")
    print(f"Time:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Setup temp project
    print("Setting up temp project...")
    temp_dir = setup_temp_project()
    print(f"  Temp dir: {temp_dir}")
    print(f"  greet.py exists: {os.path.exists(os.path.join(temp_dir, 'greet.py'))}")
    print(f"  test_greet.py exists: {os.path.exists(os.path.join(temp_dir, 'test_greet.py'))}")
    print()

    # Step 2: Create the AgentLoop
    print("Creating AgentLoop with all 10 tools and gemma4:e4b...")
    loop = create_standalone_loop(temp_dir)
    print()

    # Step 3: Run the agent
    task = f"Please add a docstring to the greet function in {temp_dir}/greet.py, then run the tests."
    print(f'Task: "{task}"')
    print(f"Starting agent loop...")
    print()

    start_time = time.time()
    try:
        result = await loop.run(task)
        elapsed = time.time() - start_time
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Agent loop raised exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        result = {"phase": "ERROR", "episode_id": "N/A", "files_changed": [], "summary": str(e)}

    # Step 4: Collect telemetry from the loop
    status = loop.get_status()
    turns = loop.episode.turns if loop.episode else []
    phase_seq = print_turn_phases(turns)
    tool_calls = print_turn_tools(turns)
    tool_stats = loop.tool_stats

    # Memory store check
    memory_index_path = Path("/tmp/localite_e2e_memory") / "sessions" / "index.json"
    memory_saved = memory_index_path.exists()

    # File modification check — look for a docstring on the greet function
    greet_path = Path(temp_dir) / "greet.py"
    has_docstring = False
    if greet_path.exists():
        content = greet_path.read_text()
        # Check if the greet function has a docstring (triple-quoted string inside function body)
        lines = content.split("\n")
        in_greet = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def greet("):
                in_greet = True
                continue
            if in_greet:
                if '"""' in stripped:
                    has_docstring = True
                    break
                if stripped.startswith("def ") or stripped.startswith("class "):
                    break
                if stripped and not stripped.startswith("#") and '"""' not in stripped:
                    break
        print(f"\n  greet.py content:\n{content}")
    else:
        print(f"\n  greet.py MISSING!")

    # Tests check
    tests_pass = check_tests_pass(temp_dir)

    # =====================================================================
    # REPORT
    # =====================================================================
    print_separator("RESULTS")
    print()

    print(f"1. DID THE AGENT COMPLETE THE FULL CYCLE?")
    final_phase = result.get("phase", "?")
    print(f"   Final phase: {final_phase}")
    if final_phase == "COMPLETE":
        print(f"   ✅ YES — Agent completed the full cycle")
    else:
        print(f"   ❌ NO — Agent ended in phase {final_phase}")
    print()

    print(f"2. HOW MANY TURNS?")
    print(f"   Total turns: {len(turns)}")
    print(f"   Phase sequence: {phase_seq}")
    print()

    print(f"3. WHAT TOOLS WERE CALLED AND IN WHAT ORDER?")
    if tool_calls:
        for tc in tool_calls:
            args_preview = str(tc.get("args", {}))
            if len(args_preview) > 120:
                args_preview = args_preview[:120] + "..."
            print(f"   Turn {tc['turn']} [{tc['phase']}]: {tc['tool']}({args_preview})")
    else:
        print(f"   (no tool calls detected)")
    print()

    print(f"4. WAS TASK_COMPLETE USED?")
    any_complete = any(tc.get("tool") == "task_complete" for tc in tool_calls)
    if any_complete:
        print(f"   ✅ YES — task_complete was called by the model")
    else:
        print(f"   ℹ️  NO — Loop ended naturally via VERIFY→COMPLETE or other mechanism")
    print()

    print(f"5. WERE ADAPTIVE PHASE SKIPS TRIGGERED?")
    print(f"   (Check stderr logs above for 'Skipping phase' messages)")
    print()

    print(f"6. TOOL STATS SNAPSHOT:")
    if tool_stats:
        for tname, stats in sorted(tool_stats.items()):
            print(f"   {tname}: calls={stats['calls']}, "
                  f"success={stats['successes']}, fail={stats['failures']}, "
                  f"trust={stats['trust_score']:.2f}, "
                  f"duration_ms={stats['total_duration_ms']}")
    else:
        print(f"   (tool_stats is empty)")
    print()

    print(f"7. MEMORY SESSION SAVED?")
    print(f"   Memory index.json exists: {memory_saved}")
    if memory_saved:
        mem_data = json.loads(memory_index_path.read_text())
        print(f"   Sessions count: {len(mem_data)}")
        for m in mem_data:
            print(f"     - {m['session_id'][:12]}... | {m['task'][:60]} | {m['status']}")
    else:
        print(f"   Memory directory exists: {os.path.exists('/tmp/localite_e2e_memory')}")
        print(f"   Sessions dir exists: {memory_index_path.parent.exists()}")
    print()

    print(f"8. FILE MODIFIED?")
    if has_docstring:
        print(f"   ✅ YES — greet.py has a docstring on the greet function")
    else:
        print(f"   ❌ NO — greet.py does NOT have a docstring on greet()")
    print()

    print(f"9. DO TESTS PASS?")
    if tests_pass:
        print(f"   ✅ YES — pytest passes")
    else:
        print(f"   ❌ NO — pytest failed")
    print()

    print(f"10. RESULT DICT:")
    print(f"    phase: {result.get('phase')}")
    print(f"    episode_id: {result.get('episode_id', 'N/A')[:20]}...")
    print(f"    files_changed: {result.get('files_changed', [])}")
    print(f"    summary: {result.get('summary', '')}")
    print()

    print(f"11. GET_STATUS():")
    print(f"    {json.dumps(status, indent=4, default=str)}")
    print()

    print(f"12. TOTAL ELAPSED TIME:")
    print(f"    {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print()

    # Summary verdict
    print_separator("VERDICT")
    all_tests_ok = tests_pass
    file_modified = has_docstring
    completed = final_phase == "COMPLETE"
    tool_stats_populated = len(tool_stats) > 0

    passed_checks = sum([completed, all_tests_ok, file_modified, memory_saved, tool_stats_populated])
    total_checks = 5

    print(f"  ✅ Completed cycle:       {'PASS' if completed else 'FAIL'}")
    print(f"  ✅ Tests pass:            {'PASS' if all_tests_ok else 'FAIL'}")
    print(f"  ✅ File modified:         {'PASS' if file_modified else 'FAIL'}")
    print(f"  ✅ Memory saved:          {'PASS' if memory_saved else 'FAIL'}")
    print(f"  ✅ Tool stats populated:  {'PASS' if tool_stats_populated else 'FAIL'}")
    print(f"  Overall: {passed_checks}/{total_checks} checks passed")
    print(f"  Turns: {len(turns)}, Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())