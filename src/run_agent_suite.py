"""
CLI orchestrator for the AgentLoop evaluation suite.

Loads a model profile, creates client/tools/AutoGate/AgentLoop, runs each
test task through AgentLoop.run(), collects AgentEvalResult metrics, and
saves per-task JSON results + a summary report.

Usage:
    python3 src/run_agent_suite.py --profile gemma4_e4b
    python3 src/run_agent_suite.py --profile gemma4_e4b --runs 2
    python3 src/run_agent_suite.py --profile gemma4_e4b --output my_results
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localite.config import ConfigLoader, ModelProfile
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

from src.agent_eval_harness import AgentEvalHarness, AgentEvalResult, AutoGate
from src.agent_test_tasks import TASKS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "agent_suite")
DEFAULT_TASK_TIMEOUT = 1800  # 30 minutes per task
AGENT_TASKS_DIR = PROJECT_ROOT  # Agent writes files to CWD (project root)


def _ensure_dirs():
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(AGENT_TASKS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_tools() -> dict[str, BaseTool]:
    """Create the default set of tools (instance-first pattern)."""
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


def load_profile(profile_name: str) -> ModelProfile:
    """Load a model profile by name."""
    config_loader = ConfigLoader()
    try:
        profile = config_loader.load_profile(profile_name)
    except FileNotFoundError:
        available = config_loader.list_profiles()
        print(f"Error: Profile '{profile_name}' not found. Available: {available}")
        sys.exit(1)
    return profile


def build_report(
    task_results: list[AgentEvalResult],
    profile_name: str,
    total_elapsed: float,
) -> str:
    """Build a summary report markdown string from task results.

    Args:
        task_results: List of AgentEvalResult from each task run.
        profile_name: The model profile name used.
        total_elapsed: Total wall-clock time for the suite.

    Returns:
        Markdown report string.
    """
    lines = []
    lines.append(f"# Agent Loop Evaluation Report — {profile_name}")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Profile**: {profile_name}")
    lines.append(f"**Tasks**: {len(task_results)}")
    lines.append(f"**Total Duration**: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Task | Status | Duration | Turns | Tool Calls | Format Avg | Files Changed |")
    lines.append("|------|--------|----------|-------|------------|------------|---------------|")

    passed = 0
    failed = 0
    for r in task_results:
        status = "✅ PASS" if r.success else "❌ FAIL"
        if r.error and not r.success:
            status += f" ({r.error[:60]})"
        d = r.duration_s
        lines.append(
            f"| {r.task_id} | {status} | {d:.1f}s | {r.turns_used} | "
            f"{r.tool_calls} | {r.format_monitor_avg:.3f} | "
            f"{', '.join(r.files_changed) if r.files_changed else '-'} |"
        )
        if r.success:
            passed += 1
        else:
            failed += 1

    lines.append("")
    lines.append(f"**Passed**: {passed}/{len(task_results)}")
    lines.append(f"**Failed**: {failed}/{len(task_results)}")
    lines.append(f"**Pass Rate**: {passed / len(task_results) * 100:.1f}%")
    lines.append("")

    # Aggregate metrics
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(f"| Metric | Average | Min | Max |")
    lines.append(f"|--------|---------|-----|-----|")

    durations = [r.duration_s for r in task_results]
    turns = [r.turns_used for r in task_results]
    tool_calls = [r.tool_calls for r in task_results]
    format_avgs = [r.format_monitor_avg for r in task_results]
    refreshes = [r.context_refreshes for r in task_results]

    if durations:
        lines.append(f"| Duration (s) | {sum(durations)/len(durations):.1f} | {min(durations):.1f} | {max(durations):.1f} |")
    if turns:
        lines.append(f"| Turns Used | {sum(turns)/len(turns):.1f} | {min(turns)} | {max(turns)} |")
    if tool_calls:
        lines.append(f"| Tool Calls | {sum(tool_calls)/len(tool_calls):.1f} | {min(tool_calls)} | {max(tool_calls)} |")
    if format_avgs:
        lines.append(f"| Format Avg | {sum(format_avgs)/len(format_avgs):.4f} | {min(format_avgs):.4f} | {max(format_avgs):.4f} |")
    if refreshes:
        lines.append(f"| Context Refreshes | {sum(refreshes)/len(refreshes):.1f} | {min(refreshes)} | {max(refreshes)} |")

    lines.append("")

    # Per-task details
    lines.append("## Per-Task Details")
    lines.append("")

    for r in task_results:
        lines.append(f"### {r.task_id}")
        lines.append("")
        lines.append(f"**Description**: {r.task_description}")
        lines.append(f"**Success**: {r.success}")
        lines.append(f"**Error**: {r.error or 'None'}")
        lines.append(f"**Duration**: {r.duration_s:.1f}s")
        lines.append(f"**Turns**: {r.turns_used}")
        lines.append(f"**Tool Calls**: {r.tool_calls}")
        lines.append(f"**Tool Breakdown**: {json.dumps(r.tool_calls_breakdown, indent=2)}")
        lines.append(f"**Format Monitor Avg**: {r.format_monitor_avg:.4f}")
        lines.append(f"**Context Refreshes**: {r.context_refreshes}")
        lines.append(f"**Phases Reached**: {', '.join(sorted(r.phases_reached)) if r.phases_reached else 'N/A'}")
        lines.append(f"**Files Changed**: {', '.join(r.files_changed) if r.files_changed else 'None'}")
        lines.append("")

    # Tool usage summary
    lines.append("## Tool Usage Summary")
    lines.append("")
    all_tool_counts: dict[str, int] = {}
    for r in task_results:
        for tool_name, count in r.tool_calls_breakdown.items():
            all_tool_counts[tool_name] = all_tool_counts.get(tool_name, 0) + count

    lines.append("| Tool | Total Calls |")
    lines.append("|------|------------|")
    for tool_name, count in sorted(all_tool_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {tool_name} | {count} |")

    lines.append("")
    return "\n".join(lines)


def save_results(
    task_results: list[AgentEvalResult],
    profile_name: str,
    output_dir: str,
    total_elapsed: float,
):
    """Save per-task JSON results and summary report.

    Args:
        task_results: List of AgentEvalResult from each task run.
        profile_name: The model profile name used.
        output_dir: Directory to save results.
        total_elapsed: Total wall-clock time for the suite.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save per-task JSON files
    for r in task_results:
        filename = f"agent_{r.task_id}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(r.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved {filepath}")

    # Save combined results as JSON
    combined = {
        "profile": profile_name,
        "date": datetime.now().isoformat(),
        "total_duration_s": round(total_elapsed, 2),
        "tasks": [r.to_dict() for r in task_results],
    }
    combined_path = os.path.join(output_dir, "agent_results_combined.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    logger.info(f"Saved {combined_path}")

    # Save summary report markdown
    report = build_report(task_results, profile_name, total_elapsed)
    report_path = os.path.join(output_dir, "summary.md")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved {report_path}")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - {combined_path}")
    print(f"  - {report_path}")
    print(f"  - {len(task_results)} per-task JSON files")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_suite(
    profile_name: str,
    runs: int = 1,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    task_timeout: int = DEFAULT_TASK_TIMEOUT,
    profile: Optional[ModelProfile] = None,
):
    """Run the full agent evaluation suite.

    Args:
        profile_name: Name of the model profile to use.
        runs: Number of times to run each task (default 1).
        output_dir: Directory to save results.
        task_timeout: Maximum seconds per task run.
        profile: Pre-loaded profile (optional, else loaded from name).
    """
    _ensure_dirs()

    # Load profile
    if profile is None:
        profile = load_profile(profile_name)

    print(f"Agent Loop Evaluation Suite")
    print(f"{'=' * 50}")
    print(f"Profile: {profile_name}")
    print(f"Model: {profile.name}")
    print(f"Tasks: {len(TASKS)}")
    print(f"Runs per task: {runs}")
    print(f"Task timeout: {task_timeout}s")
    print(f"Output dir: {output_dir}")
    print()

    # Create harness
    harness = AgentEvalHarness(profile=profile, max_iterations=3)

    all_results: list[AgentEvalResult] = []
    suite_start = time.perf_counter()

    for run_idx in range(1, runs + 1):
        print(f"\n--- Run {run_idx}/{runs} ---")
        for task in TASKS:
            task_id = task["task_id"]
            task_desc = task["task_description"]
            verifier = task["verify"]

            print(f"\n  [{task_id}] Starting task...")

            result = await harness.run_task(
                task_id=task_id,
                task_description=task_desc,
                verifier=verifier,
                output_dir=AGENT_TASKS_DIR,
                run_timeout=task_timeout,
            )

            all_results.append(result)

            status_icon = "✅" if result.success else "❌"
            print(f"  [{task_id}] {status_icon} Done: {result.duration_s:.1f}s, "
                  f"{result.turns_used} turns, {result.tool_calls} tool calls"
                  f"{' - ' + result.error if result.error else ''}")

    suite_elapsed = time.perf_counter() - suite_start
    print(f"\n{'=' * 50}")
    print(f"Suite complete in {suite_elapsed:.1f}s ({suite_elapsed / 60:.1f} min)")

    # Save results
    save_results(all_results, profile_name, output_dir, suite_elapsed)

    return all_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Loop Evaluation Suite — tests the localite coding agent through its 5-phase loop",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="gemma4_e4b",
        help="Model profile name (default: gemma4_e4b)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per task (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=DEFAULT_TASK_TIMEOUT,
        help=f"Timeout per task in seconds (default: {DEFAULT_TASK_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


async def main_async(argv: list[str] | None = None):
    """Async main entry point."""
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    await run_suite(
        profile_name=args.profile,
        runs=args.runs,
        output_dir=args.output,
        task_timeout=args.task_timeout,
    )


def main(argv: list[str] | None = None):
    """Synchronous entry point."""
    asyncio.run(main_async(argv))


if __name__ == "__main__":
    main()