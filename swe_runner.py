#!/usr/bin/env python3
"""SWE-bench Lite evaluation harness for localite agent.

Runs a localite agent on SWE-bench Lite dev instances,
collects git diff output, scores against reference patches,
and writes results to results/swe_bench/.

Usage:
    python3 swe_runner.py --instances instance_id1 instance_id2
    python3 swe_runner.py --max-instances 3
"""

import argparse
import asyncio
import difflib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger("SWE-RUNNER")

# Per-logger DEBUG for localite submodules
logging.getLogger('localite.loop.agent_loop').setLevel(logging.DEBUG)
logging.getLogger('localite.model.client').setLevel(logging.DEBUG)
logging.getLogger('SWE-RUNNER').setLevel(logging.DEBUG)

# Suppress noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# Results directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "swe_bench")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Localite imports ---
from localite.config import ConfigLoader
from localite.model.client import AsyncOllamaClient
from localite.permissions.gate import PermissionGate
from localite.episodes.store import EpisodeStore
from localite.code_index import CodeIndex
from localite.loop.agent_loop import AgentLoop
from localite.tools.base import BaseTool
from localite.tools.read import ReadFileTool
from localite.tools.write import WriteFileTool
from localite.tools.edit import EditFileTool
from localite.tools.search import GrepSearchTool
from localite.tools.shell import RunShellTool
from localite.tools.list_files import ListFilesTool
from localite.tools.test_executor import TestExecutorTool
from localite.tools.diff_view import DiffViewTool
from localite.tools.task_complete import TaskCompleteTool
from localite.tools.memory_tools import MemoryReadTool, MemoryWriteTool
from localite.memory.memory_store import EpisodicMemoryStore
from localite.context.standing_instructions import StandingInstructions
from swebench.harness.grading import (
    compute_fail_to_pass,
    compute_pass_to_pass,
    get_resolution_status,
)


# ============================================================
# Dataset loading
# ============================================================

def load_dev_instances(max_instances: int = None, specific_ids: list[str] = None) -> list[dict]:
    """Load SWE-bench Lite dev split instances.

    Args:
        max_instances: Maximum number of instances to load (None = all).
        specific_ids: If given, only load these instance IDs.

    Returns:
        List of instance dicts from the dataset.
    """
    from datasets import load_dataset

    logger.info("Loading SWE-bench Lite dev split...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
    logger.info(f"Loaded {len(ds)} dev instances")

    instances = []
    for inst in ds:
        instance_id = inst["instance_id"]
        if specific_ids and instance_id not in specific_ids:
            continue
        instances.append(inst)
        if max_instances and len(instances) >= max_instances:
            break

    logger.info(f"Selected {len(instances)} instances for evaluation: {[i['instance_id'] for i in instances]}")
    return instances


# ============================================================
# Repo management
# ============================================================

REPOS_DIR = os.path.join(PROJECT_ROOT, "results", "swe_bench", "repos")
os.makedirs(REPOS_DIR, exist_ok=True)


def clone_repo(repo_name: str, base_commit: str) -> str:
    """Clone a GitHub repo at a specific commit.

    Args:
        repo_name: Full repo name (e.g. "marshmallow-code/marshmallow").
        base_commit: Git commit hash to check out.

    Returns:
        Path to the cloned repo workdir.
    """
    safe_name = repo_name.replace("/", "__")
    workdir = os.path.join(REPOS_DIR, safe_name)

    if os.path.exists(workdir):
        logger.info(f"Repo {repo_name} already exists at {workdir}, resetting...")
        # Reset to base_commit
        subprocess.run(
            ["git", "checkout", "--force", base_commit],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=30,
        )
    else:
        logger.info(f"Cloning {repo_name} to {workdir}...")
        clone_url = f"https://github.com/{repo_name}.git"
        result = subprocess.run(
            ["git", "clone", clone_url, workdir],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone {repo_name}: {result.stderr}")

        # Checkout base commit
        result = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to checkout {base_commit}: {result.stderr}")

    logger.info(f"Repo ready at {workdir} (commit {base_commit})")
    return workdir


def get_git_diff(workdir: str) -> str:
    """Get git diff (unstaged changes) from the repo workdir.

    Returns:
        String of the diff, or empty string if no changes.
    """
    result = subprocess.run(
        ["git", "diff"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def apply_patch(workdir: str, patch_text: str) -> bool:
    """Apply a patch to the repo workdir.

    Args:
        workdir: Path to repo.
        patch_text: The patch content to apply.

    Returns:
        True if patch applied successfully, False otherwise.
    """
    if not patch_text.strip():
        return False
    try:
        result = subprocess.run(
            ["git", "apply", "-"],
            cwd=workdir,
            input=patch_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"Patch apply failed: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Patch apply exception: {e}")
        return False


def install_repo_deps(workdir: str, timeout: int = 120) -> dict:
    """Install the repo's package as editable in the current environment.

    Checks for setup.py/pyproject.toml/setup.cfg and runs pip install -e .
    This is critical for cross-repo evaluation — fresh repos need their
    package installed before pytest can find the module.

    Args:
        workdir: Path to the cloned repo.
        timeout: Timeout in seconds for the pip install subprocess.

    Returns:
        Dict with keys: success (bool), command_used (str), output (str).
    """
    setup_files = []
    for f in ["setup.py", "pyproject.toml", "setup.cfg"]:
        if os.path.isfile(os.path.join(workdir, f)):
            setup_files.append(f)

    if not setup_files:
        logger.info(f"  No setup files found in {workdir}, skipping pip install")
        return {"success": False, "command_used": None, "output": "No setup files found"}

    cmd = [sys.executable, "-m", "pip", "install", "-e", workdir]
    logger.info(f"  Installing repo deps: {' '.join(cmd)} (timeout={timeout}s)")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        if success:
            logger.info(f"  Repo deps installed successfully")
        else:
            logger.warning(f"  pip install failed (rc={result.returncode}): {result.stderr[:200]}")
        return {
            "success": success,
            "command_used": " ".join(cmd),
            "output": output[-1000:],
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"  pip install timed out after {timeout}s")
        return {"success": False, "command_used": " ".join(cmd), "output": f"Timed out after {timeout}s"}
    except Exception as e:
        logger.warning(f"  pip install exception: {e}")
        return {"success": False, "command_used": " ".join(cmd), "output": str(e)}


def run_pytest(workdir: str, timeout: int = 120) -> dict:
    """Run pytest in the repo workdir.

    Uses sys.executable (the current Python interpreter, e.g. from venv)
    so that dependencies like simplejson are found.

    Returns:
        Dict with keys: passed (bool), output (str), returncode (int), tests_passed, tests_failed.
    """
    try:
        python_exe = sys.executable
        result = subprocess.run(
            [python_exe, "-m", "pytest", "-x", "--tb=short", "-q"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + "\n" + result.stderr

        # Parse test counts from pytest output
        tests_passed = 0
        tests_failed = 0
        for line in result.stdout.split("\n"):
            if "passed" in line and "failed" in line:
                # e.g. "3 passed, 0 failed"
                parts = line.strip().split(",")
                for part in parts:
                    if "passed" in part:
                        tests_passed = int(part.strip().split()[0])
                    if "failed" in part:
                        tests_failed = int(part.strip().split()[0])
            elif "passed" in line and "failed" not in line:
                try:
                    tests_passed = int(line.strip().split()[0])
                except (ValueError, IndexError):
                    pass

        return {
            "passed": result.returncode == 0,
            "output": output,
            "returncode": result.returncode,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "output": "TIMEOUT",
            "returncode": -1,
            "tests_passed": 0,
            "tests_failed": 0,
        }
    except Exception as e:
        return {
            "passed": False,
            "output": str(e),
            "returncode": -1,
            "tests_passed": 0,
            "tests_failed": 0,
        }


def compute_patch_similarity(agent_diff: str, reference_patch: str) -> float:
    """Compute similarity between agent's diff and reference patch.

    Uses difflib.SequenceMatcher ratio on the diff texts.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not agent_diff and not reference_patch:
        return 1.0
    if not agent_diff or not reference_patch:
        return 0.0
    matcher = difflib.SequenceMatcher(None, agent_diff, reference_patch)
    return matcher.ratio()


def evaluate_instance(instance: dict, workdir: str, timeout: int = 120) -> dict:
    """Evaluate an instance using SWE-bench F2P/P2P grading.

    Parses FAIL_TO_PASS and PASS_TO_PASS from the instance dict,
    runs pytest -v on the relevant test files, determines which
    specific test IDs passed/failed, builds a SWE-bench report dict,
    and computes scores via swebench.harness.grading.

    Args:
        instance: Dataset instance dict with FAIL_TO_PASS and PASS_TO_PASS.
        workdir: Repo workdir to run tests in.
        timeout: Timeout in seconds for pytest.

    Returns:
        Dict with:
            f2p_report: {success: [...], failure: [...], total: int}
            p2p_report: {success: [...], failure: [...], total: int}
            f2p_score: float (ratio of F2P that passed)
            p2p_score: float (ratio of P2P that passed)
            resolution_status: "FULL"|"PARTIAL"|"NO"
            stdout: raw pytest output
    """
    # Parse F2P and P2P test ID lists from the instance
    f2p_raw = instance.get("FAIL_TO_PASS", "[]")
    p2p_raw = instance.get("PASS_TO_PASS", "[]")
    if isinstance(f2p_raw, str):
        f2p_ids = json.loads(f2p_raw)
    else:
        f2p_ids = f2p_raw or []
    if isinstance(p2p_raw, str):
        p2p_ids = json.loads(p2p_raw)
    else:
        p2p_ids = p2p_raw or []

    logger.info(f"  F2P IDs ({len(f2p_ids)}): {f2p_ids}")
    logger.info(f"  P2P IDs ({len(p2p_ids)}): {p2p_ids[:3]}... ({len(p2p_ids)} total)")

    # Collect unique test file paths from test IDs
    test_files = set()
    for tid in f2p_ids + p2p_ids:
        # test ID format: "path/to/test_file.py::TestClass::test_method" or "path/to/test_file.py::test_func"
        parts = tid.split("::")
        if parts:
            test_files.add(parts[0])

    logger.info(f"  Test files to run ({len(test_files)}): {sorted(test_files)}")

    # Build the pytest command for the specific test files
    # Use -v for verbose output with test IDs, --tb=short for tracebacks, -q for summary
    python_exe = sys.executable
    cmd = [python_exe, "-m", "pytest", "-v", "--tb=short"] + sorted(test_files)

    default_report = {
        "f2p_report": {"success": [], "failure": [], "total": len(f2p_ids)},
        "p2p_report": {"success": [], "failure": [], "total": len(p2p_ids)},
        "f2p_score": 0.0,
        "p2p_score": 0.0,
        "resolution_status": "NO",
        "stdout": "No tests to run (F2P/P2P both empty)",
    }

    if not f2p_ids and not p2p_ids:
        logger.info("  No F2P or P2P tests to evaluate")
        return default_report

    try:
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout + "\n" + result.stderr
        lines = result.stdout.split("\n")

        # Build a set of tests that passed/failed from verbose output
        # In pytest -v output, we see lines like:
        #   tests/test_fields.py::TestParentAndName::test_datetime_list_inner_format PASSED [ 27%]
        #   tests/test_fields.py::TestField::test_repr FAILED
        #   tests/test_fields.py::TestField::test_error_raised ERROR
        # The progress indicator [xx%] is appended after PASSED/FAILED/ERROR, so
        # we need to check for the keyword within the line, not just at the end.
        passed_tests = set()
        failed_tests = set()

        for line in lines:
            line_stripped = line.strip()
            # Extract test ID by finding PASSED/FAILED/ERROR keywords within the line
            # and taking everything before the keyword as the test ID
            for keyword in (" PASSED", " FAILED", " ERROR"):
                if keyword in line_stripped:
                    idx = line_stripped.find(keyword)
                    test_id = line_stripped[:idx].strip()
                    if test_id:
                        if keyword == " PASSED":
                            passed_tests.add(test_id)
                        else:
                            failed_tests.add(test_id)
                    break

        # Build report dict as expected by swebench.harness.grading
        # compute_fail_to_pass and compute_pass_to_pass expect:
        #   { "FAIL_TO_PASS": {"success": [...], "failure": [...]}, "PASS_TO_PASS": {...} }
        # Use flexible matching: for parametrized tests (containing '['), the
        # dataset IDs may be truncated (e.g., "test_make_error[required-Missing"
        # instead of "test_make_error[required-Missing data for required field.]"),
        # so we match by prefix.
        def match_tid(dataset_id: str, pytest_id: str) -> bool:
            """Match a dataset test ID to a pytest output test ID."""
            if dataset_id == pytest_id:
                return True
            # For parametrized tests, dataset ID may be a prefix
            if "[" in dataset_id:
                return pytest_id.startswith(dataset_id)
            return False

        f2p_success = [tid for tid in f2p_ids if any(match_tid(tid, pt) for pt in passed_tests)]
        f2p_failure = [tid for tid in f2p_ids if tid not in f2p_success]

        p2p_success = [tid for tid in p2p_ids if any(match_tid(tid, pt) for pt in passed_tests)]
        p2p_failure = [tid for tid in p2p_ids if any(match_tid(tid, ft) for ft in failed_tests) and tid not in p2p_success]

        report = {
            "FAIL_TO_PASS": {
                "success": f2p_success,
                "failure": f2p_failure,
                "total": len(f2p_ids),
            },
            "PASS_TO_PASS": {
                "success": p2p_success,
                "failure": p2p_failure,
                "total": len(p2p_ids),
            },
        }

        # Compute scores using swebench grading functions
        f2p_score = compute_fail_to_pass(report)
        p2p_score = compute_pass_to_pass(report)
        resolution = get_resolution_status(report)

        logger.info(f"  F2P: {len(f2p_success)}/{len(f2p_ids)} passed, "
                    f"P2P: {len(p2p_success)}/{len(p2p_ids)} passed, "
                    f"Resolution: {resolution}")

        # Normalize test IDs that might have different formats
        # (e.g., tests collected via parametrize may show different IDs than expected)
        # Check if we got zero matches but tests actually ran
        all_known_ids = set(f2p_ids + p2p_ids)
        matched_ids = set(f2p_success + f2p_failure + p2p_success + p2p_failure)
        if len(matched_ids) < len(all_known_ids):
            # Some test IDs were not matched in the output
            # Try matching by test name (last part of :: separated ID)
            # First check if any tests were found at all
            if not passed_tests and not failed_tests:
                logger.warning("  No test results parsed from pytest output — output may have different format")
                logger.warning(f"  pytest stdout (first 20 lines): {output[:1000]}")

        return {
            "f2p_report": {"success": f2p_success, "failure": f2p_failure, "total": len(f2p_ids)},
            "p2p_report": {"success": p2p_success, "failure": p2p_failure, "total": len(p2p_ids)},
            "f2p_score": f2p_score,
            "p2p_score": p2p_score,
            "resolution_status": resolution,
            "stdout": output,
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"  pytest timed out after {timeout}s for F2P/P2P evaluation")
        return {
            "f2p_report": {"success": [], "failure": f2p_ids, "total": len(f2p_ids)},
            "p2p_report": {"success": [], "failure": p2p_ids, "total": len(p2p_ids)},
            "f2p_score": 0.0,
            "p2p_score": 0.0,
            "resolution_status": "NO",
            "stdout": "TIMEOUT",
        }
    except Exception as e:
        logger.error(f"  F2P/P2P evaluation error: {e}")
        return {
            "f2p_report": {"success": [], "failure": f2p_ids, "total": len(f2p_ids)},
            "p2p_report": {"success": [], "failure": p2p_ids, "total": len(p2p_ids)},
            "f2p_score": 0.0,
            "p2p_score": 0.0,
            "resolution_status": "NO",
            "stdout": str(e),
        }


# ============================================================
# Agent creation
# ============================================================

def create_swe_agent(workdir: str) -> AgentLoop:
    """Create a localite AgentLoop configured for SWE-bench.

    Follows the test_e2e_realtime.py pattern:
    - Create tools with workdir set to the repo
    - PermissionGate with auto_approve=True
    - AsyncOllamaClient for gemma4:e4b
    - AgentLoop with max 20 turns

    Args:
        workdir: Repo workdir path to set as agent's working directory.

    Returns:
        Configured AgentLoop instance.
    """
    # 1. Create all tools
    tools: dict[str, BaseTool] = {}

    read_tool = ReadFileTool()
    read_tool.workdir = workdir
    tools[read_tool.name] = read_tool

    write_tool = WriteFileTool()
    write_tool.workdir = workdir
    tools[write_tool.name] = write_tool

    edit_tool = EditFileTool()
    edit_tool.workdir = workdir
    tools[edit_tool.name] = edit_tool

    search_tool = GrepSearchTool()
    search_tool.workdir = workdir
    tools[search_tool.name] = search_tool

    shell_tool = RunShellTool()
    shell_tool.workdir = workdir
    tools[shell_tool.name] = shell_tool

    list_tool = ListFilesTool()
    list_tool.workdir = workdir
    tools[list_tool.name] = list_tool

    # Update shell_tool description to discourage file listing
    shell_tool.description = (
        "Execute a shell command (e.g., running scripts, installing packages, git commands). "
        "Do NOT use for file listing — use list_files instead."
    )

    test_tool = TestExecutorTool()
    test_tool.workdir = workdir
    tools[test_tool.name] = test_tool

    diff_tool = DiffViewTool()
    diff_tool.workdir = workdir
    tools[diff_tool.name] = diff_tool

    tools["task_complete"] = TaskCompleteTool()
    tools["memory_read"] = MemoryReadTool()
    tools["memory_write"] = MemoryWriteTool()

    # Create ctags-based CodeIndex for symbol-guided navigation
    code_index = CodeIndex(workdir)

    # 2. Permission gate: auto_approve so no user prompts
    gate = PermissionGate(auto_approve=True)

    # 3. Episode store
    store = EpisodeStore()

    # 4. Load profile for gemma4_e4b
    config_loader = ConfigLoader()
    profile = config_loader.load_profile("gemma4_e4b")

    # 5. Model client
    model = AsyncOllamaClient(
        model_name="gemma4:e4b",
        base_url=profile.base_url,
        timeout=profile.timeout,
        has_thinking_tags=profile.has_thinking_tags,
    )

    # 6. Standing instructions
    standing_instructions = StandingInstructions()

    # 7. Memory store
    memory_dir = os.path.join(PROJECT_ROOT, "results", "swe_bench", "memory")
    memory_store = EpisodicMemoryStore(base_dir=memory_dir)

    # Wire memory store into memory tools
    if "memory_read" in tools:
        tools["memory_read"]._memory_store = memory_store
    if "memory_write" in tools:
        tools["memory_write"]._memory_store = memory_store

    # 8. Build loop with profile max_turns
    loop = AgentLoop(
        model_client=model,
        tools=tools,
        permission_gate=gate,
        episode_store=store,
        model_profile=profile,
        standing_instructions=standing_instructions,
        max_iterations=profile.max_turns,
        memory_store=memory_store,
        code_index=code_index,
    )

    logger.info(f"AgentLoop created for workdir={workdir}")
    return loop


# ============================================================
# Instance execution
# ============================================================

async def run_instance(instance: dict, agent_timeout: int = 900) -> dict:
    """Run a single SWE-bench instance.

    Args:
        instance: Dataset instance dict.
        agent_timeout: Maximum time (seconds) for the agent to run.

    Returns:
        Result dict with all scoring data.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    issue_text = instance.get("problem_statement", instance.get("issue_text", ""))
    reference_patch = instance.get("patch", "")
    test_patch = instance.get("test_patch", "")

    logger.info(f"\n{'='*60}")
    logger.info(f"Running instance: {instance_id}")
    logger.info(f"  Repo: {repo} @ {base_commit[:12]}")
    logger.info(f"  Issue: {issue_text[:100]}...")
    logger.info(f"{'='*60}")

    result = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "status": "running",
        "turns_used": 0,
        "wall_time_sec": 0,
        "agent_diff": "",
        "patch_similarity": 0.0,
        "test_patch_applied": False,
        "tests_before": None,
        "tests_after": None,
        "test_pass_rate": 0.0,
        "agent_errors": [],
        "reference_patch_length": len(reference_patch),
        "agent_diff_length": 0,
        "phase_sequence": [],
        "files_changed": [],
        "summary": "",
    }

    start_time = time.time()

    try:
        # Step 1: Clone repo at base_commit
        workdir = clone_repo(repo, base_commit)
        result["workdir"] = workdir

        # Step 1b: Install repo as editable so pytest can find the module
        repo_install = install_repo_deps(workdir)
        result["repo_install"] = repo_install
        if not repo_install["success"]:
            logger.warning(f"  Repo install skipped/failed — tests may fail with ModuleNotFoundError")

        # Step 2: Run tests before (without test_patch) to see baseline
        # Skip if install failed — tests are guaranteed to fail with ModuleNotFoundError
        if repo_install.get("success", False):
            logger.info(f"  Running baseline tests...")
            tests_before = run_pytest(workdir, timeout=120)
        else:
            logger.warning(f"  Skipping tests_before (install failed — module not importable)")
            tests_before = {
                "passed": False,
                "output": "SKIPPED — install_repo_deps() failed, module not importable",
                "returncode": -1,
                "tests_passed": 0,
                "tests_failed": 0,
            }
        result["tests_before"] = tests_before

        # Step 3: Create and run the agent
        loop = create_swe_agent(workdir)

        # Prepare issue text for the agent
        agent_task = f"""You are working in the repository at {workdir}.

Your task is to resolve the following GitHub issue:

{issue_text}

Work in the repository directory. Use the available tools to explore, understand,
and modify the code. After making changes, run the tests to verify.

When you are done, call task_complete with a summary of what you changed."""

        logger.info(f"  Starting agent loop (timeout={agent_timeout}s)...")

        # Run with timeout via asyncio.wait_for
        try:
            agent_result = await asyncio.wait_for(
                loop.run(agent_task),
                timeout=agent_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"  Agent timed out after {agent_timeout}s")
            result["status"] = "timeout"
            result["agent_errors"].append(f"Agent timed out after {agent_timeout}s")
            # Get partial results if possible
            result["turns_used"] = len(loop.episode.turns) if loop.episode else 0
            result["phase_sequence"] = [
                getattr(t, "phase", "?") for t in (loop.episode.turns if loop.episode else [])
            ]
        except Exception as e:
            logger.warning(f"  Agent error: {type(e).__name__}: {e}")
            result["status"] = "agent_error"
            result["agent_errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
            result["turns_used"] = len(loop.episode.turns) if loop.episode else 0
        else:
            result["status"] = "completed"
            # Collect agent results
            if loop.episode:
                result["turns_used"] = len(loop.episode.turns)
                result["phase_sequence"] = [getattr(t, "phase", "?") for t in loop.episode.turns]
                result["files_changed"] = loop.episode.files_changed or []
                result["summary"] = loop.episode.summary or agent_result.get("summary", "")
            result["agent_result_phase"] = agent_result.get("phase", "")

        # Step 4: Collect git diff from agent's changes
        agent_diff = get_git_diff(workdir)
        result["agent_diff"] = agent_diff
        result["agent_diff_length"] = len(agent_diff)

        # Step 5: Compute patch similarity
        result["patch_similarity"] = compute_patch_similarity(agent_diff, reference_patch)

        # Step 6: Run tests after agent (no test_patch) — legacy baseline
        logger.info(f"  Running baseline tests after agent changes (full suite)...")
        tests_after = run_pytest(workdir, timeout=120)
        result["tests_after"] = tests_after

        # Compute legacy test pass rate (tests that passed / total tests)
        total_after = tests_after.get("tests_passed", 0) + tests_after.get("tests_failed", 0)
        if total_after > 0:
            result["test_pass_rate"] = tests_after.get("tests_passed", 0) / total_after

        # Step 6b: F2P/P2P evaluation on agent's changes
        logger.info(f"  Running F2P/P2P evaluation on agent's changes...")
        f2p_p2p_result = evaluate_instance(instance, workdir, timeout=120)
        result["f2p_p2p_report"] = f2p_p2p_result
        result["resolution_status"] = f2p_p2p_result["resolution_status"]
        result["f2p_score"] = f2p_p2p_result["f2p_score"]
        result["p2p_score"] = f2p_p2p_result["p2p_score"]

        # Step 7: Try applying test_patch and running F2P/P2P evaluation with gold tests
        if test_patch.strip():
            # First reset the repo to base_commit for clean testing
            subprocess.run(
                ["git", "checkout", "--force", base_commit],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Re-apply agent's diff
            if agent_diff:
                apply_patch(workdir, agent_diff)

            # Now apply test_patch
            test_applied = apply_patch(workdir, test_patch)
            result["test_patch_applied"] = test_applied

            if test_applied:
                logger.info(f"  Running F2P/P2P evaluation with test_patch applied (gold tests)...")
                # Run full test suite for legacy comparison
                tests_with_test_patch = run_pytest(workdir, timeout=120)
                result["tests_with_test_patch"] = tests_with_test_patch

                total = tests_with_test_patch.get("tests_passed", 0) + tests_with_test_patch.get("tests_failed", 0)
                if total > 0:
                    result["test_patch_pass_rate"] = tests_with_test_patch.get("tests_passed", 0) / total

                # Also run F2P/P2P evaluation with test_patch to see if gold tests pass
                gold_eval = evaluate_instance(instance, workdir, timeout=120)
                result["gold_f2p_p2p_report"] = gold_eval
            else:
                result["tests_with_test_patch"] = {"passed": False, "output": "Failed to apply test_patch", "returncode": -1}
                result["test_patch_pass_rate"] = 0.0
                result["gold_f2p_p2p_report"] = None
        else:
            result["test_patch_pass_rate"] = None
            result["tests_with_test_patch"] = None
            result["gold_f2p_p2p_report"] = None

        if result["status"] == "running":
            result["status"] = "completed"

    except Exception as e:
        logger.error(f"  Fatal error on {instance_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        result["status"] = "error"
        result["agent_errors"].append(f"Fatal: {type(e).__name__}: {str(e)[:500]}")

    finally:
        result["wall_time_sec"] = time.time() - start_time
        logger.info(f"  Instance {instance_id} finished: status={result['status']}, "
                    f"wall_time={result['wall_time_sec']:.0f}s, "
                    f"turns={result['turns_used']}, "
                    f"diff_len={result['agent_diff_length']}, "
                    f"patch_sim={result['patch_similarity']:.3f}")

    return result


# ============================================================
# Results saving
# ============================================================

def save_result(result: dict):
    """Save a single instance result to a JSON file."""
    instance_id = result["instance_id"]
    safe_id = instance_id.replace("/", "__")
    path = os.path.join(RESULTS_DIR, f"{safe_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Result saved to {path}")


def save_combined_results(all_results: list[dict]):
    """Save all results combined into a single JSON file."""
    path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Combined results saved to {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite evaluation harness for localite agent"
    )
    parser.add_argument(
        "--instances", nargs="+",
        help="Specific instance IDs to run (e.g., marshmallow__marshmallow-1083)"
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="Maximum number of instances to run"
    )
    parser.add_argument(
        "--agent-timeout", type=int, default=900,
        help="Timeout per instance in seconds (default: 900)"
    )
    args = parser.parse_args()

    # Add file handler for debug log capture
    debug_log_path = os.path.join(RESULTS_DIR, "debug_run_1359.log")
    file_handler = logging.FileHandler(debug_log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Debug log will be saved to {debug_log_path}")

    logger.info(f"SWE-bench Lite Evaluation Harness")
    logger.info(f"  Instances: {args.instances or 'all'}")
    logger.info(f"  Max instances: {args.max_instances or 'unlimited'}")
    logger.info(f"  Agent timeout: {args.agent_timeout}s")
    logger.info(f"  Results dir: {RESULTS_DIR}")

    # Load instances
    instances = load_dev_instances(
        max_instances=args.max_instances,
        specific_ids=args.instances,
    )

    if not instances:
        logger.error("No instances to evaluate!")
        sys.exit(1)

    logger.info(f"Running {len(instances)} instances...")

    # Run each instance sequentially
    all_results = []
    for i, instance in enumerate(instances):
        logger.info(f"\n{'#'*60}")
        logger.info(f"Instance {i+1}/{len(instances)}: {instance['instance_id']}")
        logger.info(f"{'#'*60}")

        result = asyncio.run(run_instance(instance, agent_timeout=args.agent_timeout))
        save_result(result)
        all_results.append(result)

    # Save combined results
    save_combined_results(all_results)

    # Print summary table
    logger.info(f"\n{'='*70}")
    logger.info(f"{'SUMMARY':^70}")
    logger.info(f"{'='*70}")
    logger.info(f"{'Instance':35} {'Status':12} {'Turns':6} {'Time(s)':8} {'Patch Sim':10}")
    logger.info(f"{'-'*70}")
    for r in all_results:
        sim = r.get("patch_similarity", 0)
        logger.info(f"{r['instance_id']:34} {r['status']:12} {r['turns_used']:6} "
                    f"{r['wall_time_sec']:7.0f} {sim:8.3f}")
    logger.info(f"{'='*70}")

    resolved = sum(1 for r in all_results if r.get("patch_similarity", 0) > 0.5)
    logger.info(f"Resolved (sim > 0.5): {resolved}/{len(all_results)}")

    logger.info(f"\nDone! Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""SWE-bench Lite evaluation harness for localite agent.

Runs a localite agent on SWE-bench Lite dev instances,
collects git diff output, scores against reference patches,
and writes results to results/swe_bench/.

Usage:
    python3 swe_runner.py --instances instance_id1 instance_id2
    python3 swe_runner.py --max-instances 3
"""

