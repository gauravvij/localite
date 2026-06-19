"""Test suite for AgentLoop._parse_tool_call — all 6 parser formats.

Run with:
    python -m pytest tests/test_tool_call_parsing.py -v

No real API key or network connection required.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localite.config import ConfigLoader
from localite.loop.agent_loop import AgentLoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROFILE_NAME = os.environ.get("LOCALITE_TEST_PROFILE", "deepseek_v4_flash_v6")

TOOL_NAMES = [
    "list_files",
    "read_file",
    "grep_search",
    "edit_file",
    "write_file",
    "run_shell",
    "test_executor",
    "task_complete",
]


def _make_mock_tool(name: str) -> MagicMock:
    """Create a minimal mock tool object."""
    t = MagicMock()
    t.name = name
    return t


@pytest.fixture(scope="module")
def loop():
    """Instantiate a minimal AgentLoop with mock dependencies."""
    loader = ConfigLoader(profiles_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "profiles",
    ))
    profile = loader.load_profile(PROFILE_NAME)

    mock_tools = {name: _make_mock_tool(name) for name in TOOL_NAMES}

    mock_client = MagicMock()
    mock_gate = MagicMock()
    mock_gate.check.return_value = MagicMock(decision="approved", modified_tool_call=None)
    mock_store = MagicMock()

    agent = AgentLoop(
        model_client=mock_client,
        tools=mock_tools,
        permission_gate=mock_gate,
        episode_store=mock_store,
        model_profile=profile,
    )
    return agent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def parse(loop_fixture, text: str):
    """Shorthand to call _parse_tool_call."""
    return loop_fixture._parse_tool_call(text)


# ---------------------------------------------------------------------------
# Format 1 — flat JSON with "tool" key
# ---------------------------------------------------------------------------

class TestFormat1:
    def test_read_file(self, loop):
        result = parse(loop, '{"tool": "read_file", "arguments": {"path": "/foo.py"}}')
        assert result is not None
        assert result["name"] == "read_file"
        assert result["args"]["path"] == "/foo.py"

    def test_edit_file(self, loop):
        result = parse(loop, '{"tool": "edit_file", "arguments": {"path": "/bar.py", "search_text": "old", "replace_text": "new"}}')
        assert result is not None
        assert result["name"] == "edit_file"
        assert result["args"]["search_text"] == "old"
        assert result["args"]["replace_text"] == "new"

    def test_list_files(self, loop):
        result = parse(loop, '{"tool": "list_files", "arguments": {"path": "/src", "depth": 2}}')
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "/src"


# ---------------------------------------------------------------------------
# Format 1b — flat JSON with "tool_name" key
# ---------------------------------------------------------------------------

class TestFormat1b:
    def test_list_files(self, loop):
        result = parse(loop, '{"tool_name": "list_files", "params": {"path": "/src"}}')
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "/src"

    def test_grep_search(self, loop):
        result = parse(loop, '{"tool_name": "grep_search", "params": {"pattern": "def foo", "path": "."}}')
        assert result is not None
        assert result["name"] == "grep_search"
        assert result["args"]["pattern"] == "def foo"


# ---------------------------------------------------------------------------
# Format 2 — LFM2.5 native <|tool_call_start|>
# ---------------------------------------------------------------------------

class TestFormat2:
    def test_read_file(self, loop):
        result = parse(loop, "<|tool_call_start|>[read_file(path='/foo.py')]<|tool_call_end|>")
        assert result is not None
        assert result["name"] == "read_file"
        assert result["args"]["path"] == "/foo.py"

    def test_list_files(self, loop):
        result = parse(loop, "<|tool_call_start|>[list_files(path='/src', depth='2')]<|tool_call_end|>")
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "/src"


# ---------------------------------------------------------------------------
# Format 3 — Qwen tools[]
# ---------------------------------------------------------------------------

class TestFormat3:
    def test_list_files(self, loop):
        result = parse(loop, '{"tools": [{"list_files": {"path": "."}}]}')
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "."

    def test_grep_search(self, loop):
        result = parse(loop, '{"tools": [{"grep_search": {"pattern": "L031", "path": "/src"}}]}')
        assert result is not None
        assert result["name"] == "grep_search"
        assert result["args"]["pattern"] == "L031"


# ---------------------------------------------------------------------------
# Format 4 — Qwen tool_calls[]
# ---------------------------------------------------------------------------

class TestFormat4:
    def test_grep_search(self, loop):
        result = parse(loop, '{"tool_calls": [{"grep_search": {"pattern": "L031"}}]}')
        assert result is not None
        assert result["name"] == "grep_search"
        assert result["args"]["pattern"] == "L031"

    def test_read_file(self, loop):
        result = parse(loop, '{"tool_calls": [{"read_file": {"path": "/foo.py", "max_lines": 200}}]}')
        assert result is not None
        assert result["name"] == "read_file"
        assert result["args"]["path"] == "/foo.py"


# ---------------------------------------------------------------------------
# Format 5 — key-as-name
# ---------------------------------------------------------------------------

class TestFormat5:
    def test_read_file(self, loop):
        result = parse(loop, '{"read_file": {"path": "/foo.py"}}')
        assert result is not None
        assert result["name"] == "read_file"
        assert result["args"]["path"] == "/foo.py"

    def test_list_files(self, loop):
        result = parse(loop, '{"list_files": {"path": "/src", "depth": 3}}')
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "/src"


# ---------------------------------------------------------------------------
# Format 6 — naked args (signature inference)
# ---------------------------------------------------------------------------

class TestFormat6:
    def test_read_file_naked(self, loop):
        """{"path": "/foo.py", "max_lines": 100} → read_file"""
        result = parse(loop, '{"path": "/foo.py", "max_lines": 100}')
        assert result is not None
        assert result["name"] == "read_file"
        assert result["args"]["path"] == "/foo.py"
        assert result["args"]["max_lines"] == 100

    def test_list_files_naked(self, loop):
        """{"path": "/src", "depth": 2} → list_files"""
        result = parse(loop, '{"path": "/src", "depth": 2}')
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "/src"

    def test_grep_search_naked(self, loop):
        """{"pattern": "L031", "path": "/src", "glob_pattern": "*.py"} → grep_search"""
        result = parse(loop, '{"pattern": "L031", "path": "/src", "glob_pattern": "*.py", "max_results": 20}')
        assert result is not None
        assert result["name"] == "grep_search"
        assert result["args"]["pattern"] == "L031"

    def test_run_shell_naked(self, loop):
        """{"command": "pytest tests/"} → run_shell"""
        result = parse(loop, '{"command": "pytest tests/", "timeout": 60}')
        assert result is not None
        assert result["name"] == "run_shell"
        assert result["args"]["command"] == "pytest tests/"


class TestFormat6b:
    def test_edit_file_with_thought(self, loop):
        """{"thought": "...", "path": "/foo.py", "search_text": "old", "replace_text": "new"} → edit_file"""
        result = parse(loop, '{"thought": "I will edit", "path": "/foo.py", "search_text": "old", "replace_text": "new"}')
        assert result is not None
        assert result["name"] == "edit_file"
        assert result["args"]["path"] == "/foo.py"
        assert result["args"]["search_text"] == "old"
        assert result["args"]["replace_text"] == "new"
        # thought key should NOT be in args
        assert "thought" not in result["args"]

    def test_list_files_with_message(self, loop):
        """{"message": "exploring", "path": "/src", "depth": 2} → list_files"""
        result = parse(loop, '{"message": "exploring", "path": "/src", "depth": 2}')
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "/src"
        assert "message" not in result["args"]


class TestFormat6c:
    def test_double_brace_bug(self, loop):
        """{"tool": "list_files", "arguments": {"path": "/src"}}} (trailing }}) → should parse correctly"""
        result = parse(loop, '{"tool": "list_files", "arguments": {"path": "/src"}}}')
        assert result is not None
        assert result["name"] == "list_files"
        assert result["args"]["path"] == "/src"

    def test_double_brace_naked(self, loop):
        """{"path": "/foo.py", "max_lines": 50}} → read_file"""
        result = parse(loop, '{"path": "/foo.py", "max_lines": 50}}')
        assert result is not None
        assert result["name"] == "read_file"
        assert result["args"]["path"] == "/foo.py"


class TestFormat6d:
    def test_task_complete_naked(self, loop):
        """{"status": "success", "reason_code": "tests_passing", "summary": "done"} → task_complete"""
        result = parse(loop, '{"status": "success", "reason_code": "tests_passing", "summary": "done"}')
        assert result is not None
        assert result["name"] == "task_complete"
        assert result["args"]["status"] == "success"
        assert result["args"]["reason_code"] == "tests_passing"

    def test_task_complete_with_thought(self, loop):
        """task_complete with thought prefix"""
        result = parse(loop, '{"thought": "All done", "status": "success", "reason_code": "tests_passing", "summary": "Fixed the bug"}')
        assert result is not None
        assert result["name"] == "task_complete"
        assert result["args"]["status"] == "success"
        assert "thought" not in result["args"]


# ---------------------------------------------------------------------------
# Non-tool-call responses — should return None
# ---------------------------------------------------------------------------

class TestNonToolCall:
    def test_pure_message(self, loop):
        """{"thought": "thinking", "message": "hello"} → None"""
        result = parse(loop, '{"thought": "thinking", "message": "hello"}')
        assert result is None

    def test_message_only(self, loop):
        """{"message": "I need to explore first"} → None"""
        result = parse(loop, '{"message": "I need to explore first"}')
        assert result is None

    def test_reasoning_only(self, loop):
        """{"reasoning": "Let me think..."} → None"""
        result = parse(loop, '{"reasoning": "Let me think..."}')
        assert result is None


# ---------------------------------------------------------------------------
# Malformed JSON — should return None, no exception
# ---------------------------------------------------------------------------

class TestMalformedJSON:
    def test_truncated_json(self, loop):
        result = parse(loop, '{"tool": "read_file", "arguments": {"path":')
        assert result is None

    def test_empty_string(self, loop):
        result = parse(loop, "")
        assert result is None

    def test_plain_text(self, loop):
        result = parse(loop, "I will now read the file to understand the issue.")
        assert result is None

    def test_invalid_json(self, loop):
        result = parse(loop, "{not valid json at all!!!}")
        assert result is None


# ---------------------------------------------------------------------------
# Main — print PASS/FAIL summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    sys.exit(result.returncode)
