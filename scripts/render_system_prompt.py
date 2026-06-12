#!/usr/bin/env python3
"""Render the system prompt the same way AgentLoop does and save to file."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from localite.loop.agent_loop import SYSTEM_PROMPT

# Import all tools
from localite.tools.read import ReadFileTool
from localite.tools.edit import EditFileTool
from localite.tools.write import WriteFileTool
from localite.tools.shell import RunShellTool
from localite.tools.list_files import ListFilesTool
from localite.tools.search import GrepSearchTool
from localite.tools.test_executor import TestExecutorTool
from localite.tools.task_complete import TaskCompleteTool
from localite.tools.diff_view import DiffViewTool
from localite.tools.memory_tools import MemoryReadTool, MemoryWriteTool

# Build the tools dict as AgentLoop does
tools = {
    "read_file": ReadFileTool(),
    "edit_file": EditFileTool(),
    "write_file": WriteFileTool(),
    "run_shell": RunShellTool(),
    "list_files": ListFilesTool(),
    "grep_search": GrepSearchTool(),
    "test_executor": TestExecutorTool(),
    "task_complete": TaskCompleteTool(),
    "diff_view": DiffViewTool(),
    "memory_read": MemoryReadTool(),
    "memory_write": MemoryWriteTool(),
}

# This mirrors _get_tool_descriptions() without the trust/demote logic
lines = []
for t in tools.values():
    lines.append(f"### {t.name}")
    lines.append(t.description)
    lines.append("")

tool_descriptions = "\n".join(lines)

# Render the system prompt
rendered = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

# Save to file
output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "swe_bench")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "rendered_system_prompt.txt")
with open(output_path, "w") as f:
    f.write(rendered)

print(f"System prompt written to {output_path}")
print(f"Total length: {len(rendered)} chars, {len(rendered.splitlines())} lines")