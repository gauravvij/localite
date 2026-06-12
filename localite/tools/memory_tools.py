"""Memory read/write tools for episodic memory access."""

from localite.tools.base import BaseTool, ToolResult, measure_duration


class MemoryReadTool(BaseTool):
    """Tool for reading saved episodic memory about a topic."""

    def __init__(self, memory_store=None):
        self._memory_store = memory_store

    @property
    def name(self) -> str:
        return "memory_read"

    @property
    def description(self) -> str:
        return (
            "Read saved memory about a topic from previous sessions. Use this to recall what was "
            "learned about a specific topic. Results are capped at 500 tokens. "
            "WHEN TO USE: At the start of a session to recall project structure, dependencies, "
            "bug patterns, or decisions from previous sessions. "
            "WHEN NOT TO USE: For reading files (use read_file), for searching code (use grep_search), "
            "for saving information (use memory_write instead). "
            "PARAMETERS: 'topic' (required, string, e.g., 'project_structure', 'dependencies', "
            "'bug_patterns', 'config_decisions'). "
            "EXAMPLE: {\"topic\": \"project_structure\"} "
            "COMMON MISTAKES: Reading a topic that doesn't exist yet (first time, no data saved); "
            "using generic single-word topics that are too broad to be useful."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to read from memory (e.g., 'project_structure', 'dependencies', 'bug_patterns')",
                },
            },
            "required": ["topic"],
        }

    @measure_duration
    async def execute(self, topic: str) -> ToolResult:
        """Read a topic from episodic memory."""
        if not self._memory_store:
            return ToolResult(success=False, output="", error="Memory store not available")
        content = self._memory_store.read_topic(topic)
        return ToolResult(success=True, output=content)


class MemoryWriteTool(BaseTool):
    """Tool for saving information to episodic memory for future sessions."""

    def __init__(self, memory_store=None):
        self._memory_store = memory_store

    @property
    def name(self) -> str:
        return "memory_write"

    @property
    def description(self) -> str:
        return (
            "Save important information to episodic memory. This persists across sessions. Use this "
            "to remember project structure, learned patterns, configurations, or decisions for future tasks. "
            "WHEN TO USE: After discovering important project structure, solving a tricky bug, finding "
            "a working configuration, or making a design decision worth remembering across sessions. "
            "WHEN NOT TO USE: For temporary information that only matters this session; "
            "for reading memory (use memory_read instead); for storing file contents (use write_file). "
            "PARAMETERS: 'topic' (required, string, e.g., 'project_structure', 'dependencies', "
            "'bug_patterns', 'config_decisions'), 'content' (required, string, max 2000 chars). "
            "EXAMPLE: {\"topic\": \"bug_patterns\", "
            "\"content\": \"Setting num_workers>2 with persistent_workers=True causes DataLoader hangs\"} "
            "COMMON MISTAKES: Writing overly long content (capped at 2000 chars); "
            "using vague topic names that are hard to retrieve later."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic name to save under (e.g., 'project_structure', 'dependencies', 'bug_patterns')",
                },
                "content": {
                    "type": "string",
                    "description": "Content to remember (max 2000 chars)",
                },
            },
            "required": ["topic", "content"],
        }

    @measure_duration
    async def execute(self, topic: str, content: str) -> ToolResult:
        """Write information to episodic memory."""
        if not self._memory_store:
            return ToolResult(success=False, output="", error="Memory store not available")
        # Cap content at 2000 chars
        content = content[:2000]
        self._memory_store.write_topic(topic, content)
        return ToolResult(
            success=True,
            output=f"Saved {len(content)} chars to memory topic: {topic}",
        )