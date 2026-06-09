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
        return "Read saved memory about a topic from previous sessions. Use this to recall what was learned about a specific topic. Results are capped at 500 tokens."

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
        return "Save important information to episodic memory. This persists across sessions. Use this to remember project structure, learned patterns, configurations, or decisions for future tasks."

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