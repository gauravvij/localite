"""TaskComplete tool — signals task completion and exits the agent loop."""

from localite.tools.base import BaseTool, ToolResult, measure_duration


class TaskCompleteTool(BaseTool):
    """Tool for signaling that the task is complete and the agent can exit."""

    @property
    def name(self) -> str:
        return "task_complete"

    @property
    def description(self) -> str:
        return "Signal that the task is complete. Call this when the user's request has been fully satisfied — all code written, tests passing, changes verified. This exits the agent loop."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "partial", "failed"],
                    "description": "Overall status of the completed task",
                },
                "reason_code": {
                    "type": "string",
                    "enum": ["tests_passing", "changes_applied", "user_approved", "max_iterations", "irresolvable_error"],
                    "description": "Reason the task is being completed",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished (max 200 chars)",
                },
            },
            "required": ["status", "reason_code", "summary"],
        }

    @measure_duration
    async def execute(self, status: str, reason_code: str, summary: str) -> ToolResult:
        """Record the completion and return a signal result.

        The AgentLoop detects this tool call and sets phase to COMPLETE
        without actually executing the tool. This execute method exists
        for testability and fallback.
        """
        return ToolResult(
            success=True,
            output=f"Task complete: status={status}, reason={reason_code}, summary={summary[:200]}",
            data={"status": status, "reason_code": reason_code, "summary": summary[:200]},
        )