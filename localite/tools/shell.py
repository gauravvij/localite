"""Shell execution tool - runs shell commands and captures output."""

import subprocess
import shlex
from localite.tools.base import BaseTool, ToolResult, measure_duration


class RunShellTool(BaseTool):
    """Tool for executing shell commands."""

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and capture its output."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30,
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command",
                    "default": None,
                },
            },
            "required": ["command"],
        }

    @measure_duration
    async def execute(
        self,
        command: str,
        timeout: int = 30,
        workdir: str | None = None,
    ) -> ToolResult:
        """Execute a shell command and return its output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Exit code: {result.returncode}",
                data={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error=f"Command not found: {shlex.split(command)[0]}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error executing command: {e}",
            )