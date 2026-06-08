"""Permission gate — interactive approval for tool calls.

Supports y/approve, s/skip, n/reject+reason, e/edit modes with Rich display.
Can operate in step_mode (approve one at a time) or block mode (batch approval).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text


PermissionDecision = Literal["approved", "skipped", "rejected", "edited"]


@dataclass
class PermissionResult:
    """Result of a permission gate decision."""
    decision: PermissionDecision
    modified_tool_call: Optional[dict] = None
    reason: Optional[str] = None


class PermissionGate:
    """Interactive permission gate for tool call approval.

    Modes:
        step_mode=True: Propose one tool call at a time, wait for user input.
        step_mode=False (block mode): Collect proposals, submit all at once.
    """

    def __init__(self, console: Optional[Console] = None, step_mode: bool = True):
        self.console = console or Console()
        self.step_mode = step_mode
        self._pending_proposals: list[tuple[str, dict]] = []

    def propose(
        self,
        action_description: str,
        tool_call: dict,
    ) -> PermissionResult:
        """Propose a tool call for approval.

        In step_mode, prompts the user immediately.
        In block mode, queues the proposal for batch approval.

        Args:
            action_description: Human-readable description of the action.
            tool_call: The tool call dict (e.g. {"name": "read_file", "args": {...}}).

        Returns:
            PermissionResult with the user's decision.
        """
        if self.step_mode:
            return self._prompt_single(action_description, tool_call)
        else:
            self._pending_proposals.append((action_description, tool_call))
            return PermissionResult(decision="approved")  # placeholder until flush

    def flush_pending(self) -> list[PermissionResult]:
        """Flush all pending proposals in block mode.

        Returns:
            List of PermissionResults for all queued proposals.
        """
        if not self._pending_proposals:
            return []

        results = []
        self.console.print(
            Panel(
                f"[bold yellow]Batch Approval: {len(self._pending_proposals)} pending actions[/]",
                title="Permission Gate",
                border_style="yellow",
            )
        )

        for i, (desc, call) in enumerate(self._pending_proposals, 1):
            self.console.print(f"\n[bold]Proposal {i}/{len(self._pending_proposals)}[/]")
            result = self._prompt_single(desc, call)
            results.append(result)

        self._pending_proposals.clear()
        return results

    def _prompt_single(
        self,
        action_description: str,
        tool_call: dict,
    ) -> PermissionResult:
        """Prompt the user for a single tool call decision."""
        tool_name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})

        # Build display
        args_items = list(args.items())
        args_str = ", ".join(f"{k}={v!r}" for k, v in args_items[:5])
        if len(args) > 5:
            args_str += f", ... ({len(args)} params)"

        display_panel = Panel(
            Text.assemble(
                ("Action: ", "bold"),
                (action_description, ""),
                "\n\n",
                ("Tool: ", "bold"),
                (f"[cyan]{tool_name}[/]", ""),
                "\n",
                ("Arguments: ", "bold"),
                (f"[green]{args_str}[/]", ""),
                "\n\n",
                ("[dim]Full tool call:[/]", ""),
            ),
            title="[bold]🔧 Tool Call Request[/]",
            border_style="blue",
        )
        self.console.print(display_panel)

        # Show the full tool call as JSON
        syntax = Syntax(
            str(tool_call),
            "json",
            theme="monokai",
            line_numbers=False,
        )
        self.console.print(syntax)

        # Prompt for decision
        decision = Prompt.ask(
            "\n[bold]Approve?[/] ([green]y[/]/[yellow]s[/]kip/[red]n[/]o/[blue]e[/]dit)",
            default="y",
        )

        if decision.lower() == "y":
            return PermissionResult(decision="approved", modified_tool_call=tool_call)

        elif decision.lower() == "s":
            return PermissionResult(
                decision="skipped",
                modified_tool_call=None,
                reason="User skipped the action",
            )

        elif decision.lower() == "n":
            reason = Prompt.ask("[red]Reason for rejection[/]")
            return PermissionResult(
                decision="rejected",
                modified_tool_call=None,
                reason=reason,
            )

        elif decision.lower() == "e":
            self.console.print("[bold blue]Edit mode:[/] Enter the modified tool call as JSON")
            self.console.print("[dim]Current tool call for reference:[/]")
            self.console.print(str(tool_call))
            edited_json = Prompt.ask("[bold blue]Modified tool call[/]")
            import json as _json
            try:
                edited = _json.loads(edited_json)
                return PermissionResult(
                    decision="edited",
                    modified_tool_call=edited,
                    reason="User edited the tool call",
                )
            except (_json.JSONDecodeError, ValueError):
                self.console.print("[red]Invalid JSON. Using original tool call.[/]")
                return PermissionResult(
                    decision="approved",
                    modified_tool_call=tool_call,
                    reason="Failed to parse edit, using original",
                )

        else:
            self.console.print(f"[yellow]Unknown choice '{decision}', defaulting to approve[/]")
            return PermissionResult(decision="approved", modified_tool_call=tool_call)

    def set_step_mode(self, enabled: bool):
        """Toggle between step mode and block mode."""
        self.step_mode = enabled