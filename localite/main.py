"""localite — Terminal UI entry point with Rich.

Usage:
    python3 localite/main.py --model "hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M"
    python3 localite/main.py --profile gemma4_e4b
    python3 localite/main.py --resume <session_id>
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

from localite.config import ConfigLoader, ModelProfile
from localite.model.client import AsyncOllamaClient
from localite.permissions.gate import PermissionGate
from localite.episodes.store import EpisodeStore
from localite.episodes.model import Episode
from localite.loop.agent_loop import AgentLoop
from localite.loop.phases import Phase
from localite.tools.base import BaseTool
from localite.tools.read import ReadFileTool
from localite.tools.write import WriteFileTool
from localite.tools.edit import EditFileTool
from localite.tools.search import GrepSearchTool
from localite.tools.shell import RunShellTool
from localite.tools.test_executor import TestExecutorTool
from localite.tools.diff_view import DiffViewTool

logger = logging.getLogger(__name__)


class LocaliteTerminal:
    """Rich-powered terminal UI for localite agent."""

    def __init__(
        self,
        model_client: AsyncOllamaClient,
        agent_loop: AgentLoop,
        permission_gate: PermissionGate,
        console: Console | None = None,
    ):
        self.client = model_client
        self.loop = agent_loop
        self.gate = permission_gate
        self.console = console or Console()

        # Rich layout components
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self):
        """Setup the Rich layout structure."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="status", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="conversation", ratio=2),
            Layout(name="tool_output", ratio=1),
        )

    def _render_header(self) -> Panel:
        """Render the header panel."""
        model_name = self.client.model_name
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        status = self.loop.get_status()
        return Panel(
            Text.assemble(
                ("🐚 localite", "bold cyan"),
                ("  ⚡ ", ""),
                (f"Model: {short_name}", "yellow"),
                ("  |  ", "dim"),
                (f"Phase: {status.get('phase', 'INIT')}", "green"),
                ("  |  ", "dim"),
                (f"Turn: {status.get('turn', '0/4')}", "blue"),
            ),
            border_style="bright_blue",
        )

    def _render_conversation(self) -> Panel:
        """Render the conversation panel from episode turns."""
        tree = Tree("💬 Conversation")
        if self.loop.episode:
            for turn in self.loop.episode.turns:
                phase_label = f"[{turn.phase}]"
                if turn.tool_call:
                    name = turn.tool_call.get("name", "?")
                    tree.add(f"{phase_label} 🛠 {name} ({turn.user_approval or 'pending'})")
                elif turn.model_output:
                    preview = turn.model_output[:80].replace("\n", " ")
                    tree.add(f"{phase_label} {preview}")
        else:
            tree.add("(waiting for input...)")
        return Panel(tree, title="Conversation")

    def _render_tool_output(self) -> Panel:
        """Render the latest tool output panel."""
        if self.loop.episode and self.loop.episode.turns:
            last_turn = self.loop.episode.turns[-1]
            if last_turn.tool_result:
                output = last_turn.tool_result.get("output", "")
                if output:
                    return Panel(
                        Syntax(output[:1000], "text", theme="monokai"),
                        title="Latest Tool Output",
                    )
        return Panel("(no tool output yet)", title="Latest Tool Output")

    def _render_status_bar(self) -> Panel:
        """Render the status bar at the bottom."""
        status = self.loop.get_status()
        model_name = self.client.model_name.split("/")[-1] if "/" in self.client.model_name else self.client.model_name
        mode = "Step" if self.gate.step_mode else "Batch"
        return Panel(
            Text.assemble(
                (f"Model: {model_name}", "bold"),
                ("  |  ", "dim"),
                (f"Phase: {status.get('phase', 'INIT')}", "green"),
                ("  |  ", "dim"),
                (f"Turns: {status.get('turn', '0/4')}", "blue"),
                ("  |  ", "dim"),
                (f"Mode: {mode}", "magenta"),
                ("  |  ", "dim"),
                (f"Episode: {status.get('episode_id', 'N/A')[:8]}...", "dim"),
            ),
            border_style="white",
        )

    def render(self) -> Layout:
        """Render the full live layout."""
        self.layout["header"].update(self._render_header())
        self.layout["conversation"].update(self._render_conversation())
        self.layout["tool_output"].update(self._render_tool_output())
        self.layout["status"].update(self._render_status_bar())
        return self.layout

    async def run_interactive(self):
        """Run the interactive terminal loop."""
        self.console.print(
            Panel(
                "[bold cyan]Welcome to localite — Fully Local AI Coding Agent[/]\n\n"
                "[yellow]Enter your coding task below. Ctrl+D to exit, Ctrl+C to interrupt.[/]\n"
                "[dim]Type 'exit' or 'quit' to quit. | Default: Gemma 4 E4B[/]",
                title="🐚 localite v0.1.0",
                border_style="green",
            )
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/]")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]Goodbye![/]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            # Run the agent loop for this request
            with Status("[bold blue]Thinking...", spinner="dots") as status:
                try:
                    result = await self.loop.run(user_input)
                    status.update("[green]Done![/]")

                    # Show result summary
                    self.console.print(
                        Panel(
                            Text.assemble(
                                ("✅ Episode Complete\n\n", "bold green"),
                                (f"Phase: {result.get('phase', '?')}", ""),
                                ("\n", ""),
                                (f"Files changed: {len(result.get('files_changed', []))}", ""),
                                ("\n", ""),
                                (f"Episode: {result.get('episode_id', '?')[:12]}...", "dim"),
                            ),
                            title="Result",
                            border_style="green",
                        )
                    )

                    # Show diff of files changed if any
                    if result.get("files_changed"):
                        table = Table(title="Files Changed")
                        table.add_column("File", style="cyan")
                        table.add_column("Status", style="yellow")
                        for f in result["files_changed"]:
                            table.add_row(f, "modified")
                        self.console.print(table)

                except (ConnectionError, TimeoutError) as e:
                    self.console.print(f"[red]Error: {e}[/]")
                except Exception as e:
                    self.console.print(f"[red]Unexpected error: {e}[/]")
                    logger.exception("Error in agent loop")


def create_default_tools() -> dict[str, BaseTool]:
    """Create and return the default set of tools."""
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="localite — Fully Local AI Coding Agent (default: Gemma 4 E4B)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (overrides profile)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="gemma4_e4b",
        help="Model profile name (default: gemma4_e4b)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Session ID to resume",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        default=False,
        help="Enable batch approval mode instead of step-by-step",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


async def main_async(argv: list[str] | None = None):
    """Main async entry point."""
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load profile
    config_loader = ConfigLoader()
    try:
        profile = config_loader.load_profile(args.profile)
    except FileNotFoundError:
        console = Console()
        console.print(f"[red]Profile '{args.profile}' not found. Available: {config_loader.list_profiles()}[/]")
        sys.exit(1)

    # Override model name if specified
    model_name = args.model or profile.name

    # Initialize components
    client = AsyncOllamaClient(
        model_name=model_name,
        base_url=profile.base_url,
        timeout=profile.timeout,
        has_thinking_tags=profile.has_thinking_tags,
    )

    tools = create_default_tools()

    gate = PermissionGate(step_mode=not args.batch_mode)

    store = EpisodeStore()

    loop = AgentLoop(
        model_client=client,
        tools=tools,
        permission_gate=gate,
        episode_store=store,
        model_profile=profile,
        max_iterations=3,
    )

    # Build and run terminal UI
    terminal = LocaliteTerminal(
        model_client=client,
        agent_loop=loop,
        permission_gate=gate,
    )

    await terminal.run_interactive()


def main(argv: list[str] | None = None):
    """Synchronous entry point."""
    asyncio.run(main_async(argv))


if __name__ == "__main__":
    main()