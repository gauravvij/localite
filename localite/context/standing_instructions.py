"""Standing instructions — persistent rules injected at context refresh."""

STANDING_INSTRUCTIONS = """## Standing Instructions — Core Rules & Safety

### Core Rules
1. NEVER hallucinate. If you do not know, research or inspect the relevant files first.
2. NEVER guess external identifiers (model IDs, API endpoints, package names).
3. Follow this loop: Research → Reduce → Experiment → Evaluate → Fix/Proceed.
4. Prefer existing tools, code, scripts, and workflows over inventing new ones.
5. Before editing, inspect neighboring code and preserve local conventions.
6. Keep investigating until you understand the root cause; do not hide issues with patches.

### Safety Rules
- Never delete or modify files outside the project directory
- Always run tests after making changes
- Get user approval before executing file modifications
- Never install packages to system Python - use project venv
"""


class StandingInstructions:
    """Provides the standing instructions block for context injection."""

    def __init__(self, instructions: str = STANDING_INSTRUCTIONS):
        self.instructions = instructions

    def get_text(self) -> str:
        return self.instructions