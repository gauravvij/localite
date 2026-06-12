"""Test executor tool - auto-detects test frameworks and runs tests."""

import os
import subprocess
from localite.tools.base import BaseTool, ToolResult, measure_duration


class TestExecutorTool(BaseTool):
    """Tool for discovering and running tests."""

    @property
    def name(self) -> str:
        return "test_executor"

    @property
    def description(self) -> str:
        return (
            "Auto-detect and run tests. Supports pytest, unittest, and cargo-test. "
            "Returns pass/fail counts and test output. "
            "WHEN TO USE: After making code changes to verify nothing is broken; before submitting "
            "code changes to ensure existing tests pass; to reproduce test failures. "
            "WHEN NOT TO USE: For running arbitrary shell commands (use run_shell), "
            "for compiling/building projects (use run_shell with make or cargo build). "
            "PARAMETERS: 'path' (optional, directory or test file), 'framework' (optional, "
            "force specific framework: 'pytest', 'unittest', 'cargo-test'), "
            "'timeout' (optional, int seconds, default 120). "
            "EXAMPLE: {\"path\": \"/home/user/project/tests\", \"timeout\": 60} "
            "COMMON MISTAKES: Not setting a long enough timeout for large test suites; "
            "expecting test output in a specific format (it varies by framework); "
            "calling test_executor before running ANY verification (use on changed files first)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory or test file path to run tests in",
                    "default": None,
                },
                "framework": {
                    "type": "string",
                    "description": "Force a specific framework: 'pytest', 'unittest', or 'cargo-test'",
                    "default": None,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)",
                    "default": 120,
                },
            },
        }

    @measure_duration
    async def execute(
        self,
        path: str | None = None,
        framework: str | None = None,
        timeout: int = 120,
    ) -> ToolResult:
        """Auto-detect and run tests."""
        try:
            test_dir = path or os.getcwd()

            if not os.path.exists(test_dir):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Path not found: {test_dir}",
                )

            if framework is None:
                framework = self._detect_framework(test_dir)

            if framework == "pytest":
                return await self._run_pytest(test_dir, timeout)
            elif framework == "unittest":
                return await self._run_unittest(test_dir, timeout)
            elif framework == "cargo-test":
                return await self._run_cargo_test(test_dir, timeout)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unsupported framework: {framework}",
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Error running tests: {e}",
            )

    def _detect_framework(self, path: str) -> str:
        """Auto-detect the test framework to use."""
        if os.path.isfile(path):
            path = os.path.dirname(path)

        # Check for Cargo.toml
        if os.path.exists(os.path.join(path, "Cargo.toml")):
            return "cargo-test"

        # Check for pytest config or conftest
        for marker in ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"]:
            if os.path.exists(os.path.join(path, marker)):
                return "pytest"

        # Check for test files
        has_pytest = False
        has_unittest = False
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    try:
                        with open(os.path.join(root, f), "r") as fh:
                            content = fh.read()
                            if "pytest" in content:
                                has_pytest = True
                            if "unittest" in content or "TestCase" in content:
                                has_unittest = True
                    except Exception:
                        continue
                    if "test_" in f:
                        if not has_unittest and not has_pytest:
                            has_pytest = True  # default assumption

        if has_pytest:
            return "pytest"
        if has_unittest:
            return "unittest"
        return "pytest"  # default

    async def _run_pytest(self, path: str, timeout: int) -> ToolResult:
        """Run pytest."""
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", path, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout

            # Parse results
            passed = result.stdout.count("PASSED") + result.stdout.count("passed")
            failed = result.stdout.count("FAILED") + result.stdout.count("failed")

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Tests failed ({failed} failed, {passed} passed)",
                data={"passed": passed, "failed": failed, "returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"pytest timed out after {timeout}s",
            )

    async def _run_unittest(self, path: str, timeout: int) -> ToolResult:
        """Run unittest discovery."""
        try:
            result = subprocess.run(
                ["python3", "-m", "unittest", "discover", "-s", path, "-v"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout

            passed = result.stdout.count("ok")
            failed = result.stdout.count("FAIL") + result.stdout.count("ERROR")

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Tests failed ({failed} failed, {passed} passed)",
                data={"passed": passed, "failed": failed, "returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"unittest timed out after {timeout}s",
            )

    async def _run_cargo_test(self, path: str, timeout: int) -> ToolResult:
        """Run cargo test."""
        try:
            result = subprocess.run(
                ["cargo", "test"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=path,
            )
            output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout

            from re import search
            passed = 0
            failed = 0
            m = search(r"(\d+) passed", output)
            if m:
                passed = int(m.group(1))
            m = search(r"(\d+) failed", output)
            if m:
                failed = int(m.group(1))

            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=None if result.returncode == 0 else f"Tests failed",
                data={"passed": passed, "failed": failed, "returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"cargo test timed out after {timeout}s",
            )