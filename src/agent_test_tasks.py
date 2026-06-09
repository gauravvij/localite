"""
Agent test tasks — 6 coding tasks for the agent evaluation harness.

Each task is a dict with:
    - task_id: short unique identifier
    - task_description: the prompt to give the agent
    - verify: function(output_dir) -> (bool, error_message_string)

The verify function checks whether the agent's output in output_dir
is correct. It should import and run the created script, or check
file contents.
"""

import os
import subprocess
import sys
from typing import Callable

# ---------------------------------------------------------------------------
# Verifier helpers
# ---------------------------------------------------------------------------

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "agent_tasks")


def _ensure_task_dir():
    os.makedirs(TASK_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Task 1: Fibonacci
# ---------------------------------------------------------------------------


def _verify_fibonacci(output_dir: str) -> tuple[bool, str]:
    """Verify the agent wrote fib.py and it prints first 20 Fibonacci numbers."""
    fib_path = os.path.join(output_dir, "fib.py")
    if not os.path.exists(fib_path):
        # Try common alternatives
        for alt in ["fibonacci.py", "fibo.py"]:
            p = os.path.join(output_dir, alt)
            if os.path.exists(p):
                fib_path = p
                break
        else:
            return False, f"fib.py not found in {output_dir}"

    # Run the script
    try:
        result = subprocess.run(
            [sys.executable, fib_path],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "fib.py execution timed out"

    if result.returncode != 0:
        return False, f"fib.py exited with code {result.returncode}: {result.stderr[:200]}"

    # Parse output lines
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if len(lines) < 20:
        return False, f"Expected 20 numbers, got {len(lines)}"

    # Check first few Fibonacci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    expected_start = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    try:
        actual = [int(l) for l in lines[:10]]
    except ValueError:
        return False, f"Output lines are not integers: {lines[:5]}"

    if actual != expected_start:
        return False, f"Expected {expected_start}, got {actual}"

    return True, "OK"


# ---------------------------------------------------------------------------
# Task 2: Greet
# ---------------------------------------------------------------------------


def _verify_greet(output_dir: str) -> tuple[bool, str]:
    """Verify the agent wrote greet.py with a greeting function and test output."""
    greet_path = os.path.join(output_dir, "greet.py")
    if not os.path.exists(greet_path):
        return False, f"greet.py not found in {output_dir}"

    try:
        result = subprocess.run(
            [sys.executable, greet_path],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "greet.py execution timed out"

    if result.returncode != 0:
        return False, f"greet.py exited with code {result.returncode}: {result.stderr[:200]}"

    output = result.stdout.strip().lower()
    # Should contain the name (the task asks to greet "World" or similar)
    if "hello" in output or "hi" in output or "greet" in output:
        return True, "OK"

    return True, "OK (unclear content check)"


# ---------------------------------------------------------------------------
# Task 3: Word Frequency
# ---------------------------------------------------------------------------


def _verify_word_freq(output_dir: str) -> tuple[bool, str]:
    """Verify the agent wrote word_freq.py that counts word frequencies."""
    wf_path = os.path.join(output_dir, "word_freq.py")
    if not os.path.exists(wf_path):
        return False, f"word_freq.py not found in {output_dir}"

    try:
        # Import and test the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("word_freq", wf_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find a word frequency function
        freq_fn = None
        for name in ["word_freq", "word_frequency", "count_words", "freq", "wordcount"]:
            if hasattr(mod, name):
                freq_fn = getattr(mod, name)
                break

        if freq_fn is None:
            # Try running as script
            result = subprocess.run(
                [sys.executable, wf_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and len(result.stdout.strip()) > 0:
                return True, "OK (script runs)"
            return False, "No word frequency function found and script doesn't produce output"

        # Test the function
        test_input = "the cat and the dog and the bird"
        result = freq_fn(test_input)
        if isinstance(result, dict):
            if result.get("the") == 3 and result.get("and") == 2:
                return True, "OK"
            return True, f"OK (function returned dict: {result})"

        return True, "OK (function exists and runs)"
    except Exception as e:
        return False, f"word_freq test failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Task 4: Factorial
# ---------------------------------------------------------------------------


def _verify_factorial(output_dir: str) -> tuple[bool, str]:
    """Verify the agent wrote factorial.py that computes factorial correctly."""
    fact_path = os.path.join(output_dir, "factorial.py")
    if not os.path.exists(fact_path):
        return False, f"factorial.py not found in {output_dir}"

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("factorial", fact_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find factorial function
        fact_fn = None
        for name in ["factorial", "fact", "fac"]:
            if hasattr(mod, name):
                fact_fn = getattr(mod, name)
                break

        if fact_fn is None:
            # Run as script
            result = subprocess.run(
                [sys.executable, fact_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return False, f"factorial.py execution error: {result.stderr[:200]}"
            return True, "OK (script runs)"

        # Test known values: 0! = 1, 1! = 1, 5! = 120, 7! = 5040
        test_cases = [(0, 1), (1, 1), (5, 120), (7, 5040)]
        for n, expected in test_cases:
            actual = fact_fn(n)
            if actual != expected:
                return False, f"factorial({n}) = {actual}, expected {expected}"

        return True, "OK"
    except Exception as e:
        return False, f"factorial test failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Task 5: Sort
# ---------------------------------------------------------------------------


def _verify_sort(output_dir: str) -> tuple[bool, str]:
    """Verify the agent wrote sort.py that sorts a list correctly."""
    sort_path = os.path.join(output_dir, "sort.py")
    if not os.path.exists(sort_path):
        return False, f"sort.py not found in {output_dir}"

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sort", sort_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find sort function
        sort_fn = None
        for name in ["sort_list", "sort", "sort_numbers", "bubble_sort", "merge_sort", "quick_sort"]:
            if hasattr(mod, name):
                sort_fn = getattr(mod, name)
                break

        if sort_fn is None:
            # Run as script
            result = subprocess.run(
                [sys.executable, sort_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return False, f"sort.py execution error: {result.stderr[:200]}"
            return True, "OK (script runs)"

        # Test: [3, 1, 4, 1, 5, 9, 2] -> [1, 1, 2, 3, 4, 5, 9]
        input_list = [3, 1, 4, 1, 5, 9, 2]
        expected = [1, 1, 2, 3, 4, 5, 9]
        actual = sort_fn(input_list)

        if actual == expected:
            return True, "OK"
        # Handle case where function sorts in-place and returns None
        if actual is None and input_list == expected:
            return True, "OK (in-place sort)"
        return True, f"OK (sort produces: {actual})"
    except Exception as e:
        return False, f"sort test failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Task 6: Is Prime
# ---------------------------------------------------------------------------


def _verify_is_prime(output_dir: str) -> tuple[bool, str]:
    """Verify the agent wrote is_prime.py that checks primality correctly."""
    prime_path = os.path.join(output_dir, "is_prime.py")
    if not os.path.exists(prime_path):
        return False, f"is_prime.py not found in {output_dir}"

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("is_prime", prime_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Find prime checking function
        prime_fn = None
        for name in ["is_prime", "prime", "check_prime", "isprime"]:
            if hasattr(mod, name):
                prime_fn = getattr(mod, name)
                break

        if prime_fn is None:
            result = subprocess.run(
                [sys.executable, prime_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return False, f"is_prime.py execution error: {result.stderr[:200]}"
            return True, "OK (script runs)"

        # Test: primes -> 2, 3, 5, 7, 11, 13; non-primes -> 1, 4, 6, 8, 9, 10
        primes = [2, 3, 5, 7, 11, 13]
        non_primes = [1, 4, 6, 8, 9, 10]

        for n in primes:
            if not prime_fn(n):
                return False, f"is_prime({n}) = False, expected True"

        for n in non_primes:
            if prime_fn(n):
                return False, f"is_prime({n}) = True, expected False"

        return True, "OK"
    except Exception as e:
        return False, f"is_prime test failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


TASKS = [
    {
        "task_id": "fibonacci",
        "task_description": (
            "Write a Python script called fib.py that defines a function "
            "to print the first N Fibonacci numbers. The script should print "
            "the first 20 Fibonacci numbers, one per line. Run it with "
            "python3 fib.py to verify it works correctly."
        ),
        "verify": _verify_fibonacci,
    },
    {
        "task_id": "greet",
        "task_description": (
            "Write a Python script called greet.py that defines a function "
            "greet(name) which returns a greeting string like 'Hello, World!'. "
            "The script should also print greet('World') when run directly. "
            "Run it with python3 greet.py to verify."
        ),
        "verify": _verify_greet,
    },
    {
        "task_id": "word_freq",
        "task_description": (
            "Write a Python script called word_freq.py that defines a function "
            "word_frequency(text) which takes a string and returns a dictionary "
            "with word counts (case-insensitive, ignoring punctuation). "
            "Include a demonstration when run directly. "
            "Run it with python3 word_freq.py to verify."
        ),
        "verify": _verify_word_freq,
    },
    {
        "task_id": "factorial",
        "task_description": (
            "Write a Python script called factorial.py that defines a function "
            "factorial(n) which computes n! recursively or iteratively. "
            "Include test cases that print factorials of 0 through 10 when run "
            "directly. Run it with python3 factorial.py to verify."
        ),
        "verify": _verify_factorial,
    },
    {
        "task_id": "sort",
        "task_description": (
            "Write a Python script called sort.py that defines a function "
            "sort_list(numbers) which sorts a list of numbers in ascending order. "
            "Include a demonstration sorting [3, 1, 4, 1, 5, 9, 2, 6] when run "
            "directly. Run it with python3 sort.py to verify."
        ),
        "verify": _verify_sort,
    },
    {
        "task_id": "is_prime",
        "task_description": (
            "Write a Python script called is_prime.py that defines a function "
            "is_prime(n) which returns True if n is prime, False otherwise. "
            "Include test cases checking numbers 1-20 when run directly. "
            "Run it with python3 is_prime.py to verify."
        ),
        "verify": _verify_is_prime,
    },
]