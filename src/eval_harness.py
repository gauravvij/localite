"""
Core evaluation harness for multi-turn SLM agent degradation testing.

Tests a model served via Ollama across multiple dimensions of degradation:
instruction drift, memory retrieval, hallucination onset, tool call format
drift, persona consistency, and recency bias.

Supports:
- Filler conversation caching (reuse across runs)
- Multi-run loop support (N runs per depth)
- Per-test timing
- Failure categorization (scorers return (score, category) tuples)
- Adaptive stopping (skip deeper depths when model has fully degraded)
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = "/home/azureuser/local_llm_eval"
CACHE_DIR = os.path.join(PROJECT_ROOT, "results", "cache")


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Thinking / reasoning block stripping
# ---------------------------------------------------------------------------


def strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks from model output.

    LFM2.5 uses <thinking>...</thinking> ... <response>...</response> XML format.
    Gemma 4 may use <|channel>thought</channel|> tags.
    Also handles standalone thought tags.
    """
    if not text:
        return ""

    # Remove Gemma 4 style: <|channel>thought</channel|> blocks
    text = re.sub(r'<\|channel>thought<\|?[^>]*>.*?</channel\|?>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|channel>thought.*?</channel\|?>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|channel\|?>.*?</channel\|?>', '', text, flags=re.DOTALL)

    # Remove everything between <thinking> and </thinking> (including tags)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)

    # Handle orphaned </thinking> tags
    text = text.replace('</thinking>', '')
    text = text.replace('<thinking>', '')

    # If <response> tag exists, take everything after it
    if '<response>' in text:
        text = text.split('<response>', 1)[1]

    # Remove closing </response> tag if present
    text = text.replace('</response>', '')

    # Handle standalone angle-bracket tags that might appear (but keep JSON)
    text = re.sub(r'<[a-zA-Z_/][^>]*>', '', text)

    return text.strip()


def count_tokens(text: str) -> int:
    """Rough token count heuristic (~4 chars per token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(model_name: str, dimension: str, depth: int,
               depth_unit: str, system_prompt: str, run_num: int,
               seed: int = 0) -> str:
    """Generate a deterministic cache key for a filler conversation."""
    raw = f"{model_name}|{dimension}|{depth}|{depth_unit}|{system_prompt}|r{run_num}|s{seed}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{model_name.replace('/', '_')}_{dimension}_d{depth}{depth_unit[0]}_r{run_num}_{h}.json"


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key)


def load_cached_filler(model_name: str, dimension: str, depth: int,
                       depth_unit: str, system_prompt: str,
                       run_num: int, seed: int = 0) -> Optional[List[Dict]]:
    """Load cached filler conversation if it exists."""
    _ensure_cache_dir()
    key = _cache_key(model_name, dimension, depth, depth_unit, system_prompt, run_num, seed)
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return data
    return None


def save_cached_filler(conversation: List[Dict], model_name: str,
                       dimension: str, depth: int, depth_unit: str,
                       system_prompt: str, run_num: int, seed: int = 0):
    """Save filler conversation to cache."""
    _ensure_cache_dir()
    key = _cache_key(model_name, dimension, depth, depth_unit, system_prompt, run_num, seed)
    path = _cache_path(key)
    with open(path, "w") as f:
        json.dump(conversation, f, indent=2, default=str)


def clear_cache(model_name: Optional[str] = None, dimension: Optional[str] = None):
    """Clear cache entries, optionally filtered by model/dimension."""
    _ensure_cache_dir()
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        if model_name and model_name.replace("/", "_") not in fname:
            continue
        if dimension and dimension not in fname:
            continue
        os.remove(os.path.join(CACHE_DIR, fname))


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES = {
    "pass": "Behaviour matches expectation — no degradation",
    "empty_response": "Model returned empty or whitespace-only response",
    "plain_text": "Response is plain text when structured output (JSON/xml) was expected",
    "wrong_json": "Response is JSON but with incorrect keys or structure",
    "partial_json": "Response contains JSON mixed with plain text",
    "wrong_function": "Valid JSON function call but wrong function name",
    "incoherent": "Response is garbled, repetitive, or incoherent",
    "hallucinated_fact": "Response contradicts known facts from context",
    "hallucinated_name": "Response fabricates a name or fact not in context",
    "generic_evasion": "Response is an 'I don't know' or generic non-answer",
    "ignore_instruction": "Response ignores a clear system/user instruction",
    "follows_override": "Response follows a later override instruction instead of the original",
    "follows_original": "Response follows the original instruction despite a valid override",
    "mixed_signal": "Response shows partial compliance with conflicting instructions",
    "persona_drop": "Response drops persona and responds as generic AI",
    "persona_inconsistent": "Response partially maintains persona but with contradictions",
    "timeout": "Model failed to respond within timeout",
    "error": "Runtime error during evaluation",
}


@dataclass
class ScoredResult:
    """Result of scoring a model response, including failure category."""
    score: float
    category: str  # One of FAILURE_CATEGORIES keys
    details: str = ""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Structured result for a single test case."""
    dimension: str
    depth: int
    depth_unit: str  # 'turns' or 'tokens'
    run: int
    system_prompt: str
    messages: List[Dict]  # conversation history (excl. system and test query)
    test_query: str
    raw_output: str
    stripped_output: str
    score: float
    failure_category: str = "pass"  # One of FAILURE_CATEGORIES keys
    expected: Any = None
    actual: Any = None
    error: Optional[str] = None
    filler_stats: Optional[Dict] = None  # e.g. {"num_turns": 5, "approx_tokens": 200}
    eval_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Client for Ollama's /api/chat endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M",
        keep_alive: Optional[str] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.keep_alive = keep_alive  # None = default Ollama behavior

    def chat(
        self,
        messages: List[Dict],
        options: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> Dict:
        """Send a chat request to Ollama with retry logic."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if options:
            payload["options"] = options

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=1800,  # 30 min timeout for long contexts on CPU
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout: {e}"
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"  [Retry {attempt + 1}] Timeout, waiting {wait}s...")
                    time.sleep(wait)
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 2
                    print(f"  [Retry {attempt + 1}] Error, waiting {wait}s...")
                    time.sleep(wait)
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                if attempt < max_retries - 1:
                    time.sleep(2)

        raise RuntimeError(
            f"Ollama chat failed after {max_retries} attempts: {last_error}"
        )

    def generate_filler(
        self,
        system_prompt: str,
        conversation: List[Dict],
        user_question: str,
    ) -> Tuple[str, str]:
        """Generate a natural filler turn: user question + assistant response.

        Returns (user_question, assistant_response).
        The user_question is returned as-is (it was the input).
        The assistant_response is the model's stripped output.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_question})

        result = self.chat(messages)
        raw = result.get("message", {}).get("content", "")
        stripped = strip_thinking(raw)
        return user_question, stripped


# ---------------------------------------------------------------------------
# Failure categorization helpers
# ---------------------------------------------------------------------------


def categorize_instruction_adherence(response: str, stripped: str) -> str:
    """Categorize instruction adherence failures."""
    if not stripped:
        return "empty_response"
    # Check for valid JSON with required keys
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "thought" in data and "action" in data:
            return "pass"
        return "wrong_json"
    except (json.JSONDecodeError, ValueError):
        pass
    # Check for JSON embedded in text
    json_match = re.search(r'\{[^{}]*"thought"[^{}]*"action"[^{}]*\}', stripped, re.DOTALL)
    if json_match:
        return "partial_json"
    return "plain_text"


def categorize_memory(response: str, stripped: str, expected: str) -> str:
    """Categorize memory retrieval failures."""
    if not stripped:
        return "empty_response"
    if expected.lower() in stripped.lower():
        return "pass"
    # Check for hallucinated wrong name
    common_names = ["Rex", "Max", "Buddy", "Charlie", "Cooper", "Jack", "Rocky", "Bear", "Milo"]
    if any(n.lower() in stripped.lower() for n in common_names):
        return "hallucinated_name"
    # Check for generic evasion
    evasion_patterns = ["i don't know", "i don't recall", "i'm not sure",
                        "i cannot", "i don't have", "i don't remember"]
    if any(p in stripped.lower() for p in evasion_patterns):
        return "generic_evasion"
    return "hallucinated_fact"


def categorize_tool_call(response: str, stripped: str) -> str:
    """Categorize tool call drift failures."""
    if not stripped:
        return "empty_response"
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "function" in data and "arguments" in data:
            fn = data["function"]
            if fn in ("get_weather", "search_web"):
                return "pass"
            return "wrong_function"
        return "wrong_json"
    except (json.JSONDecodeError, ValueError):
        pass
    # Check for JSON embedded in text
    json_match = re.search(r'\{[^{}]*"function"[^{}]*\}', stripped, re.DOTALL)
    if json_match:
        return "partial_json"
    if "get_weather" in stripped or "search_web" in stripped:
        return "partial_json"
    return "plain_text"


def categorize_persona(response: str, stripped: str) -> str:
    """Categorize persona consistency failures."""
    if not stripped:
        return "empty_response"
    keywords = [
        "marine biolog", "deep sea", "deep-sea", "dr. sarah",
        "sarah chen", "oceanograph", "marine ecolog", "marine science",
        "ocean", "sea life", "aquatic", "marine",
    ]
    kw_matches = sum(1 for kw in keywords if kw.lower() in stripped.lower())
    if kw_matches >= 3:
        return "pass"
    if kw_matches >= 1:
        return "persona_inconsistent"
    return "persona_drop"


def categorize_recency(response: str, stripped: str) -> str:
    """Categorize recency bias failures."""
    if not stripped:
        return "empty_response"
    text = stripped.strip().lower()
    follows_original = text.startswith("aye aye") or "aye aye" in text[:40]
    # Check for override indicators
    override_indicators = ["normally", "just respond", "without the prefix",
                          "no longer", "ignore previous"]
    follows_override = any(ind in text for ind in override_indicators)

    if follows_original and not follows_override:
        return "pass"  # Original instruction followed
    if follows_override and not follows_original:
        return "follows_override"
    if follows_original and follows_override:
        return "mixed_signal"
    return "incoherent"


def categorize_hallucination(response: str, stripped: str, expected: str) -> str:
    """Categorize hallucination onset failures."""
    if not stripped:
        return "empty_response"
    if expected.lower() in stripped.lower():
        return "pass"
    evasion_patterns = ["i don't know", "i don't recall", "i'm not sure",
                        "i cannot", "i don't have"]
    if any(p in stripped.lower() for p in evasion_patterns):
        return "generic_evasion"
    return "hallucinated_fact"


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Core evaluation harness orchestrating multi-turn degradation tests."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        results_dir: str = os.path.join(PROJECT_ROOT, "results"),
        cache_enabled: bool = True,
    ):
        self.client = client or OllamaClient()
        self.results_dir = results_dir
        self.cache_enabled = cache_enabled
        self.results: List[TestResult] = []

    def run_single(
        self,
        dimension: str,
        depth: int,
        depth_unit: str,
        run_num: int,
        system_prompt: str,
        conversation: List[Dict],
        test_query: str,
        scorer: Callable[[str, Any], Tuple[float, str]],
        expected: Any = None,
        category_fn: Optional[Callable[[str, str, Any], str]] = None,
    ) -> TestResult:
        """Run a single test case at a given depth.

        scorer should return (score, category) tuple.
        category_fn is an optional dedicated categorizer (used if scorer returns
        incomplete categorization).
        """
        # Build the full message list for the test query
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation)
        messages.append({"role": "user", "content": test_query})

        # Compute filler stats
        filler_stats = {
            "num_turns": len([m for m in conversation if m["role"] == "user"]),
            "approx_tokens": sum(count_tokens(m.get("content", "")) for m in conversation),
        }

        start_time = time.time()
        error = None
        raw_output = ""
        stripped_output = ""
        score = 0.0
        failure_category = "error"
        actual = None

        try:
            result = self.client.chat(messages)
            raw_output = result.get("message", {}).get("content", "")
            stripped_output = strip_thinking(raw_output)
            actual = stripped_output

            if callable(scorer):
                score_result = scorer(stripped_output, expected)
                if isinstance(score_result, tuple):
                    score, failure_category = score_result
                else:
                    score = score_result
                    # Use category_fn if provided, else auto-detect
                    if category_fn and callable(category_fn):
                        failure_category = category_fn(stripped_output, stripped_output, expected)
                    else:
                        failure_category = "pass" if score >= 0.8 else "plain_text"
            else:
                score = 0.0
                failure_category = "error"
        except Exception as e:
            error = str(e)

        elapsed = time.time() - start_time

        test_result = TestResult(
            dimension=dimension,
            depth=depth,
            depth_unit=depth_unit,
            run=run_num,
            system_prompt=system_prompt,
            messages=conversation,
            test_query=test_query,
            raw_output=raw_output,
            stripped_output=stripped_output,
            score=score,
            failure_category=failure_category,
            expected=expected,
            actual=actual,
            error=error,
            filler_stats=filler_stats,
            eval_time_seconds=elapsed,
        )

        self.results.append(test_result)
        return test_result

    def save_results(self, filename: str = "results.json") -> str:
        """Save all collected results to a JSON file."""
        os.makedirs(self.results_dir, exist_ok=True)
        filepath = os.path.join(self.results_dir, filename)

        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.client.model,
                "total_tests": len(self.results),
                "dimensions_tested": list(set(r.dimension for r in self.results)),
            },
            "results": [
                {
                    "dimension": r.dimension,
                    "depth": r.depth,
                    "depth_unit": r.depth_unit,
                    "run": r.run,
                    "score": r.score,
                    "failure_category": r.failure_category,
                    "expected": r.expected,
                    "actual": r.actual,
                    "error": r.error,
                    "filler_stats": r.filler_stats,
                    "eval_time_seconds": r.eval_time_seconds,
                    "system_prompt": r.system_prompt,
                    "messages": r.messages,
                    "test_query": r.test_query,
                    "raw_output": r.raw_output,
                    "stripped_output": r.stripped_output,
                }
                for r in self.results
            ],
            "summary": self._compute_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def _compute_summary(self) -> Dict:
        """Compute summary statistics across dimensions and depths."""
        from collections import defaultdict

        dim_depth_scores = defaultdict(list)
        dim_depth_categories = defaultdict(list)
        for r in self.results:
            key = (r.dimension, r.depth, r.depth_unit)
            dim_depth_scores[key].append(r.score)
            dim_depth_categories[key].append(r.failure_category)

        summary = {}
        for (dim, depth, unit), scores in sorted(dim_depth_scores.items()):
            if dim not in summary:
                summary[dim] = {}
            mean_score = sum(scores) / len(scores)
            cat_counts = defaultdict(int)
            for c in dim_depth_categories[(dim, depth, unit)]:
                cat_counts[c] += 1
            dominant_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "unknown"

            key_str = f"{depth}_{unit}"
            summary[dim][key_str] = {
                "depth": depth,
                "unit": unit,
                "num_runs": len(scores),
                "mean_score": round(mean_score, 4),
                "std_score": round(
                    (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5,
                    4,
                ) if len(scores) > 1 else 0.0,
                "scores": [round(s, 4) for s in scores],
                "dominant_failure_category": dominant_cat,
                "category_counts": dict(cat_counts),
                "status": "PASS" if mean_score >= 0.8 else (
                    "PARTIAL" if mean_score >= 0.3 else "FAIL"
                ),
            }

        return summary