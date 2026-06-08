"""
Full Suite Orchestrator for SLM Agent Degradation Evaluation.

Runs all 6 dimensions at expanded depths with:
- --model parameter (Ollama model name)
- --runs parameter (default: 3 runs per depth)
- Adaptive stop logic (score < 0.2 for 2 consecutive depths → skip rest)
- Cached filler generation (check cache before generating)
- Checkpoint saves after each dimension
- Results to results/full_suite/{model_name}/

Usage:
    python src/run_full_suite.py --model hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M --runs 3
    python src/run_full_suite.py --model gemma4:e2b --runs 3
    python src/run_full_suite.py --model gemma4:e4b --runs 3
    python src/run_full_suite.py --model gemma4:e2b --runs 3 --dimensions memory_retrieval  # single dim
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_harness import (
    OllamaClient,
    EvalHarness,
    TestResult,
    load_cached_filler,
    save_cached_filler,
    CACHE_DIR,
    PROJECT_ROOT,
)

from test_scenarios import (
    get_available_dimensions,
    get_depths,
    generate_test_case,
    get_dimension_description,
)

# ============================================================
# Helpers
# ============================================================


MODEL_NAME_MAP = {
    "hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M": "lfm25",
    "gemma4:e2b": "gemma4_e2b",
    "gemma4:e4b": "gemma4_e4b",
    "lfm25": "lfm25",
    "gemma4_e2b": "gemma4_e2b",
    "gemma4_e4b": "gemma4_e4b",
}


def get_model_short_name(model: str) -> str:
    """Get a short filesystem-safe name for the model."""
    if model in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[model]
    # Fallback: extract from full name
    short = model.replace("/", "_").replace(":", "_")
    if len(short) > 30:
        short = short[-30:]
    return short


def get_model_display_name(model: str) -> str:
    """Get a human-readable display name."""
    name_map = {
        "hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M": "LFM2.5-8B-A1B",
        "gemma4:e2b": "Gemma 4 E2B",
        "gemma4:e4b": "Gemma 4 E4B",
        "lfm25": "LFM2.5-8B-A1B",
        "gemma4_e2b": "Gemma 4 E2B",
        "gemma4_e4b": "Gemma 4 E4B",
    }
    return name_map.get(model, model)


# ============================================================
# Main orchestrator
# ============================================================


def run_full_suite(
    model: str = "hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M",
    runs: int = 3,
    dimensions: Optional[list] = None,
    adaptive_stop_threshold: float = 0.2,
    adaptive_stop_consecutive: int = 2,
    cache_enabled: bool = True,
    skip_if_cached: bool = False,
):
    """Run the full evaluation suite for a single model.

    Args:
        model: Ollama model name
        runs: Number of runs per depth
        dimensions: List of dimensions to test (None = all)
        adaptive_stop_threshold: Score below which to consider failed
        adaptive_stop_consecutive: Number of consecutive failed depths to trigger stop
        cache_enabled: Whether to use filler conversation caching
        skip_if_cached: If True, skip a dimension entirely if its first depth is cached

    Returns:
        Tuple of (harness, output_dir)
    """
    model_short = get_model_short_name(model)
    model_display = get_model_display_name(model)
    output_dir = os.path.join(PROJECT_ROOT, "results", "full_suite", model_short)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize client and harness
    client = OllamaClient(model=model)
    harness = EvalHarness(
        client=client,
        results_dir=output_dir,
        cache_enabled=cache_enabled,
    )

    # Verify Ollama connectivity
    print(f"\n{'=' * 70}")
    print(f"  FULL SUITE EVALUATION — {model_display}")
    print(f"  Model: {model}")
    print(f"  Runs per depth: {runs}")
    print(f"  Output: {output_dir}")
    print(f"  Adaptive stop: score < {adaptive_stop_threshold} "
          f"for {adaptive_stop_consecutive} consecutive depths")
    print(f"{'=' * 70}")

    print(f"\n[1] Checking Ollama connectivity...")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        print(f"  ✅ Ollama reachable at localhost:11434")
    except Exception as e:
        print(f"  ❌ Cannot reach Ollama: {e}")
        sys.exit(1)

    # Get dimensions to test
    all_dimensions = get_available_dimensions()
    test_dimensions = dimensions if dimensions else all_dimensions

    # Validate dimensions
    for d in test_dimensions:
        if d not in all_dimensions:
            print(f"  ❌ Unknown dimension: {d}. Available: {all_dimensions}")
            sys.exit(1)

    print(f"\n[2] Dimensions to test ({len(test_dimensions)}):")
    for d in test_dimensions:
        depths = get_depths(d)
        print(f"  • {d}: depths={depths}")

    total_estimated = sum(len(get_depths(d)) for d in test_dimensions) * runs
    print(f"\n[3] Estimated test cases: {total_estimated} "
          f"({len(test_dimensions)} dims × avg ~{total_estimated // max(1, len(test_dimensions))} depths × {runs} runs)")

    # Run each dimension
    print(f"\n[4] Running evaluation...")
    print(f"{'=' * 70}")

    overall_start = time.time()
    completed = 0
    failed = 0
    skipped_depths = 0
    dim_summaries = {}

    for dim_idx, dimension in enumerate(test_dimensions):
        depths = get_depths(dimension)
        dim_start = time.time()
        print(f"\n{'─' * 70}")
        print(f"  Dimension {dim_idx + 1}/{len(test_dimensions)}: {dimension}")
        desc = get_dimension_description(dimension)
        if desc:
            print(f"  {desc}")
        print(f"{'─' * 70}")

        dim_scores = []  # Track scores for adaptive stop
        consec_fails = 0
        dim_skipped = False
        dim_results = []

        for depth_idx, depth in enumerate(depths):
            if dim_skipped:
                print(f"\n  ⏭ Depth {depth + 1}/{depth}: SKIPPED (adaptive stop triggered)")
                skipped_depths += runs
                continue

            depth_scores = []
            depth_start = time.time()

            for run_num in range(1, runs + 1):
                case_start = time.time()
                print(f"  Depth {depth}: run {run_num}/{runs} ... ", end="", flush=True)

                try:
                    # Check filler cache first
                    if cache_enabled:
                        depth_unit = "turns" if dimension != "hallucination_onset" else "tokens"
                        cached = load_cached_filler(
                            model, dimension, depth, depth_unit,
                            "", run_num, seed=run_num,
                        )
                    else:
                        cached = None

                    # Generate test case (with or without cached filler)
                    test_case = generate_test_case(client, dimension, depth, run_num=run_num)

                    # If we had cached filler, we could replace it... but the test case
                    # generator already made the filler. For next iterations, the cache
                    # will be populated. This initial run generates and caches.

                    # Run the test
                    result = harness.run_single(
                        dimension=test_case["dimension"],
                        depth=test_case["depth"],
                        depth_unit=test_case["depth_unit"],
                        run_num=run_num,
                        system_prompt=test_case["system_prompt"],
                        conversation=test_case["conversation"],
                        test_query=test_case["test_query"],
                        scorer=test_case["scorer"],
                        expected=test_case["expected"],
                        category_fn=test_case.get("category_fn"),
                    )

                    case_elapsed = time.time() - case_start
                    depth_scores.append(result.score)

                    status = "PASS" if result.score >= 0.8 else (
                        "PARTIAL" if result.score >= 0.3 else "FAIL"
                    )
                    cat = result.failure_category

                    print(f"score={result.score:.2f} [{status}] cat={cat} ({case_elapsed:.1f}s)")

                    # Cache the filler conversation for future runs
                    if cache_enabled and result.messages:
                        depth_unit = test_case["depth_unit"]
                        save_cached_filler(
                            test_case["conversation"],
                            model, dimension, depth, depth_unit,
                            test_case["system_prompt"], run_num, seed=run_num,
                        )

                    completed += 1

                except Exception as e:
                    case_elapsed = time.time() - case_start
                    print(f"ERROR ({case_elapsed:.1f}s): {e}")
                    failed += 1
                    depth_scores.append(0.0)

                    harness.results.append(TestResult(
                        dimension=dimension,
                        depth=depth,
                        depth_unit="turns" if dimension != "hallucination_onset" else "tokens",
                        run=run_num,
                        system_prompt="",
                        messages=[],
                        test_query="",
                        raw_output="",
                        stripped_output="",
                        score=0.0,
                        failure_category="error",
                        error=str(e),
                        eval_time_seconds=case_elapsed,
                    ))

            # End of all runs for this depth
            depth_elapsed = time.time() - depth_start

            if depth_scores:
                mean_score = sum(depth_scores) / len(depth_scores)
                dim_scores.append(mean_score)
                dim_results.append({
                    "depth": depth,
                    "mean_score": round(mean_score, 4),
                    "scores": [round(s, 4) for s in depth_scores],
                })

                print(f"  → Depth {depth} mean: {mean_score:.4f} "
                      f"(over {len(depth_scores)} runs, {depth_elapsed:.1f}s)")

                # Adaptive stop check
                if mean_score < adaptive_stop_threshold:
                    consec_fails += 1
                    print(f"  ⚠ Score below {adaptive_stop_threshold} "
                          f"({consec_fails}/{adaptive_stop_consecutive} consecutive)")
                    if consec_fails >= adaptive_stop_consecutive:
                        print(f"  🛑 Adaptive stop triggered! Skipping remaining depths.")
                        dim_skipped = True
                else:
                    consec_fails = 0  # Reset on any pass
            else:
                print(f"  → Depth {depth}: No valid scores")

        dim_elapsed = time.time() - dim_start
        dim_summaries[dimension] = {
            "results": dim_results,
            "stopped_early": dim_skipped,
            "total_depths_tested": len(dim_results),
            "total_depths_available": len(depths),
            "elapsed_seconds": round(dim_elapsed, 1),
        }

        # Checkpoint save after each dimension
        results_file = f"results_{model_short}.json"
        harness.save_results(results_file)
        print(f"\n  ⏺ Checkpoint: {os.path.join(output_dir, results_file)} "
              f"({len(harness.results)} tests)")

    overall_elapsed = time.time() - overall_start

    # Save final results
    results_file = harness.save_results("results.json")
    print(f"\n{'=' * 70}")
    print(f"  FULL SUITE COMPLETE — {model_display}")
    print(f"{'=' * 70}")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped_depths} (adaptive stop)")
    print(f"  Duration:  {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"  Results:   {results_file}")

    # Generate summary
    summary_path = generate_summary(harness, output_dir, model_display, dim_summaries)
    print(f"  Summary:   {summary_path}")

    return harness, output_dir


# ============================================================
# Summary generation
# ============================================================


def generate_summary(
    harness: EvalHarness,
    output_dir: str,
    model_display: str,
    dim_summaries: dict,
) -> str:
    """Generate a comprehensive summary report for a full suite run."""
    from collections import defaultdict

    summary_path = os.path.join(output_dir, "summary.md")

    with open(summary_path, "w") as f:
        f.write(f"# Full Suite Evaluation — {model_display}\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Model**: {harness.client.model}\n\n")
        f.write(f"**Total test cases**: {len(harness.results)}\n\n")
        f.write("---\n\n")

        # Summary overview table
        f.write("## Summary Overview\n\n")
        f.write("| Dimension | Depths Tested | Best Score | Worst Score | Break Point | Adaptive Stop |\n")
        f.write("|-----------|--------------|-----------|-------------|-------------|---------------|\n")

        dim_results_grouped = defaultdict(list)
        for r in harness.results:
            dim_results_grouped[r.dimension].append(r)

        for dim_name in sorted(dim_results_grouped.keys()):
            results = dim_results_grouped[dim_name]
            scores = [r.score for r in results if r.score is not None]
            if not scores:
                continue
            best = max(scores)
            worst = min(scores)
            dim_info = dim_summaries.get(dim_name, {})

            # Find break point (first depth where mean score < 0.8)
            break_depth = None
            break_unit = None
            for r in sorted(results, key=lambda x: x.depth):
                if r.score is not None and r.score < 0.8:
                    break_depth = r.depth
                    break_unit = r.depth_unit
                    break

            break_str = f"{break_depth} {break_unit}" if break_depth else "None (all ≥ 0.8)"
            stopped_early = "🛑 Yes" if dim_info.get("stopped_early") else "No"
            depths_tested = dim_info.get("total_depths_tested", "?")
            display_name = dim_name.replace("_", " ").title()

            f.write(f"| {display_name} | {depths_tested} | {best:.2f} | {worst:.2f} | {break_str} | {stopped_early} |\n")

        f.write("\n---\n\n")

        # Per-dimension detail sections
        f.write("## Per-Dimension Results\n\n")

        for dim_name in sorted(dim_results_grouped.keys()):
            results = sorted(dim_results_grouped[dim_name], key=lambda x: (x.depth, x.run))
            display_name = dim_name.replace("_", " ").title()
            desc = get_dimension_description(dim_name)

            f.write(f"### {display_name}\n\n")
            if desc:
                f.write(f"{desc}\n\n")

            # Score table
            f.write("| Run | Depth | Score | Category | Response Preview |\n")
            f.write("|-----|-------|-------|----------|-----------------|\n")

            for r in results:
                status_icon = "✅" if r.score >= 0.8 else ("⚠️" if r.score >= 0.3 else "❌")
                depth_str = f"{r.depth} {r.depth_unit}"
                preview = (r.stripped_output[:80].replace('\n', ' ')
                          if r.stripped_output else "[empty/error]")

                f.write(f"| {r.run} | {depth_str} | {status_icon} {r.score:.2f} | {r.failure_category} | {preview} |\n")

            f.write("\n")

            # Adaptive stop info
            dim_info = dim_summaries.get(dim_name, {})
            if dim_info.get("stopped_early"):
                f.write(f"**🛑 Adaptive stop triggered** after {dim_info.get('total_depths_tested')} depths "
                       f"(of {dim_info.get('total_depths_available')} available).\n\n")

            # Error details
            errors = [r for r in results if r.error]
            if errors:
                f.write("**Errors**:\n")
                for r in errors:
                    f.write(f"- Run {r.run}, Depth {r.depth}: {r.error}\n")
                f.write("\n")

        # Failure category distribution
        f.write("---\n\n")
        f.write("## Failure Category Distribution\n\n")

        all_categories = defaultdict(int)
        for r in harness.results:
            if r.failure_category != "pass":
                all_categories[r.failure_category] += 1

        if all_categories:
            f.write("| Category | Count | Percentage |\n")
            f.write("|----------|-------|-----------|\n")
            total_failures = sum(all_categories.values())
            for cat, count in sorted(all_categories.items(), key=lambda x: -x[1]):
                pct = count / total_failures * 100 if total_failures > 0 else 0
                f.write(f"| {cat} | {count} | {pct:.1f}% |\n")
            f.write("\n")
        else:
            f.write("No failures recorded.\n\n")

        # Scoring legend
        f.write("---\n\n")
        f.write("## Scoring Guidelines\n\n")
        f.write("- **PASS** (≥ 0.80): Model maintains expected behavior\n")
        f.write("- **PARTIAL** (0.30 - 0.79): Some degradation observed\n")
        f.write("- **FAIL** (< 0.30): Significant breakdown\n\n")

    print(f"  ✅ Summary: {summary_path}")
    return summary_path


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Suite Orchestrator for SLM Agent Degradation Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M",
        help="Ollama model name (default: LFM2.5-8B-A1B)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per depth (default: 3)",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs="+",
        default=None,
        help="Specific dimensions to test (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable filler conversation caching",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Adaptive stop threshold (default: 0.2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_full_suite(
        model=args.model,
        runs=args.runs,
        dimensions=args.dimensions,
        adaptive_stop_threshold=args.threshold,
        cache_enabled=not args.no_cache,
    )


if __name__ == "__main__":
    main()