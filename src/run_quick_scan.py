"""
Quick Scan Entry Point for SLM Agent Degradation Evaluation.

Runs each dimension once per depth level (quick scan mode).
Saves raw results and generates a human-readable summary.
"""

import json
import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_harness import OllamaClient, EvalHarness, TestResult
from test_scenarios import (
    get_available_dimensions,
    get_depths,
    generate_test_case,
)

# Project root
PROJECT_ROOT = "/home/azureuser/local_llm_eval"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def run_quick_scan():
    """Execute quick scan: 1 run per depth for all dimensions."""
    print("=" * 70)
    print("  SLM AGENT DEGRADATION EVALUATION — QUICK SCAN")
    print(f"  Started at: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Initialize client and harness
    client = OllamaClient()
    harness = EvalHarness(client=client, results_dir=RESULTS_DIR)
    
    # Verify Ollama is reachable
    print("\n[1/5] Checking Ollama connectivity...")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        print(f"  ✅ Ollama reachable at localhost:11434")
        print(f"  Model: {client.model}")
    except Exception as e:
        print(f"  ❌ Cannot reach Ollama: {e}")
        sys.exit(1)
    
    # Get dimensions and depths
    dimensions = get_available_dimensions()
    print(f"\n[2/5] Loaded {len(dimensions)} evaluation dimensions:")
    for dim in dimensions:
        depths = get_depths(dim)
        print(f"  • {dim}: depths={depths}")
    
    total_cases = sum(len(get_depths(d)) for d in dimensions)
    print(f"\n[3/5] Total test cases to run: {total_cases}")
    
    # Run each dimension
    print(f"\n[4/5] Running evaluation...")
    print("=" * 70)
    
    overall_start = time.time()
    completed = 0
    failed = 0
    
    for dim_idx, dimension in enumerate(dimensions):
        depths = get_depths(dimension)
        print(f"\n{'─' * 70}")
        print(f"  Dimension {dim_idx + 1}/{len(dimensions)}: {dimension}")
        print(f"{'─' * 70}")
        
        for depth_idx, depth in enumerate(depths):
            case_start = time.time()
            print(f"\n  Depth {depth_idx + 1}/{len(depths)}: ", end="", flush=True)
            
            try:
                # Generate the test case
                test_case = generate_test_case(client, dimension, depth)
                
                print(f"generating filler... ", end="", flush=True)
                
                # Run the test
                result = harness.run_single(
                    dimension=test_case["dimension"],
                    depth=test_case["depth"],
                    depth_unit=test_case["depth_unit"],
                    run_num=1,
                    system_prompt=test_case["system_prompt"],
                    conversation=test_case["conversation"],
                    test_query=test_case["test_query"],
                    scorer=test_case["scorer"],
                    expected=test_case["expected"],
                )
                
                case_elapsed = time.time() - case_start
                status = "PASS" if result.score >= 0.8 else (
                    "PARTIAL" if result.score >= 0.3 else "FAIL"
                )
                
                print(f"score={result.score:.2f} [{status}] ({case_elapsed:.1f}s)")
                
                # Print brief output preview
                if result.stripped_output:
                    preview = result.stripped_output[:120].replace('\n', ' ')
                    print(f"  → Response: \"{preview}...\"")
                if result.error:
                    print(f"  ⚠ Error: {result.error}")
                
                completed += 1
                
            except Exception as e:
                case_elapsed = time.time() - case_start
                print(f"ERROR ({case_elapsed:.1f}s): {e}")
                failed += 1
                
                # Still record the failure
                harness.results.append(TestResult(
                    dimension=dimension,
                    depth=depth,
                    depth_unit="turns" if "turns" in str(depth) else "tokens",
                    run=1,
                    system_prompt="",
                    messages=[],
                    test_query="",
                    raw_output="",
                    stripped_output="",
                    score=0.0,
                    error=str(e),
                    eval_time_seconds=case_elapsed,
                ))
        
        # Checkpoint save after each dimension (survives timeout)
        print(f"\n  ⏺ Checkpoint saving after dimension {dim_idx + 1}...")
        harness.save_results("quick_scan_results.json")
        print(f"     Results saved ({len(harness.results)} tests so far)")
    
    overall_elapsed = time.time() - overall_start
    
    # Save results
    print(f"\n{'=' * 70}")
    print(f"[5/5] Saving results...")
    results_file = harness.save_results("quick_scan_results.json")
    print(f"  ✅ Results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  QUICK SCAN COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Completed: {completed}/{total_cases} test cases")
    print(f"  Failed:    {failed}")
    print(f"  Duration:  {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"  Results:   {results_file}")
    print(f"{'=' * 70}")
    
    return harness


def generate_summary(harness: EvalHarness):
    """Generate a human-readable summary report."""
    summary_path = os.path.join(RESULTS_DIR, "quick_scan_summary.md")
    
    with open(summary_path, "w") as f:
        f.write("# SLM Agent Degradation Evaluation — Quick Scan Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Model**: {harness.client.model}\n\n")
        f.write(f"**Total test cases**: {len(harness.results)}\n\n")
        f.write("---\n\n")
        
        # Group results by dimension
        from collections import defaultdict
        dim_results = defaultdict(list)
        for r in harness.results:
            dim_results[r.dimension].append(r)
        
        # Summary overview table
        f.write("## Summary Overview\n\n")
        f.write("| Dimension | Depths | Best Score | Worst Score | Break Point |\n")
        f.write("|-----------|--------|-----------|-------------|-------------|\n")
        
        for dim_name in sorted(dim_results.keys()):
            results = dim_results[dim_name]
            scores = [r.score for r in results if r.score is not None]
            if not scores:
                continue
            best = max(scores)
            worst = min(scores)
            depths_str = f"{min(r.depth for r in results)}-{max(r.depth for r in results)}"
            
            # Find break point (first depth where score < 0.8)
            break_depth = None
            break_unit = None
            for r in sorted(results, key=lambda x: x.depth):
                if r.score is not None and r.score < 0.8:
                    break_depth = r.depth
                    break_unit = r.depth_unit
                    break
            
            break_str = f"{break_depth} {break_unit}" if break_depth else "None (all ≥ 0.8)"
            display_name = dim_name.replace("_", " ").title()
            
            f.write(f"| {display_name} | {depths_str} | {best:.2f} | {worst:.2f} | {break_str} |\n")
        
        f.write("\n---\n\n")
        
        # Per-dimension detail sections
        f.write("## Per-Dimension Results\n\n")
        
        for dim_name in sorted(dim_results.keys()):
            results = sorted(dim_results[dim_name], key=lambda x: x.depth)
            display_name = dim_name.replace("_", " ").title()
            
            f.write(f"### {display_name}\n\n")
            
            # Dimension description
            descriptions = {
                "instruction_adherence_decay": (
                    "Tests whether the model maintains JSON format adherence "
                    "after N turns of non-JSON conversation."
                ),
                "memory_retrieval": (
                    "Tests whether the model can recall a specific fact "
                    "('My dog is named Einstein') injected at turn 1, "
                    "after N turns of unrelated filler."
                ),
                "hallucination_onset": (
                    "Tests factual recall accuracy from early context "
                    "at escalating token depths (1K-8K tokens of filler)."
                ),
                "tool_call_drift": (
                    "Tests whether JSON tool-call format degrades after "
                    "many turns of alternating tool-call and normal responses."
                ),
                "persona_consistency": (
                    "Tests whether the model maintains its assigned "
                    "Dr. Sarah Chen / marine biologist persona after N turns."
                ),
                "recency_bias": (
                    "Tests whether the model follows an early instruction "
                    "('Aye aye, captain!' prefix) or a later override, "
                    "with varying distance between the two instructions."
                ),
            }
            f.write(f"{descriptions.get(dim_name, '')}\n\n")
            
            # Score table
            f.write("| Depth | Score | Status | Response Preview |\n")
            f.write("|-------|-------|--------|-----------------|\n")
            
            for r in results:
                status = "✅ PASS" if r.score >= 0.8 else (
                    "⚠️ PARTIAL" if r.score >= 0.3 else "❌ FAIL"
                )
                depth_str = f"{r.depth} {r.depth_unit}"
                preview = (r.stripped_output[:100].replace('\n', ' ')
                          if r.stripped_output else "[empty/error]")
                
                f.write(f"| {depth_str} | {r.score:.2f} | {status} | {preview} |\n")
            
            f.write("\n")
            
            # Error details
            errors = [r for r in results if r.error]
            if errors:
                f.write("**Errors**:\n")
                for r in errors:
                    f.write(f"- Depth {r.depth}: {r.error}\n")
                f.write("\n")
        
        # Scoring Guidelines
        f.write("---\n\n")
        f.write("## Scoring Guidelines\n\n")
        f.write("- **PASS** (≥ 0.80): Model maintains expected behavior\n")
        f.write("- **PARTIAL** (0.30 - 0.79): Some degradation observed\n")
        f.write("- **FAIL** (< 0.30): Significant breakdown\n\n")
        
        # Recommendations
        f.write("## Key Observations\n\n")
        
        # Find overall break points
        all_results = sorted(harness.results, key=lambda r: r.depth)
        low_scores = [(r.dimension, r.depth, r.depth_unit, r.score)
                      for r in all_results if r.score is not None and r.score < 0.8]
        
        if low_scores:
            f.write("### Degradation Detected At:\n\n")
            f.write("| Dimension | Depth | Score |\n")
            f.write("|-----------|-------|-------|\n")
            for dim, depth, unit, score in low_scores:
                f.write(f"| {dim.replace('_', ' ').title()} | {depth} {unit} | {score:.2f} |\n")
            
            # Earliest break point
            earliest = min(low_scores, key=lambda x: x[1])
            f.write(f"\n**Earliest degradation**: {earliest[0].replace('_', ' ').title()} "
                   f"at depth {earliest[1]} {earliest[2]} (score: {earliest[3]:.2f})\n\n")
        else:
            f.write("All dimensions maintained acceptable performance (≥ 0.80) "
                   "across all tested depths.\n\n")
    
    print(f"  ✅ Summary report saved to: {summary_path}")
    return summary_path


def main():
    try:
        harness = run_quick_scan()
        print("\nGenerating summary report...")
        summary_path = generate_summary(harness)
        print(f"\nDone! Results: {os.path.join(RESULTS_DIR, 'quick_scan_results.json')}")
        print(f"Summary: {summary_path}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Saving partial results...")
        harness = globals().get('harness')
        if harness and harness.results:
            harness.save_results("quick_scan_results_interrupted.json")
            print("Partial results saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()