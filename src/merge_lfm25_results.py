"""
Merge LFM2.5 Results — Keep 24 IAD tests from results_lfm25.json,
reconstruct 5 original dimensions (78 tests) from run.log.

Outputs:
  - results/full_suite/lfm25/results_merged.json (6 dims, 102 tests)
  - Overwrites results/full_suite/lfm25/results.json with same merged data
  - Updated results/full_suite/lfm25/summary.md
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "full_suite", "lfm25")

MODEL_ID = "hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M"

# Depth unit per dimension: hallucination_onset uses "tokens", all others use "turns"
DEPTH_UNIT = {
    "memory_retrieval": "turns",
    "hallucination_onset": "tokens",
    "tool_call_drift": "turns",
    "persona_consistency": "turns",
    "recency_bias": "turns",
    "instruction_adherence_decay": "turns",
}


# ─── 1. Parse run.log to reconstruct the 5 original dimensions ───

def parse_run_log(log_path: str) -> list:
    """
    Parse the entire run.log and reconstruct all test results from the 5 original dimensions.

    Returns: list of result dicts
    """
    if not os.path.exists(log_path):
        print(f"  ❌ run.log not found at {log_path}")
        return []

    with open(log_path) as f:
        content = f.read()

    all_results = []

    # Find each dimension section by its header "Dimension N/5: DIMENSION_NAME"
    dim_pattern = r"Dimension\s+\d+/5:\s+(\w+)"
    dim_matches = list(re.finditer(dim_pattern, content))

    for i, dim_match in enumerate(dim_matches):
        dim_name = dim_match.group(1)
        start_pos = dim_match.start()

        # End position: next dimension header or end of content
        if i + 1 < len(dim_matches):
            end_pos = dim_matches[i + 1].start()
        else:
            end_pos = len(content)

        section = content[start_pos:end_pos]

        # Map log dimension name to our canonical name
        if dim_name not in DEPTH_UNIT:
            print(f"  ⚠ Unknown dimension '{dim_name}' in log, skipping")
            continue

        actual_name = dim_name
        unit = DEPTH_UNIT[actual_name]

        # Pattern: Depth N: run M/3 ... score=X.XX [STATUS] cat=CATEGORY (XXX.Xs)
        pattern = r"Depth\s+(\d+):\s+run\s+(\d+)/\d+\s+.*?score=([\d.]+)\s+\[(PASS|FAIL|PARTIAL)\]\s+cat=(\S+)\s+\(([\d.]+)s\)"

        results = []
        for match in re.finditer(pattern, section):
            depth = int(match.group(1))
            run_num = int(match.group(2))
            score = float(match.group(3))
            status = match.group(4)
            category = match.group(5)
            eval_time = float(match.group(6))

            result = {
                "dimension": actual_name,
                "depth": depth,
                "depth_unit": unit,
                "run": run_num,
                "score": score,
                "failure_category": category,
                "expected": "",
                "actual": "",
                "error": None,
                "filler_stats": {"num_turns": depth, "approx_tokens": depth * 120},
                "eval_time_seconds": eval_time,
                "system_prompt": "",
                "messages": [],
                "test_query": "",
                "raw_output": "",
                "stripped_output": "",
            }
            results.append(result)

        print(f"  → {actual_name}: {len(results)} tests")
        all_results.extend(results)

    return all_results


# ─── 2. Load IAD results from results_lfm25.json ───

def load_iad_results(filepath: str) -> list:
    """Load instruction_adherence_decay results from the checkpoint JSON."""
    if not os.path.exists(filepath):
        print(f"  ❌ IAD results not found at {filepath}")
        return []

    with open(filepath) as f:
        data = json.load(f)

    all_results = data.get("results", [])
    iad_results = [r for r in all_results if r.get("dimension") == "instruction_adherence_decay"]
    print(f"  → Loaded {len(iad_results)} instruction_adherence_decay tests from {filepath}")
    return iad_results


# ─── 3. Build summary ───

def build_summary(all_results: list) -> dict:
    """Build per-dimension summary from all results."""
    dims_in_results = set(r["dimension"] for r in all_results)
    dims_sorted = sorted(dims_in_results)

    dim_summary = {}
    for dim in dims_sorted:
        dim_tests = [r for r in all_results if r["dimension"] == dim]
        scores = [r.get("score", 0) or 0 for r in dim_tests]
        categories = defaultdict(int)
        for r in dim_tests:
            cat = r.get("failure_category", "unknown") or "unknown"
            categories[cat] += 1

        # Group by depth
        depths = defaultdict(list)
        for r in dim_tests:
            depths[r["depth"]].append(r)

        depth_means = []
        break_point = None
        for d in sorted(depths.keys()):
            d_scores = [r["score"] for r in depths[d]]
            mean = sum(d_scores) / len(d_scores)
            depth_means.append({
                "depth": d,
                "mean_score": round(mean, 4),
                "num_runs": len(d_scores),
            })
            if mean < 0.8 and break_point is None:
                break_point = d

        unit = DEPTH_UNIT.get(dim, "turns")
        dim_summary[dim] = {
            "total_tests": len(dim_tests),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "break_point": break_point,
            "break_point_label": f"{break_point} {unit}" if break_point else "None (all \u2265 0.8)",
            "depth_means": depth_means,
            "failure_categories": dict(categories),
        }

    return dim_summary


# ─── 4. Merge ───

def merge_results(iad_results: list, log_results: list) -> dict:
    """Merge IAD results with reconstructed log results."""
    all_results = iad_results + log_results
    dims_in_results = sorted(set(r["dimension"] for r in all_results))
    dim_summary = build_summary(all_results)

    merged_metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "total_tests": len(all_results),
        "dimensions_tested": dims_in_results,
    }

    return {
        "metadata": merged_metadata,
        "results": all_results,
        "summary": dim_summary,
    }


# ─── 5. Generate Summary Markdown ───

def generate_summary_md(merged_data: dict) -> str:
    """Generate a comprehensive summary.md from merged results."""
    md = "# Full Suite Evaluation \u2014 LFM2.5-8B-A1B\n\n"
    md += f"**Date**: {merged_data['metadata']['timestamp']}\n\n"
    md += f"**Model**: {merged_data['metadata']['model']}\n\n"
    md += f"**Total test cases**: {merged_data['metadata']['total_tests']}\n\n"

    # Summary Overview table
    dim_summary = merged_data["summary"]
    dims_sorted = sorted(dim_summary.keys())

    md += "## Summary Overview\n\n"
    md += "| Dimension | Tests | Mean Score | Break Point |\n"
    md += "|-----------|-------|------------|-------------|\n"
    for dim in dims_sorted:
        s = dim_summary[dim]
        total = s["total_tests"]
        mean = f"{s['mean_score']:.4f}"
        bp = s.get("break_point_label", "N/A")
        dim_label = dim.replace("_", " ").title()
        md += f"| {dim_label} | {total} | {mean} | {bp} |\n"

    # Per-Dimension Results
    md += "\n---\n\n## Per-Dimension Results\n"

    for dim in dims_sorted:
        dim_tests = [r for r in merged_data["results"] if r["dimension"] == dim]
        s = dim_summary[dim]

        md += f"\n### {dim.replace('_', ' ').title()}\n\n"
        md += f"**Tests**: {s['total_tests']} | **Mean Score**: {s['mean_score']:.4f} | **Break Point**: {s.get('break_point_label', 'N/A')}\n\n"

        # Depth means table
        md += "| Depth | Mean Score | Runs |\n"
        md += "|-------|-----------|------|\n"
        for dm in s["depth_means"]:
            md += f"| {dm['depth']} | {dm['mean_score']:.4f} | {dm['num_runs']} |\n"

        md += "\n**Per-Test Results:**\n\n"
        md += "| Run | Depth | Score | Category | Eval Time |\n"
        md += "|-----|-------|-------|----------|-----------|\n"

        for r in sorted(dim_tests, key=lambda x: (x["depth"], x["run"])):
            depth = r["depth"]
            run_num = r["run"]
            score = r["score"]
            cat = r.get("failure_category", "?")
            ev_time = r.get("eval_time_seconds", 0)
            score_icon = "\U00002705" if score >= 0.80 else ("\u26a0\ufe0f" if score >= 0.30 else "\u274c")
            md += f"| {run_num} | {depth} | {score_icon} {score:.2f} | {cat} | {ev_time:.1f}s |\n"

        # Failure category distribution
        cats = s["failure_categories"]
        md += "\n**Failure Category Distribution:**\n\n"
        md += "| Category | Count |\n"
        md += "|----------|-------|\n"
        for cat_name, count in sorted(cats.items(), key=lambda x: -x[1]):
            md += f"| {cat_name} | {count} |\n"

        md += "\n---\n"

    # Overall failure category distribution
    md += "\n## Overall Failure Category Distribution\n\n"
    all_cats = defaultdict(int)
    for r in merged_data["results"]:
        cat = r.get("failure_category", "unknown") or "unknown"
        all_cats[cat] += 1
    md += "| Category | Count | Percentage |\n"
    md += "|----------|-------|-----------|\n"
    total = len(merged_data["results"])
    for cat_name, count in sorted(all_cats.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        md += f"| {cat_name} | {count} | {pct:.1f}% |\n"

    md += "\n## Scoring Guidelines\n\n"
    md += "- **PASS** (\u2265 0.80): Model maintains expected behavior\n"
    md += "- **PARTIAL** (0.30 - 0.79): Some degradation observed\n"
    md += "- **FAIL** (< 0.30): Significant breakdown\n"

    return md


# ─── 6. Main ───

def main():
    print("=" * 70)
    print("  MERGE LFM2.5 RESULTS")
    print("=" * 70)

    iad_path = os.path.join(RESULTS_DIR, "results_lfm25.json")
    log_path = os.path.join(RESULTS_DIR, "run.log")

    # Step 1: Load IAD results (full data from results_lfm25.json)
    print(f"\n[1] Loading IAD results from {iad_path}...")
    iad_results = load_iad_results(iad_path)
    print(f"  \u2192 {len(iad_results)} instruction_adherence_decay tests loaded")

    # Step 2: Parse run.log to reconstruct 5 original dimensions
    print(f"\n[2] Parsing run.log from {log_path}...")
    log_results = parse_run_log(log_path)
    print(f"  \u2192 {len(log_results)} tests reconstructed from run.log")

    # Step 3: Check dimension coverage
    log_dims = set(r["dimension"] for r in log_results)
    expected_original_dims = {
        "memory_retrieval", "hallucination_onset", "tool_call_drift",
        "persona_consistency", "recency_bias",
    }
    missing = expected_original_dims - log_dims
    if missing:
        print(f"  \u26a0 Missing dimensions from log: {missing}")
    else:
        print(f"  \u2705 All 5 original dimensions found in run.log")

    # Step 4: Merge
    total = len(iad_results) + len(log_results)
    print(f"\n[3] Merging {len(iad_results)} IAD + {len(log_results)} log-reconstructed = {total} total...")
    merged_data = merge_results(iad_results, log_results)

    merged_dims = set(r["dimension"] for r in merged_data["results"])
    print(f"  \u2192 Final: {len(merged_data['results'])} tests across {sorted(merged_dims)}")

    # Step 5: Save to results_merged.json
    merged_path = os.path.join(RESULTS_DIR, "results_merged.json")
    with open(merged_path, "w") as f:
        json.dump(merged_data, f, indent=2, default=str)
    print(f"\n[4] \u2705 Saved merged results to {merged_path}")

    # Step 6: Overwrite results.json
    results_json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(merged_data, f, indent=2, default=str)
    print(f"  \u2705 Overwrote {results_json_path} with merged data")

    # Step 7: Generate summary.md
    summary_md = generate_summary_md(merged_data)
    summary_md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(summary_md_path, "w") as f:
        f.write(summary_md)
    print(f"  \u2705 Wrote updated summary to {summary_md_path}")

    # Step 8: Verify
    print("\n[5] Verification:")
    with open(merged_path) as f:
        verify = json.load(f)
    verify_dims = set(r["dimension"] for r in verify["results"])
    expected_dims = {
        "instruction_adherence_decay",
        "memory_retrieval",
        "hallucination_onset",
        "tool_call_drift",
        "persona_consistency",
        "recency_bias",
    }
    print(f"  Dimensions: {sorted(verify_dims)}")
    print(f"  Total tests: {len(verify['results'])}")
    print(f"  metadata.total_tests: {verify['metadata']['total_tests']}")
    print(f"  All 6 dimensions present: {verify_dims == expected_dims}")

    if verify_dims == expected_dims and verify["metadata"]["total_tests"] == len(verify["results"]):
        dim_counts = defaultdict(int)
        for r in verify["results"]:
            dim_counts[r["dimension"]] += 1
        for d in sorted(dim_counts):
            print(f"    {d}: {dim_counts[d]} tests")
        print(f"\n  \u2705 Merge complete \u2014 all 6 dimensions present, total={verify['metadata']['total_tests']}!")
    else:
        missing_dims = expected_dims - verify_dims
        print(f"\n  \u274c Missing dimensions: {missing_dims}")
        sys.exit(1)

    # Step 9: Print summary
    print("\n" + "=" * 70)
    print("  MERGE SUMMARY")
    print("=" * 70)
    for dim in sorted(verify["summary"]):
        s = verify["summary"][dim]
        bp = s.get("break_point_label", "N/A")
        mean = s.get("mean_score", "N/A")
        cats = s.get("failure_categories", {})
        print(f"  {dim}:")
        print(f"    Tests: {s['total_tests']}, Mean: {mean}, Break: {bp}")
        if cats:
            cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
            print(f"    Categories: {cat_str}")
        print()


if __name__ == "__main__":
    main()