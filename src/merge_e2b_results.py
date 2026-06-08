"""
Merge E2B Results — Reconstruct instruction_adherence_decay from run.log,
merge with 5 completed dimensions from results.json.

Outputs:
  - results/full_suite/gemma4_e2b/results_merged.json (6 dims, 96 tests)
  - Overwrites results/full_suite/gemma4_e2b/results.json with same merged data
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "full_suite", "gemma4_e2b")

# ─── 1. Parse run.log to reconstruct instruction_adherence_decay ───

def parse_instruction_adherence_decay(log_path: str) -> list:
    """
    Parse run.log to extract instruction_adherence_decay results.

    Expected log format:
      Depth N: run M/3 ... score=X.XX [PASS/FAIL] cat=CATEGORY (XXX.Xs)
    """
    results = []
    log_path = os.path.join(RESULTS_DIR, "run.log")

    if not os.path.exists(log_path):
        print(f"  ❌ run.log not found at {log_path}")
        return results

    with open(log_path) as f:
        content = f.read()

    # Find the instruction_adherence_decay section by locating its header and
    # the start of the next dimension (or EOF)
    iad_start = content.find("Dimension 1/6: instruction_adherence_decay")
    iad_end = content.find("Dimension 2/6:")
    if iad_start == -1:
        # Try finding the section by the line with the dimension name
        for line in content.split("\n"):
            if "instruction_adherence_decay" in line:
                iad_start = content.find(line)
                break
        if iad_start == -1:
            print("  ❌ Could not find instruction_adherence_decay section in run.log")
            return results

    if iad_end == -1:
        iad_section = content[iad_start:]
    else:
        iad_section = content[iad_start:iad_end]

    # Pattern: Depth N: run M/3 ... score=X.XX [STATUS] cat=CATEGORY (XXX.Xs)
    pattern = r"Depth\s+(\d+):\s+run\s+(\d+)/\d+\s+.*?score=([\d.]+)\s+\[(PASS|FAIL|PARTIAL)\]\s+cat=(\w+)\s+\(([\d.]+)s\)"
    for match in re.finditer(pattern, iad_section):
        depth = int(match.group(1))
        run_num = int(match.group(2))
        score = float(match.group(3))
        status = match.group(4)
        category = match.group(5)
        eval_time = float(match.group(6))

        result = {
            "dimension": "instruction_adherence_decay",
            "depth": depth,
            "depth_unit": "turns",
            "run": run_num,
            "score": score,
            "failure_category": category if category != "pass" else "pass",
            "expected": "JSON format with award field",
            "actual": "",  # Not available from log
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

    return results


# ─── 2. Load existing results ───

def load_results(filepath: str) -> dict:
    """Load a results JSON file."""
    with open(filepath) as f:
        return json.load(f)


# ─── 3. Merge ───

def merge_results(iad_results: list, existing_data: dict) -> dict:
    """
    Merge instruction_adherence_decay with the 5 existing dimensions.

    Args:
        iad_results: Reconstructed instruction_adherence_decay results list
        existing_data: Full dict from results.json (metadata, results, summary)

    Returns:
        Merged dict with metadata, results (6 dims, ~96 tests), summary
    """
    existing_results = existing_data.get("results", [])
    existing_metadata = existing_data.get("metadata", {})

    # Combine results
    all_results = iad_results + existing_results

    # Get all dimensions
    dims_in_results = set(r["dimension"] for r in all_results)
    dims_sorted = sorted(dims_in_results)

    # Compute per-dimension summary
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
            depth_means.append({"depth": d, "mean_score": round(mean, 4), "num_runs": len(d_scores)})
            if mean < 0.8 and break_point is None:
                break_point = d

        # Use correct label suffix per dimension
        bp_unit = "tokens" if dim == "hallucination_onset" else "turns"
        dim_summary[dim] = {
            "total_tests": len(dim_tests),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "break_point": break_point,
            "break_point_label": f"{break_point} {bp_unit}" if break_point else "None (all ≥ 0.8)",
            "depth_means": depth_means,
            "failure_categories": dict(categories),
        }

    # Build merged metadata with fresh timestamp
    merged_metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": existing_metadata.get("model", "gemma4:e2b"),
        "total_tests": len(all_results),
        "dimensions_tested": dims_sorted,
    }

    return {
        "metadata": merged_metadata,
        "results": all_results,
        "summary": dim_summary,
    }


# ─── 4. Main ───

def main():
    print("=" * 70)
    print("  MERGE E2B RESULTS")
    print("=" * 70)

    # Step 1: Load the CLEAN checkpoint (results_gemma4_e2b.json = 72 tests, 5 dims, no IAD)
    # NOTE: results.json may already be contaminated with double-counted IAD from a prior merge
    clean_path = os.path.join(RESULTS_DIR, "results_gemma4_e2b.json")
    results_path = os.path.join(RESULTS_DIR, "results.json")
    base_path = clean_path if os.path.exists(clean_path) else results_path
    print(f"\n[1] Loading clean results from {base_path}...")
    existing_data = load_results(base_path)
    existing_results = existing_data.get("results", [])
    existing_dims = set(r["dimension"] for r in existing_results)
    print(f"  → Loaded {len(existing_results)} tests across {sorted(existing_dims)}")

    # Step 2: Check if instruction_adherence_decay is already present
    has_iad = "instruction_adherence_decay" in existing_dims
    if has_iad:
        print("\n[2] instruction_adherence_decay already present — skipping reconstruction")
        merged_data = existing_data
    else:
        print("\n[2] instruction_adherence_decay MISSING — reconstructing from run.log...")
        iad_results = parse_instruction_adherence_decay(os.path.join(RESULTS_DIR, "run.log"))
        print(f"  → Reconstructed {len(iad_results)} instruction_adherence_decay test cases")
        if iad_results:
            depths = set(r["depth"] for r in iad_results)
            scores_by_depth = defaultdict(list)
            for r in iad_results:
                scores_by_depth[r["depth"]].append(r["score"])
            for d in sorted(depths):
                scores = scores_by_depth[d]
                mean = sum(scores) / len(scores)
                print(f"    Depth {d}: mean={mean:.2f}, runs={len(scores)}, cats={[r['failure_category'] for r in iad_results if r['depth']==d]}")

        # Step 3: Merge
        print("\n[3] Merging...")
        merged_data = merge_results(iad_results, existing_data)

    merged_dims = set(r["dimension"] for r in merged_data["results"])
    print(f"  → Final: {len(merged_data['results'])} tests across {sorted(merged_dims)}")

    # Step 4: Save to results_merged.json
    merged_path = os.path.join(RESULTS_DIR, "results_merged.json")
    with open(merged_path, "w") as f:
        json.dump(merged_data, f, indent=2, default=str)
    print(f"\n[4] Saved merged results to {merged_path}")

    # Step 5: Also overwrite results.json
    results_json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(merged_data, f, indent=2, default=str)
    print(f"  ✅ Overwrote {results_json_path} with merged data")

    # Step 6: Verify
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
    print(f"  All 6 dimensions present: {verify_dims == expected_dims}")

    if verify_dims == expected_dims:
        dim_counts = defaultdict(int)
        for r in verify["results"]:
            dim_counts[r["dimension"]] += 1
        for d in sorted(dim_counts):
            print(f"    {d}: {dim_counts[d]} tests")
        print("\n  ✅ Merge complete — all 6 dimensions present!")
    else:
        missing = expected_dims - verify_dims
        print(f"\n  ❌ Missing dimensions: {missing}")
        sys.exit(1)

    # Step 7: Summary overview
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