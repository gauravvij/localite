# SLM Multi-Turn Agent Degradation Evaluation — Full Suite & Multi-Model Comparison

## Goal
Systematically characterize and compare degradation failure modes across 3 small language models (LFM2.5-8B-A1B, Gemma 4 E2B, Gemma 4 E4B) in multi-turn agent contexts, producing a taxonomy of common failure patterns and actionable context engineering recommendations.

## Research Summary

### Models Available (All local via Ollama on 8-core CPU, 62GB RAM)

| Model | Active Params | Total Params | Architecture | Context | Quant | Disk |
|-------|--------------|-------------|-------------|---------|-------|------|
| **LFM2.5-8B-A1B** (Liquid) | ~1.5B | 8.3B | MoE | 128K | Q4_K_M | 5.3 GB |
| **Gemma 4 E2B** (Google) | ~2B | 2.3B | Dense | 256K | Q4_K_M | 7.2 GB |
| **Gemma 4 E4B** (Google) | ~4B | 8.1B | Dense | 256K | Q4_K_M | 9.6 GB |

Architectural diversity (MoE vs dense, 1.5B→4B active params) makes this a solid comparison set.

### Key Learnings from Phase 1 (Quick Scan)
- LFM2.5 uses ` thinking...response` XML CoT format — harness must strip it
- Ollama /api/chat endpoint works with system/user/assistant message roles
- Filler turn generation is the bottleneck (6-10s per turn on CPU)
- Total quick scan runtime for 1 pass at limited depths: ~74 min
- Tool call format fails immediately (score ≤ 0.30 at depth 1)
- Recency bias fails immediately (score 0.00 at depth 1)
- Non-monotonic degradation observed (instruction adherence recovers at depth 20)
- `keep_alive="-1"` causes 400 error from Ollama API — use `keep_alive=None`

### Existing Files (from Phase 1)
- `src/eval_harness.py` — Ollama client, thinking stripping, filler generation, result collection
- `src/test_scenarios.py` — 6 dimension generators with scorers
- `src/run_quick_scan.py` — Quick scan orchestrator (1 run per depth)
- `results/quick_scan_results.json` — 22 raw test results
- `results/quick_scan_summary.md` — Phase 1 summary report

## Approach

### Phase 2: Full Suite on LFM2.5 (baseline)

Enhance the existing harness with:

**1. Multi-run support** — 3 runs per depth (mean ± std), with random seed variation in filler generation

**2. Expanded depths with adaptive stop** — Run depths in increasing order. If score < 0.2 for 2 consecutive depths, skip remaining higher depths and label the dimension "fully degraded." This saves ~40-50% runtime while still capturing the break point precisely.

| Dimension | Full Suite Depths | Adaptive Stop |
|-----------|------------------|---------------|
| Instruction Adherence Decay | 1, 3, 5, 8, 10, 15, 20, 30 | Yes |
| Memory Retrieval | 1, 3, 5, 8, 10, 15, 20, 30 | Yes |
| Hallucination Onset | 1K, 2K, 4K, 8K, 16K, 32K | Yes (dense depth sweep at key break zones) |
| Tool Call Drift | 1, 3, 5, 10, 20 | Yes (expects immediate failure) |
| Persona Consistency | 1, 5, 10, 20, 30 | Yes |
| Recency Bias | 1, 3, 5, 10, 15, 20 | Yes |

**3. Filler conversation caching** — Filler turns for a given model + system prompt can be cached and reused across runs of the same dimension. Store as `results/cache/{model_name}_{dimension}_{depth}.json`. For different runs at the same depth, rotate through shuffled question pools rather than regenerating filler.

**4. Failure taxonomy** — Beyond binary pass/fail, categorize failure types:
- **Tool call failures**: plain text / wrong JSON structure / correct JSON wrong function / correct JSON missing fields
- **Memory failures**: blank / hallucinated wrong name / "I don't know" / generic answer
- **Instruction adherence failures**: plain text / wrong format / partial JSON
- **Recency failures**: follows override / follows original / mixed / incoherent

**5. Structured output format** — Each test case records:
- Raw output (preserved for inspection)
- Stripped output
- Score
- Failure category (string label)
- Filler stats (num turns, approx tokens)
- Timing info

### Phase 3: Multi-Model Comparison

**1. Run identical full suite** on Gemma 4 E2B and Gemma 4 E4B using the same harness (no code changes needed — just pass different model name)

**2. Generate comparison matrix** showing:
- Break point per (dimension × model)
- Mean score per depth per dimension per model
- Which dimensions break universally vs model-specific
- Unique failure patterns per architecture (MoE vs dense)

### Runtime Estimation
- Full suite on 1 model (expanded depths, 3 runs, caching): ~2-3 hours
- 3 models total: ~6-9 hours
- All models run independently — results accumulate progressively
- Each model run is idempotent (can resume)

## Subtasks

### Phase 2: Full Suite on LFM2.5

1. **Enhance `eval_harness.py`** — Add:
   - Filler conversation cache (store/load from `results/cache/`)
   - Multi-run loop support
   - Per-test timing
   - Failure categorization infrastructure (scorers return categories)
   
2. **Enhance `test_scenarios.py`** — Update:
   - DIMENSION_DEPTHS to expanded values
   - Scorers to categorize failure types (return (score, category) tuples)
   - Filler generation to use shuffled question pool variation across runs

3. **Build `run_full_suite.py`** — New orchestrator:
   - Accepts `--model` parameter (default: the LFM2.5 GGUF)
   - Accepts `--runs` parameter (default: 3)
   - Adaptive stop logic (2 consecutive depths < 0.2 → skip rest)
   - Cached filler generation (check cache before generating)
   - Checkpoint saves after each dimension
   - Results to `results/full_suite/{model_name}/` directory
   - Summary generation at end

4. **Run full suite on LFM2.5-8B-A1B** — Execute with 3 runs per depth, expanded depths.
   - Estimated: 2-3 hours on CPU
   - Output: `results/full_suite/lfm25/` with per-dimension results + summary

### Phase 3: Multi-Model Comparison

5. **Run full suite on Gemma 4 E2B** — Same harness, `--model gemma4:e2b`
   - Output: `results/full_suite/gemma4_e2b/`

6. **Run full suite on Gemma 4 E4B** — `--model gemma4:e4b`
   - Output: `results/full_suite/gemma4_e4b/`

7. **Build `generate_comparison_report.py`** — Cross-model comparison:
   - Reads all `results/full_suite/{model}/results.json`
   - Produces comparison table: break point per (dimension × model)
   - Highlights universal failure dimensions vs model-specific ones
   - Failure pattern comparison (do all models fail the same way?)
   - Output: `results/multi_model_comparison_report.md`

8. **Generate comparison report** — Run the report generator

### Phase 4: Analysis

9. **Write `common_ground_failures.md`** — Final analysis:
   - List of failure dimensions that are universal across all 3 models
   - List of model-specific advantages/disadvantages
   - Categorized failure taxonomy with examples
   - Recommended context engineering mitigations per failure type
   - "Rules of thumb" for building multi-turn agents with SLMs

## Deliverables

| File Path | Description |
|-----------|-------------|
| `src/eval_harness.py` | Enhanced core harness with caching, multi-run, failure categorization |
| `src/test_scenarios.py` | Updated with expanded depths and failure category scorers |
| `src/run_full_suite.py` | Full suite orchestrator with adaptive stop |
| `src/generate_comparison_report.py` | Cross-model comparison report generator |
| `results/full_suite/lfm25/results.json` | Full suite results for LFM2.5 |
| `results/full_suite/gemma4_e2b/results.json` | Full suite results for Gemma 4 E2B |
| `results/full_suite/gemma4_e4b/results.json` | Full suite results for Gemma 4 E4B |
| `results/multi_model_comparison_report.md` | Cross-model comparison with universal failure analysis |
| `results/common_ground_failures.md` | Final analysis: universal failures, taxonomy, mitigations |

## Evaluation Criteria
- **For each dimension × model**: identify the precise depth (turns or tokens) where degradation exceeds 20% of baseline
- **Statistical**: 3 runs per depth, report mean ± std
- **Failure categorization**: every failed test case categorized into a failure type
- **Universal vs specific**: clear taxonomy of which failures appear across all 3 models vs model-unique
- **Raw outputs preserved**: all model responses available for manual inspection

## Notes
- All models run locally via Ollama on 8-core CPU — expect ~2-10 tok/s depending on model size
- Filler conversation caching is critical for keeping runtime manageable
- Adaptive stop prevents wasting runs on already-degraded dimensions
- Each model's suite is independent — results accumulate progressively
- No GPU available — all local inference is CPU-based