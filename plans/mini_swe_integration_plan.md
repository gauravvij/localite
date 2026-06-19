# Plan: Replace Localite Agent Loop with mini-swe-agent

## Goal
Replace Localite's custom agent loop (`localite/loop/agent_loop.py` — 1741 lines of JSON tool-calling, 5-phase state machine, format monitor, stall detection) with mini-swe-agent's `DefaultAgent` (~100 lines, bash-only, single loop) for all SWE-bench evaluations, while keeping the repo management and scoring infrastructure intact.

## Research Summary
- **mini-swe-agent** scores >74% on SWE-bench Verified with a radically simple design:
  - **Bash-only**: the model produces raw bash commands, not structured JSON tool calls. Compatible with any LLM — no tool-calling capability needed.
  - **Linear history**: every step (thought, command, output) is appended to a single message list. No state machine, no phase tracking, no format monitor.
  - **Stateless execution**: each command runs in a fresh subprocess (`subprocess.run`). No persistent shell, no "flaky shell" issues.
- **DefaultAgent** (agents/default.py): ~100 lines. Loop: query model → execute bash → observe output → repeat. Handles limits, format errors, and trajectory saving.
- **Python API**: `DefaultAgent(model=LitellmModel(...), env=LocalEnvironment(...), config={...})`.agent.run(task) → returns dict with exit_status and submission.
- **Config-driven prompts**: YAML configs with Jinja2 templates (swebench.yaml provides the exact SWE-bench prompt template referencing `/testbed` with reproduce-first workflow).
- **LocalEnvironment** supports setting `cwd` programmatically.
- **marshmallow-1359 validation**: mini-swe-agent (55 API calls, DeepSeek V4 Flash) correctly resolved the issue (submitted working patch). Localite (16 turns, same model) failed F2P with `RESOLVED_NO`.

## Approach
Write a clean new runner `mswe_runner.py` that:
1. Uses mini-swe-agent's Python API (`DefaultAgent` + `LitellmModel` + `LocalEnvironment`) for the agent loop
2. Imports shared utilities from `swe_runner.py` (clone_repo, install_repo_deps, evaluate_instance, save_result, etc.) by importing the shared functions
3. Keeps the same JSON result format so downstream reporting tools continue working
4. Adds a `--backend` CLI flag so users can compare `localite` vs `mini` backends

## Architecture Comparison

| Aspect | Localite (agent_loop.py) | mini-swe-agent (DefaultAgent) |
|--------|-------------------------|-------------------------------|
| **Lines of agent code** | ~1741 | ~100 |
| **Tool interface** | JSON tool calls (bash, read, write, edit, search, list_files, task_complete, etc.) | Raw bash commands |
| **State machine** | 5 phases (analyze, implement, test, reflect, review) | Single flat loop |
| **Format enforcement** | FormatMonitor (window=10, threshold=0.3) + ping-pong detection | Max consecutive format errors (simple counter) |
| **Context management** | Char-based eviction, truncation | Linear message append (all context preserved) |
| **Observation handling** | Custom tool output parsing | Jinja2 template (truncates >10K chars) |
| **Stall detection** | Complex stall threshold (N turns with no diff changes) | None (step_limit / cost_limit / wall_time_limit) |
| **Prompt templates** | Hardcoded in agent_loop.py | YAML config files (Jinja2 templates) |

## Subtasks

### Subtask 1: Install mini-swe-agent into the local_llm_eval venv
Install mini-swe-agent (and its dependencies) into `/home/azureuser/local_llm_eval/venv/` so `mswe_runner.py` can import it directly. This avoids needing to activate a separate venv.
- **Expected output**: `pip install mini-swe-agent` succeeds in the local_llm_eval venv
- **Verify**: `source venv/bin/activate && python3 -c "from minisweagent.agents.default import DefaultAgent; print('ok')"` exits 0

### Subtask 2: Create mswe_runner.py
Write `/home/azureuser/local_llm_eval/mswe_runner.py` that:

**Imports & shared utilities:**
- Import shared functions from `swe_runner.py` via `sys.path` + `from swe_runner import clone_repo, install_repo_deps, evaluate_instance, save_result, save_combined_results, load_dev_instances, compute_patch_similarity, get_git_diff, REPOS_DIR, RESULTS_DIR`

**Key additions:**
- `create_mini_swe_agent(workdir, model_name, profile_config)` function that:
  - Creates `LitellmModel(model_name=..., model_kwargs={...})`
  - Creates `LocalEnvironment(cwd=workdir, timeout=60, env=PAGER=cat, ...)`
  - Creates `DefaultAgent(model=..., env=..., config={...})` with:
    - `system_template`: from swebench.yaml (guide model to use bash, produce THOUGHT+command format)
    - `instance_template`: from swebench.yaml (SWE-bench task format with reproduce-first workflow, MODIFY files in workdir, git patch submission)
    - `observation_template`: from swebench.yaml (truncate long output, show return code)
    - `format_error_template`: from swebench.yaml (guide model to fix format)
    - `step_limit`: 100
    - `cost_limit`: 3.0 (adjustable via `--cost-limit`)
    - `wall_time_limit_seconds`: agent_timeout
    - `max_consecutive_format_errors`: 3
    - `output_path`: per-instance trajectory path
  - Returns the agent
- `run_instance(instance, agent_timeout, model_name, profile_config)` async function that:
  1. Clones repo at base_commit (same as current)
  2. Installs repo deps (same as current)
  3. Runs baseline tests (same as current)
  4. Creates mini-swe-agent, runs with the SWE-bench task prompt
  5. Collects git diff
  6. Computes patch similarity
  7. Runs F2P/P2P evaluation (same as current)
  8. Saves result in same JSON format
- `main()` CLI with:
  - `--instances`, `--max-instances`, `--agent-timeout`, `--profile` (but profile maps to OpenRouter model name)
  - `--model` (OpenRouter model name, default: deepseek/deepseek-v4-flash)
  - `--cost-limit` (default: 3.0)
  - `--step-limit` (default: 100)
  - `--skip-install` (skip repo dep installation for speed)
  - `--output-dir` (default: results/swe_bench/)
  - `--backend` choices=["localite", "mini"] default="mini" — to allow easy comparison

**Prompt template details:**
The system_template and instance_template should be adapted from mini-swe-agent's swebench.yaml but with `/testbed` replaced by the actual workdir path. The key elements:
1. System: "You are a helpful assistant that can interact with a computer shell to solve programming tasks."
2. User: PR description + instructions to reproduce, fix, and submit as git patch
3. The submission protocol: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`
4. Guidance to first reproduce the issue, then fix source files, then verify

**Expected output**: `/home/azureuser/local_llm_eval/mswe_runner.py` (~500 lines)
**Verify**: `python3 mswe_runner.py --help` exits 0 and shows all options

### Subtask 3: Single-instance smoke test on marshmallow-1359
Run `mswe_runner.py` on marshmallow-code__marshmallow-1359 with DeepSeek V4 Flash to validate end-to-end.
- **Expected output**: Instance resolves with `resolution_status` = "FULL" or "PARTIAL", patch similarity > 0.5
- **Verify**: Check result JSON file has `resolution_status` != "NO" and patch is correct

### Subtask 4: Run on all 18 non-sqlfluff instances
Run the full 18-instance set (the same set as the previous Localite evaluation) to get a proper comparison benchmark.
- **Expected output**: 18 result JSON files + combined results + summary table
- **Verify**: Compare resolution rate vs Localite's 44.4% (GLM 5.2) and 50.0% (DeepSeek V4 Flash)

### Subtask 5: Generate comparison report
Write a report comparing mini-swe-agent results vs Localite results on the same 18 instances.
- **Expected output**: `results/swe_bench/mini_vs_localite_comparison_report.md`
- **Verify**: Report shows per-instance comparison, aggregate stats, and conclusions about harness impact

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/home/azureuser/local_llm_eval/mswe_runner.py` | New SWE-bench runner using mini-swe-agent backend |
| `/home/azureuser/local_llm_eval/results/swe_bench/{instance_id}.json` | Per-instance results (same format as current) |
| `/home/azureuser/local_llm_eval/results/swe_bench/mini_18instances_report.md` | Results report for 18 instances |
| `/home/azureuser/local_llm_eval/results/swe_bench/mini_vs_localite_comparison_report.md` | Side-by-side comparison |

## Evaluation Criteria
- mini-swe-agent achieves **≥50% resolution rate** on the same 18 non-sqlfluff instances (matching or exceeding Localite + DeepSeek V4 Flash)
- All result JSON files have the same schema as current swe_runner.py output
- CLI is user-friendly with sensible defaults
- Trajectory files are saved for debugging

## Notes
- The `/testbed` path issue is handled by setting `cwd=workdir` on `LocalEnvironment` — the prompt template references the actual workdir path
- The agent's task prompt should tell it the working directory is the repo root, so `cd /testbed` isn't needed
- The `pip install` step for the repo (install_repo_deps) is critical — without it, pytest can't find the module
- We reuse swe_runner.py's F2P/P2P evaluation logic verbatim — it's battle-tested and correct
- mini-swe-agent's cost tracking is advisory (OpenRouter rounds to nearest cent), so cost_limit is a safety net, not a precision tool