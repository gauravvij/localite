# SWE-bench Lite Evaluation Harness for localite

## Goal
Build an automated evaluation pipeline that runs localite (gemma4:e4b, 4B params, CPU) on real SWE-bench Lite issues and measures patch quality vs reference solutions. Establish the first known baseline for a fully-local 4B agent on SWE-bench.

## Research Summary
- SWE-bench Lite: 300 test + 23 dev instances across 11 Python repos (Django, Flask, SymPy, requests, marshmallow, etc.)
- Dev split is ideal for development: smaller, designed for debugging harnesses
- Each instance: `instance_id`, `repo`, `base_commit`, `issue` (the bug report text), `patch` (gold solution), `test_patch` (verification tests), `hints` (optional)
- Official eval harness uses Docker + test isolation — we cannot use Docker (no Docker in sandbox)
- Alternative scoring: apply reference `test_patch` to the repo, run agent, then run pytest on the modified repo to check if tests pass
- Gemma 4 E4B on CPU: ~20-60s/turn. 20 turns → ~7-15 min per instance. 5 dev instances → ~35-75 min total eval run
- Smallest repos (marshmallow, flask, requests) have simpler codebases → better starting point

## Approach
1. Increase `max_turns` in gemma4_e4b profile from 5 → 20
2. Add `auto_approve` mode to `PermissionGate` (skips all user prompts for headless eval)
3. Create `swe_runner.py` — standalone eval harness that:
   - Downloads SWE-bench Lite dev instances from HuggingFace
   - Clones each repo at `base_commit`
   - Feeds `issue` text + repo directory to localite agent in headless mode
   - Collects agent's final patch (via git diff after agent finishes)
   - Scores using: reference patch overlap + test_patch execution
4. Run on 3-5 simplest dev instances to establish feasibility baseline

## Subtasks
1. **Profile update**: Increase `max_turns` from 5 → 20 in `profiles/gemma4_e4b.toml`. (verify: inspect file shows `max_turns = 20`)
2. **PermissionGate headless mode**: Add `auto_approve` option to `PermissionGate.__init__` that makes `propose()` return `PermissionResult(decision="approved")` without prompting. (verify: run unit test or import check)
3. **Create `swe_runner.py`**: Full eval harness with SWE-bench Lite loading, repo cloning, headless agent loop integration, patch generation, scoring. (verify: script runs python3 locally without import errors)
4. **Install SWE-bench-lite deps**: `pip install datasets` + clone logic support. (verify: `python3 -c "from datasets import load_dataset"`)
5. **E2E run on 3 dev instances**: Run swe_runner.py on `marshmallow__marshmallow-1083` (smallest), `flask__flask-4478`, `requests__requests-1932`. Capture results to `results/swe_bench/`. (verify: output files exist with scores)
6. **Report results**: Generate `results/swe_bench/summary_report.md` with per-instance scores, agent behavior notes, runtime stats.

## Deliverables
| File Path | Description |
|-----------|-------------|
| /home/azureuser/local_llm_eval/profiles/gemma4_e4b.toml | Updated max_turns=20 |
| /home/azureuser/local_llm_eval/localite/permissions/gate.py | Added auto_approve mode |
| /home/azureuser/local_llm_eval/swe_runner.py | SWE-bench Lite eval harness |
| /home/azureuser/local_llm_eval/results/swe_bench/ | Per-instance results + summary |

## Evaluation Criteria
- swe_runner.py loads a SWE-bench instance and feeds it to the agent without any manual interaction
- Agent completes a full EXPLORE → PLAN → EXECUTE → VERIFY → COMPLETE cycle on a real repo bug
- Patch is collected and compared to reference (even if score is low — baseline is valuable)
- Summary report documents findings for each instance

## Notes
- No Docker available — scoring uses `git diff` + `test_patch` approach
- CPU-only inference — each instance takes ~7-15 min
- This is exploratory: we're finding the baseline for a 4B local agent, not competing on the leaderboard
- If a specific instance fails (model can't solve 500K-line Django), log the failure and move on