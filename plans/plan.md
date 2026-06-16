# SWE-bench Harness: Stages 0–5 — GPU Endpoint Routing + Guidance Fixes + v6/v7 Rerun

## Goal
Wire up the Qwen GPU endpoint with automatic Ollama fallback (Stage 0), fix the guidance injection bugs that prevent the model from ever calling edit_file (Stages 1–2), add token-aware context trimming and system-prompt trimming (Stages 3–4), then rerun v6 and v7 SWE-bench Lite on the Qwen3.5-4B GPU endpoint and report improvements (Stage 5).

## Research Summary
- New Qwen GPU endpoint confirmed live: `https://specifies-format-herald-successfully.trycloudflare.com/v1`
- Model ID confirmed: `Qwen/Qwen3.5-4B`, max_model_len=16384, served via vLLM
- Profile already updated by user: `profiles/qwen35_4b_gpu.toml` has the new base_url
- Root cause of 0/5 resolve rate identified from v6/v7 logs:
  1. Guidance heuristic extracts `task_complete` (our own tool name) as a "task identifier" → ctags resolves it to a test file → model is told to read irrelevant test files 6+ times
  2. Same wrong guidance injected repeatedly (no dedup), eating the model's ~5-turn competent window
  3. No forced-edit trigger: model explores indefinitely, never calls edit_file/write_file
  4. V7 HTTP 400s: char-based trimmer pops from index 1 (may remove system prompt or task), not tool outputs first
- `swe_runner.py` lines 624–644: provider-based client selection (openai_compatible → AsyncOpenAIClient, else → AsyncOllamaClient)
- `agent_loop.py` lines 638–913: progressive guidance injection after every list_files call
- `agent_loop.py` lines 478–490: char-based context trimmer (pops context[1] blindly)
- `agent_loop.py` lines 630–640: stall path — no-tool-call just increments stall_count, no re-prompt

## Approach
Staged surgical edits to `agent_loop.py` and `swe_runner.py`, verified with a 1-instance smoke run between Stage 2 and Stage 3, then full v6+v7 reruns.

## Subtasks

### Stage 0 — GPU endpoint routing + health-check fallback
1. Add `LOCALITE_BASE_URL` env-var override in `swe_runner.py` `create_swe_agent()`: if set, override `profile.base_url` before client construction. Log which URL is used.
2. Add a synchronous health-check helper `_check_endpoint_health(base_url, model_name, timeout=10)` in `swe_runner.py` that hits `{base_url}/models` and returns True/False.
3. In `create_swe_agent()`, after resolving `base_url`: call health-check; if it fails and provider is `openai_compatible`, log a warning and fall back to the Ollama localhost profile (`http://localhost:11434`) with the same model name. Log the fallback decision clearly.
4. Verify: run `python -c "from swe_runner import _check_endpoint_health; print(_check_endpoint_health('https://specifies-format-herald-successfully.trycloudflare.com/v1', 'Qwen/Qwen3.5-4B'))"` — should print True.

### Stage 1 — Fix guidance injection (highest leverage)
5. In `agent_loop.py`, define a module-level `HARNESS_TOOL_NAMES` frozenset containing all tool names the harness itself uses: `{'task_complete', 'read_file', 'edit_file', 'list_files', 'grep_search', 'write_file', 'run_shell', 'test_executor', 'bash', 'python'}` plus common harness path tokens: `{'swe_bench', 'repos', 'workdir', 'localite', 'agent_loop'}`.
6. In the ctags identifier extractor (around line 776–792), add a filter: after extracting identifiers, remove any token that is in `HARNESS_TOOL_NAMES` (case-insensitive). Also add a confidence gate: if ctags resolves an identifier ONLY to files under `test/` or `tests/` directories (not any `src/` file), suppress that guidance (set `ctags_guidance = None`).
7. Add a `_guidance_seen: set[str]` instance variable (init in `__init__`) and a `_guidance_count: int = 0` counter. Before injecting any guidance message, check: if the message is already in `_guidance_seen` OR `_guidance_count >= 2`, skip injection entirely. Otherwise add to `_guidance_seen`, increment counter, and inject.
8. Verify: grep the agent log for a 1-instance run — confirm zero occurrences of `task_complete` or `swe_bench` in `[GUIDANCE]` lines, and no repeated guidance strings.

### Stage 2 — Forced-edit trigger + message-mode re-prompt
9. Add instance variable `_edit_calls: int = 0` in `__init__`. In the tool-call handler, after a successful `edit_file` or `write_file` call, increment `_edit_calls`.
10. In the no-tool-call (stall) path (around line 632): instead of just incrementing `stall_count`, check if `self.current_phase == Phase.EXECUTE` — if so, inject a sharp re-prompt as a user message: `"[REQUIRED] You are in EXECUTE phase. You MUST call a tool now — either edit_file or write_file to make the code change, or task_complete if done. Do NOT send a message. Call a tool."` Do this on the FIRST stall in EXECUTE; on the second stall, increment stall_count as before.
11. Add a forced-edit trigger: after every tool call in EXECUTE phase, if `_edit_calls == 0` AND `turn_counter.count >= 8` AND the model has called `read_file` or `grep_search` at least 3 times (track with `_investigate_calls: int`), inject a user message: `"[REQUIRED] You have explored enough. Now call edit_file or write_file to apply your fix. Do not explore further."` — inject this at most once (guard with `_forced_edit_injected: bool`).
12. Verify: in a 1-instance smoke run log, confirm at least one `[REQUIRED]` line appears and that `edit_file` or `write_file` is called at least once per instance where a file was identified.

### Smoke run (between Stage 2 and Stage 3)
13. Run 1 instance of v6 on the Qwen GPU endpoint: `python swe_runner.py --profile qwen35_4b_gpu --instances 1 --benchmark v6`. Grep the log for `[GUIDANCE]`, `[REQUIRED]`, `edit_file`, `write_file`. Confirm: guidance count ≤ 2, no harness-tool guidance, at least one `[REQUIRED]` or `edit_file` call. Log to `results/smoke_test_stage2.log`.

### Stage 3 — Token-aware context trimming (kills v7 HTTP 400s)
14. Replace the blind `context.pop(1)` trimmer in `agent_loop.py` (lines ~487–490) with a smarter eviction strategy: keep indices 0 (system prompt) and 1 (standing instructions if present) pinned; evict from the OLDEST tool-result messages first (role == "tool"), then oldest assistant messages, never touching the first 2 messages or the last 2 messages (most recent exchange). Cap single tool output at 8000 chars (down from 32000) when context is already near budget.
15. Also add a per-read_file output cap: if a `read_file` tool result exceeds 6000 chars, truncate it and append `\n[File truncated — use grep_search to find specific sections]`.
16. Verify: rerun the 3 v7 instances that 400'd (sqlfluff-core-2419, sqlfluff-core-1733, sqlfluff-core-1763) — zero HTTP 400 errors in log.

### Stage 4 — Trim system prompt
17. In `agent_loop.py`, audit `SYSTEM_PROMPT` and `OUTPUT_FORMAT_STANDARD`: collapse redundant anti-pattern lists (lines like "NEVER do X" repeated 3+ times), merge the 5-phase description into a 2-state summary (INVESTIGATE → EDIT), keep tool descriptions and one format example. Target: reduce system prompt from ~150 lines to ~80 lines. Preserve all tool descriptions and the output format block exactly.
18. Verify: `len(system_prompt)` logged at start of run is meaningfully smaller (target < 3000 chars vs current); 1-instance behavior unchanged or better.

### Stage 5 — Full v6 + v7 rerun on Qwen GPU + reports
19. Run full v6 evaluation: `python swe_runner.py --profile qwen35_4b_gpu --benchmark v6`. Capture to `results/v6_gpu_eval.log`.
20. Run full v7 evaluation: `python swe_runner.py --profile qwen35_4b_gpu --benchmark v7`. Capture to `results/v7_gpu_eval.log`.
21. Write three reports:
    - `results/swe_bench/v6_qwen35_gpu_summary.md` — v6 results post-fix
    - `results/swe_bench/v7_qwen35_gpu_summary.md` — v7 results post-fix
    - `results/swe_bench/v6_v7_comparison_report.md` — side-by-side: old (CPU, pre-fix) vs new (GPU, post-fix): resolve rate, guidance injection count, edit_file call count, HTTP 400 count, wall time
22. Each report must include: resolve rate, per-instance table (instance ID, status, edit_file called Y/N, guidance count, HTTP 400 Y/N), key observations, recommendations.

## Deliverables
| File | Description |
|------|-------------|
| `/home/azureuser/local_llm_eval/swe_runner.py` | Health-check helper + env-var override + Ollama fallback |
| `/home/azureuser/local_llm_eval/localite/loop/agent_loop.py` | Guidance filter/dedup, forced-edit trigger, EXECUTE re-prompt, smarter context trimmer, trimmed system prompt |
| `/home/azureuser/local_llm_eval/results/smoke_test_stage2.log` | 1-instance smoke run log |
| `/home/azureuser/local_llm_eval/results/v6_gpu_eval.log` | Full v6 run log |
| `/home/azureuser/local_llm_eval/results/v7_gpu_eval.log` | Full v7 run log |
| `/home/azureuser/local_llm_eval/results/swe_bench/v6_qwen35_gpu_summary.md` | v6 post-fix summary |
| `/home/azureuser/local_llm_eval/results/swe_bench/v7_qwen35_gpu_summary.md` | v7 post-fix summary |
| `/home/azureuser/local_llm_eval/results/swe_bench/v6_v7_comparison_report.md` | Before/after comparison |

## Evaluation Criteria
- Stage 0: `_check_endpoint_health(...)` returns True for the GPU endpoint; fallback logs correctly when endpoint is down
- Stage 1: Zero `task_complete`/`swe_bench` in `[GUIDANCE]` lines; no repeated guidance string; guidance count ≤ 2 per instance
- Stage 2: At least one `[REQUIRED]` or `edit_file`/`write_file` call per instance in EXECUTE phase
- Stage 3: Zero HTTP 400 errors in v7 rerun
- Stage 4: System prompt char count reduced meaningfully
- Stage 5: Reports exist, are non-empty, resolve rate reported (any improvement over 0/5 is a win; primary goal is edit_file being called)

## Notes
- GPU endpoint: `https://specifies-format-herald-successfully.trycloudflare.com/v1`, model `Qwen/Qwen3.5-4B`, max_model_len=16384
- Profile: `profiles/qwen35_4b_gpu.toml` (already updated by user with new base_url)
- Ollama fallback: `http://localhost:11434`
- Prior run results for comparison: v6 CPU pre-fix: 0/5 resolved, 0 agent_errors, 155 turns, 396s; v7 CPU pre-fix: 0/5 resolved, 3 agent_errors (HTTP 400), 117 turns, 380s
- The `_guidance_seen` set and `_guidance_count` must be reset per-instance (they live on the AgentLoop instance which is recreated per instance in swe_runner.py — so this is automatic)
- Do NOT change the tool descriptions or output format block in SYSTEM_PROMPT — only trim redundant prose
- After editing agent_loop.py, always run `python -m py_compile localite/loop/agent_loop.py` to verify syntax
