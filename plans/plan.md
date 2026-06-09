# Switch to Gemma 4 E4B + Close 5 MVP Gaps

## Goal
Switch the localite coding agent from LFM2.5-8B-A1B to Gemma 4 E4B (objectively better across 4/6 degradation dimensions), and close the 5 critical gaps that block real coding evaluation: format monitor, standing instructions injection, profile-driven config, context refresh repair, and plan anchoring.

## Research Summary
- **E4B vs LFM2.5 comparison**: E4B dominates on Tool Call Drift (1.00 vs 0.26 avg), Instruction Adherence Decay (30-turn horizon vs 10), Persona Consistency (perfect vs perfect). LFM2.5 only wins on Memory Retrieval (8 turns vs 5). Recency bias is universal. Full data at `/home/azureuser/local_llm_eval/results/multi_model_comparison_report.md`.
- **E4B output format**: No `<thinking>/<response>` XML tags. Clean natural text. No thinking tags detected. `has_thinking_tags=false`.
- **E4B tool call format**: Scores 1.00 ± 0.00 at all depths through 20 turns — natively produces valid JSON tool calls. No format guard needed.
- **E4B memory horizon**: Break point at 5 turns → `memory_horizon=5`.
- **E4B IAD horizon**: Break point at 30 turns → `iad_horizon=30`.
- **E4B available in Ollama**: `ollama list` confirms `gemma4:e4b` present, 9.6 GB, Q4_K_M quantization. Works on CPU.
- **Hardware**: 8-core CPU, 62.8 GB RAM, no GPU. E4B runs fine (4B dense params, ~10-15s/turn on CPU).
- **Current profiles**: Only `lfm25.toml` exists at `profiles/lfm25.toml`.

## Approach
Seven subtasks, ordered by dependency:

1. Create E4B profile TOML with correct degradation parameters
2. Wire ModelProfile into AgentLoop — load at init, drive all thresholds from profile
3. Add standing instructions injection into `_build_context()` (every turn, counters recency bias)
4. Build Format Monitor — inspect tool call outputs, track drift, trigger early refresh
5. Repair context refresh — re-inject standing instructions, session facts, active plan, last tool results
6. Add plan anchoring — store plan text when Plan phase completes, inject during Execute/Verify
7. Update main.py `--profile` default to "gemma4_e4b", update `--help` text
8. Run full test suite + real-model E2E to verify no regressions

## Subtasks
1. Create `/home/azureuser/local_llm_eval/profiles/gemma4_e4b.toml` with:
   - name=`gemma4:e4b`, provider=ollama
   - max_turns=5 (slightly above memory_horizon to let refresh catch it)
   - memory_horizon=5 (from eval data: break point at 5 turns)
   - format_guard=false (native 1.00 tool call score — no guard needed)
   - recency_protection=true (still vulnerable to recency bias)
   - has_thinking_tags=false (no XML tags in output)
   - iad_horizon=30 (break point at 30 turns)
   - base_url=localhost:11434, timeout=30
   (verify: `python3 -c "from localite.config import ConfigLoader; p=ConfigLoader().load_profile('gemma4_e4b'); print(p)"`)

2. Update `AgentLoop.__init__` to accept `model_profile: ModelProfile` parameter. Use it to set:
   - `TurnCounter(hard_limit=profile.max_turns)`
   - `self.format_guard = profile.format_guard`
   - `self.memory_horizon = profile.memory_horizon`
   - `self.recency_protection = profile.recency_protection`
   - Pass `has_thinking_tags` to `AsyncOllamaClient` already handled via `_strip_thinking`
   (verify: unit test creating AgentLoop with profile, check turn_counter.hard_limit matches profile)

3. Modify `_build_context()` to inject standing instructions as a user message block after the system prompt. This counters recency bias by re-instating core rules every turn:
   ```python
   if self.recency_protection:
       messages.append({"role": "user", "content": f"[STANDING INSTRUCTIONS]\n{self.standing_instructions.get_text()}"})
   ```
   (verify: E2E test — check context construction includes standing instructions in returned messages)

4. Create `localite/context/format_monitor.py` with:
   - `FormatMonitor` class — tracks tool call format quality over recent turns
   - `record_tool_call(tool_call: dict, response_text: str)`: parses output, scores JSON adherence (0.0-1.0)
   - `should_refresh() -> bool`: returns True if running average dips below threshold (0.3)
   - `reset()`: clear tracking window
   - Wire into `_check_degradation()` — if `self.format_guard` is True, check format monitor before turn limit
   (verify: unit test — feed bad JSON outputs, confirm should_refresh returns True)

5. Repair `_refresh_context()` so it re-injects:
   - Standing instructions block (as in subtask 3)
   - Session facts block (current objective + last tool used + last result)
   - Active plan summary (if one exists from subtask 6)
   - Last tool result as a tool-role message
   - Keep last N conversation turns (N from profile memory_horizon, not hardcoded 4)
   (verify: unit test on AgentLoop — call refresh, inspect conversation_history for all expected blocks)

6. Modify `_execute_phase()` to detect when Phase switches to PLAN, capture plan text from model output, store as `self.active_plan`. Modify `_build_context()` to inject as:
   ```python
   if self.active_plan:
       messages.append({"role": "user", "content": f"[ACTIVE PLAN]\n{self.active_plan}"})
   ```
   Clear plan when phase moves past EXECUTE or refresh triggers.
   (verify: unit test — set phase to PLAN, capture plan, confirm it appears in subsequent context)

7. Update `main.py`:
   - Change `default="lfm25"` to `default="gemma4_e4b"` in `--profile` argument
   - Update help text and welcome message to reflect E4B primary
   (verify: `python3 localite/main.py --help` shows E4B as default)

8. Run full test suite + real-model E2E:
   - All unit tests pass (24 passed previously)
   - Real-model E2E with E4B (not LFM2.5) completes at least 2 turns
   - No regressions from LFM2.5 baseline
   (verify: `TEST_E2E=1 python3 -m pytest tests/test_agent_loop.py -v --tb=short` — all green)

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/home/azureuser/local_llm_eval/profiles/gemma4_e4b.toml` | E4B model profile from degradation data |
| `/home/azureuser/local_llm_eval/localite/loop/agent_loop.py` | Updated with profile, standing instructions, plan anchoring, format monitor wiring |
| `/home/azureuser/local_llm_eval/localite/context/format_monitor.py` | New format monitor module |
| `/home/azureuser/local_llm_eval/localite/main.py` | Updated default profile to gemma4_e4b |

## Evaluation Criteria
- All 24 existing tests pass (no regressions)
- Real-model E2E with E4B completes successfully (≥2 turns with tool calls)
- `_build_context()` returns messages including standing instructions (proven via test assertion)
- `_refresh_context()` re-injects standing instructions + session facts (proven via test assertion)
- Format monitor detects bad JSON format and triggers refresh (proven via unit test)
- Profile drives turn counter hard_limit (not hardcoded)

## Notes
- E4B does NOT have thinking tags. `has_thinking_tags=false` means strip_thinking is a no-op.
- E4B scores 1.00 on tool call format at ALL depths — format_guard=false is correct for this model.
- Memory retrieval break point at 5 turns drives memory_horizon=5. Refresh should trigger before this point.
- Recency bias is universal (all models score 0.0 at depth 1) — standing instructions re-injection every turn is the primary countermeasure.