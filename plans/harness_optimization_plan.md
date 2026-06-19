# Localite Harness Optimization Plan (Path B)

## Goal
Modify the existing Localite agent harness to close the ~30-point SWE-bench performance gap vs official scores without adopting mini-swe-agent wholesale. Target: reach parity with mini-swe-agent scores for the same models (e.g., 75-80% for DeepSeek V4 Flash, 55-62% for GLM 5.2) while maintaining Localite's architecture where useful.

## Current State
- **Models tested**: DeepSeek V4 Flash v10: ~48% (23 SWE-bench Lite instances). GLM 5.2: ~43% (23 instances).
- **Official baselines**: DeepSeek V4 Flash: 79% SWE-bench Verified (mini-swe-agent). GLM 5.2: 62.1% SWE-bench Pro (mini-swe-agent).
- **Harness architecture**: 1741-line agent loop, 8 tools (JSON tool-calling), 5-phase state machine (EXPLORE/PLAN/EXECUTE/VERIFY/ITERATE/COMPLETE), FormatMonitor, standling instructions, phase-guidance system.

## Root Causes (from investigation)

### P0 — JSON Tool-Calling vs Bash-Only Interface
- mini-swe-agent uses **bash only** — model writes `sed`, `grep`, `python` commands as raw bash. Zero format overhead.
- Localite requires structured JSON: `{"tool": "edit_file", "arguments": {"path": "...", "search_text": "...", ...}}`.
- Each tool call costs 80-120 extra tokens AND cognitive switching cost (code reasoning → JSON assembly).
- Frontier models (DeepSeek V4 Flash, GLM 5.2) were trained on code first, structured function calls second. The bash interface plays to their strengths.

### P0 — 5-Phase State Machine Over-constrains
- Enforced phase progression prevents natural workflows (e.g., "find bug → immediately fix it" is blocked by PLAN phase).
- Stall_threshold=4 force-transitions EXECUTE→VERIFY mid-edit.
- mini-swe-agent has ONE loop: query → execute → observe. The model decides its own workflow.
- Different instances need different workflows. Phase enforcement averages 2-3 wasted turns per instance.

### P1 — Missing "Reproduce First" Workflow Guidance
- Localite prompt: "Fix bugs and implement changes using the tools below."
- mini-swe-agent prompt: 5 explicit steps including "Create a script to reproduce the error and execute it."
- Without explicit reproduce-first guidance, models code blind — our data shows agent_F2P=0 on most near-misses.
- F2P (Fail-to-Pass) tests are the most diagnostic signal for whether the fix is correct.

### P1 — FormatMonitor / Degradation Detection Creates Noise
- FormatMonitor triggers full context refresh when structured output drops below 30% quality.
- Models deeply engaged in code reasoning produce less structured output → degradation detected → refresh → model confused → wasted turns.
- This creates a negative reinforcement loop, especially hitting models mid-reasoning.

### P2 — No "Tests Already Handled" Statement
- Not telling the model that test files are already modified wastes potential turns.
- The model may analyze or edit `test_*.py` files unnecessarily.

### P2 — Standing Instructions (Recency-Biased User Message)
- Standing instructions are injected as a user message for recency protection.
- This adds ~200-300 tokens of unchanged text every context refresh.
- Takes attention away from the actual task context.

## Proposed Changes (Priority-Ordered)

### Change 1 (P0): Replace JSON Tool Interface with Bash-Only
**Rationale**: This is the single highest-impact change. mini-swe-agent's entire design philosophy is "bash only = works."

**Implementation**:
- Reduce tool set to ONE tool: `run_shell(command: str, timeout: int = 120)`.
- Remove `edit_file`, `write_file`, `read_file`, `list_files`, `grep_search`, `test_executor`, `task_complete`, `diff_view`, memory tools.
- The model writes bash commands directly. `run_shell` executes them with `subprocess.run`.
- The model can read files (`cat`, `head`, `grep`), edit files (`sed`, `patch`), create reproductions (`python test_script.py`), run tests (`python -m pytest`), and submit (`git diff`), all through bash.
- System prompt becomes: "You have access to a bash shell. Write commands in markdown code blocks."

**Trade-off**: Lose structured output parsing, gain zero-overhead interaction. The model speaks native Linux, not JSON.

### Change 2 (P0): Collapse 5-Phase to Single Loop
**Rationale**: The state machine is actively harmful. Models know how to code. Let them.

**Implementation**:
- Replace `_execute_phase` with a single `_run_turn()` that:
  1. Sends messages to model
  2. Parses response (just bash code blocks)
  3. Executes commands
  4. Returns output as observation
- No phase transitions. No stall detection. No phase-specific guidance.
- Keep ONLY: max_turns limit, cost limit, and basic error handling (empty response retry).
- Remove `_check_degradation`, `_should_skip_phase`, phase_seq logging.

### Change 3 (P1): Redesign Prompt with Reproduce-First Workflow
**Rationale**: Most powerful lever after bash-only interface.

**Implementation**: Use a prompt modeled on mini-swe-agent's `instance_template`:

```
📁 Repository: {workdir} (at commit {base_commit})
📋 Issue: {problem_statement}

Your task is to fix the issue described above.

Follow these steps:
1. First, find and read code relevant to the issue description
2. Create a Python script to reproduce the error and execute it with `python <filename>.py`
3. Edit the source code to fix the issue 
4. Re-run your reproduce script and confirm the error is fixed
5. Think about edge cases and make sure your fix handles them

Important: All test files (test_*.py) have already been updated by the issue author.
DO NOT modify test files — only modify the source code.

When you believe the issue is fixed, run: echo 'TASK_COMPLETE'
```

### Change 4 (P1): Remove FormatMonitor (or Make Advisory)
**Rationale**: With bash-only interface, format parsing is trivial (just extract ```bash blocks). No need for quality scoring.

**Implementation**:
- Either delete `format_monitor.py` entirely, or make it log-only with no side effects.
- Remove FormatMonitor integration from `agent_loop.py`.

### Change 5 (P2): Simplify Message Pipeline
- Remove standing instructions as injected user message (the new prompt covers it).
- Remove phase-guidance lookup table.
- Remove format-error templates.

## Deliverables

| File | Change |
|------|--------|
| `/home/azureuser/local_llm_eval/localite/loop/agent_loop.py` | Collapse to single-loop. Remove phases, stall, format monitor. New prompt. |
| `/home/azureuser/local_llm_eval/localite/context/format_monitor.py` | Delete or disable. |
| `/home/azureuser/local_llm_eval/localite/context/standing_instructions.py` | Delete or simplify. |
| `/home/azureuser/local_llm_eval/profiles/*.toml` | Remove phase-related config (stall_threshold, format_guard). |
| `/home/azureuser/local_llm_eval/swe_runner.py` | Update task formatting to use new `instance_template`. |
| New: `localite/loop/bash_agent_loop.py` | (Optional) Clean-slate minimal agent loop ~100 lines. |

## Evaluation Criteria

1. **Same models, same 18-instance subset** → DeepSeek V4 Flash reaches ≥65% (up from 50%), GLM 5.2 reaches ≥55% (up from 44%).
2. **No format degradation triggers** in any run (FormatMonitor gone).
3. **Agent always reproduces bug before fixing** — verified by examining agent traces.
4. **Agent_F2P > 0 on resolved instances** — no more "fixes that don't touch failing tests."
5. **Turns per instance ≤20 avg** (currently ~15, so maintaining or improving).

## Anti-Patterns to Avoid

- **Do not keep JSON tool-calling as an option** — mixing bash and JSON creates confusion. Go all-in on bash.
- **Do not add a "plan" phase** — the model can plan implicitly in its reasoning (if the model supports CoT/thinking). For non-thinking models, the 5-step prompt provides enough structure.
- **Do not keep format-error retries** — if bash output fails to parse, just retry the prompt with "Please output a bash command in a code block."
- **Do not keep stall detection** — models don't stall meaningfully with bash. If they're silent, they're thinking. Let them.
- **Do not half-implement** — partial changes (e.g., keep format monitor but remove phases) won't move the needle enough. The 30-point gap requires radical simplification.

## Decision
**This path is NOT recommended** (see Path A reasoning below). Path B requires significant refactoring of ~2000+ lines across 5+ files with uncertain payoff — the mini-swe-agent team already proved the bash-only approach is optimal, and trying to beat it by modifying Localite is re-inventing the wheel under constraints.