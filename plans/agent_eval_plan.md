# AgentLoop Evaluation Harness

## Goal
Build an evaluation suite that tests the AgentLoop system end-to-end — instead of calling raw Ollama. This measures whether the guardrails (FormatMonitor, standing instructions injection, context refresh, plan anchoring) actually improve task completion and prevent degradation in a multi-turn coding agent.

## Why This Matters
The existing eval harness calls raw Ollama with synthetic filler turns. That tests the **model**, not the **agent system**. The agent has:
- FormatMonitor (detects + recovers from format drift)
- Standing instructions re-injected every turn
- Context refresh (resets memory when degradation detected)
- Plan anchoring (captures + re-injects the plan)
- Permission gate with batch mode

If we're building an agent, we should evaluate the agent.

## Approach
1. Create an `AutoGate(PermissionGate)` that auto-approves all tool calls — no interactive prompts needed for batch evaluation.
2. Create `agent_eval_harness.py` — harness that creates AgentLoop, runs coding tasks, collects internal metrics (format monitor scores, context refreshes, turns used).
3. Create `agent_test_tasks.py` — 6 coding tasks with verifier functions:
   - fibonacci.py: write+run Fibonacci printer
   - greet.py: write+test command-line greeting script
   - word_freq.py: write+test word frequency counter
   - factorial.py: write+test recursive factorial with edge cases
   - sort.py: write+test bubble sort on given array
   - is_prime.py: write+test prime checker with multiple test cases
4. Create `run_agent_suite.py` — orchestrator running all tasks for given profile(s), saving results + generating comparison report.

## Subtasks
1. Create AutoGate class in src/agent_eval_harness.py
2. Create AgentEvalResult dataclass and AgentEvalHarness class
3. Create agent_test_tasks.py with 6 tasks + verifiers
4. Create run_agent_suite.py CLI orchestrator
5. Run agent suite for E4B (primary target)
6. Generate agent-level comparison report
7. (Optional) Run agent suite for LFM2.5 for comparison

## Deliverables
- `/home/azureuser/local_llm_eval/src/agent_eval_harness.py` — Core harness class
- `/home/azureuser/local_llm_eval/src/agent_test_tasks.py` — Task definitions with verifiers
- `/home/azureuser/local_llm_eval/src/run_agent_suite.py` — CLI orchestrator
- `/home/azureuser/local_llm_eval/results/agent_suite/gemma4_e4b/` — Results for E4B
- `/home/azureuser/local_llm_eval/results/agent_suite/agent_comparison_report.md` — Final report

## Success Criteria
- Harness runs 6 tasks through AgentLoop completely unattended
- Each task result includes: success (bool), turns_used (int), format_monitor_avg (float), context_refreshes (int), tool_calls (int), duration (float)
- Report shows per-task pass/fail with metrics
- Report compares agent E4B vs raw E4B (from existing data) if available