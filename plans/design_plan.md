# Technical Design Plan — Remaining Agent Features

## Completed Work Reference
Commit 70c2ee4 closes all Week 1 gaps. This plan covers Week 2+ features.

## Research Summary

### Episodic Memory Patterns (3-tier landscape)

**Tier 1 — File-based (Claude Code pattern)**
- `CLAUDE.md` + `MEMORY.md` per project
- First 200 lines / 25KB of `MEMORY.md` loaded at session start
- Detailed notes in topic files read on demand (progressive disclosure)
- Auto-generated from corrections and observations
- Low-friction, no vector DB, no infrastructure
- **⚠️ Problem for small models**: 25KB of memory at session start consumes 60-70% of an 8B model's working context, causing attention dilution + context rot (confirmed by our own Memory Retrieval test: model fails at 10+ turns)

**Tier 2 — Tool-based (SmallCode pattern) ✅ RECOMMENDED**
- `memory_load` / `memory_remember` tool pair — **on-demand, not auto-injected**
- `knowledge/` directory available via tool, not loaded by default
- Context Budget Engine aggressively caps/evicts content
- Evidence store per-turn observations
- **Why this wins for small models**: Context is never polluted with memory the model doesn't need. The model calls memory tools *when it needs context* — not every turn.

**Tier 3 — DB-backed (Claude-Mem pattern)**
- Lifecycle hooks: SessionStart, UserPromptSubmit, PostToolUse, Stop, SessionEnd
- SQLite + Chroma vector DB for hybrid search
- MCP search tools: `search` (compact index), `timeline` (chronological), `get_observations` (full detail)
- Progressive disclosure with token cost visibility
- Most sophisticated but highest complexity — overkill for 8B models

**Our context**: Local 8B models, CPU-only, no vector DB infrastructure. Our own experiment data shows: Memory Retrieval fails at 10+ turns (0.00 score) on LFM2.5-8B. The 8B Wall research confirms small models struggle with context >10-15 turns. **Therefore: Tier 2 (SmallCode pattern) is correct — tool-based, on-demand access. NEVER auto-inject large memory blocks into context.**

### Completion Signal Pattern (convergent industry pattern)
- Dedicated tool with `status` enum + `reason_code` + optional `summary`
- Framework convergence: OpenAI Agents SDK, Microsoft Agent Framework, LangGraph all use structured termination
- Prevents false positives/negatives and silent quits
- Must be gated at orchestrator level (loop stops when tool detected)

### Adaptive Phase Transitions
- **SmallCode**: Early-stop detection (repetition, patch spirals), Per-tool trust score decay (demote after 3 failures, drop after 5), Adaptive retry temperature, TODO-driven planning
- **Anthropic harness**: Initializer agent, feature list JSON, one-feature-at-a-time, claude-progress.txt
- Our loop has 5 phases (EXPLORE→PLAN→EXECUTE→VERIFY→ITERATE→COMPLETE)
- Skip conditions: no files exist → must EXPLORE; trivial task (seen before) → skip EXPLORE; EXECUTE made no changes → skip VERIFY; task is a single file write → merge PLAN+EXECUTE

### Delegation Telemetry
- Per-tool success/failure counts, average latency, trust score
- SmallCode's per-tool trust score decay is the reference pattern
- Track in AgentLoop, emit in `get_status()`, reset per session

---

## Design Decisions

### 1. Exit Tool (`task_complete`)

**Schema:**
```python
tool task_complete(
    status: Literal["completed", "partial", "blocked", "failed"],
    reason: str = "",      # human-readable summary
    summary: str = "",     # what was accomplished
    files_changed: list[str] = [],  # files created/modified
)
```

**Behavior:**
- Registered as a first-class tool in `create_default_tools()` alongside write_file, read_file, run_shell, etc.
- `AgentLoop._execute_phase()` detects `task_complete` tool call → immediately exits loop → sets `self.current_phase = Phase.COMPLETE`
- No tool execution needed — the call itself IS the termination signal
- Result captured in episode data for eval suite
- Standing instructions reinforced with: "When you have completed the task, call task_complete() rather than saying 'done' in prose"

**Verification:** Same E2E test but model reaches COMPLETE faster (no wasted ITERATE cycles)

### 2. Episodic Memory (Tier 2 — Tool-based, no auto-injection ✅)

**Critical constraint from experiment data:** Our own Memory Retrieval scan showed LFM2.5-8B-A1B fails at 10+ turns (score 0.00). The 8B Wall research confirms small models suffer from **context window starvation** at >10-15 turns as tools + conversation + system prompt consume context. Therefore:

> **Memory content is NEVER auto-injected into `_build_context()` or `_refresh_context()`.**
> Memory is purely **tool-accessed on demand**. The model calls a tool when it needs context.

**Storage:**
```
~/.localite/memory/
  └── <project_hash>/
      ├── sessions/
      │   └── index.json              # Compact session summaries (latest 10, ~50 bytes each)
      │   ├── 2026-06-09_12-30-15.md   # Full session log (written but NOT read back)
      │   └── ...
      └── topics/
          ├── architecture.md           # Topic files loaded on demand via memory_read
          ├── common-errors.md
          └── ...
```
No MEMORY.md index (avoids auto-loading). Session summaries live in `index.json`. Full session logs are written for human inspection but never auto-injected.

**Lifecycle:**

1. **Session end** (`AgentLoop.run()` returning COMPLETE or timeout):
   - Compress the episode into a **compact summary** (~50 bytes): `{"task":"<truncated>","files":[...],"outcome":"passed|failed|timeout"}`
   - Append to `sessions/index.json` (keep latest 10)
   - Write full session log to `sessions/<timestamp>.md` (for human review, NOT read by model)

2. **Session start** (`AgentLoop.__init__`):
   - Load `sessions/index.json` — **do NOT inject into context**
   - Store in `self.memory_store` for tool access only
   - **The only "injection" is a single compact line in system prompt**: `[PREVIOUS SESSION: <task> → <outcome>]` if there's a recent session for the same project — this is <50 tokens

3. **On-demand retrieval** — `memory_read(topic)` tool:
   - Model calls this *when it needs context* — not every turn
   - Reads `topics/<topic>.md` and returns content
   - **Result is gated by context budget** (max 500 tokens returned; if larger, returns first 500 + "[truncated]")
   - Tool description explains: "Use this when you need to recall previous decisions, design patterns, or gotchas"

4. **Writing learnings** — `memory_write(topic, content)` tool:
   - Model explicitly saves learnings
   - Appends or overwrites `topics/<topic>.md`
   - Updates `index.json` with a note that topic exists
   - Tool description explains: "Use this after completing a subtask to save important context for future sessions"

**Key design constraints:**
- No vector DB, no embeddings, no external services
- **Memory is tool-accessed, never auto-injected** (avoids attention dilution for small models)
- Session start only injects a 1-line <50-token summary, if at all
- Context budget enforcement: memory_read results capped at 500 tokens
- File-based so it's inspectable, debuggable, and portable
- Progressive disclosure: tools available always, data loaded only on demand

### 3. Remove iad_horizon

- Delete field from `ModelProfile` dataclass in `config.py`
- Delete from `profiles/gemma4_e4b.toml`
- The degradation system now uses `stall_threshold` + format monitor thresholds instead

### 4. Adaptive Phase Transitions

**Principle:** Phases are optional when their input conditions are already satisfied.

**Transition rules (in `next_phase()` logic):**

| From | Skip condition | Skip to |
|------|---------------|---------|
| EXPLORE | Task is a known pattern + episodic memory has relevant session(s) | PLAN |
| PLAN | Task is a single-file write (no exploration needed, plan is trivial) | EXECUTE |
| VERIFY | No files were changed in EXECUTE phase | ITERATE or COMPLETE |
| ITERATE | No files changed AND no test_executor/run_shell called | COMPLETE (can't fix what didn't change) |
| VERIFY | `task_complete` was called already | COMPLETE |

**Implementation:**
- `AgentLoop.run()` checks skip conditions before entering each phase
- Phase enum gains `is_skippable()` method
- Standing instructions tell the model it can call `task_complete` at any time

### 5. Delegation Telemetry

**Data structure** (on `AgentLoop`):
```python
self.tool_stats: dict[str, dict] = {
    "write_file": {"calls": 0, "successes": 0, "failures": 0, "avg_duration_ms": 0},
    "run_shell": {"calls": 0, "successes": 0, "failures": 0, "avg_duration_ms": 0},
    ...
}
```

**Update point:** After each `_handle_tool_call()` returns — increment calls, update success/failure, track duration.

**Trust score:** `trust = successes / max(calls, 1)`. If trust < 0.3 and calls >= 3, demote tool in schema description (append `⚠️ unreliable` to description). If failures >= 5, drop from tool schema entirely.

**SmallCode reference:** Per-tool trust score decay — soft-demotes after 3 failures, drops after 5.

**Emission:** Included in `get_status()` output and `run()` return dict for eval suite consumption.

---

## Implementation Order

1. **`task_complete` tool** — simplest, highest impact (fixes the looping issue directly)
2. **Adaptive phase transitions** — uses `task_complete` as a signal, adds skip conditions
3. **Remove `iad_horizon`** — trivial cleanup
4. **Delegation telemetry** — small, contained, wires into _handle_tool_call
5. **Episodic memory** — most complex, needs careful implementation

---

## Subtasks

1. **`task_complete` tool**
   - Create `localite/tools/task_complete.py` with `TaskCompleteTool` class
   - Register in `create_default_tools()` in `main.py`
   - Detect in `AgentLoop._execute_phase()` — intercept tool call, set phase to COMPLETE, don't execute
   - Update standing instructions to mention `task_complete`
   - Unit tests: mock tool call detection, verify COMPLETE transition
   - Real-model E2E: verify gemma4:e4b uses it (may need prompt reinforcement)

2. **Adaptive phase transitions**
   - Add `skip_phases` logic to `AgentLoop.run()` — check conditions before each phase
   - EXPLORE skip: check if session index has matching prior session (from memory_store, tier after subtask 5)
   - PLAN skip: check if task is trivial (heuristic: single file, no search needed)
   - VERIFY skip: check if `episode.files_changed` is empty
   - COMPLETE shortcut: if `task_complete` was detected, skip all remaining phases
   - Unit tests: verify skip conditions for each phase

3. **Remove iad_horizon**
   - Delete from `localite/config.py` ModelProfile
   - Delete from `profiles/gemma4_e4b.toml`

4. **Delegation telemetry**
   - Add `tool_stats` dict to `AgentLoop.__init__`
   - Track calls/successes/failures/duration in `_handle_tool_call`
   - Compute trust score per tool
   - Demote/drop tools with low trust scores in `_build_context()` tool descriptions
   - Include in `get_status()` and `run()` return

5. **Episodic memory system**
   - Create `localite/memory/memory_store.py` with `EpisodicMemoryStore` class
   - Methods: `load_session_index()`, `save_session_summary()`, `read_topic()`, `write_topic()`
   - Create `localite/tools/memory_tools.py` with `MemoryReadTool` and `MemoryWriteTool`
   - Register memory tools in `create_default_tools()`
   - Wire session-end summary generation in `AgentLoop.run()` on COMPLETE
   - Wire session-start index loading in `AgentLoop.__init__` — store in `self.memory_store`, **do NOT auto-inject into context**
   - The only injection is a single `<50-token` line in system prompt: `[PREVIOUS SESSION: <task> → <outcome>]` for most recent session only
   - Context budget cap: `memory_read` results truncated at 500 tokens

---

## Deliverables

| File Path | Description |
|-----------|-------------|
| `localite/tools/task_complete.py` | TaskCompleteTool — exit tool with status/reason/summary |
| `localite/loop/agent_loop.py` | Updated: task_complete detection, adaptive skips, tool_stats, memory tools wiring (no auto-injection) |
| `localite/loop/phases.py` | Updated: Phase.should_skip(), Phase transitions |
| `localite/config.py` | Updated: remove iad_horizon |
| `profiles/gemma4_e4b.toml` | Updated: remove iad_horizon |
| `localite/main.py` | Updated: register TaskCompleteTool, memory tools |
| `localite/context/standing_instructions.py` | Updated: task_complete instruction |
| `localite/memory/memory_store.py` | EpisodicMemoryStore class |
| `localite/tools/memory_tools.py` | MemoryReadTool, MemoryWriteTool |
| `localite/tools/__init__.py` | Updated exports |
| `tests/test_task_complete.py` | Unit tests for exit tool |
| `tests/test_phase_skips.py` | Unit tests for adaptive transitions |
| `tests/test_memory_store.py` | Unit tests for episodic memory |
| `tests/test_tool_telemetry.py` | Unit tests for delegation telemetry |

## Success Criteria
- [ ] `task_complete` tool call terminates the loop immediately (verified by E2E test)
- [ ] Phase skips reduce turn count for trivial tasks (verified by unit test mock)
- [ ] `iad_horizon` removed without breaking profile loading (verified by unit test)
- [ ] Tool telemetry tracks calls/successes/failures accurately (verified by unit test)
- [ ] Episodic memory persists across sessions (verified by integration test: run → save → load → verify memory_read retrieves saved data)
- [ ] Memory content is never auto-injected into _build_context() (verified by asserting no 'PREVIOUS SESSIONS' block in mock AgentLoop context output)