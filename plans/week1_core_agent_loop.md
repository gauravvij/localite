# Week 1: Core Agent Loop (MVP Foundation)

## Goal
Build a running single-model conversational coding agent with approval gating, turn counting, context refresh, diff display, test-after-every-write, episode persistence, and terminal UI.

## Tech Stack
- **Language**: Python 3.10+
- **Model serving**: Ollama (local, via /api/chat) — existing integration confirmed
- **UI**: Rich (terminal) for streaming, progress bars, status bars
- **Config**: TOML (model profiles)
- **Persistence**: JSON files on disk (~/.local-code-agent/sessions/)
- **Libraries**: httpx (async HTTP), rich (TUI), pydantic (config schemas)
- **Inherited patterns from SmallCode**: PlanTracker, QualityMonitor, tool parsing patterns (adapted to Python)

## Project Structure (to be created)
```
localite/
├── pyproject.toml
├── localite/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── config.py             # Model profile config (TOML loading)
│   ├── model/
│   │   ├── __init__.py
│   │   └── client.py         # Ollama/LLM client (reuse from eval_harness)
│   ├── loop/
│   │   ├── __init__.py
│   │   ├── agent_loop.py     # 5-phase agent loop
│   │   ├── phases.py         # Phase definitions (Explore/Plan/Execute/Verify/Iterate)
│   │   └── turn_counter.py   # Turn counter + hard limit
│   ├── permissions/
│   │   ├── __init__.py
│   │   └── gate.py           # y/s/n/e permission gate
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── read.py           # read file tool
│   │   ├── write.py          # write file tool
│   │   ├── edit.py           # edit file tool
│   │   ├── search.py         # grep/search tool
│   │   ├── shell.py          # shell command tool
│   │   ├── test_executor.py  # auto-detect and run tests
│   │   └── diff_view.py      # unified diff display
│   ├── context/
│   │   ├── __init__.py
│   │   ├── buffer.py         # Context buffer management
│   │   ├── refresh.py        # Context refresh logic
│   │   └── standing_instructions.py  # Standing instructions (W2 foundation)
│   └── episodes/
│       ├── __init__.py
│       ├── store.py          # Episode persistence
│       └── model.py          # Episode data model
├── profiles/
│   └── lfm25.toml            # LFM2.5 model profile (our primary test model)
└── tests/
    └── test_agent_loop.py    # Integration tests
```

## Subtasks

### 1. Project scaffold + dependencies
- Create directory structure under /home/azureuser/local_llm_eval/localite/
- Create pyproject.toml with dependencies: httpx, rich, pydantic, tomli
- Create __init__.py files
- Install dependencies in venv

### 2. Model client (reuse + generalize from eval_harness.py)
- Read existing OllamaClient from src/eval_harness.py
- Create localite/model/client.py with:
  - Async OllamaClient (local /api/chat, /api/generate)
  - Support for streaming (SSE from /api/chat)
  - Model profile loading from TOML
  - strip_thinking() for models with <thinking> tags
  - Error handling, timeout config
- **Verify**: Can call Ollama model and get a streaming response

### 3. Tool system
- Create base Tool interface/abstract class
- Implement: read_file, write_file, edit_file, grep_search, run_shell
- Each tool: name, description, arguments schema, execute function
- Tool output capture with stdout/stderr
- Unified diff generation for any proposed file change
- **Verify**: Each tool can be called independently and produces correct output

### 4. Permission gate (y/s/n/e)
- Display proposed action (tool call + args) to user
- Accept input: y=approve, s=skip, n=reject+explain, e=edit command
- Support block mode: show all proposed steps, user approves/rejects individually
- Support for editing proposed command before execution
- **Verify**: y/s/n/e all work, edit mode allows modifying command

### 5. 5-phase agent loop with 4-turn hard limit
- Implement 5-phase loop: Explore → Plan → Execute → Verify → Iterate
- Phase transitions:
  - Explore: model reads files, searches codebase
  - Plan: model proposes plan (displayed as structured plan)
  - Execute: model makes changes (tool calls), each approved via permission gate
  - Verify: run tests, show results
  - Iterate: on failure, loop back to Execute (max 3 iterations)
- Hard turn counter: 4 turns per segment, after which force context refresh + user approval
- Context refresh: re-inject system prompt + standing instructions after approval gate
- **Verify**: Full user→model→approve→tool→result→verify cycle works end-to-end

### 6. Episode persistence
- Episode data model: session_id, episode_id, objective, turns[], plan, files_changed[], summary, timestamp
- Save to ~/.local-code-agent/sessions/<session_id>/episodes/<episode_id>.json
- Load/resume: list sessions, reload last session
- Episode compression: key facts, decisions, files changed (simple extract)
- **Verify**: Episode saves correctly, can be reloaded in new session

### 7. Terminal UI with Rich
- Streaming model output (render tokens as they arrive)
- Status bar: model name, turn counter (3/4), permission mode, phase
- Inline diff display (with syntax highlighting via Rich)
- Test results display (pass/fail with output)
- Progress indicators for tool execution
- Keyboard shortcuts: Ctrl+C to interrupt, Ctrl+D to exit
- **Verify**: UI renders correctly, streaming works, status bar updates

### 8. End-to-end integration test
- Launch agent in test mode (model responds with canned outputs)
- Walk through: user request → explore → plan → approve → execute → verify → complete
- Verify episode persistence
- Measure: time per cycle, turns used, user approvals
- **Verify**: Acceptance test passes

## Deliverables
| Path | Description |
|------|-------------|
| localite/pyproject.toml | Project config with dependencies |
| localite/localite/main.py | Entry point with CLI args |
| localite/localite/config.py | TOML config loading |
| localite/localite/model/client.py | Async Ollama client with streaming |
| localite/localite/tools/*.py | All tool implementations |
| localite/localite/permissions/gate.py | y/s/n/e permission gate |
| localite/localite/loop/agent_loop.py | 5-phase agent loop |
| localite/localite/episodes/store.py | Episode persistence |
| localite/localite/context/*.py | Context buffer management |
| localite/profiles/lfm25.toml | LFM2.5 profile |
| Complete terminal UI | Streaming, status bar, diffs |

## Evaluation Criteria
- [ ] Agent can read a file, propose a change, get approval, write the change, run tests, and save an episode record — in one session
- [ ] Turn counter hard-stops at 4 turns and forces refresh
- [ ] Permission gate: y/s/n/e all function correctly
- [ ] Streaming model output renders in terminal
- [ ] Episode persists to disk and can be reloaded