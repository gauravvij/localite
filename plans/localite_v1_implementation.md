# localite_v1 Implementation Plan

## Goal
Create a lean SWE-bench agent in `localite_v1/` that builds on mini-swe-agent's architecture (native tool_calls, clean agent loop, Jinja2 templates) with minimal overhead.

## Design Philosophy
- **Thin wrapper pattern** — import and reuse mini-swe-agent's proven components (LitellmModel, DefaultAgent, LocalEnvironment, BASH_TOOL)
- **Our layer = templates + configuration + evaluation workflow** — the three things we actually need to customize
- **No over-engineering** — no FormatMonitor, no context refresh, no 5-phase guidance

## Architecture

```
localite_v1/
├── __init__.py          # Package marker
├── agent.py             # create_agent() builder — 60 lines
├── templates.py         # Jinja2 template strings — 80 lines
├── runner.py            # run_instance() — SWE-bench execution — 250 lines
├── evaluate.py          # Batch eval + comparison reports — 200 lines
└── README.md            # Usage docs
```

**Total: ~590 lines of purpose-written code.**

## Subtasks

1. **Create `localite_v1/__init__.py`** — empty package marker
2. **Create `localite_v1/templates.py`** — Jinja2 templates optimized for both capable models and SLMs:
   - SYSTEM_TEMPLATE: clean, minimal (~5 lines)
   - INSTANCE_TEMPLATE: PR description, task, submission protocol (~30 lines)
   - OBSERVATION_TEMPLATE: output with smart truncation (5K head + 5K tail) (~20 lines)
   - FORMAT_ERROR_TEMPLATE: informative error with finish_reason handling (~15 lines)
3. **Create `localite_v1/agent.py`** — single builder function `create_agent()`:
   - Creates AgentConfig with our templates
   - Creates LitellmModel with BASH_TOOL, OpenRouter auth, cost tracking
   - Creates LocalEnvironment with configurable cwd, timeout
   - Returns ready-to-use DefaultAgent
4. **Create `localite_v1/runner.py`** — `run_instance()` function:
   - Clone SWE-bench repo → install deps → run baseline → create_agent → agent.run() → collect diff → compute patch_similarity → F2P/P2P → determine resolution → save result JSON + trajectory
   - CLI with --model, --instances, --step-limit, --agent-timeout, --skip-install
5. **Create `localite_v1/evaluate.py`** — batch evaluation:
   - `run_instances()` — run multiple instances sequentially
   - `generate_report()` — aggregate results, produce comparison markdown
   - `compare_runs()` — side-by-side with another result set
6. **Verify** — run a quick smoke test: `python -m localite_v1.runner --model deepseek/deepseek-v4-flash --instances marshmallow-code__marshmallow-1359 --step-limit 50`

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tool call format | Native API `tool_calls` (via LitellmModel + BASH_TOOL) | 100% parse rate for capable models |
| SLM fallback | LitellmTextbasedModel (regex) — configurable per model | For models without native tool call support |
| Context management | None — let API handle it | mini-swe-agent proves this works |
| Format recovery | Informative error → retry (max 3) | No context refresh needed |
| Output truncation | 10K char (5K head + 5K tail) | Proven in mini-swe-agent |
| Template engine | Jinja2 (reuse mini-swe-agent's rendering) | Already a dependency |

## Deliverables
| File | Purpose |
|------|---------|
| localite_v1/__init__.py | Package marker |
| localite_v1/agent.py | Agent builder |
| localite_v1/templates.py | Jinja2 templates |
| localite_v1/runner.py | Instance runner + CLI |
| localite_v1/evaluate.py | Batch eval + reporting |
| localite_v1/README.md | Usage documentation |

## Success Criteria
- `python -m localite_v1.runner --help` shows proper CLI
- Smoke test on marshmallow-1359 completes with resolution_status and patch_similarity > 0.5
- Result JSON + trajectory saved correctly