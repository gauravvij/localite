# Local Code Agent — Architectural Hypothesis

**A degradation-aware, approval-gated multi-turn coding agent for fully local use.**
**The architecture for a local alternative to Claude Code / Cursor, built from first principles.**

**Author**: Synthesized from SLM degradation evaluation data (3 models, 291 tests, 6 dimensions)
**Date**: 2026-06-08
**Status**: Design hypothesis — ready for implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary) *(Build Stages: MVP vs Post-MVP)*
2. [Philosophy & Design Principles](#2-philosophy--design-principles)
3. [The Core Insight: Degradation-Adaptive Interaction](#3-the-core-insight-degradation-adaptive-interaction)
4. [System Architecture Overview (MVP)](#4-system-architecture-overview)
5. [Layer 0: Model Compatibility & Degradation Profiles](#5-layer-0-model-compatibility--degradation-profiles)
    - 5.3 [Estimated Profiles (Post-MVP)](#53-estimated-profiles--post-mvp)
    - 5.4 [Auto-Detection Protocol (Post-MVP)](#54-auto-detection-protocol-post-mvp)
6. [Layer 1: Interactive Permission Subsystem (MVP)](#6-layer-1-interactive-permission-subsystem-mvp)
    - 6.6 [Post-MVP: Extended Permission Modes](#66-post-mvp-extended-permission-modes)
7. [Layer 2: Degradation-Aware Context Engine](#7-layer-2-degradation-aware-context-engine)
    - 7.6 [Episodic Session Model](#76-episodic-session-model)
    - 7.7 [Format Monitor (MVP)](#77-format-monitor-mvp)
    - 7.8 [Selective Reference Episode Loading (Post-MVP)](#78-selective-reference-episode-loading-post-mvp)
8. [Layer 3: Phase-Based Agent Loop (MVP)](#8-layer-3-phase-based-agent-loop-mvp)
    - 8.3 [Adaptive Refresh (Post-MVP — Week 3)](#83-mvp-agent-loop-details)
9. [Layer 4: Tool Execution & Validation](#9-layer-4-tool-execution--validation)
    - 9.5 [Test Execution Tool (MVP)](#95-test-execution-tool-mvp)
    - 9.6 [Format Monitor as Built-in QC Check (MVP)](#96-format-monitor-as-built-in-qc-check-mvp)
    - 9.7 [Correction Frequency Tracker (MVP → Week 3)](#97-correction-frequency-tracker-mvp--week-3)
10. [Layer 5: External Integration Layer (MVP)](#10-layer-5-external-integration-layer-mvp)
    - 10.2 [Post-MVP: Cloud Escalation Path](#102-post-mvp-cloud-escalation-path)
    - 10.3 [Post-MVP: VS Code / Platform Integration](#103-post-mvp-vs-code--platform-integration)
11. [External Dependencies & Inheritance Strategy](#11-external-dependencies--inheritance-strategy)
12. [Implementation Roadmap](#12-implementation-roadmap) *(3-week MVP build plan)*
13. [Evaluation Criteria & Success Metrics (MVP)](#13-evaluation-criteria--success-metrics-mvp)
    - 13.6 [Post-MVP Metrics](#136-post-mvp-metrics)
14. [Appendix A: Degradation Data Cheat Sheet](#14-appendix-a-degradation-data-cheat-sheet)
15. [Appendix B: Model Profile Format](#15-appendix-b-model-profile-format)
16. [Appendix C: Session Fact Buffer Schema](#16-appendix-c-session-fact-buffer-schema)

---

## 1. Executive Summary

### What We're Building

A **fully local, interactive multi-turn coding agent** — a local alternative to Claude Code and Cursor — that:

- **Understands** the user's codebase, proposes changes, and gets approval before executing
- **Handles** multi-file refactoring, bug fixing, and feature implementation through interactive approval
- **Supports** any local model (tested on LFM2.5, Gemma 4 E2B, Gemma 4 E4B, Qwen3-8B)
- **Accounts for** the fundamental degradation patterns that cause local models to fail in long sessions
- **Works** entirely offline, on CPU (with GPU acceleration when available)

### The Core Thesis

> **Local SLMs can match cloud coding agents for most practical coding tasks — but only if the architecture compensates for their known degradation patterns. The interactive approval loop is not a UX compromise; it's a structural necessity that resets the degradation clock and keeps the model within its competency horizon.**

### Build Stages

This document describes the architecture in two development stages:

| Stage | Scope | Timeline |
|-------|-------|----------|
| **MVP** (this document's primary focus) | Single-model agent loop, 3 permission modes, standing instructions buffer, format monitor, episode management, test-after-every-write, adaptive refresh | **3 weeks** |
| **Post-MVP Enhancements** (marked throughout) | Two-model routing, 5-mode permissions (Batch/Auto), auto-detection protocol, selective reference episode loading, VS Code extension | After MVP proven |

**The MVP is the core thesis test**: can a properly degraded-compensated single local model handle real multi-turn coding tasks? Advanced features layer on later.

### Key Numbers from Evaluation

| Finding | Data Point | Architectural Impact |
|---------|-----------|---------------------|
| Memory retrieval fails after | 5–8 turns (all models) | Context refresh every ≤5 turns |
| Recency bias is universal | 0.000 at depth 1 (all models) | Standing instructions block |
| Tool call format drifts | 1 turn on LFM2.5, 20 on E2B | Format adherence monitor |
| Instruction adherence decays | 10–30 turns depending on model | Plan re-injection + approval gates |
| Persona drops | 10 turns on E2B, never on others | Persona guard for affected models |
| Run-to-run variance | Up to ±0.471 std dev at failure edge | Multi-run evaluation, not single-shot |

---

## 2. Philosophy & Design Principles

### Principle 1: Don't Fight Degradation — Design Around It

Trying to squeeze 100-turn reliability from an 8B model is a losing battle. Instead:
- **Limit consecutive agentic turns** to the model's proven horizon
- **Inject context refreshes** before the horizon, not after failure
- **Prefer many short accurate segments** over one long degraded session

### Principle 2: The User Is the Coherence Anchor

The user's approval is not just a safety feature — it's a **structural reset** that:
- Breaks the turn chain (counters IAD)
- Provides a natural insertion point for context refresh (counters memory failure)
- Lets the user correct the model without the model needing self-awareness

### Principle 3: Model Profiles Over Generic Configs

Different models have radically different degradation signatures. A generic "max_turns" setting means one model hits its cliff and another wastes potential. Every model needs a **degradation profile** derived from systematic evaluation (or auto-detected during first session).

### Principle 4: Inherit, Don't Rewrite

SmallCode has solved hard problems: tool parsers, plan tracking, hybrid search, snapshot/rollback, quality monitoring, thinking budget control. We inherit these patterns and add our degradation-aware layer on top — not rebuild them.

### Principle 5: Measure Everything

Every session collects degradation telemetry: turn count at first format failure, memory retrieval accuracy at each refresh point, persona consistency checks. Over time, this builds a per-model degradation profile that tunes the system automatically.

### Principle 6: Episodes, Not Sessions

A coding agent session is **never one task**. Users stack tasks, ask mid-stream questions, backtrack, and context-switch. The architecture must segment the conversation into **episodes**:

- Each **episode** has one coherent objective (a plan, a question, a fix)
- Completed episodes are **compressed into summaries** and archived
- The model sees only the **current episode** + a **cross-episode summary**
- Episodes are the unit of memory: the model works within an episode, the orchestrator manages across episodes
- Episodes persist across terminal restarts — last night's work is this morning's archived episode

> **The model should never have to "remember" what happened in episode 3 of a session with 12 episodes. The system records and injects that knowledge selectively.**

This principle enables sessions that span hours, days, and dozens of distinct tasks without the model ever exceeding its context horizon.

---

## 3. The Core Insight: Degradation-Adaptive Interaction

### The Problem Schematic

```
Cloud model (Claude 4, GPT-5):
  ┌──────────────────────────────────────────────────────────────┐
  │ [System] [User] [Tool] [Tool] [Tool] [Tool] [Tool] [Tool]   │
  │ 200K context, no degradation across 50+ tool turns           │
  └──────────────────────────────────────────────────────────────┘

Local SLM without adaptation:
  ┌──────────────────────────────────────────────────────────────┐
  │ [System] [User] [Tool] [Tool] [Tool] [TOOL_FAILS turn 8]    │
  │ 32K context, model forgets facts, format decays, persona     │
  │ drops. Session becomes useless.                              │
  └──────────────────────────────────────────────────────────────┘

Local SLM with degradation-adaptive architecture:
  ┌────APPROVAL GATE────┐    ┌────APPROVAL GATE────┐    ┌────┐
  │ [System] [Plan/Perm] │    │ [RefreshSystem]  [Pl]│    │ ...│
  │ [Explore] [Tool]  ..│    │ [Execute] [Tool] [..]│    │    │
  │ max 5 turns          │    │ max 5 turns          │    │    │
  └──────────────────────┘    └──────────────────────┘    └────┘
```

### The Refresh Cycle

Each approval-gated segment:
1. **Re-injects** standing instructions, session facts, active plan position
2. **Resets** the turn counter that drives degradation detection
3. **Runs** up to `model.memory_horizon - 2` turns (safety margin)
4. **Requires** user approval before the next segment

This means a task that requires 20 agentic turns gets broken into 4 segments of 5 turns each, with context refreshes between them. The model never exceeds its proven competency horizon.

### The Critical Gap: Unbounded Sessions

The refresh cycle above solves degradation within a **single task episode**. But real coding agent sessions are **never one task**:

```
Real session — unbounded, interleaved:
┌──────────────────────────────────────────────────────────────────┐
│ [Task A: refactor auth module — 5 episodes, complete]            │
│ [User asks: "What does the config parser look like?" — task B]   │
│ [Task C: add logging to database module]                         │
│ [User: "Actually, the auth refactor broke something, fix it"]    │
│ [Task D: fix regression in auth from 3 episodes ago]             │
│ [User pauses, comes back 2 hours later: "Continue the refactor"] │
│ → Context is now 12+ episodes deep, 100+ turns, model           │
│   has no idea what happened before.                              │
└──────────────────────────────────────────────────────────────────┘
```

Without an **episodic session model**, three failure modes emerge:

1. **Context overflow**: The raw conversation history exceeds the model's context window. History truncation (keep-last-N) drops the task definition, standing instructions, and session facts the model needs.
2. **Interleaved task confusion**: The model conflates facts from Task A ("what we decided about JWT") with Task C ("logging format") because they're all in the same flat history.
3. **Cross-session amnesia**: The user returns after a break and the model has no memory of the session state. The user has to re-explain everything from scratch.

The episodic session model (detailed in [Section 7.6](#76-episodic-session-model)) solves all three.

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                                │
│  (Terminal / TUI — primary MVP interface)                            │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Chat Buffer   │  │ Permission Panel │  │ Diff / File Preview  │  │
│  │ (streaming)   │  │ (3 modes)        │  │ (approve/skip/reject)│  │
│  └──────────────┘  └──────────────────┘  └──────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    ORCHESTRATOR (Main Loop)                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Phase Router: Explore → Plan → Execute → Verify → Iterate   │   │
│  │                                                              │   │
│  │ Degradation Monitor: turn_counter(limit=4), memory_check,    │   │
│  │                     format_qc, correction_frequency_tracker   │   │
│  │                                                              │   │
│  │ Approval Gate: propose → wait_user → proceed/abort/modify   │   │
│  │                                                              │   │
│  │ Test Executor: detect_framework → run_tests → show_results  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    CONTEXT ENGINE                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Permanant Buffer     │ State Buffer    │ History Buffer     │   │
│  │ - system prompt      │ - current files │ - last 3 tool-     │   │
│  │ - standing instrs*   │ - active plan   │   result pairs     │   │
│  │ - safety rules       │ - progress      │                     │   │
│  │ - format requirements│                  │                     │   │
│  ├──────────────────────┴──────────────────┴────────────────────┤   │
│  │ Session Facts Buffer (re-injected every N turns)              │   │
│  │ - files created/modified  - decisions made  - user prefs     │   │
│  │                                                              │   │
│  │ Format Monitor (first tool-call QC check)                     │   │
│  │ - detects JSON degradation, triggers refresh on early warning │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  * Standing instructions re-injected every turn (counters recency)  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    MODEL LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ONE LOCAL MODEL (configurable via degradation profile)      │   │
│  │  - Single model handles all phases (explore/plan/execute)    │   │
│  │  - Profile determines turn limit, refresh interval, guards   │   │
│  │  - Configurable: --model llama3.2 or set in profile config   │   │
│  │                                                              │   │
│  │  Degradation Profile (per model):                            │   │
│  │  memory_horizon, format_horizon, persona_horizon,            │   │
│  │  recency_pat, context_refresh_interval, requires_format_guard│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ╔══════════════════════════════════════════════════════════════╗   │
│  ║ Post-MVP: Multi-model routing (Fast Model / Strong Model / ║   │
│  ║ Cloud Escalation) will be added after the single-model loop ║   │
│  ║ is proven. See §5.3 and §8.3 for deferred design sketches. ║   │
│  ╚══════════════════════════════════════════════════════════════╝   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    TOOL EXECUTION LAYER                              │
│  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────────┐ │
│  │read_file ││write_file││  patch   ││  bash    ││ hybrid_search│ │
│  ├──────────┤├──────────┤├──────────┤├──────────┤├──────────────┤ │
│  │ snapshot ││ read-    ││ semantic ││ persis-  ││ BM25+token   │ │
│  │/rollback ││ before-  ││ merge    ││ tent     ││ embedding    │ │
│  │          ││ write    ││ fallback ││ shell    ││              │ │
│  └──────────┘└──────────┘└──────────┘└──────────┘└──────────────┘ │
│  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────────┐ │
│  │ Quality  ││ Early    ││ Format   ││ Verify   ││ Test         │ │
│  │ Monitor  ││ Stop     ││ Monitor  ││ Code     ││ Execution    │ │
│  ├──────────┤├──────────┤├──────────┤├──────────┤├──────────────┤ │
│  │Snapshot  ││Correction││          ││          ││              │ │
│  │ Rollback ││Frequency ││          ││          ││              │ │
│  │          ││ Tracker  ││          ││          ││              │ │
│  └──────────┘└──────────┘└──────────┘└──────────┘└──────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 5. Layer 0: Model Compatibility & Degradation Profiles

### 5.1 Degradation Profile Schema

Every model the agent supports has a degradation profile that answers one question: **"How many consecutive tool-calling turns can this model sustain before which failure mode?"**

```toml
# Standard degradation profile (TOML)
[model.degradation]
# Core turn horizons: how many consecutive turns before each failure mode triggers
memory_horizon = 5          # turns before fact retrieval fails (< 0.8 mean score)
iad_horizon = 30            # turns before instruction adherence decays
format_horizon = null       # null = no format degradation observed (e.g., E4B)
persona_horizon = null      # null = persona never drops
hallucination_horizon = 4   # turns before token-count threshold (scaled by token/s)
recency_horizon = 1         # turns before recency bias overrides original instruction

# Context refresh config
context_refresh_interval = 5  # turns between automatic context refreshes
session_fact_injection = "every_turn"  # | "every_refresh" | "never"

# Format guard
requires_format_monitor = false      # does the model need JSON format correction?
format_injection_threshold = 0.5     # inject format reminder when score < this

# Persona guard
requires_persona_guard = false       # does the model need persona re-injection?
persona_check_interval = 10          # turne between persona checks

# Model capability tiers
capability = {
    exploration = 0.8,       # 0-1 scale: ability to explore/read unfamiliar code
    planning = 0.6,          # ability to propose coherent multi-step plans
    execution = 0.9,         # ability to write accurate code changes
    debugging = 0.5,         # ability to diagnose and fix bugs
}

# Model strengths for routing decisions
strengths = ["format_adherence", "instruction_following"]
weaknesses = ["memory_retrieval", "multi_file_coordination"]
```

### 5.2 Evaluated Profiles (From Our Data)

| Field | LFM2.5-8B-A1B | Gemma 4 E2B | Gemma 4 E4B |
|-------|:-------------:|:------------:|:------------:|
| memory_horizon | 8 | 8 | **5** |
| iad_horizon | **10** | 20 | **30** |
| format_horizon | **1** | 20 | null |
| persona_horizon | null | **10** | null |
| recency_horizon | 1 | 1 | 1 |
| context_refresh_interval | 5 | 5 | 5 |
| requires_format_monitor | **true** | false | false |
| requires_persona_guard | false | **true** | false |
| strengths | efficiency, persona | stability | format, IAD |
| weaknesses | format drift, IAD | persona, memory | memory |

### 5.3 Estimated Profiles — Post-MVP

> **Post-MVP Enhancement** — Estimated profiles for untested models will be generated once the core agent loop is proven. Skip during MVP implementation.

| Field | Qwen3-8B | DeepSeek Coder V3 Lite | Llama 4 Scout |
|-------|:--------:|:---------------------:|:-------------:|
| memory_horizon | 10 (est) | 12 (est) | 8 (est) |
| iad_horizon | 15 (est) | 20 (est) | 15 (est) |
| notes | Reasoning model → needs thinking budget control | Specialized for code → better debugging | General purpose → similar to Gemma |
| requires_format_monitor | Medium | Low | Medium |
| thinking_budget | 2000 | N/A | 2000 |

### 5.4 Auto-Detection Protocol (Post-MVP)

> **Post-MVP Enhancement** — MVP ships with pre-computed profiles from the evaluation data in Section 5.2. Auto-detection via micro-calibration will be added after the core agent loop is proven in production.

For models without a pre-computed profile (post-MVP), run a **micro-calibration** during the first session:

1. **Memory quick check** (5 filler turns, inject fact, test recall) → ~60s
2. **Format quick check** (3 tool-call turns, check JSON adherence) → ~40s
3. **Recency check** (1 turn, override instruction) → ~20s

This produces initial `memory_horizon`, `format_horizon`, and `recency_horizon` estimates. Full profile is refined over subsequent sessions.

---

## 6. Layer 1: Interactive Permission Subsystem (MVP)

### 6.1 Design — MVP: Three Permission Modes

This is the **most critical architectural difference** from SmallCode's auto-approve mode. The permission subsystem gates every tool-calling turn or batch of turns.

**MVP ships with 3 permission modes** — Read-only, Plan, and Step. Batch and Auto modes are deferred to post-MVP (see Section 6.6).

```
User: "Refactor the auth module to use JWT instead of sessions"

   ↓

Phase 1: UNDERSTAND (automatic, no approval needed)
   ├── Agent reads auth.js, middleware.js, config.js
   ├── Agent reads test files, dependency files
   └── Agent summarizes: "Found 4 files affected, here's the current auth flow"

   ↓

Phase 2: PLAN (requires approval — Plan mode)
   ├── Agent: "Here's my plan:
   │     1. Create auth/jwt.js with JWT utility functions
   │     2. Update auth/middleware.js to verify JWT tokens
   │     3. Update routes to use new middleware
   │     4. Update tests
   │     5. Remove old session-based code"
   └── User: "Approved, but skip step 5 for now"

   ↓

Phase 3: EXECUTE (requires approval per step — Step mode)
   ├── Step 1: Agent writes auth/jwt.js
   ├──    → Shows diff to user → "Approve?" → User: "Yes"
   ├── Step 2: Agent edits auth/middleware.js
   ├──    → Shows diff → "Approve?" → User: "Yes"
   ├── Step 3: Agent edits routes
   ├──    → Shows diff → "Approve?" → User: "Looks wrong, keep sessions for now"
   ├── Step 4: Agent updates tests
   └──    → Shows diff → "Approve?" → User: "Yes"

   ↓

Phase 4: VERIFY (automatic, shows results)
   ├── Agent runs tests → "3/3 pass"
   └── User: "Great, done"
```

### 6.2 Permission Modes (MVP)

```
┌──────────────────────────────────────────────────────────────────┐
│ PERMISSION MODE SELECTOR (MVP — 3 modes)                          │
│                                                                  │
│  [Read-only] Read files, search code, explore — no approval     │
│              Model can read/explore but cannot write or execute  │
│                                                                  │
│  [Plan]      Propose plan — requires approval to proceed         │
│              Model proposes numbered plan, user approves or      │
│              modifies before any execution                       │
│                                                                  │
│  [Step]      Execute one step — requires approval per edit/bash  │
│              Default mode for MVP. Each tool call is proposed,   │
│              user approves/rejects/edits individually.           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 Permission Policies (TOML Config — MVP)

```toml
[permissions]
default_mode = "step"           # default permission mode for new sessions

# Per-tool approval rules — MVP
[permissions.tools]
bash = { requires_approval = true, max_consecutive = 3, dangerous_patterns = ["rm -rf", "> /dev/"] }
write_file = { requires_approval = true, max_size_bytes = 50000 }
patch = { requires_approval = true }
read_file = { requires_approval = false, max_consecutive = 20 }
search = { requires_approval = false }

# Approval shortcuts — MVP
[permissions.shortcuts]
"y" = "approve_current"
"n" = "reject_current"  
"ye" = "approve_and_explain"   # approve but ask agent to explain the change
"e" = "edit_current"           # reject but modify the proposed action
```

### 6.4 The Approval Protocol (MVP)

```
User Message → Orchestrator → Model proposes action(s)
  → Orchestrator formats as: "Proposed: [write_file path/to/file.py]
    Reason: [model's explanation]
    Diff: [+3/-1 lines]
    Approve? [y/n/ye/e]"
  → User responds
  → Orchestrator executes (or skips) based on response
  → If approved: execute tool, show result, continue to next step
  → If rejected: skip to next step or re-plan
  → If edited: modify the action based on user input, re-propose
```

### 6.5 Degradation-Adaptive Permission Tuning (MVP)

The system automatically tightens permissions as degradation is detected:

```javascript
// Inside the orchestrator loop — MVP
if (this.turnCounter >= modelProfile.degradation.memory_horizon - 2) {
  // Approaching memory cliff — require approval even for reads
  this.permissionMode = 'step'; // Force step-by-step approval
  this.injectContextRefresh(); // Re-inject session facts
}

if (formatMonitor.score < modelProfile.degradation.format_injection_threshold) {
  // Format quality dropping — inject reminder and require approval
  this.injectFormatReminder();
}

if (turnCounter >= modelProfile.degradation.context_refresh_interval) {
  // Time for a scheduled context refresh
  this.injectContextRefresh();
  this.turnCounter = 0; // Reset the counter
}
```

### 6.6 Post-MVP: Extended Permission Modes

> **Post-MVP Enhancement** — Batch and Auto modes will be implemented after the core 3-mode MVPs proven. The --auto-approve-* flags and extended approval shortcuts are deferred.

**Batch mode** (post-MVP): Execute all planned steps — requires one approval for the batch, then auto-approves each step within the batch.

**Auto mode** (post-MVP): Full autonomy — no approval required per step. Equivalent to Claude Code's default mode.

**Extended overrides** (post-MVP):

```
  Overrides (post-MVP):
    --auto-approve-reads     Auto-approve all file reads
    --auto-approve-writes    Auto-approve all file writes
    --auto-approve-patches   Auto-approve all patches
    --auto-approve-bash      Auto-approve all shell commands
    --max-auto-steps N       Auto-approve up to N consecutive
                             steps before requiring confirmation
```

**Extended shortcuts** (post-MVP):
```
"ya" = "approve_and_auto"     # approve this + auto-approve remaining in batch
```

---

## 7. Layer 2: Degradation-Aware Context Engine

### 7.1 The Four-Buffer Architecture

Rather than a single flat context window, the agent maintains four logical buffers that are assembled into the prompt on each turn.

```
┌──────────────────────────────────────────────────────────────────┐
│                    ASSEMBLED PROMPT                               │
│                                                                  │
│  [PERMANENT BUFFER — always present, never truncated]             │
│  ├── System prompt (codename, role)                               │
│  ├── Standing instructions ("Always output tool calls as JSON")   │
│  ├── Safety rules ("Never delete files without confirmation")     │
│  ├── Format requirements (JSON schema, markdown rules)            │
│  └── Output format template                                      │
│                                                                  │
│  [SESSION FACTS BUFFER — re-injected every N turns]               │
│  ├── Task objective                                              │
│  ├── Files created/modified so far                               │
│  ├── Key decisions and reasoning                                 │
│  ├── User preferences detected                                   │
│  └── Current session state summary                                │
│                                                                  │
│  [STATE BUFFER — always present]                                  │
│  ├── Active plan (numbered steps with ✓/→ markers)                │
│  ├── Current phase (explore/plan/execute/verify/iterate)          │
│  ├── Current step position                                        │
│  └── Open file/context hint ("currently viewing: auth.js")       │
│                                                                  │
│  [HISTORY BUFFER — sliding window of last 3 tool-result pairs]    │
│  └── tool_call / tool_result pairs (truncated to 4000 chars)     │
│                                                                  │
│  [CURRENT USER MESSAGE]                                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Session Facts Buffer — Detailed Schema

This is the most important innovation. The session facts buffer stores **key facts the model must remember**, structured so they can be injected at any point regardless of conversation history.

```javascript
// Session Fact — a single atomic fact the model should remember
{
  id: "fact-3",
  category: "decision",       // "file_created" | "decision" | "user_preference" | "task"
  timestamp: 1749372000,
  description: "Moved error handling to middleware.js instead of per-route",
  context: "User requested cleaner error handling during auth refactor step 2",
  confidence: "definite",     // "definite" | "inferred" | "tentative"
  requires_approval: false,   // was this fact approved by user?
}

// Fact injection on each refresh:
`[SESSION FACTS]
• Files created: auth/jwt.js, auth/middleware.js
• Files modified: routes/auth.js, tests/auth.test.js
• Decisions: Error handling moved to middleware layer (user approved)
• In progress: Step 3 of 5 (update routes)
• Latest test result: 3/3 passing

Standing instructions still active:
• Always output tool calls as valid JSON
• Verify code after every write by running pytest on affected files`
```

### 7.3 Context Refresh Protocol

Triggered by:
1. **Turn counter** reaches `context_refresh_interval`
2. **Approval gate** fires (user approves/rejects a step)
3. **Degradation detection** (format monitor drops below threshold)
4. **Phase transition** (explore → plan → execute → verify)

What the refresh injects (all at once, as a single system-level message):
```
[CONTEXT REFRESH]
→ Task: Refactor auth module from sessions to JWT
→ Status: Step 3 of 5 (Update routes/auth.js)
→ Recent: auth/jwt.js ✓ | auth/middleware.js ✓ | modifying routes/auth.js
→ Facts: JWT secret stored in .env, token expiry set to 1 hour
→ Standing: Output tool calls as JSON. Verify with pytest after each write.
→ Remaining: steps 4 (tests), 5 (cleanup old session code)
```

### 7.4 History Pruning Strategy

Instead of keeping the entire conversation history, the engine maintains:

1. **Last 3 tool-result pairs** (model input → tool output), each truncated to 4000 chars
2. **Session facts buffer** (the compressed, structured version of everything important)
3. **Plan state** (number of plan with current step)

This is enough for the model to understand "what just happened" without the full 50-turn context.

### 7.5 Recency Bias Guard — Standing Instructions Block

Since recency bias is **universal at depth 1** (all 3 models score 0.000), the system maintains a **standing instructions block** that is:

1. Injected at the TOP of the system prompt (highest priority position)
2. Re-injected on every context refresh
3. Written in **imperative, short form** (models follow short instructions better)

```markdown
## STANDING INSTRUCTIONS (always active, do not override)
- Output ALL tool calls as valid JSON wrapped in ```json``` blocks
- NEVER delete files without explicit user approval
- Run `pytest <file>` after every code modification to verify correctness
- Follow the ACTIVE PLAN below — do not skip ahead
- If you are unsure about a requirement, ASK the user — do not guess
- Do not suggest refactoring unrelated code unless asked
```

---

### 7.7 Format Monitor (MVP)

> **Added for MVP** — The format monitor checks the first tool-call output of each segment for early warning signs of JSON format degradation. This is a built-in QC check that runs before any content is processed.

The format monitor inspects the model's first tool-call output in each segment:

```javascript
class FormatMonitor {
  constructor(modelProfile) {
    this.enabled = modelProfile.requires_format_monitor || true; // Always on for MVP
    this.threshold = modelProfile.format_injection_threshold || 0.5;
    this.segmentFirstCall = null;
  }

  check(toolCallOutput) {
    // Check 1: Valid JSON structure?
    try {
      JSON.parse(toolCallOutput);
    } catch (e) {
      this.triggerRefresh("JSON parse failed on first tool call");
      return { passed: false, issue: "invalid_json" };
    }

    // Check 2: Required fields present?
    const hasRequiredFields = toolCallOutput.includes('"tool"') || 
                              toolCallOutput.includes('"action"');
    if (!hasRequiredFields) {
      this.triggerRefresh("Missing required fields in first tool call");
      return { passed: false, issue: "missing_fields" };
    }

    // Check 3: Degradation warning (partial format drift)
    const bracketsBalanced = (toolCallOutput.match(/{/g) || []).length === 
                             (toolCallOutput.match(/}/g) || []).length;
    if (!bracketsBalanced) {
      this.triggerRefresh("Unbalanced braces in output");
      return { passed: false, issue: "unbalanced_braces" };
    }

    this.segmentFirstCall = { passed: true, output: toolCallOutput };
    return { passed: true };
  }

  triggerRefresh(reason) {
    // Force context refresh + format reminder injection
    orchestrator.injectContextRefresh();
    orchestrator.injectFormatReminder();
    this.segmentFirstCall = null; // Reset for next segment
    console.log(`[Format Monitor] Refresh triggered: ${reason}`);
  }
}
```

**Early warning signs** that trigger a refresh:
- Invalid JSON (parse failure) — most severe
- Missing required fields (tool name, path, content) — moderate
- Unbalanced braces or brackets — structural degradation
- Hallmark character corruption (e.g., `\\"` instead of `"`) — model nearing hallucination horizon

**MVP behavior**: On any early warning, force a context refresh with format reminder injection before the model continues. This resets the degradation clock at minimal cost (~1 extra turn).

### 7.8 Selective Reference Episode Loading (Post-MVP)

> **Post-MVP Enhancement** — MVP uses ALL compressed episode summaries from the episode store (typically 500-2000 tokens each, fitting comfortably within a 32K context window). Selective reference loading — choosing only the most relevant prior episodes to inject — will be added post-MVP to optimize context efficiency.

The MVP approach is simple: on each episode transition, inject the compressed summaries of ALL completed episodes into the cross-episode summary block. This is straightforward to implement and avoids complex relevance-scoring logic:

```
[SESSION HISTORY SUMMARY — 5 previous episodes (MVP: ALL episodes included)]
Summary format: ~500-2000 tokens per episode
Cost for 20 episodes: ~10,000-40,000 tokens (well within 32K-128K context)
```

Post-MVP, a selective loading strategy will choose only the N most relevant prior episodes based on:
- File overlap (episodes touching the same files)
- Decision relevance (decisions referenced in current plan)
- Time recency (last 3 episodes always included)

### 7.6 Episodic Session Model

#### The Problem

The four-buffer architecture solves degradation within a single task. But real sessions look like this:

```
Episode 1: "Refactor auth module to JWT"        → 5 context refreshes, 25 turns
Episode 2: "What does the config parser look?"   → read-only, 3 turns  
Episode 3: "Add logging to database module"      → 2 refreshes, 10 turns
Episode 4: "Fix regression in auth refactor"     → 2 refreshes, 8 turns
Episode 5: "Continue the API documentation"      → 3 refreshes, 15 turns
                                                  ──────────────────────────
                                                  5 episodes, 61 total turns
```

If you feed all 61 turns into the model's context window (even with truncation), every subsequent turn:
- **Wastes tokens** on irrelevant history (the config parser question doesn't help with the API docs)
- **Confuses the model** — it sees facts from previous episodes ("we decided to skip middleware for now") that conflict with current episode instructions
- **Exceeds context** for 32K models after ~5-6 episodes

The model doesn't need the full history — it needs a **compressed, cross-episode summary** plus the **current episode's full context**.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ EPISODE STORE (persistent, on disk)                              │
│                                                                  │
│  Episode #1 — "Refactor auth to JWT"                             │
│  ├── status: completed                                           │
│  ├── duration: 25 turns                                          │
│  ├── files: auth/jwt.js (created), auth/middleware.js (modified) │
│  ├── decisions: JWT_TokenExpiry=1h, ErrorHandling=middleware     │
│  └── summary: "Replaced session auth with JWT. 3 files changed, │
│                5/5 tests passing. Used jsonwebtoken library."    │
│                                                                  │
│  Episode #2 — "Config parser question"                           │
│  ├── status: completed                                           │
│  ├── duration: 3 turns                                           │
│  ├── files: (read-only) config/parser.js                         │
│  ├── decisions: (none)                                           │
│  └── summary: "Explored config parser, no changes made."        │
│                                                                  │
│  Episode #3 — "Add database logging"                             │
│  ├── status: completed                                           │
│  ├── duration: 10 turns                                          │
│  └── summary: "Added structured logging to db queries. Won't    │
│                log auth table."                                  │
│                                                                  │
│  Episode #4 — "Fix auth regression"                              │
│  ├── status: completed                                           │
│  ├── duration: 8 turns                                           │
│  ├── references: Episode #1 (auth refactor)                      │
│  └── summary: "Fixed JWT middleware to handle expired tokens     │
│                correctly. Added token refresh endpoint."         │
│                                                                  │
│  Episode #5 — "API documentation" [ACTIVE EPISODE]               │
│  ├── status: executing                                           │
│  ├── current_phase: execute                                      │
│  ├── current_step: 3/6 steps                                     │
│  └── plan: [1. Read route files, 2. Generate OpenAPI schema...] │
└──────────────────────────────────────────────────────────────────┘
```

#### Episode Lifecycle

```
User sends message
       │
       ▼
┌─────────────────────────────────────┐
│ Episode Boundary Detector           │
│                                     │
│ Is this a new task?                 │
│  ├── User mentions different file   │
│  ├── User says "actually", "also"   │
│  ├── User asks a question (read-op) │
│  ├── User reports a bug in prev ep  │
│  └── User pauses + returns >5 min   │
│                                     │
│ On new episode detected:            │
│  1. Close active episode            │
│  2. Compress to summary             │
│  3. Save to episode store           │
│  4. Create new episode              │
│  5. Inject cross-episode summary    │
│  6. Reset turn counter (fresh ctx)  │
└─────────────────────────────────────┘
```

#### Episode Boundary Detection Heuristics

```javascript
class EpisodeBoundaryDetector {
  detect(message, currentEpisode) {
    // Case 1: Time gap → new episode (user came back after break)
    if (this.timeSinceLastMessage > 300_000) return true; // 5 min

    // Case 2: Different file/module mentioned → new episode or sub-episode
    if (currentEpisode.files && this.extractFileReferences(message)
          .some(f => !currentEpisode.files.includes(f))) {
      return this.priorityCheck(message); // sub-episode if directly related
    }

    // Case 3: Tense shift → new episode
    if (/^(actually|also|meanwhile|by the way|back to)/i.test(message)) return true;

    // Case 4: Bug report about previous episode → new episode with reference
    if (/(broke|regression|stopped working|error in)/i.test(message) &&
        this.extractFileReferences(message).some(f => this.wasModified(f))) {
      return true; // auto-inject reference to the modifying episode
    }

    // Case 5: Question mode (read-only, no plan)
    if (/^(what|how|why|where|when|who|tell me about|show me)/i.test(message) &&
        !/^(create|add|change|update|refactor|fix|implement)/i.test(message)) {
      return true; // lightweight episode, no plan needed
    }

    // Case 6: Continuation of current episode
    if (/continue|next|proceed|keep going/i.test(message)) return false;

    // Case 7: Default — stay in current episode if it has an active plan
    return currentEpisode.phase !== 'done' && currentEpisode.status === 'executing';
  }
}
```

#### Cross-Episode Summary Injection

When a model starts a new episode, it receives this in its system prompt:

```
[SESSION HISTORY SUMMARY — 5 previous episodes]
You have completed 5 episodes in this session. Here are the relevant facts:

Files created across all episodes:
  auth/jwt.js, auth/middleware.js, db/logging.js

Key architectural decisions:
  · JWT tokens expire after 1 hour (Episode 1, user approved)
  · All error handling goes through middleware.js (Episode 1, user approved)
  · Auth table excluded from database logging (Episode 3, user requested)

Active issues (in progress, unresolved):
  · (none)

References for current episode:
  · This episode references auth/jwt.js and auth/middleware.js from Episode 1.
   Refer to those for the current implementation.
  · The token refresh endpoint was added in Episode 4 (fix for expired token bug).

===== CURRENT EPISODE BEGINS HERE =====
[Full context for this episode: plan, standing instructions, session facts]
```

This gives the model just enough cross-episode context to understand what's happening without dumping the entire 61-turn history into its context window.

#### Episode Archiving & Compression

When an episode is closed, the orchestrator generates a **compressed summary**:

```javascript
function compressEpisode(episode) {
  return {
    id: 4,
    title: "Fix auth regression",
    status: "completed",
    turns: 8,
    duration_seconds: 720,
    user_request: "The auth module broke, middleware returns 403 on valid tokens",
    files_created: [],
    files_modified: ["auth/middleware.js"],
    decisions: [
      "Check token expiry before signature (was checking signature first)",
      "Added auth/middleware.test.js for edge cases",
    ],
    test_results: { passed: 5, failed: 0, total: 5 },
    outcome: "Fixed token validation order. Tests passing.",
    references: [1],  // references Episode 1 (the original auth refactor)
    rollbacks: 0,
    user_corrections: 1,  // user caught: "Check expiry before signature"
  };
}
```

The summary format is designed so 50 episodes take ~5000 tokens of injected context — well within even a 32K model's budget.

#### Session Persistence (Cross-Terminal)

Episodes are saved to `~/.local-code-agent/sessions/<session_id>/episodes/` as individual JSON files:

```
~/.local-code-agent/sessions/
└── 2026-06-08-auth-refactor/
    ├── session.json           # session metadata, model used, start time
    ├── episodes/
    │   ├── 001-refactor-jwt.json    # completed
    │   ├── 002-config-question.json # completed
    │   ├── 003-db-logging.json      # completed
    │   ├── 004-fix-regression.json  # completed
    │   └── 005-api-docs.json        # active (not yet compressed)
    ├── session_facts.json    # aggregated facts across all episodes
    └── model_profile.json    # degradation telemetry (refined each session)
```

When the user re-launches the agent after a break:

1. Agent finds the most recent session
2. Loads all completed episode summaries
3. Builds the cross-episode summary
4. Resumes the last active episode (or starts a new one if user gives new task)
5. The model has **zero recall loss** across sessions

#### Episodic Phase Router — Extended

The phase router from Section 8 gains an extra transition for inter-episode boundaries:

```
┌─────────────────────────────────────────────────────────────────────┐
│ EPISODE TRANSITION (new episode detected)                           │
│                                                                     │
│ 1. Snapshot current episode state (files, plan position, facts)     │
│ 2. Compress into summary                                            │
│ 3. Save to episode store                                            │
│ 4. Build cross-episode summary from all prior episodes              │
│ 5. Reset all buffers (fresh context for new episode)                │
│ 6. Inject cross-episode summary into system prompt                  │
│ 7. Start new episode                                                │
│ 8. If new episode references a prior episode:                       │
│    ├── Load the reference episode's full context (not compressed)   │
│    └── Include only the relevant files/decisions                    │
│                                                                     │
│ Cost: ~500-2000 tokens for cross-episode summary                    │
│ vs. 20,000+ tokens for full conversation history                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Example: Full Session Lifecycle

```
User launches agent → New session created
  ↓
User: "Refactor auth to use JWT" → Episode 1 starts
  │  └── Plan: 1. Create jwt.js  2. Update middleware  3. Update routes  4. Tests
  │  └── 5 context refreshes, 25 turns
  └── User opens a separate terminal, comes back 30 min later
  ↓
User: "What does the config parser look like?" → Episode 2 starts
  │  └── Episode 1 compressed, cross-episode summary injected
  │  └── 3 read-only turns, no plan
  ↓
User: "Add logging to the db module" → Episode 3 starts
  │  └── Episode 2 compressed
  │  └── Cross-episode summary: "Episode 1: JWT refactor. Episode 2: config question"
  │  └── 2 refreshes, 10 turns
  ↓
User: "The auth module broke after the changes" → Episode 4 starts
  │  └── Episode 3 compressed
  │  └── Cross-episode summary: "In Episode 1 (auth refactor), files modified..."
  │  └── 2 refreshes, 8 turns
  ↓
User closes terminal → All episodes saved
  └── Next launch: "Session 2026-06-08 has 4 completed episodes. Resume?"
## 8. Layer 3: Phase-Based Agent Loop (MVP)

### 8.1 The Five-Phase Protocol (MVP)

Each coding task follows this phase sequence. The MVP uses a **single local model** throughout (configurable via model profile), with the same permission mode applied per phase.

```
User Request
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: UNDERSTAND                                              │
│                                                                  │
│ Model: Single configured local model                              │
│ Permission: Read-only (auto-approve all file reads/search)       │
│ Max turns: 5 (strict turn counter)                               │
│ Context: Full four-buffer assembly                                │
│                                                                  │
│ Actions: read_file, hybrid_search, find_files                    │
│ Output: Summary of codebase state, affected files                │
│                                                                  │
│ Exit: Model produces summary → user reviews → proceed            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: PLAN                                                    │
│                                                                  │
│ Model: Single configured local model                              │
│ Permission: Read-only (auto-approve)                             │
│ Max turns: 3 (plan generation + refinement)                      │
│ Context: Session facts + codebase summary only                   │
│                                                                  │
│ Actions: None (model only produces text plan)                    │
│ Output: Numbered plan with file paths, expected changes          │
│                                                                  │
│ Exit: User approves plan (possibly with modifications)           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3: EXECUTE (The critical phase)                            │
│                                                                  │
│ Model: Single configured local model                              │
│ Permission: Step-by-step (requires approval per write/patch)     │
│ Turn counter: Hard limit of 4 turns per segment                  │
│ Context: Four-buffer architecture (max ~8K assembled)            │
│                                                                  │
│ Per-step flow:                                                    │
│   1. Model proposes tool call + explanation                      │
│   2. Orchestrator shows: [Tool] [Path] [Diff/Command]            │
│   3. User approves/rejects/edits                                  │
│   4. On approval: execute tool, run tests, show result           │
│   5. On rejection: skip step or re-plan                          │
│   6. Every 4 turns: hard context refresh                          │
│   7. Format monitor checks first tool call output                 │
│   8. On format degradation: inject format reminder + refresh      │
│                                                                  │
│ Monitoring: turn_counter(limit=4), format_score, correction_freq │
│                                                                  │
│ Exit: All steps complete OR turn limit reached → verify          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4: VERIFY (with built-in test execution)                   │
│                                                                  │
│ Model: Not needed (uses compiler/test output, not model judgment)│
│ Permission: Read + bash (auto-approve for test/compile runs)     │
│ Max turns: 3 (run tests, compile, summarize results)             │
│ Context: Minimal ("what was changed" + test output)             │
│                                                                  │
│ Required step — Runs after EVERY write:                          │
│   1. Detect project test framework (pytest, jest, unittest, etc.)│
│   2. Run tests on affected files/modules                          │
│   3. Show test results inline                                     │
│                                                                  │
│ Actions: bash(pytest), bash(npm test), read_file(check)          │
│ Output: Test results summary, any warnings/errors                │
│                                                                  │
│ Snapshots taken before execution (rollback available).           │
│                                                                  │
│ Exit: Tests pass → done. Tests fail → phase 5.                  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Tests pass?   │── Yes → ✅ DONE
                    └────────┬────────┘
                             │ No
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 5: ITERATE (Debug Loop)                                    │
│                                                                  │
│ Model: Same local model                                          │
│ Permission: Step-by-step (same as Phase 3)                       │
│ Turn counter: 4 turns per segment                                │
│ Context: Previous attempts + error output + test failure details │
│                                                                  │
│ Flow:                                                             │
│   1. Show test failure output to model (truncated)                │
│   2. Model diagnoses root cause                                   │
│   3. Model proposes fix (tool call)                               │
│   4. User approves                                               │
│   5. Execute fix                                                  │
│   6. Re-run tests (auto, after every write)                       │
│   7. After 3 failed attempts: suggest alternative approach        │
│                                                                  │
│ Auto-rollback on each failed attempt                              │
│                                                                  │
│ Exit: Tests pass, or max attempts reached (report to user).      │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 Phase Transition Rules (MVP)

| From | To | Condition | Context Action |
|------|----|-----------|---------------|
| Understand | Plan | Model produces summary + User agrees | Summarize findings into session facts |
| Plan | Execute | User approves the plan | Save plan to active state, reset turn counter |
| Execute | Verify | All steps executed OR turn limit reached | Snapshot files, detect test framework, prepare test commands |
| Execute | Plan (loop) | User rejects a step | Update plan state, re-propose with user correction |
| Verify | Done | All tests pass + user confirms | Archive session, save episode trace |
| Verify | Iterate | Any test fails | Load pre-execution snapshot, prepare error context |
| Iterate | Verify | Fix applied | Re-run tests from verify phase |
| Iterate | Execute | Model proposes mid-plan fix | Return to execute with updated plan |
| Any | Any | Turn counter hits 4 | Hard context refresh: re-inject standing instructions, session facts, plan state |

### 8.3 MVP Agent Loop Details

**Turn counter at 4** (not 5 — safety margin from the evaluation data showing memory retrieval fails at 5-8 turns):
- Hard stop after 4 tool-calling turns per segment
- Forces context refresh before the model hits its memory horizon
- Safety margin of 1-4 turns against the worst-case model's horizon

**Hard context refresh every N turns**:
- Re-injects standing instructions block (counters recency bias)
- Re-injects session facts buffer (counters memory failure)
- Re-injects active plan state (counters IAD)
- Resets turn counter to 0

**Approval gate per step**:
- Every tool call requires user approval before execution
- Shows: [Tool name] [Target path] [Diff preview / Command preview]
- User options: approve (y), skip (s), reject (n), edit (e)
- Diff display uses unified diff format for clarity

**Test execution after every write** (built-in verification):
- Detects project test framework: pytest, unittest, jest, mocha, etc.
- Runs tests on the specific file(s) modified (not full suite for speed)
- Shows test output inline: pass/fail counts, error messages
- If no test framework detected: runs syntax check (python3 -m py_compile, npm run build, etc.)

**Adaptive refresh interval** (Week 3 feature — see roadmap):
> **Post-MVP (Week 3)** — The refresh interval will adapt based on telemetry:
> - If user corrections are frequent (≥2 per segment): shrink interval to 3 turns
> - If format degradation detected early: shrink interval to 2 turns
> - If no issues for 5+ segments: stretch interval to 5 turns
> - MVP ships with hardcoded interval of 4 turns

**Progressive disclosure / streaming UI** (Week 2-3):
- Real-time token streaming in terminal as model generates
- Diff previews shown incrementally as they're produced
- Test output streams in as commands execute
- MVP (Week 1): simple block output — wait for full response, then show
- Post-MVP (Weeks 2-3): streaming progressive display

### 8.4 Phase Transition Rules Detail

| From | To | Condition | Context Action |
|------|----|-----------|---------------|
| Understand | Plan | Model says "I understand" + User agrees | Summarize findings into session facts |
| Plan | Execute | User approves the plan | Save plan to active state, reset turn counter |
| Execute | Verify | All steps executed OR batch complete | Snapshot all files, prepare test commands |
| Execute | Plan (loop) | User rejects a step | Update plan state, re-propose |
| Verify | Done | All tests pass + user confirms | Archive session, save trace |
| Verify | Iterate | Any test fails | Load snapshot, prepare error context |
| Iterate | Verify | Fix applied | Re-run tests from verify phase |
| Iterate | Execute | Model proposes mid-plan fix | Return to execute with fixed plan |
| Any | Any | Degradation monitor fires | Inject context refresh, continue |
---

## 9. Layer 4: Tool Execution & Validation

### 9.1 Inherited from SmallCode (Use As-Is)

The following SmallCode patterns should be used directly with minimal modification:

| Pattern | Source | Why Keep |
|---------|--------|----------|
| **Liquid tool parser** | `liquid_tool_parser.js` | Handles LFM2.5's non-standard format |
| **Two-stage routing** | `two_stage_router.js` | Category-first routing saves ~200 tokens/step |
| **Plan tracker + re-injection** | `plan_tracker.js` | ACTIVE PLAN anchor counters IAD |
| **Quality monitor** | `quality_monitor.js` | Empty response, hallucinated tool, repeat detection |
| **Early stop** | `early_stop.js` | Repetition loops, read-only streaks, patch spirals |
| **Snapshot/rollback** | `executor.js` snapshot logic | Safety net for bad edits |
| **Patch-first editing** | `executor.js` patch tool | Smaller diffs, fewer tokens |
| **Hybrid search** | `hybrid_search.js` | BM25 + token-embedding, no model download |
| **Thinking budget** | `thinking_budget.js` | Critical for Qwen3/DeepSeek reasoning models |
| **Trace recorder** | `trace_recorder.js` | Reproducibility and debugging |
| **Contract system** | `contract.js` | Definition of Done as testable assertions |

### 9.2 New: Format Adherence Monitor

SmallCode's quality monitor checks for **hallucinated tools** and **empty responses**. We need a specific **format adherence monitor** that catches the degradation patterns we measured:

| Failure Pattern | Our Data | Detection | Correction |
|----------------|----------|-----------|------------|
| JSON tool calls wrapped in markdown | LFM2.5: max 0.30 at all depths | regex: tool call inside ```json or prose block | "Output tool calls as raw JSON, not inside markdown blocks" |
| Tool calls replaced by plain text | All models at failure depths | No tool_call in response | "You must use a tool call. Use write_file, read_file, or bash." |
| JSON includes extra fields | E2B: partial_json failures | Validate against schema | "Only include {name, arguments} in your tool call" |
| Missing closing brace/wrapper | LFM2.5: partial_json | Count brace balance | "Complete the JSON before sending" |

```javascript
// FormatMonitor — new addition
class FormatMonitor {
  constructor(modelProfile) {
    this.score = 1.0;
    this.consecutiveFailures = 0;
    this.model = modelProfile;
  }

  inspect(response) {
    // Check 1: Does it have tool calls at all?
    // Check 2: Are tool calls parseable?
    // Check 3: Are tool calls clean JSON or wrapped in prose?
    // Check 4: Do all required keys exist?
    
    return { 
      score,        // 0.0-1.0
      failureType,  // "missing" | "wrapped" | "invalid_json" | "extra_fields"
      injection     // corrective prompt text (or null)
    };
  }

  // Inject format reminder into the next prompt
  getFormatReminder() {
    return this.model.degradation.requires_format_monitor
      ? `[FORMAT REMINDER] Always output tool calls as valid JSON. Do not wrap them in markdown code blocks or prose. Use: {"name": "tool_name", "arguments": {...}}`
      : '';
  }
}
```

### 9.3 New: Persona Guard

```javascript
// PersonaGuard — only activates for models with persona_horizon < null
class PersonaGuard {
  constructor(modelProfile) {
    this.enabled = modelProfile.degradation.persona_horizon !== null;
    this.checkCount = 0;
    this.interval = modelProfile.degradation.persona_check_interval;
    this.systemMessage = ''; // set by orchestrator
  }

  shouldCheck() {
    if (!this.enabled) return false;
    return (++this.checkCount % this.interval) === 0;
  }

  getPersonaReminder() {
    return `[PERSONA REMINDER] Remember: your role is "${this.systemMessage.slice(0, 200)}". Do not drop this persona.`;
  }
}
```

### 9.4 New: Memory Probes (MVP)

Periodically inject a low-cost memory test to detect if the model has forgotten session facts:

```
Embedded in a filler message: "Just to confirm, what file are we currently modifying?"
or
Injected into the session facts: "[MEMORY CHECK] What is the current task objective?"
```

If the model fails a memory probe, trigger an immediate context refresh.

### 9.5 New: Test Execution Tool (MVP)

> **Added for MVP** — The test execution tool runs after every write operation as a built-in verification step.

The test execution tool detects the project's test framework and runs relevant tests:

```javascript
class TestExecutor {
  async detectFramework(projectRoot) {
    // Check for common test frameworks
    if (await fileExists(path.join(projectRoot, 'pytest.ini')) ||
        await fileExists(path.join(projectRoot, 'pyproject.toml'))) {
      return 'pytest';
    }
    if (await fileExists(path.join(projectRoot, 'jest.config.js')) ||
        await fileExists(path.join(projectRoot, 'package.json'))) {
      return 'jest';
    }
    if (await fileExists(path.join(projectRoot, 'Cargo.toml'))) {
      return 'cargo_test';
    }
    // Fallback: check for test directories
    if (await dirExists(path.join(projectRoot, 'tests')) ||
        await dirExists(path.join(projectRoot, 'test'))) {
      return 'generic';
    }
    return null; // No test framework detected
  }

  async runTests(affectedFiles, framework) {
    // Run tests on the specific file(s) modified
    switch (framework) {
      case 'pytest':
        return await bash(`python -m pytest ${affectedFiles.join(' ')} -v --tb=short`);
      case 'jest':
        return await bash(`npx jest ${affectedFiles.join(' ')} --no-coverage`);
      case 'cargo_test':
        return await bash('cargo test');
      default:
        // Syntax check fallback
        return await bash(`python3 -m py_compile ${affectedFiles.join(' ')}`);
    }
  }

  async executeAfterWrite(filePath) {
    const framework = await this.detectFramework(process.cwd());
    if (!framework) {
      return { passed: true, note: 'No test framework detected. Syntax check skipped.' };
    }
    const result = await this.runTests([filePath], framework);
    return {
      passed: result.exitCode === 0,
      output: result.stdout,
      errors: result.stderr,
      framework: framework
    };
  }
}
```

**Integration into the agent loop** (Section 8):
- Auto-runs after every write_file or patch operation in the Execute phase
- Results are shown inline to the user
- If tests fail, the agent transitions from Verify → Iterate phase automatically
- The user can override (force-approve despite test failure)

### 9.6 New: Format Monitor as Built-in QC Check (MVP)

> **Added for MVP** — The format monitor (detailed in Section 7.7) also serves as a built-in quality control check that runs on every first tool-call output of each segment.

**What it checks**:
1. **First tool call output** — Is it valid JSON? Are required fields present?
2. **Brace balance** — Structural integrity of the output
3. **Required field presence** — Does the tool call have name/path/content?

**What triggers a refresh**:
- Invalid JSON (parse failure) → immediate context refresh + format reminder injection
- Missing required fields → inject format reminder, continue
- Unbalanced braces → early warning, inject format reminder

**Integration**: Runs automatically as part of the Degradation Monitor. No user configuration needed.

### 9.7 New: Correction Frequency Tracker (MVP → Week 3)

> **Added for MVP; adaptive behavior in Week 3** — Tracks user corrections to feed into the adaptive refresh interval.

```javascript
class CorrectionFrequencyTracker {
  constructor() {
    this.corrections = [];      // Timestamped correction events
    this.segmentCorrections = 0; // Corrections in current segment
    this.totalSegments = 0;
  }

  recordCorrection(type, turnNumber) {
    this.corrections.push({
      type,          // "rejected_step" | "edited_diff" | "reverted_change"
      turnNumber,
      timestamp: Date.now()
    });
    this.segmentCorrections++;
  }

  onSegmentEnd() {
    this.totalSegments++;
    const rate = this.segmentCorrections / (this.totalSegments || 1);
    
    // Week 3 adaptive logic:
    if (rate >= 0.3) {
      return { action: 'tighten_refresh', suggestedInterval: 3 };
    }
    if (rate <= 0.1 && this.totalSegments >= 5) {
      return { action: 'loosen_refresh', suggestedInterval: 5 };
    }
    this.segmentCorrections = 0;
    return { action: 'maintain' };
  }

  getCorrectionRate() {
    if (this.corrections.length === 0) return 0;
    return this.segmentCorrections / Math.max(this.totalSegments, 1);
  }
}
```

**MVP behavior**: Track corrections passively. The data feeds into the adaptive refresh system (Week 3 feature). No runtime adaptation in Week 1.

---

## 10. Layer 5: External Integration Layer (MVP)

### 10.1 Terminal / TUI — Primary MVP Interface

> **MVP primary interface** — The terminal user interface is the focus for MVP development.

The terminal/TUI interface provides:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Local Code Agent  [Session: auth-refactor]  [Step: approve?]       │
│  Model: Gemma 4 E4B  [Turns: 3/4]                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User: Add rate limiting to the auth module                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Agent: I'll add rate limiting to the auth module. Here's my     │ │
│  │ plan:                                                           │ │
│  │                                                                  │ │
│  │ 1. Create auth/rate_limit.py — configurable rate limiter        │ │
│  │ 2. Update auth/__init__.py — export new module                  │ │
│  │ 3. Add tests/test_rate_limit.py — verify limiter                │ │
│  │                                                                  │ │
│  │ Shall I proceed? [y/n/e]                                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ [Diff Preview] auth/rate_limit.py (new file)                    │ │
│  │ +import time                                                     │ │
│  │ +from collections import defaultdict                             │ │
│  │ +                                                               │ │
│  │ +class RateLimiter:                                             │ │
│  │ +    def __init__(self, max_requests=100, window_seconds=60):   │ │
│  │ +        ...                                                    │ │
│  │ [Approve: y] [Skip: s] [Reject: n] [Edit: e]                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ [Test Results — pytest tests/test_rate_limit.py -v]             │ │
│  │ ✓ test_basic_rate_limit PASSED                                  │ │
│  │ ✓ test_window_reset PASSED                                      │ │
│  │ ✓ test_concurrent_requests PASSED                               │ │
│  │ 3/3 tests passing                                               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Key MVP features**:
- Chat buffer with model responses streamed in real-time
- Permission mode selector (read-only / plan / step)
- Diff preview with unified diff format
- Approval shortcuts: y (yes), n (no), s (skip), e (edit)
- Test output shown inline after every write
- Turn counter displayed in status bar

**Streaming / Progressive Display (Weeks 2-3)**:
- **Week 1 (MVP baseline)**: Block output — wait for full model response before showing it
- **Week 2 (streaming)**: Real-time token streaming as model generates, progressive diff previews
- **Week 3 (polish)**: Streaming test output, live status indicators for tool execution

### 10.2 Post-MVP: Cloud Escalation Path

> **Post-MVP Enhancement** — Cloud escalation will be added after the core agent loop is proven. Skip during MVP implementation.

When a local model cannot solve a problem after multiple attempts (post-MVP):

```toml
[escalation]
# After N failed attempts on the local model, suggest or auto-escalate
max_local_attempts = 3
max_strong_local_attempts = 6

# Cloud provider configuration (optional, for hybrid mode)
[escalation.cloud]
provider = "openrouter"  # or "openai" or "anthropic"
model = "claude-sonnet-4"  # or "gpt-5.5" or similar
api_key_env = "OPENROUTER_API_KEY"
cost_warning = true  # warn user before sending to paid API

# Escalation strategy
strategy = "send_context"  # "send_context" | "send_snapshot" | "collaborative"
```

The escalation strategy:
1. **First 3 attempts**: Local model, any tier
2. **Attempts 4-6**: Strongest local model (if available) or retry with better context
3. **Attempt 7+**: Propose cloud escalation with cost estimate
4. **On cloud escalation**: Send only the problem context + error output + attempted fixes (not the full conversation)

### 10.3 Post-MVP: VS Code / Platform Integration

> **Post-MVP Enhancement** — VS Code extension will be implemented after the core agent loop is proven. Skip during MVP implementation.

```toml
[editor]
# Supported editor integrations
integrations = ["terminal"]  # MVP: terminal only | Post-MVP: "vscode" | "jetbrains" | "neovim"

# VS Code extension-specific
[editor.vscode]
highlight_changes = true  # highlight changed lines in diff view
inline_completions = false  # disable inline (replaced by interactive flow)
```

---

## 11. External Dependencies & Inheritance Strategy

### 11.1 What We Fork and What We Write From Scratch

```
OUR CODE (write from scratch):          ~3000 lines
  ├── Orchestrator (main loop)
  ├── Interactive permission subsystem (MVP: 3 modes)
  ├── Degradation monitor
  ├── Context engine (4-buffer architecture)
  ├── Session facts buffer
  ├── Format adherence monitor
  ├── Persona guard
  ├── Memory probes
  ├── Model degradation profiles
  ├── Auto-detection protocol (Post-MVP — MVP uses pre-computed profiles)
  └── Phase router
```
INHERITED FROM SMALLCODE (use as dependency):  ~5000 lines
  ├── Plan tracker + re-injection
  ├── Liquid tool parser
  ├── Two-stage tool router
  ├── Quality monitor
  ├── Early stop detector
  ├── Hybrid code search
  ├── Thinking budget control
  ├── Snapshot/rollback
  ├── Patch-first editing
  ├── Persistent shell sessions
  ├── Trace recorder
  └── Contract system

CAN REPLACE LATER (nice-to-have swaps):  
  ├── RTK integration (Rust binary, heavy dep)
  ├── Memory system (SQLite FTS5 — could use mem-only version)
  └── MarrowScript cognition compiler (compiled layer, very specific to SmallCode)
```

### 11.2 Integration Point: How We Hook Into SmallCode

We do NOT modify SmallCode. Instead, we build a **wrapper** that:

1. **Imports SmallCode's classes** (PlanTracker, QualityMonitor, EarlyStopDetector, hybridSearch, etc.) as npm packages or copied modules
2. **Wraps the executeTool function** with our permission gating
3. **Wraps the chatCompletion call** with our context engine (builds the four-buffer prompt)
4. **Intercepts the model response** with our degradation monitor (format check, memory check) before passing to tool exec
5. **Injects correction messages** (format reminder, persona reminder, context refresh) as user-role messages before the next model call

```javascript
// Archetype of the wrapper
class DegradationAwareAgent {
  constructor(modelName, profile) {
    this.modelProfile = profile;
    this.turnCounter = 0;
    this.permissionMode = 'read_only';
    this.currentPhase = 'understand';
    
    // Inherited from SmallCode
    this.planTracker = new PlanTracker();
    this.qualityMonitor = new QualityMonitor();
    this.earlyStop = new EarlyStopDetector();
    this.formatMonitor = new FormatMonitor(profile);
    this.personaGuard = new PersonaGuard(profile);
    
    // Our additions
    this.contextEngine = new ContextEngine(profile);
    this.permissionGate = new PermissionGate();
    this.degradationMonitor = new DegradationMonitor(profile);
    this.sessionFacts = new SessionFactsBuffer();
  }

  async processUserMessage(message) {
    // 1. Check if this is an approval response
    if (this.permissionGate.isAwaitingApproval) {
      return this.handleApprovalResponse(message);
    }

    // 2. Build context from four-buffer architecture
    const prompt = this.contextEngine.assemblePrompt({
      userMessage: message,
      sessionFacts: this.sessionFacts.getAll(),
      activePlan: this.planTracker.formatForPrompt(),
      history: this.getTruncatedHistory(),
      phase: this.currentPhase,
    });

    // 3. Apply degradation corrections (if any)
    const corrections = this.degradationMonitor.getPendingCorrections();
    if (corrections.length > 0) {
      prompt.messages.push(...corrections.map(c => ({ role: 'user', content: c })));
    }

    // 4. Call model
    const response = await this.callModel(prompt);

    // 5. Check for degradation
    const formatCheck = this.formatMonitor.inspect(response);
    if (formatCheck.injection) {
      this.degradationMonitor.scheduleCorrection(formatCheck.injection);
    }

    // 6. Extract tool calls or text
    const toolCalls = this.parseToolCalls(response);

    // 7. If tool calls and needs approval → gate
    if (toolCalls.length > 0 && this.needsApproval(toolCalls)) {
      return this.proposeAndWait(toolCalls);
    }

    // 8. Execute and validate
    const results = await this.executeTools(toolCalls);
    await this.validate(results);

    // 9. Update state
    this.turnCounter++;
    if (this.turnCounter >= this.modelProfile.degradation.context_refresh_interval) {
      await this.injectContextRefresh();
    }

    return results;
  }

  async injectContextRefresh() {
    this.turnCounter = 0;
    const refresh = this.contextEngine.buildRefreshBlock({
      sessionFacts: this.sessionFacts.getAll(),
      activePlan: this.planTracker.formatForPrompt(),
      standingInstructions: this.getStandingInstructions(),
      remainingSteps: this.planTracker.getRemainingSteps(),
    });
    await this.injectSystemMessage(refresh);
  }
}
```

### 11.3 Dependency Decision Tree

```
Q: Do we need SmallCode's Liquid Tool Parser?
  → YES: LFM2.5 uses non-standard format. Without it, the agent can't use LFM2.5.
  → Use as-is: smallcode/src/tools/liquid_tool_parser.js

Q: Do we need SmallCode's Two-Stage Router?
  → YES: Critical for 32K context models. Saves ~200 tokens per turn.
  → Use as-is: smallcode/src/tools/two_stage_router.js

Q: Do we need SmallCode's Plan Tracker?
  → YES: ACTIVE PLAN re-injection is our primary IAD countermeasure.
  → Use as-is: smallcode/src/session/plan_tracker.js

Q: Do we need SmallCode's Early Stop?
  → YES: Catches degeneracy patterns we can't predict from evaluation.
  → Use as-is: smallcode/src/governor/early_stop.js

Q: Do we need SmallCode's Quality Monitor?
  → YES: Empty responses, hallucinated tools, repeated calls.
  → Use as-is: smallcode/src/governor/quality_monitor.js

Q: Do we need SmallCode's Hybrid Search?
  → YES: No external deps, runs on CPU, BM25+token embedding.
  → Use as-is: smallcode/src/tools/hybrid_search.js

Q: Do we need SmallCode's Snapshot/Rollback?
  → YES: Critical safety net for local models that make bad edits.
  → Use as-is: pattern from smallcode/bin/executor.js

Q: Do we need SmallCode's Thinking Budget?
  → YES: Required for Qwen3, DeepSeek R1, and future reasoning models.
  → Use as-is: smallcode/src/model/thinking_budget.js

Q: Do we need SmallCode's Trace Recorder?
  → YES: Reproducibility and debugging. Also generates test replays.
  → Use as-is: smallcode/bin/trace_recorder.js

Q: Do we need SmallCode's Contract System?
  → NICE TO HAVE: Adds Definition-of-Done assertions. Good for enterprise.
  → Use as-is: smallcode/src/session/contract.js

Q: Do we need SmallCode's Adaptive Router?
  → PARTIALLY: Their failure-rate-based escalation is basic. Replace with
    our degradation-aware model selection (phase-based, not failure-based).
  → Adapt: smallcode/src/model/adaptive_router.js

Q: Do we need SmallCode's Governor (classifyTask, verifyCode, decompose)?
  → PARTIALLY: verifyCode is useful (compile + execute check). classifyTask
    is regex-based and simple. decompose strategies are good.
  → Use verifyCode and pickDecomposeStrategy from smallcode/bin/governor.js

Q: Do we need SmallCode's Executor (all tool implementations)?
  → YES: Well-tested implementations for read/write/patch/bash/search.
    Especially the patch tool with semantic merge fallback.
  → Use as-is: smallcode/bin/executor.js tool implementations
```

---

## 12. Implementation Roadmap

> **This is the 3-week MVP build plan.** All features described in this roadmap are for the MVP (single model, 3 permission modes, standing instructions, format monitor, episode persistence, test-after-every-write). Post-MVP enhancements are noted but excluded from the 3-week scope.

### Week 1: Core Agent Loop (MVP Foundation)

**Goal**: Running single-model conversational agent with approval gating, turn counting, context refresh, diff display, and test-after-every-write.

#### Deliverables

| Day | Focus | Tasks |
|-----|-------|-------|
| **Day 1-2** | Project scaffold + model integration | Set up project structure (Python/Node.js), import SmallCode dependencies (PlanTracker, QualityMonitor, EarlyStopDetector, tool executors), connect to Ollama/OpenAI-compatible API via model profile config |
| **Day 3-4** | Agent loop + approval gate | Implement 5-phase loop (Explore → Plan → Execute → Verify → Iterate) with turn counter hard-limited to 4 per segment. Implement permission gate: propose → y/s/n/e (approve/skip/reject/edit). Implement diff display (unified diff format). |
| **Day 5-6** | Test execution + episode persistence | Implement test execution tool: auto-detect pytest/jest/cargo_test, run after every write, show results inline. Implement episode persistence: save to ~/.local-code-agent/sessions/&lt;id&gt;/episodes/. |
| **Day 7** | Terminal UI + integration | Terminal UI with streaming model output, status bar (model name, turn counter 3/4, permission mode), and inline test results. End-to-end integration test. |

#### Week 1 Checklist

- [ ] Project structure with SmallCode dependency imports
- [ ] Single model integration (configurable via model profile TOML)
- [ ] Basic conversation loop: user asks → model proposes → user approves/edits/rejects
- [ ] Turn counter with hard limit of 4 turns per segment
- [ ] Hard context refresh after approval gate (re-inject standing instructions + session facts)
- [ ] Diff display with unified format + approve/skip/reject per step
- [ ] Episode persistence to disk (~/.local-code-agent/sessions/&lt;id&gt;/episodes/)
- [ ] Test execution after every write (detect framework, run, show results)
- [ ] Terminal UI with streaming output (block output mode — full response then display)
- [ ] **Acceptance test**: Agent can read a file, propose a change, get approval, write the change, run tests, and save an episode record — in one session.

### Week 2: Degradation Compensation

**Goal**: Add the standing instructions buffer (counters universal recency bias), format monitor (catches tool call drift early), episode summary compression, and cross-terminal resume.

#### Deliverables

| Day | Focus | Tasks |
|-----|-------|-------|
| **Day 8-9** | Standing instructions buffer | Implement per-turn injection of standing instructions as a separate system block. Design instruction template: basic rules (no hallucination), format rules (JSON output spec), safety rules (no destructive ops without approval). Inject before every model call. |
| **Day 10** | Format monitor | Implement format monitor: check first tool call output for valid JSON, required fields present, brace balance. Trigger soft refresh (re-inject format instructions) on early warning signs. Cross-reference with Section 7.7. |
| **Day 11-12** | Episode summary compression | Implement compressEpisode() from Section 7.6: extract decisions made, files changed, key facts from completed episode. Compress to ~500-2000 tokens. MVP: inject ALL compressed episode summaries into context (well within 32K window). |
| **Day 13** | Cross-terminal resume | List saved sessions on launch. Reload last session on restart. Restore session facts buffer from episode summaries. Show user: "Session restored — you were working on [project/task]." |

#### Week 2 Checklist

- [ ] Standing instructions buffer with per-turn injection (counters universal recency bias)
- [ ] Format monitor: checks first tool call output for degradation warning signs
- [ ] Episode summary compression (compressEpisode function, ~500-2000 tokens per episode)
- [ ] Cross-terminal resume: list saved sessions, reload on restart
- [ ] Memory retrieval probes: inject 1 fact-based check per segment (e.g., "What file are we modifying?")
- [ ] **Acceptance test**: Start a task, complete it, close terminal, reopen — session resumes correctly with accurate memory of prior work.

### Week 3: Adaptive & Iterative

**Goal**: Adaptive refresh interval (not hardcoded 4 turns), user correction frequency tracking, test-fail → auto-iteration loop, streaming UX improvements.

#### Deliverables

| Day | Focus | Tasks |
|-----|-------|-------|
| **Day 15-16** | Adaptive refresh | Implement telemetry collection: track format degradation, memory probe accuracy, user correction rate per segment. Adjust refresh interval dynamically: shrink (more frequent) when degradation is detected early, grow (less frequent) when model is stable for 5+ segments. |
| **Day 17-18** | Correction frequency tracking | Track user corrections per segment. Adaptive logic: if correction rate ≥0.3 (30% of steps corrected), tighten refresh interval by 1. If ≤0.1 for 5+ segments, loosen (allow more turns). Log correction patterns for model profile refinement. |
| **Day 19** | Test-fail → auto-iteration | When tests fail after a write, automatically enter debug-iterate cycle: re-enter Execute phase (same model, same 4-turn limit), show test error output as context. Max 3 attempts before asking user for guidance. Auto-rollback on 3rd failure to last known-good state. |
| **Day 20** | Streaming UX + polish | Progressive disclosure: show model response as it generates (token streaming). Progressive diff previews. Live status indicators for tool execution. Performance benchmarks against no-adaptation baseline. |

#### Week 3 Checklist

- [ ] Adaptive refresh interval based on telemetry (not hardcoded 4 turns)
- [ ] User correction frequency tracking: tighten at ≥0.3 rate, loosen at ≤0.1 rate for 5+ segments
- [ ] Test-fail auto-iteration loop: re-enter Execute on test failure, max 3 attempts, auto-rollback
- [ ] Progressive disclosure / streaming UX improvements (real-time token streaming)
- [ ] Quality monitor integration (from SmallCode — empty response detection, hallucinated tools)
- [ ] Performance benchmarks against baseline (no adaptation)
- [ ] **Acceptance test**: Agent adapts refresh interval based on user correction rate and format degradation telemetry. When tests fail, agent automatically re-attempts with error context.

### Post-MVP Roadmap (After Week 3)

> **Post-MVP** — The following features are deferred until the core agent loop is proven in production.

| Feature | Estimated Effort | Prerequisites |
|---------|:----------------:|:--------------|
| Two-model routing (fast + strong) | 2 weeks | Week 1 agent loop stabilized |
| 5-mode permissions (Batch + Auto) | 1 week | Permission gate + week 2 standing instructions |
| Auto-detection protocol | 2 weeks | Multiple model profiles + format/memory detection |
| Selective reference episode loading | 1 week | Episode summary compression proven in production |
| Cloud escalation path | 1 week | Two-model routing + OpenRouter integration |
| VS Code extension | 3 weeks | Terminal UI proven + API surface stable |

---

## 13. Evaluation Criteria & Success Metrics (MVP)

> **MVP focus** — The evaluation criteria below are scoped to what can be measured with the single-model, 3-permission-mode, 3-week-build system. Post-MVP metrics are deferred to a separate subsection.

### 13.1 Functional Criteria (MVP)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Multi-file refactoring | ≥80% user approval rate | Track approval/rejection ratio |
| Bug fixing accuracy | ≥70% fix pass rate | Run test suite after fix, count passes |
| Feature implementation | ≥70% user satisfaction | User survey after each feature session |
| Task completion | ≥85% of multi-turn tasks reach Verify phase | Track phase transitions through 5-phase loop |
| Rollback frequency | ≤10% of write operations | Count rollback events / total writes |
| Turn-to-failure reduction | ≥4x vs non-adaptive baseline | Compare degradation metrics with vs without refresh |

### 13.2 Degradation Metrics (MVP)

| Metric | Target | Compared To |
|--------|--------|-------------|
| Memory retrieval score | ≥0.8 at all depths | Baseline: 0.0 at depth 8 (E4B raw) |
| Tool call format score | ≥0.9 at all turns | Baseline: 0.10 at depth 1 (LFM2.5 raw) |
| Instruction adherence score | ≥0.9 at all turns | Baseline: 0.333 at depth 10 (LFM2.5 raw) |
| Persona consistency score | ≥1.0 at all turns | Baseline: 0.333 at depth 10 (E2B raw) |
| Context refresh recovery | ≥90% of refreshes restore score to 1.0 | Measure score before/after each refresh |
| User corrections per session | ≤2 per 10-turn segment | Count user rejections / modifications |

### 13.3 Performance Metrics (MVP)

| Metric | Target | Context |
|--------|--------|---------|
| Latency per tool call | ≤30s (CPU) / ≤5s (GPU) | Reasonable for local usage |
| Total session time | ≤3min for refactors (≤5 files) | Compare to Claude Code's ~1min |
| Context window usage | ≤80% of model's available context | Don't exceed model's working limit |
| Token waste (format guard) | ≤5% overhead | Format guard should be lightweight |

### 13.4 Comparison to Claude Code / Cursor (MVP)

| Dimension | Target vs Claude Code | Target vs Cursor |
|-----------|:---------------------:|:----------------:|
| Code quality | ≥80% of cloud quality | ≥80% of cloud quality |
| Latency | ≤2x cloud latency | ≤3x cloud latency |
| Task completion rate | ≥70% of cloud rate | ≥70% of cloud rate |
| User satisfaction | ≥4/5 | ≥4/5 |
| Privacy | ✅ Fully local | ✅ Fully local |
| Cost | ✅ $0 | ✅ $0 |
| Offline | ✅ Yes | ✅ Yes |

### 13.5 Episodic Session Metrics (MVP)

These metrics evaluate the system's ability to handle unbounded, interleaved sessions:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Episode boundary detection accuracy | ≥90% | Session replay: compare detected boundaries against human-judged boundaries |
| Cross-episode context injection overhead | ≤2500 tokens/summary | Average token count of injected cross-episode summary |
| Cross-episode confusion rate | ≤5% of turns | Turns where model references wrong episode facts |
| Session resumption accuracy | ≥95% | After restart, does the model correctly recall session state? |
| Context window utilization across episodes | ≤60% of model's max | The episodic model should prevent linear context growth |
| Episode count without degradation | Unlimited (bounded by episodic model) | Run 30+ episodes, check per-episode degradation doesn't trend down |

### 13.6 Post-MVP Metrics

> **Post-MVP** — The following metrics depend on features not yet built (selective reference loading, cloud escalation, two-model routing). They are tracked here for future reference but not targeted during the 3-week MVP build.

| Metric | Target | Prerequisite Feature |
|--------|--------|:--------------------|
| Cross-terminal session restore latency | ≤2s | Episode summary compression + cross-terminal resume (Week 2) |
| Mid-task interleaving handling | ≥80% user satisfaction | Phase router + episode boundary detection (Week 1) |
| User correction carry-over | ≥85% | Session facts buffer with per-episode fact extraction (Week 2) |
| Ephemeral vs persistent fact accuracy | ≥90% | Selective reference episode loading (Post-MVP) |
| Selective reference accuracy (recall@5) | ≥90% | Selective reference episode loading (Post-MVP) |
| Cloud escalation success rate | ≥80% | Cloud escalation path (Post-MVP) |
| Two-model routing latency improvement | ≥20% vs single model | Two-model routing (Post-MVP) |

---

## 14. Appendix A: Degradation Data Cheat Sheet

### Turn-Based Dimensions (Turn counts before score < 0.8)

| Dimension | LFM2.5-8B-A1B | Gemma 4 E2B | Gemma 4 E4B | Risk Priority |
|-----------|:-------------:|:-----------:|:-----------:|:-------------:|
| Memory Retrieval | **8** turns | **8** turns | **5** turns | 🔴 HIGHEST |
| Instruction Adherence | **10** turns | **20** turns | **30** turns | 🟡 MEDIUM |
| Tool Call Drift | **1** turn | **20** turns | None | 🟡 MEDIUM (LFM2.5 only) |
| Persona Consistency | None | **10** turns | None | 🟢 LOW (E2B only) |
| Recency Bias | **1** turn | **1** turn | **1** turn | 🔴 HIGHEST (universal) |

### Token-Based Dimensions (Tokens before score < 0.8)

| Dimension | LFM2.5-8B-A1B | Gemma 4 E2B | Gemma 4 E4B |
|-----------|:-------------:|:-----------:|:-----------:|
| Hallucination Onset | **8000** tokens | **8000** tokens | **8000** tokens |

### Key Safe Limits (with safety margin of -2)

| Model | Safe Consecutive Turns | Context Refresh Interval |
|-------|:----------------------:|:------------------------:|
| LFM2.5-8B-A1B | 6 | 5 |
| Gemma 4 E2B | 6 | 5 |
| Gemma 4 E4B | 3 (memory) / 8 (rest) | 5 |
| Qwen3-8B (est.) | 8 | 6 |

---

## 15. Appendix B: Model Profile Format

```toml
# Profile: gemma4-e4b
name = "gemma4-e4b"
context_length = 262144
max_output_tokens = 8192
supports_tool_calling = true
tool_format = "auto"  # "auto" | "json" | "hermes" | "liquid" | "xml"
supports_system_message = true
template = "gemma"  # chat template for this model
stop_sequences = ["<end_of_turn>"]

# Model tier (fast / strong / cloud)
tier = "fast"

# Capability ratings (0-1) — used for phase routing
[capabilities]
code_reading = 0.9        # understanding unfamiliar code
planning = 0.7            # proposing coherent plans
execution = 0.85          # writing accurate code changes
debugging = 0.6           # diagnosing and fixing bugs

# Degradation profile — derived from evaluation data
[degradation]
memory_horizon = 5          # turns before fact retrieval drops below 0.8
iad_horizon = 30            # turns before instruction adherence drops below 0.8
format_horizon = null       # null = no observed format degradation
persona_horizon = null      # null = persona never drops
hallucination_horizon_tokens = 8000  # tokens before hallucination risk increases
recency_horizon = 1         # recency bias triggers on next turn

# Context management
context_refresh_interval = 5   # inject refresh every this many turns
session_fact_injection = "every_refresh"  # | "every_turn" | "never"

# Degradation monitoring
requires_format_monitor = false  # E4B maintains format indefinitely
requires_persona_guard = false   # E4B maintains persona indefinitely
format_injection_threshold = 0.5
persona_check_interval = 10

# Thinking budget (for reasoning models)
[thinking]
budget_tokens = 2000
hard_cap_chars = 8000
supported_providers = ["llamacpp", "lmstudio"]  # providers that support thinking params
```

---

## 16. Appendix C: Session Fact Buffer Schema

```typescript
interface SessionFact {
  id: string;                    // "fact-1", "fact-2", ...
  category: FactCategory;        // Which type of fact
  timestamp: number;             // Unix ms when fact was recorded
  description: string;           // Human-readable fact text
  context: string;               // Why this fact exists (for debugging)
  confidence: 'definite' | 'inferred' | 'tentative';
  requiresApproval: boolean;     // Was this explicitly confirmed by user?
  sourceTurn: number;            // Which turn this was recorded at
}

type FactCategory = 
  | 'file_created'       // "Created auth/jwt.js"
  | 'file_modified'      // "Modified auth/middleware.js"
  | 'file_deleted'       // "Deleted old-session.js"
  | 'decision'           // "Decided to use JWT not OAuth"
  | 'user_preference'    // "User prefers snake_case"
  | 'task'               // "Task: refactor auth module"
  | 'dependency'         // "Added jsonwebtoken dependency"
  | 'test_result'        // "Tests passing: 5/5"
  | 'error'              // "Error: test_auth.js line 42 fails"
  | 'constraint';        // "Must keep backward compatibility"

interface SessionFactsBuffer {
  facts: SessionFact[];
  maxFacts: number;              // Max facts before summarization (default 20)
  
  add(fact: Omit<SessionFact, 'id' | 'timestamp'>): void;
  getByCategory(category: FactCategory): SessionFact[];
  getRecent(count: number): SessionFact[];
  summarize(maxLength: number): string;  // Compress to N chars for injection
  
  // Automatic extraction from tool results
  extractFromToolCall(toolName: string, args: any, result: any): void;
  extractFromMessage(message: string): void;
}
```

### Example Buffer State After 10 Turns

```
📋 Session Facts (7 active, 3 archived):

[task] Refactor auth module from sessions to JWT (turn 1, definite)
[file_created] auth/jwt.js with JWT utility functions (turn 3, definite)
[file_modified] auth/middleware.js to verify JWT tokens (turn 5, definite)
[decision] Error handling moved to middleware.js (turn 5, definite, approved)
[file_modified] routes/auth.js uses new middleware (turn 7, definite)
[dependency] Added jsonwebtoken to package.json (turn 8, definite)
[test_result] 3/3 tests passing (turn 9, definite)

✅ Archiving: [file_modified] auth/test.js (turn 6) → summarized into fact count
✅ Archiving: [constraint] Must support refresh tokens (turn 2) → absorbed into task
```

---

## Architectural Principles — Summary

1. **Degradation is not a bug — it's a property of the model.** Design for it, don't fight it.
2. **Short agentic segments with context refreshes** beat long degraded sessions.
3. **The user is the coherence anchor.** Every approval gate resets the degradation clock.
4. **Episodes, not sessions.** Tasks are interleaved and unbounded. Segment the session into episodes with compressed summaries and fresh context per episode.
5. **Model profiles turn evaluation data into architecture parameters.**
6. **SmallCode solved the hard parsing/search/planning problems.** Build on top, not from scratch.
7. **Session facts are the model's external memory.** The model doesn't need to remember — we tell it.
8. **Standing instructions beat system prompts.** Re-inject critical rules, don't trust they persist.
9. **Format monitors catch what quality monitors miss.** Tool call format quality degrades before tool existence.
10. **Phase-based routing uses the right model for the right cognitive load.**
11. **Measure everything.** Every session builds the degradation profile database.