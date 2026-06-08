# Restructure ARCHITECTURE_LOCAL_CODE_AGENT.md — MVP-First Focus

## Goal
Restructure the architecture document to focus on the 3-week MVP build plan, mark advanced features (two-model routing, permission modes beyond 3, auto-detection protocol, selective reference loading) as "post-MVP enhancements" with clear deferral notes.

## Sections to Modify

### 1. Executive Summary (Section 1)
- Add "Build Stages" callout box: MVP (3 weeks) vs. Post-MVP enhancements
- Keep core thesis, keep key numbers table

### 2. Architecture Overview Diagram (Section 4)
- Replace diagram: single-model, single-approval-mode focus
- Add footnote: multi-model routing layer is post-MVP

### 3. Layer 0 — Model Profiles (Section 5)
- Keep 5.1 Profile Schema (it's the foundation)
- Keep 5.2 Evaluated Profiles (critical data)
- Defer 5.3 Estimated Profiles to post-MVP
- Mark 5.4 Auto-Detection Protocol as post-MVP — replace with "MVP ships with pre-computed profiles from evaluation data"

### 4. Layer 1 — Permission Subsystem (Section 6)
- Rewrite: MVP has 3 modes (Read-only, Plan, Step) — not 5
- The 5-mode system with --auto-approve-* flags becomes a post-MVP subsection
- Keep the Phase flow (Understand → Plan → Execute → Verify) — that's the core UX

### 5. Layer 2 — Context Engine (Section 7)
- Keep 4-buffer design (Permanent, State, History, Session Facts)
- Standing instructions buffer (per-turn injection for recency) — this IS MVP
- Episode session model stays (it's core to handling multi-hour sessions)
- Mark "Selective reference episode loading" as post-MVP — MVP uses ALL episode summaries
- Format monitor goes here (check first tool call output for JSON format)

### 6. Layer 3 — Agent Loop (Section 8)
- Keep the phase loop (Explore → Plan → Execute → Verify → Iterate)
- Add: turn counter at 4, hard refresh, approval gate per step
- Add: test execution as first-class verification step (runs tests after every write)
- Add: adaptive refresh interval based on telemetry (Week 3 feature)

### 7. Layer 4 — Tools (Section 9)
- Keep SmallCode inheritance table
- Add "Test Execution" as new inherited tool (bash + test detection)
- Add: Format monitor as first tool output check
- Add: Correction frequency tracking

### 8. Layer 5 — Integration (Section 10)
- Keep terminal/TUI focus as primary MVP interface
- Mark VS Code extension as post-MVP

### 9. Implementation Roadmap (ENTIRELY REWRITE — Section 12)
Replace the current roadmap with the 3-week plan:

**Week 1**: Core agent loop (one model, turn counter at 4, hard refresh, diff/approve per step, episode persistence to disk, test execution after every write, terminal UI with streaming)

**Week 2**: Standing instructions buffer (per-turn injection for recency), format monitor (checks first tool call output), episode summary compression, cross-terminal resume

**Week 3**: Adaptive refresh interval based on telemetry, user correction frequency tracking, test-fail → auto-iteration loop, progressive disclosure / streaming UX improvements

### 10. Evaluation Criteria (Section 13)
- Simplify to MVP-relevant metrics
- Episode metrics stay if they're MVP-implementable (boundary detection, session resumption)
- Move speculative metrics (cross-terminal restore, selective reference) to post-MVP

### 11. Appendices
- Keep Appendix A (degradation cheat sheet) — essential
- Keep Appendix B (model profile format) — essential
- Keep Appendix C (session fact buffer schema) — essential

## Approach
For each section, use search-and-replace edits to rewrite content. The changes are semantic restructuring, not line-for-line changes — so each edit replaces a block.