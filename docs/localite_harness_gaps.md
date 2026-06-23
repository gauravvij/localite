# Localite Harness Gaps Analysis

**Date:** 2025-06-19  
**Model Tested:** DeepSeek V4 Flash via OpenRouter  
**Benchmark:** SWE-bench Lite dev (18 non-sqlfluff instances)  
**Localite Resolution Rate:** 50.0% (9/18)  
**Published DeepSeek V4 Flash (SWE-agent) Resolution:** ~79% on SWE-bench Verified  

> **TL;DR:** Localite was designed for small language models (SLMs) — the hand-holding, aggressive truncation, phase system, and format guards all make sense for a 1-8B model. For a 284B model with native tool calling and 1M-token context, these same features become liabilities. The harness is leaving ~30 percentage points on the table.

---

## Issue #1 (Highest Impact): No Native Tool Calling — Free-Form Text JSON Parsing

### What It Does
The harness does **not** use the model's structured tool-use API (`tools` parameter in the OpenAI-compatible chat format). Instead:

1. The system prompt tells the model to output `{"tool": "read_file", "arguments": {"path": "/foo.py"}}` as **raw text**
2. The model generates a free-form text response containing this JSON
3. `_parse_tool_call()` tries to extract the first balanced `{}` object from the text
4. It then tries **7 different parsing strategies** to handle variations (flat JSON, LFM2.5 native, Qwen `tools[]`, Qwen `tool_calls[]`, key-as-name, naked args inferred from signature, thought-stripped)

### Location
`/home/azureuser/local_llm_eval/localite/loop/agent_loop.py`, lines ~1390-1560 (`_parse_tool_call`, `_extract_first_json_object`, `_infer_tool_from_signature`, etc.)

### Why It's a Problem
- **Stall cycles**: model produces a format the parser doesn't recognize → counted as "no tool call" → stall counter increments → context refresh triggered → even more noise
- **Token waste**: the system prompt has to teach the model a custom JSON format with examples (~50 lines of prompt)
- **Format 6 (naked args) inference**: guesses the tool from argument keys like `{"path": "/foo.py", "search_text": "x", "replace_text": "y"}` → infers `edit_file`, but if `path` could match `read_file` or `list_files`, the disambiguation heuristic defaults to `read_file`
- **Every LLM-native format** (OpenAI `tool_calls`, Anthropic tool use) guarantees structured, reliable tool invocation. The text-parsing approach is strictly worse.

### Fix
Replace `_parse_tool_call()` with OpenAI-compatible `tools` parameter. DeepSeek V4 Flash, Claude, GPT-4, Gemini, and most modern models all support this natively. The model returns `choices[0].message.tool_calls` as structured JSON — no parsing needed.

---

## Issue #2: read_file Truncation at 6,000 Characters (~1,500 Tokens)

### What It Does
```python
if tool_name_called == "read_file" and len(tool_content) > 6000:
    tool_content = tool_content[:6000] + "\n\n[File truncated..."
```

When the model reads a source file, it only sees the **first 6,000 characters** (~1,500 tokens).

### Location
`agent_loop.py`, lines ~620-622

### Why It's a Problem
- If the function that needs fixing is 200 lines deep in a 500-line file, the model **never sees it**
- The model reads a file → sees only the header → can't find the bug → reads a different file → context grows → eviction kicks in → has to re-read → waste cycle
- The model has a **1M-token** context window — this truncation is pointless for capable models
- For SLMs with small context windows (4K-32K tokens), this makes sense. For DeepSeek/Claude/GPT, it's actively harmful.

### Fix
Remove the hardcoded 6000-char truncation, or increase it to at least 50K chars. Let the model's own context window manage this.

---

## Issue #3: Character-Based Context Budget (Effective ~20K Tokens)

### What It Does
```python
char_budget = int(max_ctx * 0.8)  # ~64K chars = ~16K tokens
while len(trimmed_turns) > 1:
    total_chars = sum(len(m.get("content", "")) for m in trimmed_turns)
    if total_chars <= char_budget:
        break
    # Evict oldest non-system message
```

`max_context_chars` is configured at **80,000 characters** in the DeepSeek profile, enforced at ~80% usage (64K chars).

### Location
`agent_loop.py`, lines ~1195-1235 (`_refresh_context`)

### Why It's a Problem
- Code averages **~4 chars per token**, so the effective context window is **~16-20K tokens**
- The model has a **1M-token** native context — the harness is using only ~2% of it
- The model can't see the repository structure it explored 10 turns ago
- It forgets what files it already read → re-reads the same files → burns turns → more context growth → more eviction → **death spiral**
- The eviction algorithm evicts **tool results first** (the most useful content — actual file contents), then **assistant messages** (the model's own reasoning), then **any non-pinned message**

### Fix
- Increase `max_context_chars` to at least **500,000** for models with large context windows
- Or remove the budget entirely and let the model's native context window handle it
- If budget is kept, use **token-based** counting (via `tiktoken` or model-specific tokenizer) instead of character counting

---

## Issue #4: Task Objective Compression at 15K Chars

### What It Does
```python
total_ch = sum(len(m.get("content", "")) for m in self.conversation_history[1:])
if total_ch > 15000:
    original = self.conversation_history[0]["content"]
    compressed = self._compress_objective(original, max_chars=300)
```

After ~15K characters of conversation (≈6-8 tool call turns), the **original task objective gets compressed to 300 characters**.

### Location
`agent_loop.py`, lines ~420-430 (`_execute_phase`)

### Why It's a Problem
- For SWE-bench tasks with detailed bug descriptions, this is devastating
- The model loses context of *what it was asked to do* — exact error message, expected behavior, reproduction steps
- It's the most important piece of context in the entire conversation, yet it's the first thing to get compressed
- After compression, the model can only work from memory of what it read earlier

### Fix
- Never compress the task objective — it's the most important message
- If space is needed, compress **old tool results** first (the model already processed them)
- Or don't compress anything and rely on the model's large context window

---

## Issue #5: Phase System Overhead (Wastes ~30% of Token Budget)

### What It Does
The 5-phase cycle (EXPLORE → PLAN → EXECUTE → VERIFY → ITERATE → COMPLETE) injects **~500-1000 characters of phase guidance** into every context build:

```python
if phase == "EXECUTE":
    guidance = (
        "You are in the EXECUTE phase — emit a JSON tool call NOW using this EXACT format: "
        "{\"tool\": \"tool_name\", \"arguments\": {...}}. "
        ...
```

Each phase transition injects:
- `[CURRENT PHASE: X]` block with full guidance (~500 chars)
- `[ACTIVE TASK]` block
- `[STANDING INSTRUCTIONS]` block
- `Session Facts` block
- `Active Plan` block

Total overhead: **~2-3K chars per turn** that doesn't help solve the actual problem.

### Location
`agent_loop.py`, lines ~1080-1140 and throughout ~200 lines of phase guidance

### Why It's a Problem
- DeepSeek V4 Flash was trained on examples with native tool calling — being told "Use the EXACT format" repeatedly is **actively counterproductive**
- It's like asking a fluent English speaker to read a grammar textbook every sentence they write
- The phase system was designed for SLMs that need structured scaffolding. A 284B model doesn't need to be told "you are in the EXPLORE phase, please explore the codebase"
- **Simple flat loop wins**: mini-swe-agent uses a single loop (model → bash → observe → repeat) and scores significantly higher

### Fix
- Skip all phase guidance for capable models (profile-level flag like `advanced_model: true`)
- Or remove the phase system entirely and use a flat loop with native tool calling

---

## Issue #6: Harness Guidance Distraction (6+ Intervention Types)

### What It Does
The harness injects up to **6 different guidance interventions** into the conversation:

| Intervention | Trigger | Content |
|---|---|---|
| `[GUIDANCE]` | After `list_files` | ctags, keyword scoring, strategy A/B/C/D analysis |
| `[REQUIRED]` | Stall in EXECUTE | "You are in EXECUTE phase — emit a tool call" |
| `[NUDGE]` | Post `read_file` with no edits | "Stop reading, start editing" |
| `[POST-EDIT VERIFY]` | After `edit_file`/`write_file` | "Now run test_executor" |
| `[STANDING INSTRUCTIONS]` | Every context build | Recency protection rules |
| `[ACTIVE PLAN]` | Re-injected on refresh | Current plan text |

### Location
`agent_loop.py`, various locations: lines ~646-658 (NUDGE), lines ~660-670 (POST-EDIT), lines ~1080-1140 (guidance), plus context builder

### Why It's a Problem
- Each is a plausible feature in isolation, but **together they consume ~15-20% of usable context** with meta-instructions
- The model's attention budget is finite — every guidance message pushes actual code context out of the window
- The NUDGE message ("you now have the content of X, call edit_file now") is actively counterproductive — the model may need to read multiple files to understand the full picture before editing
- The POST-EDIT VERIFY message tells the model to run tests — but DeepSeek V4 Flash already knows to run tests after editing (it's in the base training)

### Fix
- Remove all intervention messages for capable models
- Trust the model to plan its own workflow
- If intervention is needed, use it sparingly and only for SLM profiles

---

## Issue #7: The Compound Failure Cascade

These issues don't exist in isolation — they **compound** to create a characteristic failure pattern:

```
Turn 1:  Model reads file A (sees 6000 chars, misses the bug location 200 lines deep)
Turn 2:  Model reads file B (sees 6000 chars, still can't find the bug)
Turn 3:  Task objective compressed to 300 chars — forgets exact bug description
Turn 4:  Model reads more files → context grows
Turn 5:  Context eviction removes early tool results → forgets file A content
Turn 6:  Model re-reads file A (wasting a turn)
...
Turn 10: Stall in EXECUTE → [REQUIRED] guidance injection → more noise
Turn 15: Context refresh removes more context → model loses state
Turn 20: Gives up or produces wrong patch based on incomplete information
```

### Evidence from the Data

Looking at the DeepSeek V4 Flash Localite run:
- **Avg turns per instance**: 24.1
- **Failed instances avg turns**: 24.2 (same as resolved!)
- This means failed instances aren't timing out — they're producing wrong answers with normal turn counts
- The model has enough turns to solve the problem... if it could see the relevant code

Compare to mini-swe-agent on the same model (marshmallow-1359 smoke test):
- **mini-swe-agent**: 37 turns, 231s, sim=0.777
- **Localite**: 39 turns, 193s, sim=0.851
- Both resolved (PARTIAL), similar effort → the gap isn't in per-instance efficiency, it's in *how many instances the agent can even attempt correctly*

---

## Symptom: Low Patch Similarity on Failed Instances

The 9 failed instances all have **very low** patch similarity (mean ~0.21). This isn't random noise — it means the model makes edits that are structurally different from the reference patch, often in completely wrong files or with wrong approach.

| Instance | Sim | Likely Root Cause |
|---|---|---|
| astroid-1139 | 0.170 | Couldn't see the relevant AST visitor code (truncated at 6000 chars) |
| pvlib-1707 | 0.189 | Couldn't see the deep function needing modification |
| astroid-1978 | 0.181 | Similar — file too large, truncated |
| pvlib-1606 | 0.192 | Same pattern |
| astroid-1268 | 0.204 | Same pattern |

These aren't "model isn't smart enough" failures — they're "model can't see the relevant code" failures.

---

## Summary: What to Fix, In Priority Order

| Priority | Issue | Fix | Expected Impact |
|---|---|---|---|
| P0 | No native tool calling | Switch to `tools` parameter | Eliminates stall cycles, saves tokens, perfect parsing |
| P0 | read_file truncation (6K chars) | Remove or raise to 50K+ | Model can see full files, finds the right code |
| P1 | Char-based context budget | Raise to 500K+ chars or remove | Model retains history, avoids re-read death spiral |
| P1 | Task compression at 15K | Never compress task objective | Model remembers exact bug description |
| P2 | Phase system overhead | Skip for capable models | More context for actual code |
| P2 | Guidance interventions (x6) | Remove for capable models | Less noise, more focus |
| P3 | Re-evaluate SLM-vs-capable split | Profile-level flags | One harness serves both use cases |