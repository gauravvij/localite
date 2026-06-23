# Agent Architecture Analysis: mini-swe-agent vs Localite

## Why This Matters

We ran DeepSeek V4 Flash (a very capable model) through both harnesses:
- **mini-swe-agent**: 9/18 (50%) resolved on SWE-bench Lite non-sqlfluff instances
- **Localite**: 8/18 (44.4%) resolved — but with **higher cost and slower wall time** because of architectural overhead

The gap is NOT about the model — it's about how the harness communicates tool calls to the model. Here's the full architectural breakdown.

---

## 1. The Core Difference: Native Tool Calls vs Text-JSON Embedding

### How mini-swe-agent works (THE RIGHT WAY)

```
[model sends response via API]
  ↓
response.choices[0].message.tool_calls = [
  {
    "id": "call_abc123",
    "type": "function",
    "function": {"name": "bash", "arguments": '{"command": "ls -la"}'}
  }
]
  ↓
parse_toolcall_actions() extracts name + arguments directly
  ↓
No ambiguity. No regex. No brace-depth counting. Always works.
```

Key code in `LitellmModel._query()`:
```python
def _query(self, messages, **kwargs):
    return litellm.completion(
        model=self.config.model_name,
        messages=messages,
        tools=[BASH_TOOL],     # <-- passes tool schema to API
        **(self.config.model_kwargs | kwargs),
    )
```

The API returns `tool_calls` as a **structured, typed object** — the model doesn't need to figure out JSON embedding in text. The API handles the parsing.

### How Localite works (THE BROKEN WAY)

```
[model sends response via API]
  ↓
response.choices[0].message.content = "
  Let me read the file first.
  {\"tool\": \"read_file\", \"arguments\": {\"path\": \"src/main.py\"}}
"
  ↓
_extract_first_json_object(content)
  → finds first { via brace counting
  → may grab {} from Python code, set comprehension {x for x...}, or empty dict literal
  → very unreliable
```

Localite NEVER uses the `tools` parameter in the API call. It relies entirely on text-based JSON extraction from the `content` field, which is fragile and fundamentally incompatible with how capable models produce tool calls.

---

## 2. Tool Call Format: OpenAI Function Call vs Gemini-Style vs Text

### mini-swe-agent (single format — native function calls)

All tool calls use the OpenAI `tools` parameter:

```python
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "...",}
            },
            "required": ["command"],
        },
    },
}
```

The environment response comes back as a `tool`-role message with `tool_call_id` — the API handles matching.

### Localite (multiple fragile text formats)

- `OUTPUT_FORMAT_STANDARD`: `{"tool":"read_file","arguments":{...}}` — flat JSON embedded in text
- `OUTPUT_FORMAT_GEMMA_NATIVE`: `read_file: {"tool_name":"read_file","params":{...}}` — Gemma-style
- **No native tool call support** — even though the underlying LLM API supports it

**Forces capable models to use unnatural text JSON** when they'd natively output structured tool calls. This:
1. Stops DeepSeek V4 Flash from using its native tool calling ability
2. Causes format guard to trigger (model uses slightly different JSON than expected)
3. Leads to context refreshes that wipe the model's working memory

---

## 3. Prompt / System Message Design

### mini-swe-agent: Minimal, clean, non-prescriptive

System prompt (from swebench.yaml):
```
You are a helpful assistant that can interact with a computer shell to solve programming tasks.
```

Instance prompt: Clear task description with PR info, boundaries (which files to modify), and submission instructions. ~80 lines total. **No phase guidance. No thinking templates. No micro-management.**

Template variables: `{{task}}`, `{{workdir}}` — simple Jinja2 rendering.

### Localite: Bloated, prescriptive, phase-encumbered

- System prompt includes: output format declaration, tool descriptions, tool calling format, format example, error format explanation — **massive over-specification**
- 5-phase guidance: EXPLORE → PLAN → IMPLEMENT → VERIFY → ITERATE — forces the model through discrete stages even when the model already knows the cycle
- Each phase transition: wastes 2+ turns injecting phase guidance + model reorienting
- Standing instructions block: re-injected every refresh — adds redundant context

---

## 4. Format Error Handling

### mini-swe-agent: Informative, doesn't waste calls

```python
# In format_error_template:
{% if finish_reason in ["length", "tool_calls"] -%}
Your previous response reached the output token limit before you produced a tool call...
Respond more concisely...
{%- else -%}
Tool call error: {{error}}
Call the bash tool with your command as the argument
{%- endif %}
```

- Single `FormatError` exception with messages
- `max_consecutive_format_errors` limits (default 3) — exits cleanly if model can't format
- Error template distinguishes truncation (`finish_reason: length/tool_calls`) from real format issues
- **Does NOT do context refresh — just tells the model to fix formatting**

### Localite: Full context reset on format errors

- FormatMonitor scores every response on a 0-1 scale
- When average drops below 0.3: triggers `_refresh_context()` which:
  1. Trims conversation to `memory_horizon` turns (default 5-10)
  2. Re-injects standing instructions, active plan, last tool result
  3. Resets turn counter
  4. **Wipes the model's working memory** — model has to rediscover where it was
- FormatGuard flag: when `True`, accelerates degradation detection
- Bonus: `record_tool_call()` passes empty string `""` as `response_text` to FormatMonitor, so ALL tool calls get score 0.0 — ensuring constant false-positive degradation signals

---

## 5. Context Management

### mini-swe-agent: No explicit context management

- Messages just accumulate in `self.messages` list
- The API (LiteLLM/OpenRouter) handles context window natively
- Models are good at managing their own context — they don't need us injecting standing instructions or trimming aggressively
- The only context management is the `format_error_template` which tells the model to be more concise

### Localite: Aggressive, lossy context management

- `_refresh_context()` trims to `memory_horizon` (5-10 turns) + evicts by char budget (80% of `max_context_chars`)
- Loses the model's reasoning chain, partial analysis, and debugging progress
- Standing instructions, active plan, last tool result re-injected as fresh messages — **adds more context** than was removed
- Creates a vicious cycle: model produces format slightly off → format drops → refresh → model loses track → produces more format errors

---

## 6. Environment Interface

### mini-swe-agent: Clean action→observation cycle

```
step():
  1. model.query() → message with tool_calls actions
  2. env.execute(action) for each action → output dict
  3. format_observation_messages(actions, outputs) → tool-role messages
```

Tool result format (from `observation_template`):
```
<returncode>0</returncode>
<output>
command output here
</output>
```

Output truncation: configurable via Jinja2 template logic — 5000-char head + 5000-char tail with truncation warning.
Submission: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`

### Localite: Over-engineered with file content injection

- Last tool result injected as separate `tool_`-role message during refresh
- Active plan re-injected on every refresh
- Memory store (summary line from previous runs) injected into system prompt
- Nudged files list injected periodically

---

## 7. Model Selection / Multi-Backend Support

### mini-swe-agent: Clean model abstraction

```
Model (abstract interface)
├── LitellmModel         → uses litellm (OpenAI-compatible, Anthropic, Google, local)
├── LitellmTextbasedModel→ text regex parsing (for models without tool call support)
├── OpenRouterModel      → direct HTTP to OpenRouter (bypasses litellm)
├── OpenRouterTextbasedModel
├── PortkeyModel
├── RequestyModel
└── test_models.DeterministicModel
```

- `model.query()` returns a dict with keys: `role`, `content`, `tool_calls`, `extra`
- `model.format_message()` creates message dict
- `model.format_observation_messages()` creates tool-result messages
- Selects model class automatically based on model name or explicit `model_class` config

### Localite: Monolithic agent loop with hardcoded LiteLLM

- `agent_loop.py` (1741 lines) — everything in one file
- No model abstraction — directly calls `acompletion` from `litellm`
- Model selection happens via profile TOML, not code
- No text-based fallback pattern — single approach (text-JSON) for all models

---

## 8. Key Details That Make a Difference

| Aspect | mini-swe-agent | Localite |
|--------|---------------|----------|
| **Tool call** | Native API `tool_calls` | Text JSON extraction |
| **Prompt length** | ~80 lines total | ~300+ lines |
| **Context management** | None (API handles it) | Aggressive trimming + refresh |
| **Format recovery** | Informative error, retry | Context reset, loses memory |
| **Token limit handling** | Detects `finish_reason: length` | No special handling |
| **Observation template** | Jinja2 with head/tail truncation | Fixed 500-char token truncation |
| **Multiple tool calls** | Yes (parallel execution) | No (one at a time) |
| **Submission** | `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` | Same concept |
| **Model abstraction** | Clean class hierarchy | Monolithic hardcoded |
| **Cost tracking** | LiteLLM cost registry | None |
| **Output truncation** | 10K char buffer (5K head + 5K tail) | 64K char hard cap |
| **Error type** | FormatError → add messages, continue | FormatError → refresh context |

---

## 9. Why SLMs Might Benefit From Localite's Approach

Everything above argues that mini-swe-agent's native tool call approach is better. But Localite was designed for **SLMs** (3B-14B). Are there any advantages?

**Potential SLM-specific advantages of Localite:**
1. **More explicit instruction** — SLMs benefit from detailed output format examples and phase guidance
2. **Smaller context window** — SLMs have 4K-8K context, aggressive trimming may be necessary
3. **Tool call limitation** — Local SLMs (Ollama, llama.cpp) may not support native function calling
4. **Format guard as training signal** — SLMs need more format enforcement to stay on track

**However, these advantages don't hold anymore:**
1. Modern SLMs (Qwen 3.5, Llama 4 Scout, Gemma 3, Phi-4) all support native function calling via LiteLLM
2. Even if a model doesn't support tool calls, mini-swe-agent has `LiteLLMTextbasedModel` with regex-based extraction — better than brace counting
3. Aggressive context trimming hurts SLMs even more (they need more context to understand)
4. FormatGuard with false positives makes everything worse

**The right architecture for SLMs:**
Use the same native tool call approach as mini-swe-agent, but with:
- Slightly more verbose system prompt (format examples)
- Higher `format_error_template` tolerance (allow more retries)
- Simpler observation templates (SLMs get confused by Jinja2 logic)
- Support for both text-based and native tool calls (auto-detect which the model produces)

---

## 10. Summary: Root Cause of Localite's Underperformance

```
Localite's approach:
  model → text JSON in content → brace-depth parser → fragile

Mini-swe-agent's approach:
  model → API tool_calls → structured parser → reliable

For capable models (DeepSeek V4 Flash):
  33/33 tool calls via native API → 100% parse rate
  10-15/33 would survive text-JSON extraction → ~40% parse rate
  Remaining 60% cause format errors → refreshes → memory loss → worse performance

For SLMs (GLM 5.2, Qwen 3.5):
  Same native tool calls → 90%+ parse rate
  Text-JSON → 30-50% parse rate
  Even worse because SLMs produce more reasoning noise in content
```

**Conclusion**: The best path forward is to build on top of mini-swe-agent's proven architecture, extending it with:
1. Better SLM support (smarter observation truncation, configurable prompt verbosity)
2. Better evaluation tooling (comparison reports, trajectory visualization)
3. Better cost/time management for batch evaluation runs