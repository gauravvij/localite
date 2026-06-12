# Profile-Driven Tool Call Format

## Goal
Each model profile declares its JSON tool call format. The system prompt teaches that format. The parser handles all known formats defensively.

## Why
Gemma 4 E4B natively produces `{"tool_name": "...", "params": {...}}` — teaching it `{"tool": "...", "arguments": {...}}` forces it to fight its own output formatting. Different models may prefer different keys. Profile-driven approach lets each model use its native format in the prompt (and we teach the exact format to help), while the parser remains a universal fallback layer.

## Root Cause
The system prompt's "## Output Format" section hardcodes one format (`tool`/`arguments`). The model sees examples with those keys and still overrides with its native format (`tool_name`/`params`), causing parser to drop all tool calls.

## Approach
1. Add `tool_call_format: str = "standard"` to `ModelProfile` 
2. Define two named output format templates in `agent_loop.py`
3. Route template selection at the 3 SYSTEM_PROMPT assembly sites
4. Set `tool_call_format = "gemma_native"` in gemma4_e4b.toml
5. (Bonus) Add unknown-kwarg stripping at tool dispatch to prevent `TypeError` from extra model params like `"filter"`
6. Verify syntax, tests, re-render with gemma profile active

## Subtasks

1. **Add `tool_call_format` field to `ModelProfile`** in `config.py`
   - Default: `"standard"`
   - Type: `str`
   - Expected output: dataclass field added

2. **Extract and define format templates** in `agent_loop.py`
   - `OUTPUT_FORMAT_STANDARD` — the current output format block (uses `tool`/`arguments`/`tool`/`run_shell`/etc.)
   - `OUTPUT_FORMAT_GEMMA_NATIVE` — Gemma's native format (uses `tool_name`/`params`, same param names but different envelope)
   - Both templates maintain the same examples and anti-pattern rules, just different JSON key names
   - Expected output: two string constants defined near SYSTEM_PROMPT

3. **Insert `{output_format}` placeholder into SYSTEM_PROMPT**
   - Replace the current `## Output Format` block with `{output_format}` placeholder
   - Expected output: SYSTEM_PROMPT template has `{tool_descriptions}` and `{output_format}` placeholders

4. **Route template selection at 3 assembly sites**
   - Lines 183, 845, 968 each do `SYSTEM_PROMPT.format(tool_descriptions=tool_descs)`
   - Change to: `SYSTEM_PROMPT.format(tool_descriptions=tool_descs, output_format=selected_format)`
   - Where `selected_format = OUTPUT_FORMAT_GEMMA_NATIVE if self.profile and self.profile.tool_call_format == "gemma_native" else OUTPUT_FORMAT_STANDARD`
   - Expected output: all 3 assembly sites use profile-driven format

5. **Update gemma4_e4b.toml profile**
   - Add `tool_call_format = "gemma_native"`
   - Expected output: TOML file updated

6. **Strip unknown kwargs at tool dispatch** (around line 1127)
   - After `_normalize_args()` call, filter `args` to only keys the tool's `execute()` method accepts
   - Get tool's parameter set via `tool.parameters` (the @property returning JSON schema dict)
   - Filter: `valid_params = set(tool.parameters.get("properties", {}).keys())` then `{k: v for k, v in args.items() if k in valid_params}`
   - Expected output: extra kwargs like `"filter"` silently removed instead of TypeError

7. **Verify syntax + tests**
   - `python3 -m py_compile agent_loop.py config.py`
   - `python3 -m pytest tests/ -v --tb=short -q`
   - Expected output: 32 passed, 3 skipped

8. **Render system prompt with gemma native format and inspect**
   - Run render script, confirm `tool_name`/`params` appear in examples, `tool`/`arguments` do NOT (for gemma profile)
   - Expected output: rendered_system_prompt.txt with gemma_native format

## Deliverables
| File | Change |
|------|--------|
| `localite/config.py` | Add `tool_call_format` field |
| `localite/loop/agent_loop.py` | Add 2 format templates, placeholder injection, profile-driven routing, kwarg stripping |
| `profiles/gemma4_e4b.toml` | Add `tool_call_format = "gemma_native"` |
| `results/swe_bench/rendered_system_prompt.txt` | Updated with native format |

## Evaluation Criteria
- `gemma4_e4b.toml` loaded → system prompt uses `tool_name`/`params` keys in examples
- Default (no profile / no field) → system prompt uses `tool`/`arguments` keys (backwards compatible)
- Extra unknown params like `filter` silently stripped before `tool.execute()`
- 32/32 tests pass, syntax clean