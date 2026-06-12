# Plan: Rewrite Localite Agent System Prompt with Structured Tool Guidance + Examples

## Goal
Replace the minimal 60-line `SYSTEM_PROMPT` template + 1-line `_get_tool_descriptions()` with a rich structured prompt that teaches Gemma 4 (and any model) exactly how to use each tool, when to use it, and how to recover from failures. Mirror the quality of `prompt_examples.py` and `tool_utils.py` patterns.

## Root Causes This Fixes
1. **Model invents `"argument"` parameter name** because the example shows `{"key": "value"}` — a generic placeholder that trains the model to guess
2. **Model skips `read_file`** because there's no explicit "read first, edit later" reinforcement with concrete examples
3. **Model produces non-tool output in EXECUTE phase** because there's no "DO NOT send messages in EXECUTE" antipattern rule
4. **Model does not follow guidance** because the system prompt is only 60 lines of vague rules

## Approach
- Keep `SYSTEM_PROMPT` in `agent_loop.py` as a template with `{tool_descriptions}` placeholder
- Rewrite `_get_tool_descriptions()` to produce rich structured blocks (not just `name: desc`)
- Update each tool class's `description` property with longer documentation including `<example>` blocks
- Update `StandingInstructions` to be leaner (system prompt carries the main load)
- Fix the output format example to show real parameter names not `{"key": "value"}`

## Subtasks

1. **Audit current tool descriptions and identify problem patterns** — review all 9 tools' descriptions, parameters, names. Note: `edit_file` uses `search_text`/`replace_text` not `old_value`/`new_value`, `list_files` uses `path` not `directory`. Document the exact schema for each.
   *(verify: output a list of tool schemas with correct param names)*

2. **Rewrite `_get_tool_descriptions()` to produce structured rich tool docs** — each tool gets a block with:
   - Name + description
   - Required/optional params with types
   - 1-2 `<example>` usage blocks showing exact JSON syntax with REAL parameter names
   - "When to use" / "When NOT to use"
   - Common mistakes (e.g. "list_files uses 'path' not 'directory'")
   - What success looks like / what failure looks like
   *(verify: python3 -m py_compile swe_runner.py passes — wait, no, agent_loop.py. Also inspect the rendered output)*

3. **Rewrite SYSTEM_PROMPT** with:
   - Agent identity expanded
   - Output format with real parameter examples (fix the `{"key": "value"}` bug → show `{"path": "src/main.py"}`)
   - Phase protocol embedded in system prompt (not just standing_instructions)
   - Anti-pattern rules section (DO NOT use message in EXECUTE phase, DO NOT guess parameter names)
   - Exploration mandate with concrete flow (list_files → read_file → understand → edit)
   - Worst-case: "If you don't know which file to read, use grep_search to find identifiers from the task"
   *(verify: rendered system prompt looks correct — grep for "key" placeholder removed)*

4. **Update StandingInstructions** to be leaner — remove the phase protocol (moved to system prompt), keep only safety rules and phase-transition signal rules.
   *(verify: no test regressions)*

5. **Verify syntax and run existing tests** — `python3 -m py_compile localite/loop/agent_loop.py`, `python3 -m py_compile localite/context/standing_instructions.py`, then run `pytest tests/ -v --tb=short -q`.
   *(verify: 23 passed, 3 skipped as before — NO regressions)*

6. **Render the system prompt** — run a small script that constructs the system prompt as the agent loop does and print it, so we can inspect it for quality.
   *(verify: output file written at `/home/azureuser/local_llm_eval/results/swe_bench/rendered_system_prompt.txt`)*

## Files Modified
| File | Change |
|------|--------|
| `/home/azureuser/local_llm_eval/localite/loop/agent_loop.py` | Rewrite SYSTEM_PROMPT + _get_tool_descriptions() |
| `/home/azureuser/local_llm_eval/localite/context/standing_instructions.py` | Leaner — remove phase protocol, keep safety rules |
| `/home/azureuser/local_llm_eval/localite/tools/read.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/edit.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/write.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/shell.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/list_files.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/search.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/test_executor.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/task_complete.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/diff_view.py` | Rich description with examples |
| `/home/azureuser/local_llm_eval/localite/tools/memory_tools.py` | Rich description with examples |

## Evaluation Criteria
- System prompt no longer contains `"key"` or `"value"` as parameter placeholders
- Each tool has at least one `<example>` block with real parameter names
- Phase protocol is embedded in system prompt
- Anti-pattern section warns against: message in EXECUTE phase, guessing parameter names, skipping read_file
- All tool descriptions include When to use / When NOT to use / Common mistakes
- `python3 -m py_compile` passes for all modified files
- `pytest tests/` shows same pass/skip counts (23/3)