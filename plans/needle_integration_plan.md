# Needle 26M + Gemma4 E4B AgentLoop Integration

## Goal
Create an alternative AgentLoop (`agent_loop_needle.py`) that uses Needle 26M as a **tool name validator/router** in conjunction with Gemma4 E4B, eliminating tool name hallucinations (`list_directory` → `list_files`) and validating all tool calls at ~50ms per turn instead of burning 2-3 turns on wrong tool names.

## Research Summary
- **Needle 26M**: 26M-parameter "Simple Attention Network" (no FFN layers), distilled from Gemini 3.1 Flash-Lite. Pure attention encoder-decoder (12 encoder / 8 decoder layers, d_model=512). Single-shot function calling specialist. MIT license.
- **Performance**: 6K tok/s prefill, 1.2K tok/s decode on Cactus runtime. On CPU: ~50ms per inference.
- **Installation**: `git clone https://github.com/cactus-compute/needle.git && source ./setup` — creates `.venv` with JAX (CPU), auto-downloads weights from HuggingFace. Requires Python 3.11+ (system has 3.12.3 ✅).
- **Usage**: `from needle import load_checkpoint, generate, SimpleAttentionNetwork, get_tokenizer` → `generate(model, params, tokenizer, query, tools)` → JSON tool call.
- **Limitation**: No FFN layers, 8K vocabulary — CANNOT generate code content for `edit_file`/`write_file` arguments. Can route to the correct tool name + simple scalar args (file paths, patterns).
- **Tool limitation on this env**: We are on CPU (no GPU) — Needle's Python/JAX inference on CPU will be slower than the ~50ms Cactus runtime claims but still far faster than a 2.5-min Ollama turn.

## Architecture: Needle as Tool Validator, Gemma4 as Reasoner+Editor

```
Gemma4: reasoning + proposed tool call (full JSON with arguments)
  │
  ▼
Needle: validates/corrects the tool NAME only (50ms)
  │  query = objective + Gemma's reasoning summary
  │  tools = available tool schemas
  │  output = {name: "list_files", arguments: ...}
  │
  ▼
Merge: Needle's tool name + Gemma's original arguments
  │  (For edit_file/write_file, Gemma's code content args are preserved)
  ▼
Execute tool
```

**Key insight**: Needle doesn't replace Gemma's argument generation. It acts as a **tool name validator** that catches `list_directory` → `list_files` at ~50ms instead of burning 2.5 min on a wrong turn. For navigation tools, Needle also provides correct arguments (paths, depths). For edit_file/write_file, Needle routes to the correct tool name but Gemma's code arguments are used.

## Approach

### Phase 1: Needle setup + lightweight client wrapper
Clone Needle repo into project, run setup (installs JAX CPU, downloads weights), create a `needle_client.py` wrapper that:
- Loads model once at init (~1-2s)
- Provides `validate_tool_call(query, tools_json, gemma_reasoning, gemma_tool_call) → corrected_tool_call`
- Handles graceful degradation if Needle isn't available

### Phase 2: agent_loop_needle.py
Copy of `agent_loop.py` with modifications:
1. Import + init NeedleClient (passed via constructor or auto-loaded)
2. In `_execute_phase()`: after parsing Gemma's tool call, run Needle to validate/correct the tool name
3. Merge logic: Needle's tool name + Gemma's arguments (for edit_file/write_file) OR Needle's full call (for navigation tools)
4. System prompt: modified to de-emphasize tool call format (optional — can keep same for compatibility)
5. Progressive guidance: still uses CodeIndex for file recommendations

### Phase 3: Evaluation
- Profile: `gemma4_needle.toml` pointing to the new agent loop
- Run pydicom-1139 evaluation
- Compare: turns used, tool accuracy, successful code modification rate

### Phase 4: Commit + Push (optional — before or after testing)
- Commit all current changes (ctags CodeIndex + needle integration)
- Push to GitHub

## Subtasks
0. (Optional) Git commit current state with all ctags + related changes
1. Clone the Needle 26M repo and run its setup script (Python 3.12 + JAX CPU weights)
2. Create `localite/model/needle_client.py` — lightweight wrapper: loads Needle model, provides `validate_tool_call()` method, handles graceful degradation. 
3. Create `localite/loop/agent_loop_needle.py` — copy of `agent_loop.py` with:
   - NeedleClient integration in `__init__`
   - Tool name validation step between `_parse_tool_call()` and `_handle_tool_call()`
   - Merge logic: Needle-validated tool name + Gemma's original arguments
   - Modified system prompt (tool format de-emphasized, reasoning emphasized)
4. Create `profiles/gemma4_needle.toml` pointing to the new agent loop
5. Unit test Needle tool routing with a mock test script
6. Run end-to-end pydicom-1139 evaluation to validate

## Deliverables
| File Path | Description |
|-----------|-------------|
| `localite/model/needle_client.py` | Needle model wrapper with validate_tool_call() |
| `localite/loop/agent_loop_needle.py` | Alternative AgentLoop with Needle validation |
| `profiles/gemma4_needle.toml` | Profile config for Needle + Gemma4 |
| `needle/` | Cloned Needle repo (as subdirectory with weights) |
| `plans/needle_integration_plan.md` | This plan |

## Evaluation Criteria
- Tool name hallucinations eliminated (zero `list_directory`-style errors)
- Successful tool call on first attempt in each turn
- Code modification (edit_file/write_file) arguments preserved correctly from Gemma4
- pydicom-1139 turns reduced compared to baseline (currently 10-19 turns)

## Notes
- Needle requires Python 3.11+ (✅ 3.12.3 available)
- JAX on CPU will be slower than GPU but still <500ms per inference
- Needle is MIT licensed — no restrictions on commercial use
- If Needle fails to load (JAX issues on this CPU), the agent loop gracefully falls back to Gemma-only behavior