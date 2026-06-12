# SLM Agent Harness Improvements — Type Coercion, Navigation Scaffolding, Adaptive Budget & Needle Exploration

## Goal
Apply 3 harness-level fixes, increase token budget to 8192, re-run marshmallow-1359 evaluation, and research integrating Needle 26m as a dedicated tool-calling router to decouple reasoning from JSON generation.

## Research Summary
- **Needle 26m** (Cactus Compute): 26M-parameter "Simple Attention Network" with no FFNs — pure attention layers. Distilled from Gemini 3.1 Flash Lite. Beats FunctionGemma-270M and Qwen-0.6B on single-shot function calling. 6000 tok/s prefill, 1200 tok/s decode. Architecture thesis: "tool calling is retrieval-and-routing, not reasoning" — attention is the right primitive for converting natural language intent into structured tool calls. MIT licensed, weights on HuggingFace. Can run on CPU (26M params). Limitation: sensitive to phrasing, may not generalize to arbitrary schemas.
- **Key insight for our harness**: The current bottleneck is that Gemma 4 E4B must do BOTH reasoning AND JSON tool call generation in one response. Thinking consumes 2-4K tokens, leaving no budget for clean JSON output. A dedicated tool-calling model like Needle could **decouple** these: SLM reasons in natural language → Needle converts intent to JSON tool call → harness executes. This bypasses the token budget problem entirely.
- **Type coercion**: Models (including frontier models) occasionally pass string values for integer parameters (e.g. `depth="2"`). Harness should coerce silently rather than fail.
- **Adaptive token budget**: EXECUTE phase needs more tokens because model must think AND produce JSON. EXPLORE phase needs fewer (list_files is simple). Setting a single high value globally (8192) is the simplest start.

## Approach — 3 harness fixes + profile update + re-run + Needle research

### Part A: Harness Fixes (implement and commit)

1. **Type coercion for tool parameters** — Add `_coerce_param()` helper to BaseTool or directly in ListFilesTool.execute(). When model passes `depth="2"` (string), coerce to int(2). Pattern: detect type mismatch between schema declaration and runtime value → coerce silently. Apply to `depth` in list_files as immediate target.

2. **"Known files" auto-suggestion (strengthen progressive guidance)** — Current progressive guidance says "[GUIDANCE] You now see the file structure with .py source files above. Use read_file to read their CONTENTS." Strengthen this to actually propose the specific read_file call: after list_files returns output with `.py` filenames, extract those filenames and inject: `[GUIDANCE] You found .py files. Read src/marshmallow/fields.py with read_file to understand the code.`

3. **Adaptive token budget or global 8192** — Simplest approach: set `num_predict = 8192` globally in the profile. If needed later, can add per-phase dynamic adjustment. The 4096 run showed the model CAN produce read_file calls at the 7th turn; 8192 gives it more comfortable headroom.

### Part B: Profile Update and Evaluation

4. **Increase num_predict to 8192** in `gemma4_e4b.toml`.
5. **Re-run marshmallow-1359** with all fixes active, agent-timeout 900s.
6. **Report results** — comparison against 2048 and 4096 runs.

### Part C: Needle 26m Exploration

7. **Research Needle serving options** — Can we run it via its Python package directly? Is it available via Ollama/llama.cpp? What about OpenRouter? Check if the model can be loaded on CPU (8 cores, 62GB RAM — 26M params is ~100MB, trivially fits).
8. **Design dual-model architecture** — How would Needle integrate into AgentLoop? Proposal: SLM does reasoning → produces natural language "intent" → Needle converts to JSON tool call. Harness routes both models. Document the architecture design.
9. **Prototype a proof-of-concept** — If Needle can run locally via its Python package, write a small test: feed it a tool schema + intent text, verify it produces correct JSON tool calls.

## Subtasks
1. Add type coercion to ListFilesTool.execute() for `depth` parameter (string → int) (verify: write a small test calling list_files with depth="2" and confirm it coerces correctly)
2. Strengthen progressive guidance in agent_loop.py — extract .py filenames from list_files output and inject specific read_file suggestions (verify: syntax check via ast.parse)
3. Increase num_predict from 4096 to 8192 in gemma4_e4b.toml (verify: grep the file)
4. Run swe_runner.py on marshmallow-1359 with 900s timeout and all fixes active (verify: exit code 0)
5. Read result JSON and debug log, produce comparison report against 2048 and 4096 runs
6. Research Needle 26m availability: check Ollama models, HuggingFace, OpenRouter for hosted/quantized versions (verify: list what's available)
7. Design dual-model architecture for Needle integration and document design_plan.md update
8. If Needle can run locally, write a proof-of-concept test that passes intent + tool schema → gets back JSON tool call

## Deliverables
| File Path | Description |
|-----------|-------------|
| localite/tools/list_files.py | Type coercion for depth parameter |
| localite/loop/agent_loop.py | Strengthened progressive guidance (extract .py filenames, suggest specific read_file calls) |
| profiles/gemma4_e4b.toml | num_predict = 8192 |
| results/swe_bench/marshmallow-code__marshmallow-1359.json | New evaluation result with 8192 tokens |
| results/swe_bench/debug_run_1359.log | New debug log |
| Needle research notes (in design_plan.md or new doc) | Architecture design for dual-model harness |

## Evaluation Criteria
- Type coercion works: model passes depth="2" → list_files interprets as depth=2
- Progressive guidance suggests specific .py file paths, not generic hints
- num_predict=8192 in profile confirmed
- marshmallow-1359 evaluation completes (timeout or completed)
- Comparison report shows whether read_file count increased vs 4096 run
- Needle availability assessed: can it run locally? Is it via OpenRouter? Does dual-model architecture make sense for our harness?

## Notes
- Venv path: /home/azureuser/local_llm_eval/venv
- Needle is only 26M params (~100MB) — should run on CPU effortlessly at 1200 tok/s decode
- The Needle integration would be architectural: SLM does reasoning, Needle does tool-call formatting. This would make our harness fundamentally different from single-model agents
- If Needle runs locally, the prototype test should verify it works with our tool schemas (read_file, edit_file, list_files, etc.)