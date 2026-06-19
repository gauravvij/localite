# GLM 5.2 SWE-bench Lite Evaluation

## Goal
Run the Localite harness with `z-ai/glm-5.2` via OpenRouter on the same 5 sqlfluff SWE-bench Lite instances and compare against v6-v9 (DeepSeek V4 Flash) results.

## Research Summary
- **Model ID**: `z-ai/glm-5.2` via OpenRouter (verified available)
- **Pricing**: $1.40/1M input, $4.40/1M output tokens
- **Context**: 1,048,576 tokens (~1M), max output 262,144 tokens
- **Tool calling**: OpenAI-compatible format (works with our `AsyncOpenAIClient`)
- **Thinking**: GLM-5.2 produces reasoning tokens. OpenRouter supports `reasoning: {exclude: true}` to suppress these from content. Medium article confirms GLM-5's `exclude` flag works correctly — tokens go to `reasoning` field, not content.
- **GLM-5 base model**: 744B total, 40B active MoE, SWE-bench Verified 77.8%
- **GLM-5.2**: Successor released June 16, 2026, with 1M context, improved agentic capability

## Approach
Create a GLM 5.2 profile (`glm_5_2.toml`) with large context window (1M), high stall threshold, and reasoning excluded. Wire the `reasoning: {exclude: true}` parameter into `AsyncOpenAIClient` via a new profile field. Run on 5 sqlfluff instances, compare against v6-v9.

## Subtasks
1. **Add `reasoning_exclude` to ModelProfile** — Add `reasoning_exclude: Optional[bool] = None` field to the `ModelProfile` dataclass in `config.py`.
2. **Update openai_client.py** — In the `chat()` method, when `self.reasoning_exclude is True`, add `payload["reasoning"] = {"exclude": True}` alongside the existing `disable_thinking` logic.
3. **Update swe_runner.py** — Pass `reasoning_exclude=profile.reasoning_exclude` into the `AsyncOpenAIClient()` constructor call at the create_swe_agent() function (match how `disable_thinking` is wired).
4. **Update AsyncOpenAIClient.__init__** — Add `reasoning_exclude: bool = False` parameter, store as `self.reasoning_exclude`.
5. **Create glm_5_2.toml profile** — With values: name=`z-ai/glm-5.2`, provider=`openai_compatible`, base_url=`https://openrouter.ai/api/v1`, stall_threshold=20, max_context_chars=400000, memory_horizon=20, max_context_window=1048576, num_predict=8192, format_guard=false, has_thinking_tags=false, reasoning_exclude=true, timeout=600, api_key=<from existing v6 profile>.
6. **Verify profile loads and client works** — Run a quick python test: load profile, construct client, verify reasoning_exclude is set.
7. **Run pytest** — Confirm all 30 tool call parsing tests pass.
8. **Run evaluation** — `python3 swe_runner.py --instances sqlfluff__sqlfluff-1625 sqlfluff__sqlfluff-2419 sqlfluff__sqlfluff-1733 sqlfluff__sqlfluff-1517 sqlfluff__sqlfluff-1763 --profile glm_5_2 --agent-timeout 1200` (longer timeout for a potentially more reasoning-heavy model, timeout=3600s for the command).
9. **Generate comparison report** — After run, load `all_results.json` and generate `results/swe_bench/glm_5_2_comparison_report.md` comparing v6, v7, v8, v9, and GLM-5.2 results. Include executive summary, per-instance comparison table, aggregate metrics, newly resolved analysis.

## Deliverables
| File Path | Description |
|-----------|-------------|
| profiles/glm_5_2.toml | GLM 5.2 profile |
| results/swe_bench/glm_5_2_comparison_report.md | Comparison report (v6-v9 vs GLM-5.2) |
| (modified) localite/config.py | Updated ModelProfile with reasoning_exclude |
| (modified) localite/model/openai_client.py | Updated client to pass reasoning: {exclude: true} |
| (modified) swe_runner.py | Updated constructor to pass reasoning_exclude |

## Evaluation Criteria
- Evaluation runs successfully on all 5 instances
- GLM-5.2 resolves ≥3 instances (matching v9 DeepSeek) or shows improvement
- Total cost < $5 for the 5-instance run
- No API errors, no parser failures

## Notes
- Use the existing API key from OpenRouter config (sk-or-v1-...)
- GLM-5.2 is a more capable model than DeepSeek V4 Flash (SWE-bench Verified 77.8% vs likely lower for DeepSeek V4 Flash)
- If GLM-5.2 significantly outperforms DeepSeek (4-5/5 resolved), that validates the harness is clean and the earlier poor results were truly model-related
- If GLM-5.2 performs similarly (~3/5), it suggests there are remaining harness issues or the 5-instance set has a ceiling