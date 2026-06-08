# E2B Results Summary — Gemma 4 E2B

**Generated**: 2026-06-05
**Model**: gemma4:e2b (Ollama)
**Total tests**: 96 across 6 dimensions
**Run duration**: ~8.4 hours (30,136s) for 5 new dims + prior instruction_adherence_decay

## Executive Summary

Gemma 4 E2B shows **strong performance** in instruction adherence and tool call drift, but **significant degradation** in memory retrieval, persona consistency, and recency bias.

| Dimension | Mean Score | Break Point | Verdict |
|-----------|-----------|-------------|---------|
| Instruction Adherence Decay | **0.83** | Depth 20 | 🟢 Good — maintains JSON format through 15 turns |
| Tool Call Drift | **0.89** | Depth 20 | 🟢 Good — maintains tool call format through 10 turns |
| Hallucination Onset | **0.78** | 8000 tokens | 🟡 Moderate — starts hallucinating at 8K context |
| Persona Consistency | **0.60** | Depth 10 | 🟠 Weak — drops persona as early as 10 turns |
| Memory Retrieval | **0.56** | Depth 8 | 🔴 Poor — forgets early facts after 8 turns |
| Recency Bias | **0.00** | Depth 1 | 🔴 Critical — never follows the override instruction |

## Per-Dimension Breakdown

### 1. Instruction Adherence Decay (24 tests)
- **Mean score**: 0.83
- **Break point**: Depth 20 turns (score drops below 0.8)
- **Depths tested**: 1, 3, 5, 8, 10, 15, 20, 30

| Depth | Mean Score | Categories |
|-------|-----------|-----------|
| 1 | 1.00 ✅ | partial_json |
| 3 | 1.00 ✅ | partial_json |
| 5 | 1.00 ✅ | pass, partial_json |
| 8 | 1.00 ✅ | partial_json |
| 10 | 1.00 ✅ | partial_json |
| 15 | 1.00 ✅ | partial_json |
| 20 | 0.33 ⚠️ | partial_json, plain_text, wrong_json |
| 30 | 0.33 ⚠️ | partial_json, plain_text, plain_text |

**Observation**: Strong up to 15 turns (perfect 1.0). At depth 20-30, 2/3 runs fail outputting plain text or wrong JSON format. Non-monotonic behavior: depth 30 shows recovery on run 1.

### 2. Tool Call Drift (15 tests)
- **Mean score**: 0.89
- **Break point**: Depth 20 turns
- **Depths tested**: 1, 3, 5, 10, 20

| Depth | Mean Score | Categories |
|-------|-----------|-----------|
| 1 | 1.00 ✅ | pass |
| 3 | 1.00 ✅ | pass |
| 5 | 1.00 ✅ | pass |
| 10 | 1.00 ✅ | pass |
| 20 | 0.67 ⚠️ | pass, partial_json, plain_text |

**Observation**: Excellent through 10 turns (perfect). At depth 20, 2/3 runs still pass, 1 run degrades to partial_json/plain_text.

### 3. Hallucination Onset (18 tests)
- **Mean score**: 0.78
- **Break point**: 8000 tokens
- **Depths tested**: 1000, 2000, 4000, 8000, 16000, 32000

| Depth | Mean Score | Categories |
|-------|-----------|-----------|
| 1000 tokens | 1.00 ✅ | pass |
| 2000 tokens | 1.00 ✅ | pass |
| 4000 tokens | 1.00 ✅ | pass |
| 8000 tokens | 0.33 ⚠️ | pass, hallucinated_fact |
| 16000 tokens | 0.67 ⚠️ | pass, hallucinated_fact |
| 32000 tokens | 0.67 ⚠️ | pass, hallucinated_fact |

**Observation**: Perfect accuracy through 4K tokens. At 8K, 2/3 runs hallucinate. Accuracy partially recovers at 16K-32K (2/3 pass). This is a known pattern — the model may "forget" the hard fact at specific context sizes but re-access it later.

### 4. Persona Consistency (15 tests)
- **Mean score**: 0.60
- **Break point**: Depth 10 turns
- **Depths tested**: 1, 5, 10, 20, 30

| Depth | Mean Score | Categories |
|-------|-----------|-----------|
| 1 | 1.00 ✅ | pass |
| 5 | 1.00 ✅ | pass |
| 10 | 0.33 ⚠️ | persona_drop, pass, persona_drop |
| 20 | 0.33 ⚠️ | pass, persona_drop, empty_response |
| 30 | 0.33 ⚠️ | persona_drop, pass, persona_drop |

**Observation**: Perfect at depths 1-5. From depth 10 onward, model drops persona ("I am Gemma 4, a Large Language Model developed by Google DeepMind") in 2/3 runs. Non-monotonic: depth 20 run 1 and depth 30 run 2 still pass.

### 5. Memory Retrieval (18 tests)
- **Mean score**: 0.56
- **Break point**: Depth 8 turns
- **Depths tested**: 1, 3, 5, 8, 10, 15 — **adaptive stop triggered at depth 15**

| Depth | Mean Score | Categories |
|-------|-----------|-----------|
| 1 | 1.00 ✅ | pass |
| 3 | 1.00 ✅ | pass |
| 5 | 1.00 ✅ | pass |
| 8 | 0.33 ⚠️ | pass, hallucinated_fact, hallucinated_fact |
| 10 | 0.00 ❌ | hallucinated_fact |
| 15 | 0.00 ❌ | hallucinated_fact |

**Observation**: Perfect recall of "Einstein (dog name)" through 5 turns. At depth 8, model starts claiming it doesn't have personal information. By depth 10, complete failure. ⚠️ Adaptive stop triggered — deeper depths (20, 30) not tested.

### 6. Recency Bias (6 tests)
- **Mean score**: 0.00
- **Break point**: Depth 1 turn (immediate failure)
- **Depths tested**: 1, 3 — **adaptive stop triggered at depth 3**

| Depth | Mean Score | Categories |
|-------|-----------|-----------|
| 1 | 0.00 ❌ | ignore_instruction |
| 3 | 0.00 ❌ | ignore_instruction |

**Observation**: **Complete failure** — model always follows the early "Aye aye, captain!" prefix instruction and ignores the override at depth. All 6 runs fail with `ignore_instruction`. This is the weakest dimension for Gemma 4 E2B.

## Failure Category Distribution (All Dimensions)

| Category | Count | Percentage | Dimensions Affected |
|----------|-------|-----------|-------------------|
| pass | 47 | 49.0% | All |
| partial_json | 20 | 20.8% | instruction_adherence_decay |
| hallucinated_fact | 12 | 12.5% | memory_retrieval, hallucination_onset |
| ignore_instruction | 6 | 6.2% | recency_bias |
| persona_drop | 5 | 5.2% | persona_consistency |
| plain_text | 4 | 4.2% | instruction_adherence_decay, tool_call_drift |
| wrong_json | 1 | 1.0% | instruction_adherence_decay |
| empty_response | 1 | 1.0% | persona_consistency |

**Total failures** (score < 0.8): 30 out of 96 tests (31.2%)

## Comparison with LFM2.5-8B-A1B

| Dimension | Gemma 4 E2B | LFM2.5-8B-A1B |
|-----------|-----------|-----------|
| Instruction Adherence Decay | **Break: 20 turns** (mean 0.83) | ⏳ Not yet tested |
| Tool Call Drift | **Break: 20 turns** (mean 0.89) | Break: 1 turn (mean 0.26) |
| Hallucination Onset | **Break: 8000 tokens** (mean 0.78) | Break: 8000 tokens (mean 0.83) |
| Persona Consistency | **Break: 10 turns** (mean 0.60) | Never broke (mean 1.00) |
| Memory Retrieval | **Break: 8 turns** (mean 0.56) | Break: 8 turns (mean 0.58) |
| Recency Bias | **Break: 1 turn** (mean 0.00) | Break: 1 turn (mean 0.00) |

**Key insights**:
- Gemma 4 E2B outperforms LFM2.5 on **tool call drift** (break at 20 vs 1 turn)
- LFM2.5 outperforms Gemma 4 E2B on **persona consistency** (never breaks vs break at 10)
- Both models are equally poor at **recency bias** and **memory retrieval**
- **Hallucination onset** and **memory retrieval** are near-identical between models
- Instruction adherence decay for LFM2.5 still needs to be tested

## Key Takeaways

1. **Strongest**: Instruction Adherence Decay (0.83) and Tool Call Drift (0.89)
2. **Weakest**: Recency Bias (0.00) — complete failure pattern
3. **Memory window**: ~8 turns before facts are lost
4. **Persona retention**: ~10 turns before model reverts to "I am Gemma 4"
5. **Context limit**: 8K tokens before hallucination onset (consistent with LFM2.5)
6. **Non-monotonic behavior**: Multiple dimensions show recovery at deeper depths, suggesting the degradation is not always monotonic