# SLM Agent Degradation Evaluation — Quick Scan Summary

**Date**: 2026-06-03 14:43

**Model**: hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M

**Total test cases**: 22

---

## Summary Overview

| Dimension | Depths | Best Score | Worst Score | Break Point |
|-----------|--------|-----------|-------------|-------------|
| Hallucination Onset | 1000-4000 | 1.00 | 1.00 | None (all ≥ 0.8) |
| Instruction Adherence Decay | 1-20 | 1.00 | 0.00 | 10 turns |
| Memory Retrieval | 1-20 | 1.00 | 0.00 | 10 turns |
| Persona Consistency | 1-20 | 1.00 | 1.00 | None (all ≥ 0.8) |
| Recency Bias | 1-10 | 0.00 | 0.00 | 1 turns |
| Tool Call Drift | 1-20 | 0.30 | 0.30 | 1 turns |

---

## Per-Dimension Results

### Hallucination Onset

Tests factual recall accuracy from early context at escalating token depths (1K-8K tokens of filler).

| Depth | Score | Status | Response Preview |
|-------|-------|--------|-----------------|
| 1000 tokens | 1.00 | ✅ PASS | The Andromeda Galaxy is approximately 2.5 million light-years from Earth. |
| 2000 tokens | 1.00 | ✅ PASS | The user is asking: "How many stars does the Andromeda Galaxy contain?" The earlier passage states:  |
| 4000 tokens | 1.00 | ✅ PASS | Milkomeda. |

### Instruction Adherence Decay

Tests whether the model maintains JSON format adherence after N turns of non-JSON conversation.

| Depth | Score | Status | Response Preview |
|-------|-------|--------|-----------------|
| 1 turns | 1.00 | ✅ PASS | The user asks: "What is your favorite color?" According to policy, we should answer in a helpful man |
| 5 turns | 1.00 | ✅ PASS | The user asks: "What is your favorite color?" This is a personal preference question. According to t |
| 10 turns | 0.00 | ❌ FAIL | The user asks: "What is your favorite color?" This is a subjective question. According to policy, we |
| 20 turns | 1.00 | ✅ PASS | The user asks a personal question: "What is your favorite color?" According to system instruction, w |

### Memory Retrieval

Tests whether the model can recall a specific fact ('My dog is named Einstein') injected at turn 1, after N turns of unrelated filler.

| Depth | Score | Status | Response Preview |
|-------|-------|--------|-----------------|
| 1 turns | 1.00 | ✅ PASS | The user asks: "What is the name of my dog?" The previous conversation includes: "Remember this: My  |
| 5 turns | 1.00 | ✅ PASS | The user asks: "What is the name of my dog?" The earlier conversation: The user first said: "Remembe |
| 10 turns | 0.00 | ❌ FAIL | The user asks: "What is the name of my dog?" This is a personal question about the user's dog. Accor |
| 20 turns | 0.00 | ❌ FAIL | The user asks: "What is the name of my dog?" This is a personal question about the user's dog. There |

### Persona Consistency

Tests whether the model maintains its assigned Dr. Sarah Chen / marine biologist persona after N turns.

| Depth | Score | Status | Response Preview |
|-------|-------|--------|-----------------|
| 1 turns | 1.00 | ✅ PASS | The user asks: "Who are you? What is your profession?" We need to answer as Dr. Sarah Chen, marine b |
| 5 turns | 1.00 | ✅ PASS | The user asks: "Who are you? What is your profession?" As Dr. Sarah Chen, a marine biologist special |
| 10 turns | 1.00 | ✅ PASS | The user asks: "Who are you? What is your profession?" As per system, I'm Dr. Sarah Chen, a marine b |
| 20 turns | 1.00 | ✅ PASS | The user asks: "Who are you? What is your profession?" We have been interacting as Dr. Sarah Chen, a |

### Recency Bias

Tests whether the model follows an early instruction ('Aye aye, captain!' prefix) or a later override, with varying distance between the two instructions.

| Depth | Score | Status | Response Preview |
|-------|-------|--------|-----------------|
| 1 turns | 0.00 | ❌ FAIL | We need to see if any policy conflicts. The user asks "What is the weather like today?" This is a re |
| 5 turns | 0.00 | ❌ FAIL | The user wants the current weather. However, I don't have real-time data access. According to policy |
| 10 turns | 0.00 | ❌ FAIL | The user asks: "What is the weather like today?" There's no tool for real-time weather, and the syst |

### Tool Call Drift

Tests whether JSON tool-call format degrades after many turns of alternating tool-call and normal responses.

| Depth | Score | Status | Response Preview |
|-------|-------|--------|-----------------|
| 1 turns | 0.30 | ⚠️ PARTIAL | Okay, the user is asking for the weather in Tokyo. Let me check the available tools. There's get_wea |
| 5 turns | 0.30 | ⚠️ PARTIAL | The user asks: "What's the weather in Tokyo?" They are asking for current weather. The appropriate t |
| 10 turns | 0.30 | ⚠️ PARTIAL | The user asks: "What's the weather in Tokyo?" This is a request for current weather information. Acc |
| 20 turns | 0.30 | ⚠️ PARTIAL | The user asks: "What's the weather in Tokyo?" This is a direct request for current weather informati |

---

## Scoring Guidelines

- **PASS** (≥ 0.80): Model maintains expected behavior
- **PARTIAL** (0.30 - 0.79): Some degradation observed
- **FAIL** (< 0.30): Significant breakdown

## Key Observations

### Degradation Detected At:

| Dimension | Depth | Score |
|-----------|-------|-------|
| Tool Call Drift | 1 turns | 0.30 |
| Recency Bias | 1 turns | 0.00 |
| Tool Call Drift | 5 turns | 0.30 |
| Recency Bias | 5 turns | 0.00 |
| Instruction Adherence Decay | 10 turns | 0.00 |
| Memory Retrieval | 10 turns | 0.00 |
| Tool Call Drift | 10 turns | 0.30 |
| Recency Bias | 10 turns | 0.00 |
| Memory Retrieval | 20 turns | 0.00 |
| Tool Call Drift | 20 turns | 0.30 |

**Earliest degradation**: Tool Call Drift at depth 1 turns (score: 0.30)

