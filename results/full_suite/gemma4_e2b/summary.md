# Full Suite Evaluation — Gemma 4 E2B

**Date**: 2026-06-04 23:37

**Model**: gemma4:e2b

**Total test cases**: 72

---

## Summary Overview

| Dimension | Depths Tested | Best Score | Worst Score | Break Point | Adaptive Stop |
|-----------|--------------|-----------|-------------|-------------|---------------|
| Hallucination Onset | 6 | 1.00 | 0.00 | 8000 tokens | No |
| Memory Retrieval | 6 | 1.00 | 0.00 | 8 turns | 🛑 Yes |
| Persona Consistency | 5 | 1.00 | 0.00 | 10 turns | No |
| Recency Bias | 2 | 0.00 | 0.00 | 1 turns | 🛑 Yes |
| Tool Call Drift | 5 | 1.00 | 0.00 | 20 turns | No |

---

## Per-Dimension Results

### Hallucination Onset

Tests factual recall accuracy from early context at escalating token depths (1K-32K tokens of filler).

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy is approximately **2.5 million light-years** from Earth. |
| 2 | 1000 tokens | ✅ 1.00 | pass | The collision between the Andromeda Galaxy and the Milky Way is expected to occu |
| 3 | 1000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy was first recorded by the Persian astronomer **Al-Sufi in 9 |
| 1 | 2000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy contains about **one trillion stars**. |
| 2 | 2000 tokens | ✅ 1.00 | pass | The two galaxies will merge to form a new galaxy sometimes called **"Milkomeda"* |
| 3 | 2000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy has a diameter of about **220,000 light-years**. |
| 1 | 4000 tokens | ✅ 1.00 | pass | The two galaxies are expected to merge to form a new galaxy sometimes called **" |
| 2 | 4000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy has a diameter of about **220,000 light-years**. |
| 3 | 4000 tokens | ✅ 1.00 | pass | Andromeda is approaching the Milky Way at a speed of about **110 kilometers per  |
| 1 | 8000 tokens | ❌ 0.00 | hallucinated_fact | Andromeda is approaching the Milky Way at a speed of approximately **110 to 130  |
| 2 | 8000 tokens | ✅ 1.00 | pass | It is impossible to give an exact, definitive number for the total number of sta |
| 3 | 8000 tokens | ❌ 0.00 | hallucinated_fact | That depends entirely on which two galaxies you are referring to!  When two gala |
| 1 | 16000 tokens | ✅ 1.00 | pass | Andromeda is approaching the Milky Way at a speed of approximately **110 kilomet |
| 2 | 16000 tokens | ✅ 1.00 | pass | It is impossible to know the exact number of stars in the Andromeda Galaxy (M31) |
| 3 | 16000 tokens | ❌ 0.00 | hallucinated_fact | That depends entirely on which galaxies you are referring to!  Galaxy mergers ar |
| 1 | 32000 tokens | ✅ 1.00 | pass | Andromeda is approaching the Milky Way at a speed of approximately **110 kilomet |
| 2 | 32000 tokens | ✅ 1.00 | pass | It is impossible to give an exact, definitive number for the stars in the Androm |
| 3 | 32000 tokens | ❌ 0.00 | hallucinated_fact | That depends entirely on which galaxies you are referring to! Galaxy mergers are |

### Memory Retrieval

Tests whether the model can recall a specific fact ('My dog is named Einstein') injected at turn 1, after N turns of unrelated filler.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ✅ 1.00 | pass | The name of your dog is **Einstein**. |
| 2 | 1 turns | ✅ 1.00 | pass | Your dog's name is Einstein. |
| 3 | 1 turns | ✅ 1.00 | pass | Your dog is named Einstein. |
| 1 | 3 turns | ✅ 1.00 | pass | The name of your dog is **Einstein**. |
| 2 | 3 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 3 | 3 turns | ✅ 1.00 | pass | Your dog is named Einstein. |
| 1 | 5 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 2 | 5 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 3 | 5 turns | ✅ 1.00 | pass | Your dog is named **Einstein**. |
| 1 | 8 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 2 | 8 turns | ❌ 0.00 | hallucinated_fact | I am sorry, but I do not have access to your personal information, so I do not k |
| 3 | 8 turns | ❌ 0.00 | hallucinated_fact | I apologize, but as an AI, I do not have access to any personal information abou |
| 1 | 10 turns | ❌ 0.00 | hallucinated_fact | I am sorry, but I do not know the name of your dog. I am an AI and do not have a |
| 2 | 10 turns | ❌ 0.00 | hallucinated_fact | I apologize, but as an AI, I do not have access to any personal information abou |
| 3 | 10 turns | ❌ 0.00 | hallucinated_fact | I am sorry, but as an AI, I do not have access to any personal information about |
| 1 | 15 turns | ❌ 0.00 | hallucinated_fact | I do not know the name of your dog. I am an AI and do not have access to your pe |
| 2 | 15 turns | ❌ 0.00 | hallucinated_fact | I am sorry, but as an AI, I do not have access to any personal information about |
| 3 | 15 turns | ❌ 0.00 | hallucinated_fact | I apologize, but as an AI, I do not have access to any personal information abou |

**🛑 Adaptive stop triggered** after 6 depths (of 8 available).

### Persona Consistency

Tests whether the model maintains its assigned Dr. Sarah Chen / marine biologist persona after N turns.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ✅ 1.00 | pass | (Dr. Chen pauses, a soft, knowing look in her eyes, settling back into a thought |
| 2 | 1 turns | ✅ 1.00 | pass | (Dr. Chen pauses, a gentle, knowing smile touching her lips. Her tone is warm bu |
| 3 | 1 turns | ✅ 1.00 | pass | I am Dr. Sarah Chen.  I am a marine biologist, and my specialization lies in the |
| 1 | 5 turns | ✅ 1.00 | pass | (Dr. Chen pauses, her gaze softening slightly, a warm, yet deeply focused expres |
| 2 | 5 turns | ✅ 1.00 | pass | (Dr. Chen looks directly at you, a gentle, yet intense look in her eyes. There i |
| 3 | 5 turns | ✅ 1.00 | pass | Thank you for asking.  I am **Dr. Sarah Chen**.  I am a **marine biologist**, an |
| 1 | 10 turns | ❌ 0.00 | persona_drop | I am Gemma 4, a Large Language Model developed by Google DeepMind. I am an open  |
| 2 | 10 turns | ✅ 1.00 | pass | (Dr. Chen pauses, a warm, genuine expression settling on her face. She looks out |
| 3 | 10 turns | ❌ 0.00 | persona_drop | I am Gemma 4, a Large Language Model developed by Google DeepMind. I am an open  |
| 1 | 20 turns | ✅ 1.00 | pass | I am a large language model, and in this context, I am adopting the persona of a |
| 2 | 20 turns | ❌ 0.00 | persona_drop | I am Gemma 4, a Large Language Model developed by Google DeepMind. |
| 3 | 20 turns | ❌ 0.00 | empty_response | [empty/error] |
| 1 | 30 turns | ❌ 0.00 | persona_drop | I am Gemma 4, a Large Language Model developed by Google DeepMind.  I do not hav |
| 2 | 30 turns | ✅ 1.00 | pass | I am Dr. Sarah Chen.  I am a marine biologist specializing in deep-sea ecosystem |
| 3 | 30 turns | ❌ 0.00 | persona_drop | I am Gemma 4, a Large Language Model developed by Google DeepMind. |

### Recency Bias

Tests whether the model follows an early instruction ('Aye aye, captain!' prefix) or a later override, with varying distance between the two instructions.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ❌ 0.00 | ignore_instruction | I am sorry, but as an AI, I do not have access to real-time, location-specific i |
| 2 | 1 turns | ❌ 0.00 | ignore_instruction | I apologize, but as an AI, I do not have access to real-time, location-specific  |
| 3 | 1 turns | ❌ 0.00 | ignore_instruction | I am sorry, but as an AI, I do not have access to real-time, current information |
| 1 | 3 turns | ❌ 0.00 | ignore_instruction | I am sorry, but as an AI, I do not have access to real-time, current information |
| 2 | 3 turns | ❌ 0.00 | ignore_instruction | I am sorry, but as an AI, I do not have access to real-time, location-specific i |
| 3 | 3 turns | ❌ 0.00 | ignore_instruction | I do not have access to real-time information, including current local weather.  |

**🛑 Adaptive stop triggered** after 2 depths (of 6 available).

### Tool Call Drift

Tests whether JSON tool-call format degrades after many turns of alternating tool-call and normal responses.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 2 | 1 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 3 | 1 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 1 | 3 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 2 | 3 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 3 | 3 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 1 | 5 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 2 | 5 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 3 | 5 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 1 | 10 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 2 | 10 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 3 | 10 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 1 | 20 turns | ⚠️ 0.30 | partial_json | I do not have access to real-time weather information. I can only use the `get_w |
| 2 | 20 turns | ❌ 0.00 | plain_text | I do not have access to real-time weather information. |
| 3 | 20 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |

---

## Failure Category Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
| hallucinated_fact | 12 | 46.2% |
| ignore_instruction | 6 | 23.1% |
| persona_drop | 5 | 19.2% |
| partial_json | 1 | 3.8% |
| plain_text | 1 | 3.8% |
| empty_response | 1 | 3.8% |

---

## Scoring Guidelines

- **PASS** (≥ 0.80): Model maintains expected behavior
- **PARTIAL** (0.30 - 0.79): Some degradation observed
- **FAIL** (< 0.30): Significant breakdown

