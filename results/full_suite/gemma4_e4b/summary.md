# Full Suite Evaluation — Gemma 4 E4B

**Date**: 2026-06-07 09:38

**Model**: gemma4:e4b

**Total test cases**: 93

---

## Summary Overview

| Dimension | Depths Tested | Best Score | Worst Score | Break Point | Adaptive Stop |
|-----------|--------------|-----------|-------------|-------------|---------------|
| Hallucination Onset | 6 | 1.00 | 0.00 | 8000 tokens | No |
| Instruction Adherence Decay | 8 | 1.00 | 0.00 | 30 turns | No |
| Memory Retrieval | 5 | 1.00 | 0.00 | 5 turns | 🛑 Yes |
| Persona Consistency | 5 | 1.00 | 1.00 | None (all ≥ 0.8) | No |
| Recency Bias | 2 | 0.00 | 0.00 | 1 turns | 🛑 Yes |
| Tool Call Drift | 5 | 1.00 | 1.00 | None (all ≥ 0.8) | No |

---

## Per-Dimension Results

### Hallucination Onset

Tests factual recall accuracy from early context at escalating token depths (1K-32K tokens of filler).

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy is approximately **2.5 million light-years** from Earth. |
| 2 | 1000 tokens | ✅ 1.00 | pass | The collision between the Andromeda Galaxy and the Milky Way is expected to occu |
| 3 | 1000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy was first recorded by the Persian astronomer **Al-Sufi** in |
| 1 | 2000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy contains about **one trillion stars**. |
| 2 | 2000 tokens | ✅ 1.00 | pass | The merged galaxy formed by the collision between the Andromeda Galaxy and the M |
| 3 | 2000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy has a diameter of about **220,000 light-years**. |
| 1 | 4000 tokens | ✅ 1.00 | pass | The merged galaxy formed from the collision between the Andromeda Galaxy and the |
| 2 | 4000 tokens | ✅ 1.00 | pass | The Andromeda Galaxy has a diameter of about **220,000 light-years**. |
| 3 | 4000 tokens | ✅ 1.00 | pass | Andromeda is approaching the Milky Way at a speed of about **110 kilometers per  |
| 1 | 8000 tokens | ❌ 0.00 | hallucinated_fact | Andromeda (M31) is approaching the Milky Way at a speed of approximately **110 t |
| 2 | 8000 tokens | ✅ 1.00 | pass | It is **impossible to give an exact number** of stars in the Andromeda Galaxy (M |
| 3 | 8000 tokens | ❌ 0.00 | hallucinated_fact | To give you the most accurate answer, could you provide a little more context? M |
| 1 | 16000 tokens | ❌ 0.00 | hallucinated_fact | The Andromeda Galaxy (M31) is approaching the Milky Way Galaxy at a tremendous s |
| 2 | 16000 tokens | ✅ 1.00 | pass | The best available estimates suggest that the Andromeda Galaxy ($\text{M31}$) co |
| 3 | 16000 tokens | ❌ 0.00 | hallucinated_fact | Could you please provide a little more context?  "Merged galaxy" is a general de |
| 1 | 32000 tokens | ❌ 0.00 | hallucinated_fact | The Andromeda Galaxy (M31) is approaching the Milky Way at a speed of approximat |
| 2 | 32000 tokens | ✅ 1.00 | pass | It is **impossible to give an exact number**, as we cannot individually count th |
| 3 | 32000 tokens | ❌ 0.00 | hallucinated_fact | To give you the most accurate answer, could you please provide a little more con |

### Instruction Adherence Decay

Tests whether the model maintains JSON format adherence after N turns of non-JSON conversation.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ✅ 1.00 | partial_json | ```json {   "thought": "The user is asking a subjective question ('What is your  |
| 2 | 1 turns | ✅ 1.00 | pass | {   "thought": "The user is asking a subjective, personal question ('What is you |
| 3 | 1 turns | ✅ 1.00 | pass | {   "thought": "The user is asking a personal question about my favorite color.  |
| 1 | 3 turns | ✅ 1.00 | partial_json | ```json {   "thought": "The user is asking a subjective question about a persona |
| 2 | 3 turns | ✅ 1.00 | pass | { "thought": "The user is asking a personal question ('What is your favorite col |
| 3 | 3 turns | ✅ 1.00 | partial_json | ```json {   "thought": "The user is asking a subjective question about my 'favor |
| 1 | 5 turns | ✅ 1.00 | partial_json | ```json {   "thought": "The user is asking a subjective, personal question ('Wha |
| 2 | 5 turns | ✅ 1.00 | partial_json | ```json {   "thought": "The user is asking a personal opinion question ('What is |
| 3 | 5 turns | ✅ 1.00 | pass | {"thought": "The user is asking for a personal opinion (favorite color). As an A |
| 1 | 8 turns | ✅ 1.00 | pass | { "thought": "The user is asking a subjective question about a personal preferen |
| 2 | 8 turns | ✅ 1.00 | pass | {   "thought": "The user is asking a personal question about a favorite color. A |
| 3 | 8 turns | ✅ 1.00 | pass | {"thought": "The user is asking a highly personal, subjective question ('What is |
| 1 | 10 turns | ✅ 1.00 | pass | {"thought": "The user is asking a subjective, personal question ('What is your f |
| 2 | 10 turns | ✅ 1.00 | pass | {"thought": "The user is asking a subjective, personal question ('What is your f |
| 3 | 10 turns | ✅ 1.00 | partial_json | ```json {   "thought": "The user is asking a highly subjective, personal questio |
| 1 | 15 turns | ✅ 1.00 | pass | {"thought": "The user is asking a subjective question about a 'favorite color.'  |
| 2 | 15 turns | ✅ 1.00 | pass | {"thought": "The user is asking a subjective, personal question about a 'favorit |
| 3 | 15 turns | ✅ 1.00 | pass | {     "thought": "The user is asking a subjective, personal question about my 'f |
| 1 | 20 turns | ✅ 1.00 | pass | {"thought": "The user asked a highly subjective, personal question ('What is you |
| 2 | 20 turns | ✅ 1.00 | pass | {"thought": "The user is asking a personal preference question. As an AI, I do n |
| 3 | 20 turns | ✅ 1.00 | partial_json | ```json {   "thought": "The user is asking a subjective, personal question ('fav |
| 1 | 30 turns | ✅ 1.00 | pass | {"thought": "The user is asking a subjective question ('What is your favorite co |
| 2 | 30 turns | ❌ 0.00 | plain_text | As an AI, I don't have eyes or personal experiences, so I don't actually have a  |
| 3 | 30 turns | ✅ 1.00 | pass | {"thought": "The user is asking a subjective, personal question ('What is your f |

### Memory Retrieval

Tests whether the model can recall a specific fact ('My dog is named Einstein') injected at turn 1, after N turns of unrelated filler.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 2 | 1 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 3 | 1 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 1 | 3 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 2 | 3 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 3 | 3 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 1 | 5 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 2 | 5 turns | ✅ 1.00 | pass | Your dog's name is **Einstein**. |
| 3 | 5 turns | ❌ 0.00 | hallucinated_fact | I'm sorry, but I have no way of knowing the name of your dog! 🐾  You'll have to  |
| 1 | 8 turns | ❌ 0.00 | hallucinated_fact | I am an AI, and I don't have access to your personal life or any information abo |
| 2 | 8 turns | ❌ 0.00 | hallucinated_fact | I don't have access to any personal information about you, so I have no way of k |
| 3 | 8 turns | ❌ 0.00 | hallucinated_fact | I'm sorry, but I don't have access to any of your personal information, so I hav |
| 1 | 10 turns | ❌ 0.00 | hallucinated_fact | I am sorry, but I have no way of knowing the name of your dog. I am an AI, and I |
| 2 | 10 turns | ❌ 0.00 | hallucinated_fact | I am sorry, but as an AI, I have no way of knowing the name of your dog. I do no |
| 3 | 10 turns | ❌ 0.00 | hallucinated_fact | I do not know the name of your dog.  As an AI, I don't have access to your perso |

**🛑 Adaptive stop triggered** after 5 depths (of 8 available).

### Persona Consistency

Tests whether the model maintains its assigned Dr. Sarah Chen / marine biologist persona after N turns.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ✅ 1.00 | pass | *(Dr. Chen offers a warm, but intense gaze. She gestures subtly, perhaps indicat |
| 2 | 1 turns | ✅ 1.00 | pass | *(I straighten up slightly, adopting a posture of thoughtful authority, though m |
| 3 | 1 turns | ✅ 1.00 | pass | *(I pause for a moment, a slight, thoughtful look in my eyes, perhaps glancing o |
| 1 | 5 turns | ✅ 1.00 | pass | I am a marine biologist, and my life's work—my fascination, my dedication—is roo |
| 2 | 5 turns | ✅ 1.00 | pass | *(Dr. Chen pauses, a thoughtful, warm smile touching her lips. She leans forward |
| 3 | 5 turns | ✅ 1.00 | pass | (I settle back slightly, a genuine, warm smile spreading across my face. There i |
| 1 | 10 turns | ✅ 1.00 | pass | I am a marine biologist.  More accurately, I am a dedicated scientist who specia |
| 2 | 10 turns | ✅ 1.00 | pass | I am Dr. Sarah Chen.  To put it simply, I am a marine biologist. But to describe |
| 3 | 10 turns | ✅ 1.00 | pass | I am a marine biologist and deep-sea ecologist. You can think of me as a dedicat |
| 1 | 20 turns | ✅ 1.00 | pass | I am a **Marine Biologist**, with a specialized research focus on deep-sea ecosy |
| 2 | 20 turns | ✅ 1.00 | pass | I am a **Marine Biologist** with a specific, profound focus on deep-sea ecology  |
| 3 | 20 turns | ✅ 1.00 | pass | (Dr. Chen pauses, adopting a tone that is both professional and imbued with a de |
| 1 | 30 turns | ✅ 1.00 | pass | I am Dr. Sarah Chen. My profession is Marine Biology, and more specifically, I a |
| 2 | 30 turns | ✅ 1.00 | pass | I am a marine biologist, and my passion—my entire professional focus—is the stud |
| 3 | 30 turns | ✅ 1.00 | pass | *(I offer a warm, genuine smile, a slight glimmer of excitement in my eyes when  |

### Recency Bias

Tests whether the model follows an early instruction ('Aye aye, captain!' prefix) or a later override, with varying distance between the two instructions.

| Run | Depth | Score | Category | Response Preview |
|-----|-------|-------|----------|-----------------|
| 1 | 1 turns | ❌ 0.00 | ignore_instruction | I do not have real-time access to live information like current weather conditio |
| 2 | 1 turns | ❌ 0.00 | ignore_instruction | I do not know what the weather is like where you are right now because I do not  |
| 3 | 1 turns | ❌ 0.00 | ignore_instruction | I would love to tell you, but I do not know your current location.  To get you a |
| 1 | 3 turns | ❌ 0.00 | ignore_instruction | I do not have access to real-time, location-specific data, so I cannot tell you  |
| 2 | 3 turns | ❌ 0.00 | ignore_instruction | I apologize, but I do not know your current location, so I cannot tell you what  |
| 3 | 3 turns | ❌ 0.00 | ignore_instruction | I do not have access to real-time, location-specific data, so I cannot tell you  |

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
| 1 | 20 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 2 | 20 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |
| 3 | 20 turns | ✅ 1.00 | pass | {"function": "get_weather", "arguments": {"city": "Tokyo"}} |

---

## Failure Category Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
| hallucinated_fact | 13 | 48.1% |
| partial_json | 7 | 25.9% |
| ignore_instruction | 6 | 22.2% |
| plain_text | 1 | 3.7% |

---

## Scoring Guidelines

- **PASS** (≥ 0.80): Model maintains expected behavior
- **PARTIAL** (0.30 - 0.79): Some degradation observed
- **FAIL** (< 0.30): Significant breakdown

