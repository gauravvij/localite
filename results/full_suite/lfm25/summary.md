# Full Suite Evaluation — LFM2.5-8B-A1B

**Date**: 2026-06-06T15:18:55.899134

**Model**: hf.co/unsloth/LFM2.5-8B-A1B-GGUF:UD-Q4_K_M

**Total test cases**: 102

## Summary Overview

| Dimension | Tests | Mean Score | Break Point |
|-----------|-------|------------|-------------|
| Hallucination Onset | 18 | 0.8333 | 8000 tokens |
| Instruction Adherence Decay | 24 | 0.7083 | 10 turns |
| Memory Retrieval | 24 | 0.5833 | 8 turns |
| Persona Consistency | 15 | 1.0000 | None (all ≥ 0.8) |
| Recency Bias | 6 | 0.0000 | 1 turns |
| Tool Call Drift | 15 | 0.2600 | 1 turns |

---

## Per-Dimension Results

### Hallucination Onset

**Tests**: 18 | **Mean Score**: 0.8333 | **Break Point**: 8000 tokens

| Depth | Mean Score | Runs |
|-------|-----------|------|
| 1000 | 1.0000 | 3 |
| 2000 | 1.0000 | 3 |
| 4000 | 1.0000 | 3 |
| 8000 | 0.6667 | 3 |
| 16000 | 0.6667 | 3 |
| 32000 | 0.6667 | 3 |

**Per-Test Results:**

| Run | Depth | Score | Category | Eval Time |
|-----|-------|-------|----------|-----------|
| 1 | 1000 | ✅ 1.00 | pass | 15.2s |
| 2 | 1000 | ✅ 1.00 | pass | 3.5s |
| 3 | 1000 | ✅ 1.00 | pass | 6.1s |
| 1 | 2000 | ✅ 1.00 | pass | 11.1s |
| 2 | 2000 | ✅ 1.00 | pass | 6.6s |
| 3 | 2000 | ✅ 1.00 | pass | 6.7s |
| 1 | 4000 | ✅ 1.00 | pass | 15.2s |
| 2 | 4000 | ✅ 1.00 | pass | 3.2s |
| 3 | 4000 | ✅ 1.00 | pass | 7.5s |
| 1 | 8000 | ❌ 0.00 | hallucinated_fact | 24.6s |
| 2 | 8000 | ✅ 1.00 | pass | 13.7s |
| 3 | 8000 | ✅ 1.00 | pass | 39.9s |
| 1 | 16000 | ❌ 0.00 | hallucinated_fact | 54.1s |
| 2 | 16000 | ✅ 1.00 | pass | 14.6s |
| 3 | 16000 | ✅ 1.00 | pass | 40.8s |
| 1 | 32000 | ❌ 0.00 | hallucinated_fact | 92.6s |
| 2 | 32000 | ✅ 1.00 | pass | 16.7s |
| 3 | 32000 | ✅ 1.00 | pass | 32.6s |

**Failure Category Distribution:**

| Category | Count |
|----------|-------|
| pass | 15 |
| hallucinated_fact | 3 |

---

### Instruction Adherence Decay

**Tests**: 24 | **Mean Score**: 0.7083 | **Break Point**: 10 turns

| Depth | Mean Score | Runs |
|-------|-----------|------|
| 1 | 1.0000 | 3 |
| 3 | 1.0000 | 3 |
| 5 | 1.0000 | 3 |
| 8 | 1.0000 | 3 |
| 10 | 0.3333 | 3 |
| 15 | 0.6667 | 3 |
| 20 | 0.3333 | 3 |
| 30 | 0.3333 | 3 |

**Per-Test Results:**

| Run | Depth | Score | Category | Eval Time |
|-----|-------|-------|----------|-----------|
| 1 | 1 | ✅ 1.00 | partial_json | 21.4s |
| 2 | 1 | ✅ 1.00 | partial_json | 19.1s |
| 3 | 1 | ✅ 1.00 | partial_json | 17.3s |
| 1 | 3 | ✅ 1.00 | partial_json | 23.2s |
| 2 | 3 | ✅ 1.00 | partial_json | 24.0s |
| 3 | 3 | ✅ 1.00 | partial_json | 26.2s |
| 1 | 5 | ✅ 1.00 | partial_json | 17.5s |
| 2 | 5 | ✅ 1.00 | partial_json | 22.6s |
| 3 | 5 | ✅ 1.00 | partial_json | 28.5s |
| 1 | 8 | ✅ 1.00 | partial_json | 25.7s |
| 2 | 8 | ✅ 1.00 | partial_json | 30.5s |
| 3 | 8 | ✅ 1.00 | partial_json | 35.1s |
| 1 | 10 | ✅ 1.00 | partial_json | 30.3s |
| 2 | 10 | ❌ 0.00 | plain_text | 28.8s |
| 3 | 10 | ❌ 0.00 | plain_text | 30.9s |
| 1 | 15 | ✅ 1.00 | partial_json | 38.9s |
| 2 | 15 | ❌ 0.00 | plain_text | 33.3s |
| 3 | 15 | ✅ 1.00 | partial_json | 39.5s |
| 1 | 20 | ❌ 0.00 | plain_text | 33.8s |
| 2 | 20 | ✅ 1.00 | partial_json | 56.8s |
| 3 | 20 | ❌ 0.00 | plain_text | 31.8s |
| 1 | 30 | ✅ 1.00 | partial_json | 44.0s |
| 2 | 30 | ❌ 0.00 | plain_text | 31.4s |
| 3 | 30 | ❌ 0.00 | plain_text | 33.4s |

**Failure Category Distribution:**

| Category | Count |
|----------|-------|
| partial_json | 17 |
| plain_text | 7 |

---

### Memory Retrieval

**Tests**: 24 | **Mean Score**: 0.5833 | **Break Point**: 8 turns

| Depth | Mean Score | Runs |
|-------|-----------|------|
| 1 | 1.0000 | 3 |
| 3 | 1.0000 | 3 |
| 5 | 1.0000 | 3 |
| 8 | 0.6667 | 3 |
| 10 | 0.6667 | 3 |
| 15 | 0.3333 | 3 |
| 20 | 0.0000 | 3 |
| 30 | 0.0000 | 3 |

**Per-Test Results:**

| Run | Depth | Score | Category | Eval Time |
|-----|-------|-------|----------|-----------|
| 1 | 1 | ✅ 1.00 | pass | 17.0s |
| 2 | 1 | ✅ 1.00 | pass | 17.8s |
| 3 | 1 | ✅ 1.00 | pass | 25.3s |
| 1 | 3 | ✅ 1.00 | pass | 45.9s |
| 2 | 3 | ✅ 1.00 | pass | 54.6s |
| 3 | 3 | ✅ 1.00 | pass | 94.7s |
| 1 | 5 | ✅ 1.00 | pass | 122.4s |
| 2 | 5 | ✅ 1.00 | pass | 148.2s |
| 3 | 5 | ✅ 1.00 | pass | 153.3s |
| 1 | 8 | ✅ 1.00 | pass | 162.3s |
| 2 | 8 | ✅ 1.00 | pass | 170.8s |
| 3 | 8 | ❌ 0.00 | hallucinated_fact | 394.1s |
| 1 | 10 | ✅ 1.00 | pass | 145.8s |
| 2 | 10 | ✅ 1.00 | pass | 175.1s |
| 3 | 10 | ❌ 0.00 | hallucinated_fact | 410.1s |
| 1 | 15 | ✅ 1.00 | pass | 247.9s |
| 2 | 15 | ❌ 0.00 | hallucinated_fact | 452.2s |
| 3 | 15 | ❌ 0.00 | hallucinated_fact | 527.2s |
| 1 | 20 | ❌ 0.00 | hallucinated_fact | 711.5s |
| 2 | 20 | ❌ 0.00 | hallucinated_fact | 596.3s |
| 3 | 20 | ❌ 0.00 | hallucinated_fact | 1278.0s |
| 1 | 30 | ❌ 0.00 | hallucinated_fact | 900.6s |
| 2 | 30 | ❌ 0.00 | hallucinated_fact | 996.1s |
| 3 | 30 | ❌ 0.00 | hallucinated_fact | 1246.5s |

**Failure Category Distribution:**

| Category | Count |
|----------|-------|
| pass | 14 |
| hallucinated_fact | 10 |

---

### Persona Consistency

**Tests**: 15 | **Mean Score**: 1.0000 | **Break Point**: None (all ≥ 0.8)

| Depth | Mean Score | Runs |
|-------|-----------|------|
| 1 | 1.0000 | 3 |
| 5 | 1.0000 | 3 |
| 10 | 1.0000 | 3 |
| 20 | 1.0000 | 3 |
| 30 | 1.0000 | 3 |

**Per-Test Results:**

| Run | Depth | Score | Category | Eval Time |
|-----|-------|-------|----------|-----------|
| 1 | 1 | ✅ 1.00 | pass | 79.2s |
| 2 | 1 | ✅ 1.00 | pass | 102.0s |
| 3 | 1 | ✅ 1.00 | pass | 44.9s |
| 1 | 5 | ✅ 1.00 | pass | 213.5s |
| 2 | 5 | ✅ 1.00 | pass | 350.9s |
| 3 | 5 | ✅ 1.00 | pass | 232.9s |
| 1 | 10 | ✅ 1.00 | pass | 466.7s |
| 2 | 10 | ✅ 1.00 | pass | 678.9s |
| 3 | 10 | ✅ 1.00 | pass | 488.4s |
| 1 | 20 | ✅ 1.00 | pass | 1058.6s |
| 2 | 20 | ✅ 1.00 | pass | 1042.7s |
| 3 | 20 | ✅ 1.00 | pass | 1024.4s |
| 1 | 30 | ✅ 1.00 | pass | 1826.9s |
| 2 | 30 | ✅ 1.00 | pass | 1624.9s |
| 3 | 30 | ✅ 1.00 | pass | 1467.1s |

**Failure Category Distribution:**

| Category | Count |
|----------|-------|
| pass | 15 |

---

### Recency Bias

**Tests**: 6 | **Mean Score**: 0.0000 | **Break Point**: 1 turns

| Depth | Mean Score | Runs |
|-------|-----------|------|
| 1 | 0.0000 | 3 |
| 3 | 0.0000 | 3 |

**Per-Test Results:**

| Run | Depth | Score | Category | Eval Time |
|-----|-------|-------|----------|-----------|
| 1 | 1 | ❌ 0.00 | follows_override | 27.6s |
| 2 | 1 | ❌ 0.00 | follows_override | 28.6s |
| 3 | 1 | ❌ 0.00 | follows_override | 39.0s |
| 1 | 3 | ❌ 0.00 | follows_override | 83.0s |
| 2 | 3 | ❌ 0.00 | follows_override | 70.2s |
| 3 | 3 | ❌ 0.00 | follows_override | 94.3s |

**Failure Category Distribution:**

| Category | Count |
|----------|-------|
| follows_override | 6 |

---

### Tool Call Drift

**Tests**: 15 | **Mean Score**: 0.2600 | **Break Point**: 1 turns

| Depth | Mean Score | Runs |
|-------|-----------|------|
| 1 | 0.1000 | 3 |
| 3 | 0.3000 | 3 |
| 5 | 0.3000 | 3 |
| 10 | 0.3000 | 3 |
| 20 | 0.3000 | 3 |

**Per-Test Results:**

| Run | Depth | Score | Category | Eval Time |
|-----|-------|-------|----------|-----------|
| 1 | 1 | ⚠️ 0.30 | partial_json | 6.1s |
| 2 | 1 | ❌ 0.00 | plain_text | 3.5s |
| 3 | 1 | ❌ 0.00 | plain_text | 3.6s |
| 1 | 3 | ⚠️ 0.30 | partial_json | 21.1s |
| 2 | 3 | ⚠️ 0.30 | partial_json | 34.4s |
| 3 | 3 | ⚠️ 0.30 | partial_json | 31.1s |
| 1 | 5 | ⚠️ 0.30 | partial_json | 33.4s |
| 2 | 5 | ⚠️ 0.30 | partial_json | 58.2s |
| 3 | 5 | ⚠️ 0.30 | partial_json | 40.9s |
| 1 | 10 | ⚠️ 0.30 | partial_json | 106.5s |
| 2 | 10 | ⚠️ 0.30 | partial_json | 83.1s |
| 3 | 10 | ⚠️ 0.30 | partial_json | 69.3s |
| 1 | 20 | ⚠️ 0.30 | partial_json | 148.5s |
| 2 | 20 | ⚠️ 0.30 | partial_json | 163.2s |
| 3 | 20 | ⚠️ 0.30 | partial_json | 195.8s |

**Failure Category Distribution:**

| Category | Count |
|----------|-------|
| partial_json | 13 |
| plain_text | 2 |

---

## Overall Failure Category Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
| pass | 44 | 43.1% |
| partial_json | 30 | 29.4% |
| hallucinated_fact | 13 | 12.7% |
| plain_text | 9 | 8.8% |
| follows_override | 6 | 5.9% |

## Scoring Guidelines

- **PASS** (≥ 0.80): Model maintains expected behavior
- **PARTIAL** (0.30 - 0.79): Some degradation observed
- **FAIL** (< 0.30): Significant breakdown
