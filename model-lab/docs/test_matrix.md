# Test Matrix v1.0
# Minimal, enforceable test gates for model arsenal

## ASR Regression Guard (Golden Set)

| Dataset | Duration | Tags | Pass Threshold | Method |
|---------|----------|------|----------------|--------|
| `llm_primary` | 163s | `technical` | WER < 35% | Automated |
| `numbers_dates` | 30-45s | `numbers, dates, names` | Entity F1 > 0.8 | Automated |
| `noisy_short` | 30-60s | `noisy, far-field` | WER < 50% | Automated |

### Diagnostic Gates
- `length_ratio` must be 0.7–1.3 (no truncation/hallucination)
- `repeat_3gram_rate` < 0.2 (no loops)
- `unique_token_ratio` > 0.4 (no repetition)

---

## TTS Selection (Human A/B)

| Dataset | Lines | Purpose | Metrics |
|---------|-------|---------|---------|
| `tech_script.txt` | 20 | Technical terms | RTF, clipping_ratio |
| `numbers_dates.txt` | 20 | Numerals, dates | RTF, silence_ratio |
| `paragraph.txt` | 1 long | Sustained output | duration stability |

### Automated Gates
- `silence_ratio` < 0.3
- `clipping_ratio` < 0.01
- `has_dc_offset` = false

### Human Eval
- 5 raters, 20 clips
- Paired A/B comparison
- Winner takes production slot

---

## Conversation Smoke (Manual Pass/Fail)

| Scenario | Turns | Checks |
|----------|-------|--------|
| `booking_flow` | 10 | constraint adherence, refusal case |

### Pass Criteria
- Follows instructions
- Refuses unsafe request
- Preserves entities across turns
- Valid JSON when asked

---

## Pass/Fail Summary

| Task | Automated | Human |
|------|-----------|-------|
| ASR | WER + diagnostics | None |
| TTS | Audio gates | A/B selection |
| Chat | Scenario pass/fail | Coherence rating |
