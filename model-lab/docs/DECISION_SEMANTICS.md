# Decision Semantics Contract

**Version:** 1.0  
**Status:** Constitutional  
**Purpose:** Define how Arsenal interprets evidence to make use-case-scoped recommendations.

---

## Core Principle

> **Evidence must be strict. Decisions must be tolerant.**

Arsenal collects evidence without compromise. It interprets evidence with context.

---

## Section 1: Vocabulary (Frozen)

### Evidence
A single unit of observed behavior from a model run. Immutable. Contains:
- Metrics (WER, latency, coverage, etc.)
- Gates (validity signals)
- Metadata (dataset, device, timestamp)

### Gate
A boolean or scalar signal derived from evidence. Examples:
- `wer_valid`
- `is_hallucinating`
- `coverage_ratio`

### Severity
How a gate is interpreted within a use case. Three levels:
- **PASS**: Healthy signal
- **WARN**: Degraded but acceptable
- **FAIL_FATAL**: Disqualifying for this use case

### Use Case
A product scenario with specific requirements. Examples:
- Offline Transcription
- Meeting Intelligence
- Real-Time Voice Assistant

### Outcome
The decision result for a model in a use case:
- **RECOMMENDED**
- **ACCEPTABLE**
- **REJECTED**

---

## Section 2: Gate Severity Model

### Key Rule
> **Gates emit signals. Use cases decide severity.**

A gate is never inherently fatal. Context determines meaning.

### Severity Definitions

#### PASS
- Signal indicates healthy behavior
- Adds positive weight to model score
- Example: `coverage_ratio > 0.95`

#### WARN
- Signal indicates degradation or imperfection
- Reduces confidence score
- Does NOT disqualify alone
- Example: `wer > 0.3` in conversational context

#### FAIL_FATAL
- Signal indicates fundamental unsuitability
- Causes immediate rejection for this use case
- Example: `is_hallucinating = true` in production transcription

### Severity Assignment
Severity is assigned **per use case**, not globally.

Same gate, different severities:

| Gate | Transcription Benchmark | Meeting Notes | Voice Assistant |
|------|------------------------|---------------|-----------------|
| `wer_valid = false` | FAIL_FATAL | WARN | WARN |
| `is_hallucinating = true` | FAIL_FATAL | FAIL_FATAL | FAIL_FATAL |
| `is_truncated = true` | FAIL_FATAL | WARN | WARN |
| `coverage < 0.6` | FAIL_FATAL | FAIL_FATAL | WARN |

---

## Section 3: Mapping Existing Gates

Arsenal already collects these gates. We now define their default severity mappings.

### ASR Gates

#### `wer_valid` (from sanity_gates)
- **What it signals:** Lexical accuracy within expected bounds
- **Severity by use case:**
  - Clean verbatim transcription → FAIL_FATAL
  - Meeting notes → WARN
  - Conversational agent → WARN
  - Voice search → WARN

#### `is_truncated` (from output_quality)
- **What it signals:** Output shorter than expected
- **Severity by use case:**
  - Long-form transcription → FAIL_FATAL
  - Short assistant responses → WARN
  - Casual notes → WARN

#### `is_hallucinating` (from output_quality)
- **What it signals:** Output longer than input or repetitive
- **Severity by use case:**
  - **ANY production use → FAIL_FATAL**
  - Hallucination is universally disqualifying

### Alignment Gates

#### `monotonicity_violations > 0`
- **What it signals:** Timestamp ordering broken
- **Severity by use case:**
  - Subtitle generation → FAIL_FATAL
  - Search indexing → FAIL_FATAL
  - Casual playback → WARN

#### `coverage_ratio < threshold`
- **What it signals:** Large portions of audio unrepresented
- **Severity by use case:**
  - Search/indexing → FAIL_FATAL
  - Meeting analysis → FAIL_FATAL
  - Casual captions → WARN

### V2V Gates

#### `latency > max_latency_ms`
- **What it signals:** Response too slow
- **Severity by use case:**
  - Real-time assistant → FAIL_FATAL
  - Batch processing → WARN

#### `audio_duration = 0`
- **What it signals:** No audio generated
- **Severity by use case:**
  - **ANY V2V use → FAIL_FATAL**

### Diarization Gates

#### `speaker_count_error > 1`
- **What it signals:** Wrong number of speakers detected
- **Severity by use case:**
  - Meeting transcription → FAIL_FATAL
  - Casual conversation → WARN

---

## Section 4: Decision Outcomes

### RECOMMENDED
**Definition:** Model passes all FAIL_FATAL gates and has minimal warnings.

**Criteria:**
- Zero FAIL_FATAL violations
- WARN count ≤ 1
- Primary capability has valid evidence

**Display:**
- ✅ Icon
- "Best choice for this use case"
- List strengths

### ACCEPTABLE
**Definition:** Model passes all FAIL_FATAL gates but has degraded signals.

**Criteria:**
- Zero FAIL_FATAL violations
- WARN count > 1
- Primary capability has valid evidence

**Display:**
- ⚠️ Icon
- "Usable with trade-offs"
- List warnings explicitly
- Explain what user must accept

### REJECTED
**Definition:** Model violates one or more FAIL_FATAL gates.

**Criteria:**
- One or more FAIL_FATAL violations

**Display:**
- ❌ Icon
- "Not suitable for this use case"
- List blockers (FAIL_FATAL reasons only)
- Do NOT list warnings for rejected models

### No Viable Models
**Special case:** ALL models are REJECTED.

**Display:**
- "No models meet requirements"
- List most common FAIL_FATAL reasons
- Suggest use case relaxation

---

## Section 5: Decision Explanation Contract

Every decision must answer four questions:

1. **Why this outcome?**
   - Which gates determined the result?

2. **What failed?**
   - FAIL_FATAL violations (for REJECTED)
   - Major warnings (for ACCEPTABLE)

3. **What worked?**
   - PASS gates
   - Strengths

4. **What trade-offs are accepted?**
   - For ACCEPTABLE: explicit list of warnings
   - For RECOMMENDED: "None" or "Minimal"

### Example Explanation (ACCEPTABLE)

```
Model: whisper
Outcome: ⚠️ ACCEPTABLE

Why: Passes core functionality but has quality warnings.

What worked:
- ✅ Coverage: 98% (excellent)
- ✅ No hallucination detected
- ✅ Timestamps monotonic

What you must accept:
- ⚠️ WER: 45% (conversational compression)
- ⚠️ Truncated output (dropped fillers)

Trade-off: Usable for meeting notes if verbatim accuracy is not critical.
```

---

## Section 6: Example Walkthrough

### Scenario: Your 2-Minute Real Recording

**Evidence (same for all models):**
- Audio: 2 minutes, conversational, some background noise
- No ground truth (real-world recording)

**Models evaluated:**
- Whisper
- Faster-Whisper
- LFM2.5-Audio

---

### Use Case 1: Offline Transcription (Verbatim)

**Requirements:**
- Primary: ASR
- Constraints: offline_capable = true
- Fatal gates: `wer_valid`, `is_hallucinating`, `is_truncated`

**Results:**

| Model | Outcome | Reason |
|-------|---------|--------|
| Whisper | ❌ REJECTED | `wer_valid = false` (FAIL_FATAL) |
| Faster-Whisper | ❌ REJECTED | `wer_valid = false` (FAIL_FATAL) |
| LFM2.5-Audio | ❌ REJECTED | Missing ASR capability declaration |

**Explanation:** Real conversational audio violates verbatim transcription requirements. Use case mismatch.

---

### Use Case 2: Meeting Notes (Conversational)

**Requirements:**
- Primary: ASR
- Constraints: offline_capable = false (cloud OK)
- Fatal gates: `is_hallucinating`, `coverage < 0.6`
- Warning gates: `wer_valid`, `is_truncated`

**Results:**

| Model | Outcome | Reason |
|-------|---------|--------|
| Whisper | ⚠️ ACCEPTABLE | WER warning, but coverage good |
| Faster-Whisper | ⚠️ ACCEPTABLE | WER warning, but coverage good |
| LFM2.5-Audio | ❌ REJECTED | Missing ASR capability declaration |

**Explanation (Whisper):**
```
✅ Coverage: 95%
✅ No hallucination
⚠️ WER: 45% (conversational compression)

Trade-off: Usable if you accept paraphrasing over verbatim.
```

---

### Use Case 3: Real-Time Voice Assistant

**Requirements:**
- Primary: VAD + V2V
- Constraints: max_latency_ms = 1000
- Fatal gates: `latency > 1000`, `audio_duration = 0`

**Results:**

| Model | Outcome | Reason |
|-------|---------|--------|
| Whisper | ❌ REJECTED | Missing VAD + V2V capabilities |
| Faster-Whisper | ❌ REJECTED | Missing VAD + V2V capabilities |
| LFM2.5-Audio | ❌ REJECTED | Missing VAD capability |

**Explanation:** No model satisfies the primary capability requirements.

---

## Section 7: Implementation Checklist

To implement this semantics:

1. **Update `use_cases.yaml`:**
   - Add `fatal_gates` list per use case
   - Add `warning_gates` list per use case

2. **Update `generate_decisions.py`:**
   - Replace binary `valid` check with severity evaluation
   - Compute RECOMMENDED / ACCEPTABLE / REJECTED
   - Generate explanations per Section 5

3. **Update `DECISIONS.md` format:**
   - Show all three outcome types
   - Include trade-off explanations
   - Rank by outcome, then score

4. **No changes to evidence collection:**
   - Runners stay strict
   - Gates stay unchanged
   - Evidence schema unchanged

---

## Section 8: Invariants (Never Violate)

1. **Evidence is immutable.**
   - Never reinterpret or soften gate values.

2. **Severity is use-case-scoped.**
   - Never define "global" fatal gates.

3. **Hallucination is always fatal.**
   - Exception: none.

4. **Outcomes are deterministic.**
   - Same evidence + same use case = same outcome.

5. **Explanations are mandatory.**
   - No outcome without reasons.

---

## Conclusion

This contract defines **how Arsenal thinks**, not what it measures.

Evidence collection remains unforgiving.  
Decision interpretation becomes contextual.

Once implemented:
- Real-world audio stops being auto-rejected
- Models are judged where they shine
- Trade-offs become explicit
- Arsenal becomes a search engine, not a courtroom

**Next:** Implement in `generate_decisions.py` and `use_cases.yaml`.

---

## Section 9: NA Gates and Field Evidence (Future)

### The Problem

Real-world recordings often have **no ground truth**, making WER uncomputable.

Current system treats "WER missing" as "bad" → everything gets rejected.

### The Solution: NA (Not Applicable)

Add fourth gate result type:
- PASS
- WARN
- FAIL_FATAL
- **NA** (not applicable / not computable)

### Field Evidence Grade

Add `EvidenceGrade.FIELD` for real-world recordings:

**Characteristics:**
- No ground truth (`truth_sha256: null`)
- No WER (`wer_valid: NA`)
- Robustness gates only:
  - `coverage_ok` (timestamps cover X% of audio duration)
  - `is_hallucinating` (heuristic: length explosion, duplicate n-grams)
  - `is_truncated` (heuristic: ends mid-sentence, early stop)
  - `is_repetitive` (heuristic: repeated loops, token cycles)
  - `latency_ok` (RTF, p95 spikes)

**Use case mapping:**
- Verbatim transcription: FIELD evidence cannot yield RECOMMENDED (WER unknown)
- Meeting notes: FIELD evidence can yield ACCEPTABLE (robustness sufficient)
- Voice assistant: FIELD evidence can yield RECOMMENDED (latency + robustness)

### Implementation Checklist

1. **Add `EvidenceGrade.FIELD`** to `harness/taxonomy.py`
2. **Create field runner** (`scripts/run_asr_field.py`):
   - No ground truth required
   - Compute robustness gates only
   - Store `grade: field`
3. **Update gate evaluation** to support NA:
   - `if gate not present: return NA or FAIL_FATAL (per use case)`
4. **Add use case** "Meeting Notes (Field)":
   - `min_grade: field` for ASR
   - `allow_na_gates: [wer_valid]`
   - `fatal_gates: [is_hallucinating, coverage_low]`

This enables honest evaluation of real-world audio without forcing fake WER.
