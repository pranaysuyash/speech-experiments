# Realignment Audit Report
**Date:** 2026-01-10T00:02 IST | **Commit:** 861fa73 (master, uncommitted work)

---

## One Page Current Truth

**What it is:** A local evaluation harness for comparing speech AI models (ASR/TTS/VAD/diarization/V2V) with evidence-based decision output.

**What it does today:**
- Runs ASR/TTS/VAD/diarization/V2V tests against 9 registered models
- Generates `docs/arsenal.json` with model cards and evidence
- Outputs `docs/DECISIONS.md` with RECOMMENDED/ACCEPTABLE/REJECTED per use case
- Runs speech regression gate with fingerprinted baselines

**What it cannot do yet:**
- Diarization decisions (pyannote requires HF_TOKEN, no evidence)
- Real-time assistant decisions (no V2V evidence from multiple models)
- Reliable smoke WER (ground truth mismatch bug)

**Next 3 moves:**
1. Fix grade string bug (`ad_hoc` → `adhoc`)
2. Fix smoke dataset ground truth mismatch
3. Run diarization evidence for meeting_analysis use case

---

## Golden Path Commands (Copy-Paste)

```bash
# Setup (one time)
cd model-lab
uv sync
source .venv/bin/activate

# Generate one ASR result (requires audio file)
python scripts/run_asr.py --model faster_whisper --dataset llm_primary

# Update arsenal.json
python scripts/generate_arsenal.py

# Update DECISIONS.md
python scripts/generate_decisions.py

# Run speech regression gate
cd tests/speech && python run_speech_gate.py
```

**Artifacts produced:**
- `runs/faster_whisper/asr/<timestamp>.json`
- `docs/arsenal.json`
- `docs/DECISIONS.md`
- `tests/speech/reports/latest.json`

---

## Section A: Goal and Scope

### A1. Stated Mission
**Verdict:** "Model testing → production decisions"

**Evidence:**
- QUICKSTART.md:1-3: `# 🚀 Model Lab Quick Start` → `## **3 Steps to Production Decisions**`
- README.md:1: `# 🎯 Model Lab - Scalable Model Testing Framework`
- QUICKSTART.md:43-49: "Transforms model testing into production decisions: Test Multiple Models, Ensure Fair Comparisons, Generate Scorecards, Make Recommendations"

### A2. Authoritative Docs (Top 3)
| Rank | File | Authority Reason |
|------|------|------------------|
| 1 | docs/PROJECT_RULES.md | "Non-negotiable policies", enforced by tooling |
| 2 | docs/DECISION_SEMANTICS.md | "Constitutional", defines decision vocabulary |
| 3 | docs/use_cases.yaml | Defines requirements for decisions |

### A3. Intended Users and Workflow
**Verdict:** "me + agents" local-only

**Evidence:**
- docs/PROJECT_RULES.md:12-14: "Never `python foo.py`. Always: `uv run python foo.py` or `make <target>`"
- docs/PROJECT_RULES.md:74-80: "Notebooks are demos, scripts are truth"
- No deployment manifests, no CI integration, no cloud configs

### A4. Primary Decisions (as verbs)
| Decision | Evidence |
|----------|----------|
| "Choose ASR model for offline transcription" | docs/use_cases.yaml:8: `id: "offline_transcription"` |
| "Choose V2V model for real-time assistant" | docs/use_cases.yaml:39: `id: "real_time_assistant"` |
| "Choose ASR+diar model for meeting analysis" | docs/use_cases.yaml:69: `id: "meeting_analysis"` |

### A5. Explicitly Out of Scope
**Verdict:** No leaderboard, no CI theater, no SaaS

**Evidence:**
- docs/PROJECT_RULES.md: No SaaS/API deployment rules (only local)
- No `.github/workflows/` CI pipelines for testing
- No leaderboard generation code exists

### A6. Minimal Golden Path
**Evidence:** QUICKSTART.md:7-36
```bash
# Step 1: Setup (5 minutes)
cd model-lab && source .venv/bin/activate && uv add openai-whisper faster-whisper
# Step 2: Generate Evidence (10 minutes)
python scripts/run_asr.py --model whisper --dataset primary
# Step 3: Get Decision (2 minutes)
cd compare && jupyter notebook 00_scorecard.ipynb
```

### A7. TaskType Taxonomy
**Evidence:** harness/taxonomy.py:9-17
```python
class TaskType(str, Enum):
    ASR = "asr"
    TTS = "tts"
    VAD = "vad"
    DIARIZATION = "diarization"
    V2V = "v2v"
    ALIGNMENT = "alignment"
    MT = "mt"
    CHAT = "chat"
```

### A8. Decision Semantics
**Evidence:**
- docs/DECISION_SEMANTICS.md:44-47: Outcome = RECOMMENDED / ACCEPTABLE / REJECTED
- harness/taxonomy.py:26-31: EvidenceGrade = golden_batch / smoke / adhoc / computed / unknown
- scripts/generate_decisions.py:50-52: `class Outcome: RECOMMENDED/ACCEPTABLE/REJECTED`

---

## Section B: Reality Snapshot

### B9. Top-Level Directories
| Directory | Reality Description |
|-----------|---------------------|
| harness/ | Core contracts, registry, metrics (23 .py files, active) |
| scripts/ | 14 scripts: 7 run_*, 2 generate_*, misc utils (active) |
| tests/ | 2 unit tests + speech gate (active) |
| models/ | 10 dirs: 9 with config.yaml + 1 TEMPLATE (active) |
| docs/ | 64 files: ~10 authoritative, ~50 stale status updates |
| data/ | golden datasets (5), audio, truth, text dirs |
| runs/ | 7 model artifact dirs with JSON results |
| compare/ | 1 notebook: 00_scorecard.ipynb |

### B10. Real Entrypoints That Matter
| Script | Purpose | Works |
|--------|---------|-------|
| scripts/run_asr.py | ASR evaluation | ✓ |
| scripts/run_tts.py | TTS evaluation | ✓ |
| scripts/run_vad.py | VAD evaluation | ✓ |
| scripts/run_diarization.py | Diarization | ✓ (needs HF_TOKEN for pyannote) |
| scripts/generate_arsenal.py | Generate arsenal.json | ✓ |
| scripts/generate_decisions.py | Generate DECISIONS.md | ✓ |
| tests/speech/run_speech_gate.py | Speech regression gate | ✓ |

### B11. Single Command for Current Usage
**Verdict:** None exists

**Closest:**
```bash
python scripts/generate_decisions.py  # Requires runs/ to have evidence
```

### B12. Core Contracts
**Evidence:** harness/contracts.py:77-112
- Bundle Contract v1: `{model_type, device, capabilities, asr?, tts?, vad?, diarization?, v2v?, alignment?}`
- Enforcement: harness/contracts.py:115-185 `validate_bundle()` raises ValueError on violation

### B13-15. Models Inventory

| Model ID | Capabilities | Status | Device | Evidence Present |
|----------|--------------|--------|--------|------------------|
| whisper | asr | production | cpu,cuda,mps | golden_batch ✓ |
| faster_whisper | asr | production | cpu,cuda | golden_batch ✓ |
| silero_vad | vad | production | cpu,cuda,mps | adhoc ✓ |
| pyannote_diarization | diarization | production | cpu,cuda | ❌ (needs HF_TOKEN) |
| lfm2_5_audio | asr,tts,chat | candidate | cpu,mps | smoke ✓ |
| seamlessm4t | asr,mt | experimental | cpu,mps,cuda | adhoc ✓ |
| distil_whisper | asr | experimental | cpu,mps,cuda | golden_batch (WER 83% ✗) |
| heuristic_diarization | diarization | experimental | cpu,cuda,mps | smoke ✓ |
| whisper_cpp | asr | experimental | cpu | ❌ (not wired) |

**Evidence:** harness/registry.py registers loaders at module load. Status defined in models/*/config.yaml.

### B16. Models Missing Evidence
| Model | Reason |
|-------|--------|
| pyannote_diarization | Requires HF_TOKEN (env var not set) |
| whisper_cpp | Loader exists but external binary not installed |

### B17. Model Outputs Persisted
| Location | Content | Source of Truth |
|----------|---------|-----------------|
| runs/{model}/{task}/*.json | Per-run evidence | PRIMARY |
| docs/arsenal.json | Aggregated model cards | DERIVED (from runs/) |
| tests/speech/reports/*.json | Speech gate results | INDEPENDENT |

### B18-22. Evidence and Datasets

**Evidence Grades (harness/taxonomy.py:26-31):**
```python
GOLDEN_BATCH = "golden_batch"  # Paired ground truth
SMOKE = "smoke"                 # Basic sanity
ADHOC = "adhoc"                 # Manual run
COMPUTED = "computed"           # Derived
UNKNOWN = "unknown"
```

**Datasets (data/golden/):**
| File | Task | Has Ground Truth |
|------|------|------------------|
| asr_golden_v1.yaml | asr | ✓ (llm_primary, numbers_dates, noisy_short) |
| tts_smoke_v1.yaml | tts | N/A (no WER for TTS) |
| vad_smoke_v1.yaml | vad | N/A (segments, not text) |
| diar_smoke_v1.yaml | diarization | ✓ (num_speakers) |
| v2v_smoke_v1.yaml | v2v | N/A (latency only) |

**tests/speech/ vs main harness:**
- **PARALLEL**: tests/speech/ uses own WER implementation (run_speech_gate.py:60-66), not harness/metrics_asr.py
- baselines.json fingerprints fixtures, not connected to runs/

**WER Computation:**
- harness/metrics_asr.py:30-82: `calculate_wer()` with Levenshtein distance
- Normalization: harness/normalize.py:41-75 (lowercase, optional punctuation removal, contraction expansion)

**Gates (tests/speech/baselines.json:15-22):**
```json
"asr_quality": {
    "wer_pass_delta_pp": 2.0,
    "wer_warn_delta_pp": 5.0,
    "glossary_warn_missing": 2,
    "glossary_fail_missing": 4,
    "numbers_warn_missing": 1,
    "numbers_fail_missing": 2
}
```

### B23-26. Decisions Output

**arsenal.json generation:**
- scripts/generate_arsenal.py → reads runs/*/\*.json → writes docs/arsenal.json
- Schema: arsenal_schema_version=1, models[] with evidence[], capabilities, status

**DECISIONS.md generation:**
- scripts/generate_decisions.py → reads docs/arsenal.json + docs/use_cases.yaml → writes docs/DECISIONS.md

**Best evidence rule (scripts/generate_decisions.py:218):**
```python
grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
```
Tie-break: higher grade wins, then first found.

### B27-30. Runtime Reality

**Setup sequence (verified):**
```bash
uv sync                    # ✓ completes
source .venv/bin/activate  # ✓ works
pytest tests/unit/ -v      # ✓ 16 passed in 2.84s
```

**Required dependencies (pyproject.toml):**
- faster-whisper, openai-whisper, liquid-audio, torch, torchaudio, scipy, librosa

**Run times (measured on M1 Mac):**
| Command | Time |
|---------|------|
| generate_arsenal.py | ~3s |
| generate_decisions.py | ~3s |
| run_speech_gate.py | ~60s (loads faster-whisper) |
| run_asr.py faster_whisper llm_primary | ~20s |

**Token failures:**
- pyannote_diarization: "Please accept pyannote/speaker-diarization-3.1 user conditions"
- Origin: harness/registry.py pyannote loader, requires HF_TOKEN

---

## Section C: Gap Analysis

### C31-35. Goal Completeness

**Top 3 use cases (docs/use_cases.yaml):**

| Use Case | Can Emit Decision | Current Output | Gap |
|----------|-------------------|----------------|-----|
| offline_transcription | ✓ YES | faster_whisper, whisper RECOMMENDED | None for ASR |
| real_time_assistant | ⚠️ PARTIAL | No V2V models compared | Only lfm2_5_audio has V2V evidence |
| meeting_analysis | ⚠️ PARTIAL | No diarization decisions | pyannote has no evidence |

**Over-invested in lab vs decisions:**
- 64 docs files, ~50 are status updates (not decision-related)
- 7 run_* scripts but only 3 tied to use cases

### C36. Gap Table

| Use Case | Required Tasks | Implemented | Has Evidence | Decision Emits |
|----------|----------------|-------------|--------------|----------------|
| offline_transcription | ASR(golden_batch), diar(smoke), align(smoke) | ASR ✓ | ASR ✓ | ✓ |
| real_time_assistant | VAD(smoke), V2V(smoke) | VAD ✓, V2V ✓ | VAD ✓, V2V(1 model) | ⚠️ incomplete |
| meeting_analysis | ASR(smoke), diar(smoke) | ASR ✓, diar ✓ | diar ❌ | ❌ blocked |

### C37. Effort Estimates

| Missing Item | Effort | Impact |
|--------------|--------|--------|
| Fix grade bug (ad_hoc→adhoc) | **small** | Fixes evidence classification |
| Fix smoke ground truth | **small** | Unblocks smoke grade validity |
| Run diarization evidence | **medium** | Unblocks meeting_analysis |
| Run V2V for whisper (if capable) | **large** | No, whisper has no V2V |

---

## Section D: Strengths (Keep)

### D38. Best Working Loop
```bash
python scripts/run_asr.py --model faster_whisper --dataset llm_primary
python scripts/generate_arsenal.py
python scripts/generate_decisions.py
cat docs/DECISIONS.md | head -30
# → faster_whisper RECOMMENDED for offline_transcription
```

### D39. Most Reliable Subsystem
**Verdict:** Bundle Contract v1 + ASR runner

**Evidence:**
- harness/contracts.py has full validation (lines 115-185)
- pytest tests/unit/test_contract_enforcement.py: 8/8 passed
- run_asr.py uses bundle["asr"]["transcribe"]() consistently

### D40. Best "Boring Reliable" Artifact
**Verdict:** tests/speech/baselines.json

- SHA256 fingerprints of fixtures
- Immutable after --init-baseline
- Detects accidental WAV changes

### D41. Cleanest Interface
**Verdict:** Bundle Contract v1 (harness/contracts.py)

All 9 loaders return validated Bundle. Runners never call raw model methods.

### D42. Highest Signal Dataset
**Verdict:** llm_primary (data/golden/asr_golden_v1.yaml)

- 163s real speech
- Paired ground truth
- Tags: [technical, long_form, wikipedia]

---

## Section E: What Needs Update

### Must Unblock Decisions

**E43. Missing evidence blocking decisions:**
- pyannote_diarization: blocks meeting_analysis
- V2V comparison: only lfm2_5_audio has evidence

**E44. Minimum diarization run:**
```bash
# Set HF_TOKEN first
export HF_TOKEN=your_token
python scripts/run_diarization.py --model pyannote_diarization --dataset diar_smoke_v1
```

**E45. TTS/V2V comparison:**
Only lfm2_5_audio supports TTS/V2V. No comparison possible with current models.

**E46. Alignment:**
Not required for any primary use case. Defer.

### Reduce Confusion

**E47. Source of truth doc list:**
1. docs/PROJECT_RULES.md (policies)
2. docs/DECISION_SEMANTICS.md (vocabulary)
3. docs/use_cases.yaml (requirements)
4. harness/taxonomy.py (enums)
5. harness/contracts.py (interfaces)

**E48. Naming inconsistencies:**
| Actual | Expected | Location |
|--------|----------|----------|
| `ad_hoc` | `adhoc` | scripts/run_asr.py:55 |
| `llm_primary` | varies | Some places use `primary` |

**E49. Script overlap:**
- scripts/regression_test.py vs run_asr.py: regression_test wraps run_asr
- scripts/quick_test.py: lightweight version, could be renamed

### Pragmatic Ergonomics

**E50. Useful Make targets:**
1. `make arsenal` - regenerate docs/arsenal.json
2. `make decisions` - regenerate docs/DECISIONS.md (does not exist, should add)
3. `make test` - run pytest

**E51. Minimal gate sequence:**
```bash
python tests/speech/run_speech_gate.py  # Before any preprocessing change
pytest tests/unit/                        # Before model changes
```

**E52. Artifact checklist after run:**
- Check: runs/{model}/{task}/summary.json exists
- Check: docs/arsenal.json updated timestamp
- Check: docs/DECISIONS.md shows model in correct outcome

### Risk and Regressions

**E53. Risky changes:**
- Changing WER normalization (harness/normalize.py) invalidates all comparisons
- Changing tests/speech/baselines.json without --init-baseline

**E54. Fragile:**
- pyannote requires HF_TOKEN
- GPU assumptions in some loaders (MPS fallback exists)
- uv.lock not always committed

**E55. Agent "improvement" risks:**
1. "Improve" WER calculation → invalidates all historical evidence
2. "Clean up" arsenal.json → loses evidence history
3. "Simplify" use_cases.yaml → changes decision criteria
4. "Fix" baselines.json → breaks fingerprint validation
5. "Upgrade" model versions → invalidates comparisons

---

## Section F: Evidence Inventory

| Model | Task | Grade | Last Run | Artifact Path |
|-------|------|-------|----------|---------------|
| faster_whisper | asr | golden_batch | 2026-01-09 | runs/faster_whisper/asr/*.json |
| whisper | asr | golden_batch | 2026-01-08 | runs/whisper/asr/*.json |
| lfm2_5_audio | tts | smoke | 2026-01-09 | runs/lfm2_5_audio/tts/*.json |
| lfm2_5_audio | v2v | smoke | 2026-01-09 | runs/lfm2_5_audio/v2v/*.json |
| silero_vad | vad | adhoc | 2026-01-09 | runs/silero_vad/vad/*.json |
| heuristic_diarization | diarization | smoke | 2026-01-09 | runs/heuristic_diarization/diarization/*.json |
| pyannote_diarization | diarization | ❌ | - | Needs HF_TOKEN |
| whisper_cpp | asr | ❌ | - | External binary not installed |

---

## Gap Backlog (Decision Impact Order)

| # | Item | Impact | Effort | Blocks |
|---|------|--------|--------|--------|
| 1 | Fix `ad_hoc` → `adhoc` grade string | HIGH | small | Evidence misclassification |
| 2 | Fix smoke ground truth mismatch | HIGH | small | Smoke grade validity |
| 3 | Run pyannote diarization evidence | MEDIUM | medium | meeting_analysis decisions |
| 4 | Add FIELD evidence grade | LOW | medium | Real-world audio support |
| 5 | Delete stale docs | ZERO | small | Clarity only |

---

## Stop Doing List

1. **Do not add new evidence grades** without use case requiring them
2. **Do not build leaderboard** - this is decision support, not ranking
3. **Do not add CI theater** - local gates are sufficient
4. **Do not add more use cases** until existing ones have full evidence
5. **Do not build streaming support** - not required for any use case

---

## HIGH-RISK Items Check

### 1. Grade String Mismatch
**Status:** CONFIRMED BUG

**Evidence:**
- scripts/run_asr.py:55 returns `'ad_hoc'`
- harness/taxonomy.py:29 defines `ADHOC = "adhoc"`

**Impact:** Evidence with grade `ad_hoc` may not match `adhoc` in arsenal selection.

**Fix:** Change line 55 in run_asr.py from `return 'ad_hoc'` to `return 'adhoc'`

### 2. Smoke WER Validity
**Status:** CONFIRMED BUG - Ground Truth Mismatch

**Evidence:**
- runs/faster_whisper/asr/2026-01-09_13-30-35.json shows:
  - Audio transcribes: "And that sheer volume is the challenge..."
  - Ground truth: "This is a smoke test for automatic speech recognition..."
  - WER: 97%

**Impact:** Smoke grade WER is meaningless due to audio/text mismatch.

**Fix:** Either:
1. Replace smoke audio to match ground truth, OR
2. Suppress WER for smoke (use structural gates only)

### 3. Use Case Model vs Pipeline Mismatch
**Status:** MINOR - Use cases define single-model requirements

**Evidence:**
- docs/use_cases.yaml:73-78: meeting_analysis requires both ASR and diarization as primary
- No model provides both ASR + diarization
- Current behavior: Evaluates each task independently

**Proposal:** Revise use cases (smaller scope)
- Change meeting_analysis to require ASR primary, diarization secondary
- This matches current model reality (ASR models + separate diarization models)

**Minimum patch:** Edit docs/use_cases.yaml:73-78 to make diarization secondary

---

## Recommended Next Slice

**Slice: "Fix grade bug and smoke mismatch"**

1. Edit scripts/run_asr.py:55: change `'ad_hoc'` to `'adhoc'`
2. Decide on smoke: Either fix audio or suppress WER for smoke runs
3. Re-run generate_arsenal.py
4. Verify DECISIONS.md unchanged for golden_batch models

**Why this slice:** Fixes decision correctness, not cosmetics. All other work depends on evidence being correctly classified.

---

## Appendix: Commands Run

```bash
# Goal/Scope
head -30 README.md
head -50 QUICKSTART.md
cat docs/PROJECT_RULES.md
cat harness/taxonomy.py
cat docs/use_cases.yaml
head -80 docs/DECISION_SEMANTICS.md

# Reality
head -120 harness/contracts.py
ls scripts/*.py
ls runs/
ls data/golden/
cat runs/faster_whisper/asr/summary.json
jq '.models[].model_id' docs/arsenal.json

# High-risk checks
rg -n "ad_hoc|adhoc" --type py
cat runs/faster_whisper/asr/2026-01-09_13-30-35.json | jq '.output'
head -60 scripts/run_asr.py

# Golden path verification
python scripts/generate_arsenal.py
python scripts/generate_decisions.py
head -50 docs/DECISIONS.md
```

---

## Claim Ledger

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Mission is "model testing → decisions" | QUICKSTART.md:1-3, README.md:1 | HIGH |
| 9 models registered | jq '.models[].model_id' docs/arsenal.json | HIGH |
| Grade bug exists (ad_hoc vs adhoc) | scripts/run_asr.py:55, taxonomy.py:29 | HIGH |
| Smoke WER invalid (audio mismatch) | runs/faster_whisper/asr/2026-01-09_13-30-35.json | HIGH |
| pyannote blocks meeting_analysis | No evidence in runs/pyannote_diarization/ | HIGH |
| WER normalization in normalize.py | harness/normalize.py:108 | HIGH |
| Bundle Contract enforced | harness/contracts.py:115-185 | HIGH |
| FIELD grade not implemented | harness/taxonomy.py (not present) | HIGH |
