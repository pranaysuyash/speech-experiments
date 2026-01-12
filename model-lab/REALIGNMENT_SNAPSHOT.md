# REALIGNMENT_SNAPSHOT.md
**Date:** 2026-01-09T23:02 IST | **Commit:** 861fa73 (master, uncommitted)

---

## Agent A: Repo Map and Runtime Entrypoints

### Scope
Map executables, speech test integration, evidence runners, minimum e2e path.

### Commands Run
```
tree -L 2 -d
rg -l "if __name__|argparse" --type py
ls scripts/*.py
rg -l "golden|smoke|adhoc|field" --type py
```

### Observed Outputs
- 14 scripts in `scripts/` directory
- 7 `run_*` entrypoints: asr, tts, vad, diarization, v2v, alignment, + regression_test
- Speech gate: `tests/speech/run_speech_gate.py` (standalone, not pytest)
- Evidence runners use grades: `golden_batch`, `smoke`, `adhoc` (FIELD not implemented)

### Files Touched
None

### Broken or Missing
- `EvidenceGrade.FIELD` documented but not in taxonomy.py (only golden_batch, smoke, adhoc, computed, unknown)
- `scripts/run_asr_field.py` does not exist (mentioned in DECISION_SEMANTICS.md Section 9)

### Next Actions
1. Add `FIELD` to `EvidenceGrade` enum if needed
2. Decide if field runner is required for v1

---

## Agent B: Tests and Gates Inventory

### Scope
Inventory all tests/gates with how to run, assertions, dependencies, determinism.

### Commands Run
```
find tests -type f -name "*.py"
rg -l "pytest|unittest" --type py
cat pyproject.toml | grep dependencies
```

### Observed Outputs

| Test | Path | How to Run | Dependencies | Deterministic |
|------|------|------------|--------------|---------------|
| Arsenal Docs | `tests/unit/test_arsenal_docs.py` | `pytest tests/unit/` | arsenal.json exists | Yes |
| Contract Enforcement | `tests/unit/test_contract_enforcement.py` | `pytest tests/unit/` | harness.contracts | Yes |
| Speech Gate | `tests/speech/run_speech_gate.py` | `python run_speech_gate.py` | ffmpeg, faster-whisper, fixtures | Yes (fingerprinted) |

- pytest in `[project.optional-dependencies] dev`, not main deps
- pre-commit configured but not enforced in CI

### Files Touched
None

### Broken or Missing
- pytest not installed in main venv (need `uv sync --all-extras` or `uv add pytest`)
- No CI pipeline (`.github/` exists but minimal)
- pre-commit hooks not installed locally

### Next Actions
1. Add pytest to main dependencies or document dev install
2. Run `pre-commit install` or remove from deps

---

## Agent C: Arsenal Decision Engine Status

### Scope
Audit taxonomy, evidence selection, decision semantics, mismatches.

### Commands Run
```
rg "EvidenceGrade|RECOMMENDED|ACCEPTABLE|REJECTED" --type py
cat docs/DECISION_SEMANTICS.md
python scripts/generate_decisions.py
```

### Observed Outputs
- **Taxonomy** (`harness/taxonomy.py`):
  - TaskType: asr, tts, vad, diarization, v2v, alignment, mt, chat
  - EvidenceGrade: golden_batch, smoke, adhoc, computed, unknown
  - TaskRole: primary, secondary, undeclared

- **Decision outcomes** (`scripts/generate_decisions.py`):
  - RECOMMENDED, ACCEPTABLE, REJECTED (implemented correctly)
  - Evidence priority: golden_batch > smoke > adhoc

- **Use cases** (`docs/use_cases.yaml`):
  - offline_transcription: min_grade=golden_batch for ASR
  - real_time_assistant: min_grade=smoke for V2V/VAD
  - meeting_analysis: min_grade=smoke

### Files Touched
None

### Broken or Missing
- `FIELD` grade: documented in DECISION_SEMANTICS.md Section 9, not in code
- `NA` gate result: documented, not implemented
- `run_asr_field.py`: documented, does not exist
- docs/DECISION_SEMANTICS.md says "Version: 1.0" but code says "v2.0"

### Next Actions
1. Reconcile version numbers (DECISION_SEMANTICS.md vs generate_decisions.py)
2. Decide: implement FIELD grade now or defer to v2
3. Add NA gate support if real-world audio testing is priority

---

## Agent D: Backlog Extraction

### Scope
Extract TODOs, failing commands, prioritized backlog.

### Commands Run
```
rg "TODO|FIXME|HACK|XXX" --type py
python scripts/generate_arsenal.py
python scripts/generate_decisions.py
```

### Observed Outputs
- Only 1 TODO: `scripts/run_tts.py:193` - "add git hash"
- generate_arsenal.py: ✓ runs clean
- generate_decisions.py: ✓ runs clean (1 warning: TEMPLATE model skipped)

### Files Touched
None

### Broken or Missing
- 50+ stale docs (CHATGPT_*, FINAL_*, etc.) cluttering docs/
- ~50 uncommitted files (entire tests/, new models, new scripts)

### Next Actions
1. Commit current work to a feature branch
2. Delete stale docs (or archive to docs/archive/)
3. Fix the single TODO in run_tts.py

---

## Merged Summary

### ✅ Current Working Commands (Known-Good)

| Command | Status | Notes |
|---------|--------|-------|
| `python -c "from harness import ..."` | ✅ | All imports work |
| `make list-models` | ✅ | 9 models registered |
| `python scripts/generate_arsenal.py` | ✅ | Generates arsenal.json, ARSENAL.md |
| `python scripts/generate_decisions.py` | ✅ | Generates DECISIONS.md |
| `python tests/speech/run_speech_gate.py` | ✅ | PASS, 3 fixtures |
| `python scripts/run_asr.py --model faster_whisper --dataset smoke` | ✅ | Requires audio file |

### ❌ Broken Commands (With Errors)

| Command | Error | Fix |
|---------|-------|-----|
| `pytest tests/` | ModuleNotFoundError: pytest | `uv add pytest` or `uv sync --all-extras` |
| `make test` | Same as above | Same |
| `pre-commit run` | Not installed | `pre-commit install` |

### 📋 Backlog Triage

#### P0: Must-Fix to Run
1. **Add pytest to venv** - `uv add pytest` (blocks unit tests)

#### P1: Must-Fix to Be Trustworthy
1. **Commit uncommitted work** - ~50 files, create feature branch
2. **Speech fixture/glossary mismatch** - voice1 WER 35%, dialogue glossary 7/28 (design issue, not bug)
3. **Version reconciliation** - DECISION_SEMANTICS.md says v1.0, code says v2.0

#### P2: Nice-to-Have
1. Delete stale docs (50+ files)
2. Add FIELD evidence grade (documented, not implemented)
3. Add NA gate support
4. Fix TODO in run_tts.py (git hash)
5. CI pipeline with gates

---

## The Next Smallest Slice

**Slice: "Make unit tests runnable"**

1. `uv add pytest` 
2. Run `pytest tests/unit/ -v`
3. Verify both tests pass
4. Commit

This is the minimal slice that unblocks development workflow. Everything else can wait.
