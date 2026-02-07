# LCS Log

## LCS-04: Moonshine Tiny ASR

**Surfaces**: asr

**Runtime**: pytorch

**Devices**: cpu, mps (fallback to cpu)

**Files**:
- `models/moonshine/config.yaml` - model config with variants
- `models/moonshine/claims.yaml` - claims manifest
- `models/moonshine/requirements.txt` - isolated dependencies
- `models/moonshine/README.md` - usage docs
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_moonshine_smoke.py` - 7 tests (3 structural, 4 model)

**Commands**:
```bash
make model-install MODEL=moonshine
python -m pytest tests/integration/test_model_moonshine_smoke.py -v
```

**Notes**: 27M params, 5-15x faster than Whisper on short segments. English-only.

---

## LCS-03: Enhance + Separate Metrics + Streaming Misuse

**Surfaces**: enhance, separate (metrics support)

**Files**:
- `harness/metrics_enhance.py` - si_snr, stoi, pesq (optional deps)
- `harness/metrics_separate.py` - bss_eval, sdr, multi_source_sdr (mir_eval optional)
- `tests/unit/test_metrics_enhance.py` - 12 tests
- `tests/unit/test_metrics_separate.py` - 8 tests
- `tests/unit/test_asr_stream_contract_misuse.py` - 8 tests

**Commands**:
```bash
python -m pytest tests/unit/test_metrics_enhance.py tests/unit/test_metrics_separate.py -v  # 14 pass, 6 skip
python -m pytest tests/unit/test_asr_stream_contract_misuse.py -v  # 8 pass
```

**Notes**: 
- SI-SNR always available, STOI/PESQ optional (never fail CI)
- SDR/SIR/SAR via mir_eval (optional)
- Streaming lifecycle: finalize is idempotent, close always safe

---

## LCS-02: Classification Metrics

**Surfaces**: classify (metrics support)

**Files**:
- `harness/metrics_classify.py` - accuracy_top1, precision_recall_f1, confusion_matrix, per_class_metrics
- `tests/unit/test_metrics_classify.py` - 14 tests

**Commands**:
```bash
python -m pytest tests/unit/test_metrics_classify.py -v  # 14 passed
```

**Notes**: CI-safe, dataset-free. Macro vs micro vs weighted averaging supported.

---

## LCS-01: 6 New Capability Surfaces

**Surfaces**: classify, embed, enhance, separate, music_transcription, asr_stream

**Files**:
- `harness/contracts.py` - 6 result types, 6 namespaces, Bundle v2
- `tests/unit/test_contract_enforcement.py` - 6 new surface tests

**Commands**:
```bash
python -c "from harness.contracts import CONTRACT_VERSION; print(CONTRACT_VERSION)"  # 2.0.0
python -m pytest tests/unit/test_contract_enforcement.py::TestNewSurfaces -v  # 6 passed
```

**Notes**: CONTRACT_VERSION=2.0.0. Streaming lifecycle: start_stream → push_audio → flush → finalize → close. validate_bundle now data-driven.
