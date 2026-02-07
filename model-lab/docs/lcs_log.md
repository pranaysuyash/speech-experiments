# LCS Log

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
