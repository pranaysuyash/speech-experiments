# 🎯 Model Registry Master - Comprehensive Tracking

## **Master Model Registry Table** (All Models, All Capabilities)

| Model | Provider | Inference Type | Size on Disk | Parameters | STT | TTS | Conversation | Multi-turn | Languages | Device Support | Precision | Status | Smoke WER | Primary WER | Convers. WER | Latency (ms) | RTF | EER | Setup Date | Last Tested | Notes |
|-------|----------|---------------|-------------|-----------|-----|-----|-------------|-----------|-----------|----------------|-----------|--------|-----------|-------------|-------------|-------------|-----|-----|------------|-------------|-------|
| **LFM2.5-Audio-1.5B** | LiquidAI | Local | ~2.8GB | 1.5B | ✅ | ✅ | ✅ | ✅ | English | MPS, CUDA, CPU | bfloat16, float16, float32 | 🟢 Ready | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 2026-01-08 | — | Only model with TTS + Conversation + Multi-modal interleaving. Best for conversational AI applications. Uses official liquid-audio API. |
| **Whisper-Large-V3** | OpenAI | Local | ~3.0GB | 1.5B | ✅ | ❌ | ❌ | ❌ | 99 languages | MPS, CUDA, CPU | float16, float32 | 🟢 Ready | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 2026-01-08 | — | Original Whisper implementation. State-of-the-art ASR accuracy. Slower but most reliable baseline. Requires `openai-whisper` + `ffmpeg`. |
| **Faster-Whisper-Large-V3** | guillaumekln | Local | ~1.5GB | 1.5B | ✅ | ❌ | ❌ | ❌ | 99 languages | CPU, CUDA | float16, int8_float16 | 🟢 Ready | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 🔄 Pending | 2026-01-08 | — | 4x+ faster than original Whisper. Lower memory footprint. Same accuracy. Uses CTranslate2. Best for real-time production. Requires `faster-whisper`. No MPS support (CPU fallback). |

---

## **Dataset Registry** (All Test Datasets)

| Dataset | Audio File | Text File | Duration | Purpose | Status | Hash | Notes |
|---------|-----------|-----------|----------|---------|--------|------|-------|
| **SMOKE** | `llm_recording_pranay_10s.wav` | `llm_10s.txt` | ~10s | Quick validation | ✅ Created | TBD | Extracted from primary dataset. First ~200 chars (sentence boundary). Fast smoke test for all models. |
| **PRIMARY** | `llm_recording_pranay.m4a` | `llm.txt` | ~2min | Main ASR evaluation | ✅ Available | TBD | User's 2-minute Wikipedia reading. Ground truth text available. Main evaluation dataset. |
| **CONVERSATION** | `UX_Psychology_From_Miller_s_Law_to_AI.m4a` | None | ~15min | Multi-speaker conversation | ✅ Available | TBD | NotebookLM podcast (UX Psychology). Two speakers. No ground truth - for conversation analysis only. |

---

## **Test Results Registry** (All Runs)

| Run ID | Model | Dataset | WER | CER | EER | Latency (ms) | RTF | p95 Latency | Stability | Date | Git Hash | Config Hash | Notes |
|--------|-------|---------|-----|-----|-----|-------------|-----|-------------|-----------|------|----------|-------------|-------|
| — | — | — | — | — | — | — | — | — | — | — | — | — | **Awaiting first test runs** |

---

## **Capability Matrix** (Quick Reference)

| Model | STT | TTS | Conversation | Multi-turn | Multi-modal | Streaming | Speaker Diarization | Language Detection |
|-------|-----|-----|-------------|-----------|-------------|-----------|-------------------|-------------------|
| **LFM2.5-Audio** | ✅ | ✅ | ✅ | ✅ | ✅ | 🔄 | 🔄 | 🔄 |
| **Whisper** | ✅ | ❌ | ❌ | ❌ | ❌ | 🔄 | ❌ | ✅ |
| **Faster-Whisper** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |

---

## **Performance Comparison** (After Testing)

| Metric | LFM2.5-Audio | Whisper | Faster-Whisper | Winner |
|--------|-------------|---------|----------------|--------|
| **Accuracy (WER)** | 🔄 Pending | 🔄 Pending | 🔄 Pending | — |
| **Speed (RTF)** | 🔄 Pending | 🔄 Pending | 🔄 Pending | — |
| **Memory Usage** | 🔄 Pending | 🔄 Pending | 🔄 Pending | — |
| **Entity Accuracy** | 🔄 Pending | 🔄 Pending | 🔄 Pending | — |
| **Stability** | 🔄 Pending | 🔄 Pending | 🔄 Pending | — |
| **Production Ready** | 🔄 Pending | 🔄 Pending | 🔄 Pending | — |

**📋 See Addendum**: Latest test results available in `MODEL_REGISTRY_ADDENDUM_JAN8.md`

---

## **Protocol Versions** (Reproducibility)

| Component | Version | Description |
|-----------|---------|-------------|
| **Normalization** | v1.0 | Lowercase, strip punctuation, normalize whitespace, expand contractions |
| **Entity Extraction** | v1.0 | Numbers: `\b\d+(?:\.\d+)?\b`, Dates: MM/DD/YYYY, Currency: `$10.50` |
| **WER/CER** | v1.0 | Standard word/character error rate calculation |
| **JSON Schema** | v1.0 | Standardized output format for all providers |
| **Run Contract** | v1.0 | Git hash + config hash + dataset hash |

---

## **Installation & Setup** (Per Model)

| Model | Install Command | Additional Dependencies | Config Path | Status |
|-------|----------------|------------------------|-------------|--------|
| **LFM2.5-Audio** | `uv add liquid-audio` | PyTorch, torchaudio | `models/lfm2_5_audio/config.yaml` | ✅ Configured |
| **Whisper** | `uv add openai-whisper` | `brew install ffmpeg` | `models/whisper/config.yaml` | ✅ Configured |
| **Faster-Whisper** | `uv add faster-whisper` | None | `models/faster_whisper/config.yaml` | ✅ Configured |

---

## **Quick Reference Commands** (All Models)

```bash
# Test Commands (same for all models)
python scripts/run_asr.py --model MODEL_ID --dataset DATASET_ID

# Available Models:
# - lfm2_5_audio
# - whisper
# - faster_whisper

# Available Datasets:
# - smoke (10s quick test)
# - primary (2min main evaluation)
# - conversation (15min multi-speaker)

# Examples:
python scripts/run_asr.py --model whisper --dataset smoke
python scripts/run_asr.py --model faster_whisper --dataset primary
```

---

## **Status Legend**

- 🟢 **Ready**: Fully configured, awaiting tests
- 🟡 **Testing**: Tests in progress
- ✅ **Complete**: All tests passed
- 🔴 **Failed**: Tests failed, needs attention
- 🔄 **Pending**: Awaiting results
- ⚠️ **Partial**: Some capabilities available

---

## **Update Log**

| Date | Update | Model | Dataset |
|------|--------|-------|---------|
| 2026-01-08 | Initial model setup | All models | All datasets |
| — | **First test run** | **Awaiting execution** | **Awaiting execution** |
| — | **First scorecard generation** | **Awaiting execution** | **Awaiting execution** |

---

**📊 This comprehensive registry tracks all model capabilities, test results, and protocol versions for reproducible model comparisons.**

**Next Update**: After first smoke tests complete 🔄