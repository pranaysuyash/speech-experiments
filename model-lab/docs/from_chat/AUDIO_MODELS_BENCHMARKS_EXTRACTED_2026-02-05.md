# Extracted benchmark numbers (Chat note, 2026-02-05)

**Date received**: 2026-02-05  
**Provenance**: Pasted by user into chat as an explicit “data-extraction” section.  
**Evidence status**: **Reported** (not verified in Model-Lab runs).

Machine-readable versions:

- CSV: `model-lab/data/from_chat/audio_models_benchmarks_extracted_2026-02-05.csv`
- JSON: `model-lab/data/from_chat/audio_models_benchmarks_extracted_2026-02-05.json`

---

## Extracted Benchmark Table (verbatim-ish)

### Speech Recognition Accuracy (WER / CER)

| Model                    | Metric | Value  | Dataset / Condition                 | Hardware          | Source Line (verbatim context)                           |
| ------------------------ | ------ | ------ | ----------------------------------- | ----------------- | -------------------------------------------------------- |
| Whisper Tiny             | WER    | ~9%    | LibriSpeech Clean                   | Not stated        | “Tiny … LibriSpeech Clean WER ~9%”                       |
| Whisper Tiny             | WER    | ~13%   | LibriSpeech Other                   | Not stated        | “Tiny … LibriSpeech Other WER ~13%”                      |
| Whisper Base             | WER    | ~6%    | LibriSpeech Clean                   | Not stated        | “Base … LibriSpeech Clean WER ~6%”                       |
| Whisper Base             | WER    | ~9%    | LibriSpeech Other                   | Not stated        | “Base … LibriSpeech Other WER ~9%”                       |
| Whisper Small            | WER    | ~4%    | LibriSpeech Clean                   | Not stated        | “Small … LibriSpeech Clean WER ~4%”                      |
| Whisper Small            | WER    | ~7%    | LibriSpeech Other                   | Not stated        | “Small … LibriSpeech Other WER ~7%”                      |
| Whisper Medium           | WER    | ~3%    | LibriSpeech Clean                   | Not stated        | “Medium … LibriSpeech Clean WER ~3%”                     |
| Whisper Medium           | WER    | ~6%    | LibriSpeech Other                   | Not stated        | “Medium … LibriSpeech Other WER ~6%”                     |
| Whisper Large / Large-v3 | WER    | 2.7%   | LibriSpeech Clean                   | Not stated        | “Whisper large-v3 achieves 2.7% WER on clean test”       |
| Whisper Large / Large-v3 | WER    | 5.2%   | LibriSpeech Other                   | Not stated        | “5.2% on the more challenging ‘other’ subset”            |
| Whisper (multilingual)   | WER    | ~9.0%  | Common Voice (avg across languages) | Not stated        | “approximately 9.0% WER averaged across languages”       |
| wav2vec 2.0 Base         | WER    | 4.8%   | LibriSpeech Clean                   | GPU (unspecified) | “base model achieves 4.8% WER on LibriSpeech clean test” |
| wav2vec 2.0 Large        | WER    | 1.8%   | LibriSpeech (full fine-tuning)      | GPU (unspecified) | “large variant reaching 1.8% WER”                        |
| wav2vec 2.0              | WER    | 37.04% | Ionio Clean Speech                  | Not stated        | “Ionio evaluation … 37.04% WER on clean speech”          |
| wav2vec 2.0              | WER    | 54.69% | Ionio Noisy Speech                  | Not stated        | “54.69% WER on noisy speech”                             |
| Whisper                  | WER    | 19.96% | Ionio Clean Speech                  | Not stated        | “Whisper … 19.96%”                                       |
| Whisper                  | WER    | 29.80% | Ionio Noisy Speech                  | Not stated        | “Whisper … 29.80%”                                       |
| Fish Speech V1.5         | WER    | 3.5%   | English                             | Not stated        | “WER 3.5% (English)”                                     |
| Fish Speech V1.5         | CER    | 1.2%   | English                             | Not stated        | “CER 1.2% (English)”                                     |
| Fish Speech V1.5         | CER    | 1.3%   | Chinese                             | Not stated        | “1.3% (Chinese)”                                         |

### Real-Time Factor (RTF) / Latency

| Model                 | Metric  | Value         | Condition           | Hardware            | Source Line                                 |
| --------------------- | ------- | ------------- | ------------------- | ------------------- | ------------------------------------------- |
| Whisper Small         | RTF     | ~0.6          | Inference           | Consumer GPU        | “RTF varies … ~0.6 for small variants”      |
| Whisper Large-v3      | RTF     | ~1.0          | Inference           | Consumer GPU        | “1.0 for large-v3 on consumer GPU hardware” |
| Faster Whisper        | Speedup | 4×            | Optimized inference | CTranslate2         | “Faster Whisper achieves up to 4× speedup”  |
| Insanely Fast Whisper | Speedup | 9×            | Optimized inference | Not stated          | “reports 9× acceleration”                   |
| Whisper Turbo         | RTF     | 216× realtime | Inference           | Groq infrastructure | “achieves 216× real-time factor on … Groq”  |
| wav2vec 2.0 Base      | RTF     | ~0.3          | Inference           | Standard GPU        | “RTF of ~0.3 for the base model”            |
| Gemini Audio          | Latency | <300 ms       | End-to-end          | Google TPU          | “Latency optimization achieves sub-300ms”   |
| Chatterbox-Turbo      | Latency | <200 ms       | TTS                 | GPU                 | “sub-200ms inference”                       |

### Hardware / Resource Requirements

| Model                 | Resource | Value              | Source Line                                  |
| --------------------- | -------- | ------------------ | -------------------------------------------- |
| Whisper Large-v3      | VRAM     | ~16 GB             | “requires 16 GB VRAM for large-v3 inference” |
| Whisper Medium        | VRAM     | ~5 GB              | “medium (~5 GB)”                             |
| Whisper Small         | VRAM     | ~2 GB              | “small (~2 GB)”                              |
| Whisper Base          | VRAM     | ~1 GB              | “base (~1 GB)”                               |
| Whisper Tiny          | VRAM     | <500 MB            | “tiny (<500 MB)”                             |
| MusicGen Medium/Large | VRAM     | ≥16 GB             | “Minimum: 16 GB VRAM”                        |
| Stable Audio 2.5      | Latency  | <2 s / 3 min audio | “<2 seconds for 3-minute tracks”             |
| ACE-Step 1.5          | VRAM     | ~4 GB              | “VRAM requirement: 4 GB”                     |
| ACE-Step 1.5          | Latency  | ~2 s (A100)        | “~2 seconds on A100”                         |
| ACE-Step 1.5          | Latency  | <10 s              | “<10 seconds on RTX 3090”                    |

### Datasets Explicitly Referenced

| Dataset Name                | Context                               |
| --------------------------- | ------------------------------------- |
| LibriSpeech (Clean / Other) | Primary ASR benchmark                 |
| Common Voice                | Multilingual benchmark                |
| Ionio                       | Production-style robustness benchmark |
| AudioSet                    | AudioGen training data                |
| LibriVox                    | wav2vec 2.0 pretraining               |

