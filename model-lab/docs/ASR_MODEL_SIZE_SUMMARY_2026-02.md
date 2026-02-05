# ASR model size summary (2026-02)

**Date**: 2026-02-05  
**Provenance**: Extracted from `model-lab/ASR_MODEL_RESEARCH_2026-02.md` (ported from EchoPanel). Numbers are **Reported** until validated in Model-Lab runs.

Raw table (CSV): `model-lab/data/from_chat/asr_model_size_summary_2026-02.csv`

## Quick size/quality tradeoffs (EchoPanel-oriented, useful for lab baselines)

| Model | Download | WER | Best for |
|---|---:|---:|---|
| Moonshine Tiny | ~50MB | ~10% | Smallest bundle |
| faster-whisper `tiny.en` | ~75MB | ~12% | Fast, acceptable |
| faster-whisper `base.en` (current) | ~150MB | ~8% | Good balance |
| Vosk `small-en` | ~40MB | ~15% | Ultra-low resource |
| faster-whisper `distil-large-v3` | ~750MB | ~4% | Near-best quality |
| Voxtral-Mini-4B-Realtime | ~4GB | ~3% | Best streaming (v0.3) |

## Recommendations (Reported)

- v0.2 baseline: keep `base.en` (~150MB) for a strong quality/size balance.
- v0.3 streaming eval: evaluate Voxtral-Mini-4B-Realtime for streaming quality/latency.
- Productization idea: model selector so users pick a size/quality tradeoff.
