# Model Lab Showcase

Last updated: 2026-02-23

## What is live right now

- Reproducible TTS benchmark run with Kokoro (`kokoro_tts`) on `tts_smoke_v1`
- Sprint-wide execution report with 380 task rows across 4 agents
- ASR benchmark summaries from recent runs (WER/CER/RTF)
- Streaming ASR run artifact for `kyutai_streaming`

## 1) TTS (executed today)

Command executed:

```bash
uv run python scripts/run_tts.py --model kokoro_tts --dataset tts_smoke_v1 --device cpu
```

Result artifact:
- `runs/kokoro_tts/tts/2026-02-23_14-54-57.json`
- `runs/kokoro_tts/tts/summary.json`
- `runs/kokoro_tts/tts/audio`

Key metrics from `runs/kokoro_tts/tts/2026-02-23_14-54-57.json`:
- Prompts: 3
- Healthy prompts: 3/3
- Avg RTF: 0.4805x
- Total generated audio: 6.18s

## 2) HF Pro sprint report (executed today)

Command executed:

```bash
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```

Generated artifacts:
- `runs/hf_sprint_2026q1/reports/summary.md`
- `runs/hf_sprint_2026q1/reports/task_results.csv`

Current report snapshot:
- Total ledger rows: 380
- Agents covered: `domain_specialist`, `edge_small`, `general_baselines`, `realtime_streaming`

## 3) ASR snapshot you can show immediately

From latest summary files:

- `runs/faster_whisper/asr/2026-02-23_15-38-34.json` (executed today)
  - dataset: `asr_smoke_v1`
  - latency (p50): 13196.38 ms
  - RTF: 0.0809x
  - transcript length: 2548 chars
  - output includes segment timestamps and normalized transcript

- `runs/faster_whisper/asr/summary.json`
  - dataset: `llm_primary`
  - WER: 0.2415
  - CER: 0.0611
  - RTF: 0.1480x

- `runs/nb_whisper_small_onnx/asr/summary.json`
  - dataset: `ux_primary`
  - WER: 1.0000
  - CER: 0.7581
  - RTF: 1.8494x

- `runs/distil_whisper/asr/summary.json`
  - dataset: `ux_primary`
  - WER: 1.3621
  - CER: 0.8535
  - RTF: 0.3189x

## 4) Showcase talking points

- Model-lab already produces benchmark artifacts and run manifests per model/capability under `runs/<model>/<capability>/`.
- The HF sprint report gives one consolidated, auditable task ledger (`task_results.csv`) for public scorecards.
- TTS now has a stable Kokoro path with healthy smoke outputs and generated audio files.

## 5) Streaming ASR (executed now)

Command executed:

```bash
uv run python scripts/run_asr_stream.py --model kyutai_streaming --dataset asr_smoke_v1 --device cpu --chunk-ms 160
```

Result artifact:
- `runs/kyutai_streaming/asr_stream/asr_smoke_v1_20260223_161633_f874dc79.json`

Captured metrics:
- first token latency: 0.835 ms
- finalize latency: 0.006 ms
- WER: 1.0000 (current placeholder output path)

## 6) Quick demo commands

```bash
# Regenerate TTS proof artifact
uv run python scripts/run_tts.py --model kokoro_tts --dataset tts_smoke_v1 --device cpu

# Regenerate ASR smoke artifact
uv run python scripts/run_asr.py --model faster_whisper --dataset asr_smoke_v1 --device cpu

# Regenerate streaming ASR artifact
uv run python scripts/run_asr_stream.py --model kyutai_streaming --dataset asr_smoke_v1 --device cpu --chunk-ms 160

# Regenerate sprint report
uv run python scripts/hf_sprint_report.py --execution-root runs/hf_sprint_2026q1/execution --out-dir runs/hf_sprint_2026q1/reports --sprint-id hf_pro_2026q1
```
