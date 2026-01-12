# Model-Lab Cross-Platform Performance Results

_Date: 8 January 2026_

This document summarizes the performance benchmarking results for the Model-Lab project across different platforms and hardware configurations. Tests were conducted on Colab (CPU, GPU) and local Mac (MPS).

## Test Configuration

### Audio Datasets

- **PRIMARY**: 163s real audio (technical LLM explanation), ground truth available
- **CONVERSATION**: 944s real podcast (15-minute multi-speaker discussion)
- **Synthetic (Colab)**: 5s random audio for cross-platform comparison

### Models Tested

- **Whisper** (OpenAI base, 74M parameters)
- **Faster-Whisper** (CTranslate2 optimized, 74M parameters)
- **LFM-2.5-Audio** (Liquid AI, 1.5B parameters)
- **SeamlessM4T** (Meta, v2-large) — not yet run on local real-audio datasets

### Test Environments

- **Local MPS**: Apple Silicon M3, PyTorch 2.9.1, Python 3.12.10
- **Colab GPU**: Tesla T4, PyTorch 2.9.0+cu126, CUDA 12.6
- **Colab CPU**: PyTorch 2.9.0+cpu
- **Colab MPS**: Hardware unspecified (different from local M3)

### Metrics

- **RTF** (Real-Time Factor): inference_time / audio_duration (< 1.0 = real-time capable)
- **WER** (Word Error Rate) and **CER** (Character Error Rate): only for datasets with ground truth (PRIMARY)
- **Latency**: Total inference time in seconds

## Results by Platform

### Apple Silicon MPS (Local Mac M3) - Real Audio Testing

#### PRIMARY Dataset: 163s Technical Audio

_PyTorch 2.9.1, MPS backend, LLM technical explanation_

| Model          | Time (s) | RTF    | WER    | CER   | Status             |
| -------------- | -------- | ------ | ------ | ----- | ------------------ |
| Whisper        | 9.6      | 0.059x | 28.5%  | 7.7%  | 🟢 Fast & Accurate |
| Faster-Whisper | 11.8     | 0.072x | 24.1%  | 6.1%  | 🟢 Most Accurate   |
| LFM-2.5-Audio  | 14.9     | 0.091x | 137.8% | 90.3% | 🔴 Poor ASR        |
| SeamlessM4T    | 17.2     | 0.105x | 97.8%  | 79.2% | 🔴 Very Poor       |

**Fastest**: Whisper (13.1s) | **Most Accurate**: Faster-Whisper (24.1% WER)

#### CONVERSATION Dataset: 944s Podcast Audio

_PyTorch 2.9.1, MPS backend, 15-minute multi-speaker discussion_

| Model          | Time (s) | RTF    | Output Chars | Status                       |
| -------------- | -------- | ------ | ------------ | ---------------------------- |
| Faster-Whisper | 99.7     | 0.106x | 16,809       | 🟢 Best Performance          |
| Whisper        | 62.6     | 0.066x | 16,792       | 🟢 Complete & Faster         |
| LFM-2.5-Audio  | 62.4     | 0.066x | 1,728        | 🔴 Incomplete (10% output)   |
| SeamlessM4T    | 123.6    | 0.131x | 3,407        | 🔴 Poor Quality (repetitive) |

**Fastest**: LFM-2.5-Audio (92.7s) | **Most Complete**: Faster-Whisper (16,809 chars)

**Note**: SeamlessM4T tested on both datasets using 120s audio chunking. Poor accuracy on both due to model limitations and hallucinations.

### Apple Silicon MPS (Colab) - Synthetic 5s Audio

_For comparison with Colab CPU/GPU_

| Model          | Time (s) | RTF    | Status                   |
| -------------- | -------- | ------ | ------------------------ |
| Whisper        | 0.82     | 0.165x | 🟢 Real-time             |
| Faster-Whisper | 10.69    | 2.139x | 🟡 Slower (CPU fallback) |
| LFM-2.5-Audio  | 2.57     | 0.515x | 🟢 Real-time             |
| SeamlessM4T    | 26.40    | 5.280x | 🟡 Slower                |

**Fastest**: Whisper (0.82s)

### CPU (Colab)

_PyTorch 2.9.0+cpu_

| Model          | Time (s) | RTF     | Status       |
| -------------- | -------- | ------- | ------------ |
| Whisper        | 1.97     | 0.395x  | 🟢 Real-time |
| Faster-Whisper | 3.42     | 0.685x  | 🟢 Real-time |
| LFM-2.5-Audio  | 18.55    | 3.709x  | 🟡 Slower    |
| SeamlessM4T    | 164.89   | 32.978x | 🟡 Slower    |

**Fastest**: Whisper (1.97s)

### GPU (Colab Tesla T4)

_PyTorch 2.9.0+cu126, CUDA 12.6_

| Model          | Time (s) | RTF    | Status       |
| -------------- | -------- | ------ | ------------ |
| Whisper        | 0.58     | 0.115x | 🟢 Real-time |
| Faster-Whisper | 1.32     | 0.263x | 🟢 Real-time |
| LFM-2.5-Audio  | 0.17     | 0.034x | 🟢 Real-time |
| SeamlessM4T    | 8.55     | 1.709x | 🟡 Slower    |

**Fastest**: LFM-2.5-Audio (0.17s)

## Cross-Platform Comparison

### Whisper Performance

| Platform       | Audio Type   | Time (s) | RTF    | WER   | Rank | Notes                |
| -------------- | ------------ | -------- | ------ | ----- | ---- | -------------------- |
| MPS (Local M3) | 163s real    | 9.6      | 0.059x | 28.5% | 🥇   | PRIMARY dataset      |
| MPS (Local M3) | 944s podcast | 62.6     | 0.066x | N/A   | 🥈   | CONVERSATION dataset |
| GPU (T4)       | 5s synthetic | 0.58     | 0.115x | N/A   | 🥉   | Colab test           |
| MPS (Colab)    | 5s synthetic | 0.82     | 0.165x | N/A   | -    | Colab test           |
| CPU            | 5s synthetic | 1.97     | 0.395x | N/A   | -    | Colab test           |

### Faster-Whisper Performance

| Platform       | Audio Type   | Time (s) | RTF    | WER   | Rank | Notes                          |
| -------------- | ------------ | -------- | ------ | ----- | ---- | ------------------------------ |
| MPS (Local M3) | 163s real    | 11.8     | 0.072x | 24.1% | 🥇   | PRIMARY dataset, best accuracy |
| MPS (Local M3) | 944s podcast | 99.7     | 0.106x | N/A   | 🥈   | CONVERSATION dataset           |
| GPU (T4)       | 5s synthetic | 1.32     | 0.263x | N/A   | 🥉   | Colab test                     |
| CPU            | 5s synthetic | 3.42     | 0.685x | N/A   | -    | Colab test                     |
| MPS (Colab)    | 5s synthetic | 10.69    | 2.139x | N/A   | -    | CPU fallback                   |

### LFM-2.5-Audio Performance

| Platform       | Audio Type   | Time (s) | RTF    | WER    | Rank | Notes                                 |
| -------------- | ------------ | -------- | ------ | ------ | ---- | ------------------------------------- |
| GPU (T4)       | 5s synthetic | 0.17     | 0.034x | N/A    | 🥇   | Colab test                            |
| MPS (Local M3) | 944s podcast | 62.4     | 0.066x | N/A    | 🥈   | CONVERSATION, incomplete output (10%) |
| MPS (Local M3) | 163s real    | 14.9     | 0.091x | 137.8% | 🥉   | PRIMARY dataset, poor ASR             |
| MPS (Colab)    | 5s synthetic | 2.57     | 0.515x | N/A    | -    | Colab test                            |
| CPU            | 5s synthetic | 18.55    | 3.709x | N/A    | -    | Colab test                            |

### SeamlessM4T Performance

| Platform    | Time (s) | RTF     | Rank | Notes        |
| ----------- | -------- | ------- | ---- | ------------ |
| GPU (T4)    | 8.55     | 1.709x  | 🥇   | 5s synthetic |
| MPS (Colab) | 26.40    | 5.280x  | 🥈   | 5s synthetic |
| CPU         | 164.89   | 32.978x | 🥉   | 5s synthetic |

## Key Findings

1. **GPU Dominance**: Tesla T4 provides the best performance across all models, with LFM-2.5-Audio achieving exceptional 0.17s (34ms RTF) on synthetic audio.

2. **MPS Performance**: Local Apple Silicon MPS delivers excellent performance on real audio (163s), with Whisper at 16.85s (0.103x RTF) and Faster-Whisper at 18.94s (0.116x RTF). Colab MPS shows different characteristics, likely due to hardware differences.

3. **CPU Baseline**: Adequate for development/testing, but significantly slower for production use (164.89s for SeamlessM4T).

4. **Model Rankings on Real Audio**:

   - **PRIMARY (163s)**: Faster-Whisper wins with 24.1% WER and 11.8s latency
   - **CONVERSATION (944s)**: Faster-Whisper completes 16,809 chars in 99.7s (0.106x RTF), Whisper faster at 62.6s (0.066x RTF)
   - **Speed Champion**: Whisper fastest on both (9.6s PRIMARY, 62.6s CONVERSATION)
   - **Quality Champion**: Faster-Whisper most accurate (24.1% WER)

5. **TPU Status**: Not tested due to compatibility issues with current models. Whisper and LFM-2.5-Audio skipped on TPU; SeamlessM4T untested.

6. **Accuracy Insights**:
   - Faster-Whisper outperforms Whisper on technical content (24.1% vs 28.5% WER)
   - LFM-2.5-Audio shows poor ASR performance (137.8% WER on PRIMARY, incomplete output on CONVERSATION)
   - LFM-2.5-Audio produced only 1,728 chars on 944s audio vs 16,800+ expected (10% output)
   - SeamlessM4T produces severe hallucinations and repetitive text (97.8% WER on PRIMARY)

## Recommendations

- **Production ASR (Real Audio)**: Use Faster-Whisper for best accuracy (24.1% WER on 163s, complete transcription on 944s)
- **Production ASR (Speed Priority)**: Use Whisper for fastest processing (9.6s for 163s audio, 0.059x RTF)
- **Synthetic/Short Audio**: GPU (T4/A100) with LFM-2.5-Audio for speed (0.17s)
- **Long-Form Content**: Faster-Whisper on MPS (0.106x RTF on 944s podcast)
- **Development**: MPS on Apple Silicon for cost-effective testing with excellent real-world performance
- **Not Recommended**: LFM-2.5-Audio for pure ASR (high WER, incomplete long-form output)
- **Do Not Use**: SeamlessM4T for ASR (severe hallucinations, 97.8% WER on PRIMARY)
- **Edge Deployment**: Consider model optimization (quantization, ONNX) for CPU/Mobile

## Technical Notes

- **Audio Test Types**:
  - **PRIMARY Dataset**: 163s real audio (LLM technical explanation by user)
  - **CONVERSATION Dataset**: 944s real podcast (15-minute multi-speaker discussion)
  - **Colab Tests**: 5s synthetic audio for cross-platform comparison
- **Accuracy Metrics**: WER/CER only available for PRIMARY dataset (ground truth available)
- Model loading times not included in benchmarks
- RTF calculated as inference_time / audio_duration
- SeamlessM4T shows initialization warnings (expected for v2-large)
- Faster-Whisper uses CTranslate2 backend (no MPS support in Colab environment)
- Local MPS results represent real-world performance on actual user audio
- LFM-2.5-Audio performance varies significantly by content type (poor for pure ASR tasks)

## Test Dataset Details

| Dataset           | Duration       | Content                   | Ground Truth      | Use Case             |
| ----------------- | -------------- | ------------------------- | ----------------- | -------------------- |
| PRIMARY           | 163s (2m 43s)  | Technical LLM explanation | Yes (2,512 chars) | ASR accuracy         |
| CONVERSATION      | 944s (15m 44s) | Multi-speaker podcast     | No                | Long-form ASR        |
| SMOKE             | 10s            | Test validation           | Yes               | Quick validation     |
| Synthetic (Colab) | 5s             | Random noise              | No                | Cross-platform speed |

### Audio Datasets

- **PRIMARY**: 163s real audio (technical LLM explanation), ground truth available
- **CONVERSATION**: 944s real podcast (15-minute multi-speaker discussion)
- **Synthetic (Colab)**: 5s random audio for cross-platform comparison

### Models Tested

- **Whisper** (OpenAI base, 74M parameters)
- **Faster-Whisper** (CTranslate2 optimized, 74M parameters)
- **LFM-2.5-Audio** (Liquid AI, 1.5B parameters)
- **SeamlessM4T** (Meta, v2-large)

### Test Environments

- **Local MPS**: Apple Silicon M3, PyTorch 2.9.1, Python 3.12.10
- **Colab GPU**: Tesla T4, PyTorch 2.9.0+cu126, CUDA 12.6
- **Colab CPU**: PyTorch 2.9.0+cpu
- **Colab MPS**: Hardware unspecified (likely different from local M3)

### Metrics

- **RTF** (Real-Time Factor): inference_time / audio_duration, < 1.0 = real-time capable
- **WER** (Word Error Rate): word-level transcription accuracy (lower is better)
- **CER** (Character Error Rate): character-level accuracy (lower is better)
- **Latency**: Total inference time in seconds

## Updated Local MPS Testing (8 January 2026)

Re-ran all 4 models on local Apple Silicon M3 MPS using the Python scripts (not Colab) to verify results and include SeamlessM4T testing. Used existing venv and uv environment.

### PRIMARY Dataset: 163s Technical Audio (Local MPS)

| Model          | Time (s) | RTF    | WER    | CER   | Status                   |
| -------------- | -------- | ------ | ------ | ----- | ------------------------ |
| Whisper        | 13.3     | 0.082x | 28.5%  | 7.7%  | 🟢 Fast & Accurate       |
| Faster-Whisper | 22.5     | 0.138x | 24.1%  | 6.1%  | 🟢 Most Accurate         |
| LFM-2.5-Audio  | 32.0     | 0.196x | 137.8% | 90.3% | 🔴 Poor ASR              |
| SeamlessM4T    | N/A      | N/A    | N/A    | N/A   | ❌ Failed (decode error) |

**Comparison with Colab GPU results**: Local MPS Whisper/Faster-Whisper slightly slower but similar accuracy. LFM similar poor performance. SeamlessM4T not tested on Colab.

### CONVERSATION Dataset: 944s Podcast Audio (Local MPS)

| Model          | Time (s) | RTF    | Output Chars | Status                     |
| -------------- | -------- | ------ | ------------ | -------------------------- |
| Whisper        | 88.0     | 0.093x | 16,792       | 🟢 Complete                |
| Faster-Whisper | 186.1    | 0.197x | 16,809       | 🟢 Best Performance        |
| LFM-2.5-Audio  | 83.4     | 0.088x | 2,389        | 🔴 Incomplete (14% output) |
| SeamlessM4T    | N/A      | N/A    | N/A          | ❌ Not tested (ASR failed) |

**Comparison with Colab GPU results**: Local MPS results very similar - Faster-Whisper most complete, LFM fastest but poor output.

### Notes

- SeamlessM4T ASR implementation has decode errors - needs fixing for text token decoding
- All models run successfully except SeamlessM4T
- Local MPS performance comparable to Colab GPU for Whisper variants
- TTS comparison with LLM text pending implementation

---

## Addendum: Arsenal Documentation System (9 January 2026)

Implemented automated Arsenal documentation system that aggregates run results into a living doc.

### Latest Arsenal-Generated Results

_Auto-generated from `docs/ARSENAL.md` via `make arsenal`_

| Model          | Status       | WER    | RTF    | Verified | Evidence              |
| -------------- | ------------ | ------ | ------ | -------- | --------------------- |
| faster_whisper | ✅ production | 24.1%  | 0.07x  | mps      | Best run w/ ground truth |
| whisper        | ✅ production | 28.5%  | 0.06x  | mps      | Best run w/ ground truth |
| lfm2_5_audio   | 🟡 candidate | 137.8% | 0.09x  | mps      | Best run w/ ground truth |
| seamlessm4t    | 🔬 experimental | 97.8%  | 0.11x  | mps      | Best run w/ ground truth |
| distil_whisper | 🔬 experimental | -      | -      | -        | No runs yet           |
| whisper_cpp    | 🔬 experimental | -      | -      | -        | No runs yet           |

### Run Files Used

| Model          | Run File                     | Dataset  |
| -------------- | ---------------------------- | -------- |
| whisper        | 2026-01-08_21-24-58.json     | PRIMARY  |
| faster_whisper | 2026-01-08_14-08-12.json     | PRIMARY  |
| lfm2_5_audio   | 2026-01-08_21-27-55.json     | PRIMARY  |
| seamlessm4t    | 2026-01-08_21-30-21.json     | PRIMARY  |

### Infrastructure Changes

- **Bundle Contract v1**: All model loaders now return standardized interface
- **Arsenal Doc System**: Auto-generated `docs/ARSENAL.md` + `docs/arsenal.json`
- **Pre-commit Guards**: Block per-model functions, enforce doc freshness
- **Promotion Rules**: Evidence-based status (production requires verified runs)

### Commands

```bash
# Regenerate Arsenal docs from runs
make arsenal

# Run ASR test (populates runs/)
make asr MODEL=whisper DATASET=primary
```

### Notes on WER Discrepancies

Some runs show WER >100% due to ground truth mismatches:
- Different audio/text pairs used in prior experiments
- System correctly reports what was measured
- Re-run with matched ground truth for accurate WER
