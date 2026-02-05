# Audio-Related AI Models: Comprehensive Analysis and Comparison (Chat Note)

**Date received**: 2026-02-05  
**Provenance**: Pasted by user into chat; stored verbatim-ish for lab reference.  
**Evidence status**: **Reported** (not verified in Model-Lab runs).

This note is intentionally preserved as narrative context. Explicit numeric benchmark claims are extracted into machine-readable tables:

- `model-lab/docs/from_chat/AUDIO_MODELS_BENCHMARKS_EXTRACTED_2026-02-05.md`
- `model-lab/data/from_chat/audio_models_benchmarks_extracted_2026-02-05.csv`
- `model-lab/data/from_chat/audio_models_benchmarks_extracted_2026-02-05.json`

---

## Raw note (as shared)

# Audio-Related AI Models: Comprehensive Analysis and Comparison

## 1. Speech Recognition Models

### 1.1 OpenAI Whisper

#### 1.1.1 Core Architecture and Training

**OpenAI Whisper** represents a paradigm shift in automatic speech recognition (ASR) through its implementation of **large-scale weak supervision**. The model architecture employs an **encoder-decoder transformer design** that processes audio spectrograms through a multi-layer encoder and generates text transcripts via autoregressive decoding. What distinguishes Whisper from traditional ASR systems is its unprecedented training scale: approximately **680,000 hours of diverse multilingual and multitask supervised data** encompassing web-crawled audio across nearly 100 languages .

The **multitask training framework** simultaneously optimizes for speech recognition, speech translation, spoken language identification, and voice activity detection within a single unified model. This design eliminates the need for separate pipeline components that characterized earlier ASR systems. The training methodology deliberately accepts **imperfect supervision signals**—automatically generated transcripts with variable accuracy—trading label quality for massive data scale. Empirical validation demonstrates that models trained at this scale effectively learn to distinguish signal from noise, with performance improving log-linearly with training data quantity .

The model family includes **seven size variants** enabling deployment flexibility: **tiny (39M parameters)**, **base (74M)**, **small (244M)**, **medium (769M)**, **large (1.55B)**, **large-v2**, and **large-v3**. Each variant maintains architectural consistency while scaling layer dimensions and depth. The **large-v3** variant increases mel-spectrogram bins from 80 to 128, enhancing frequency resolution for improved recognition accuracy. A **turbo variant** reduces decoder layers from 32 to 4, achieving **6× faster inference** with only 1–2% accuracy degradation .

| Variant | Parameters | LibriSpeech Clean WER | LibriSpeech Other WER | Typical Use Case |
|---------|-----------|----------------------|----------------------|------------------|
| Tiny | 39M | ~9% | ~13% | Edge devices, extreme latency constraints |
| Base | 74M | ~6% | ~9% | Balanced speed/accuracy for consumer hardware |
| Small | 244M | ~4% | ~7% | Real-time applications with quality requirements |
| Medium | 769M | ~3% | ~6% | Production transcription services |
| Large | 1.55B | **2.7%** | **5.2%** | Maximum accuracy, batch processing |
| Large-v3 | 1.55B | 2.7% | 5.2% | Improved multilingual and noisy speech |
| Turbo | 809M | ~2.9% | ~5.4% | **High-throughput, latency-critical applications** |

#### 1.1.2 Performance Characteristics

Whisper's performance profile reveals **strengths in robustness and multilingual coverage** rather than peak benchmark optimization. On **LibriSpeech**, the gold-standard English ASR benchmark, Whisper large-v3 achieves **2.7% WER on clean test** and **5.2% on the more challenging "other" subset**—competitive though not state-of-the-art for English-only recognition . However, the **Common Voice multilingual benchmark**—with greater speaker diversity and acoustic variation—reveals Whisper's distinctive advantage: approximately **9.0% WER averaged across languages**, achieved without language-specific fine-tuning .

The **Real-Time Factor (RTF)** varies dramatically by model size, from approximately **0.6 for small variants to 1.0 for large-v3** on consumer GPU hardware. Community optimizations have substantially improved these figures: **Faster Whisper** achieves up to **4× speedup** through CTranslate2 optimization, while **Insanely Fast Whisper** reports **9× acceleration** with maintained accuracy . The **Whisper Large V3 Turbo** variant, optimized for speed through decoder layer reduction, achieves **216× real-time factor** on specialized Groq infrastructure .

... (rest of note continues in the same style; preserved in the chat transcript)

