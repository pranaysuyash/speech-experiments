# ASR / Audio Model Research (2026-02)

**Provenance**: Ported from `EchoPanel/docs/ASR_MODEL_RESEARCH_2026-02.md` into this repo on 2026-02-05 so ongoing audio-model research can live in `model-lab/`.

**Evidence key**:
- **Observed**: verified by running code / inspecting a primary source in this repo
- **Reported**: carried over from the original research doc; not re-verified in this lab yet
- **Unknown**: needs confirmation

**Date**: February 2026  
**Scope**: Complete Audio Landscape - Local, API, Browser-Based  
**Methodology**: Official docs, Hugging Face, GitHub, Twitter/X community research (**Reported**)  

---

## Executive Summary & Implementation Framework

### 1.1 Audit Scope and Objectives

The audio artificial intelligence landscape has undergone transformative expansion beyond traditional speech processing, encompassing neural audio codecs, generative music systems, multimodal language models, and specialized forensic tools. This audit systematically evaluates **60+ production-ready models** across **nine major categories**: speech processing, music generation, audio classification, enhancement and restoration, neural codecs, forensics and security, multimodal integration, and specialized applications including bioacoustics and industrial monitoring.

The primary objective centers on identifying models with demonstrated production deployment, comprehensive documentation, active community support, and clear Python integration pathways. The evaluation framework prioritizes local deployment flexibility for privacy-sensitive and latency-critical applications while acknowledging the rapid-prototyping value of managed API services.

**Key Strategic Insight**: Unified audio-language models (gpt-audio, Qwen2-Audio, Step-Audio 2 mini) are displacing pipeline architectures, enabling 10-100× latency reductions for voice agents by eliminating ASR→LLM→TTS cascades.

### 1.2 Implementation Priority Tiers

| Tier | Definition | Timeline | Selection Criteria | Representative Models |
|------|-----------|----------|-------------------|----------------------|
| **Tier 1** | Immediate implementation | 0-2 months | Mature releases, comprehensive docs, active maintenance, clear integration, production-proven | Whisper, wav2vec 2.0, Demucs, CosyVoice2-0.5B, YAMNet |
| **Tier 2** | Short-term evaluation | 3-4 months | Strong technical merit, emerging ecosystem, requires validation for specific use cases | Qwen2-Audio, gpt-audio, Step-Audio 2 mini, JASCO, MusicGen |
| **Tier 3** | Long-term monitoring | 6-12 months | Niche applications, experimental architectures, or awaiting ecosystem maturation | AudioSeal, BirdNET, neural codecs (SoundStream, Lyra), industrial audio systems |

The tiering methodology acknowledges dynamic reassignment as models mature. Key triggers for tier advancement include: stable major version release, benchmark performance validation on lab-specific data, community adoption milestones (1000+ GitHub stars, active Discord/forum), and documentation completeness (API reference, tutorials, deployment guides).

---

## 2. Speech Processing Models

### 2.1 Automatic Speech Recognition (ASR)

#### 2.1.1 wav2vec 2.0 (Meta/Facebook AI Research)

| Attribute | Specification |
|-----------|--------------|
| Version | 2.0 (September 2020, ongoing updates) |
| Type | Local |
| Parameters | 95M (BASE), 317M (LARGE), 300M-2B (XLS-R variants) |
| Hosting | Hugging Face, PyTorch Hub, fairseq |
| License | MIT (code), CC-BY-NC 4.0 (some pretrained models) |
| Hardware | GPU recommended for fine-tuning (16GB+ VRAM for LARGE); CPU inference feasible for BASE |

wav2vec 2.0 established the self-supervised paradigm for speech representation learning, demonstrating that pre-training on unlabeled audio followed by minimal fine-tuning achieves competitive ASR performance with orders of magnitude less labeled data than supervised alternatives.

**Architecture Details**:
- Multi-layer convolutional feature encoder (seven blocks with 512 channels)
- Processes 16kHz raw waveforms to 49Hz latent representations
- Transformer context network employing learned relative positional embeddings
- Contrastive pre-training objective: predicting quantized latent representations for masked audio segments

**Benchmark Performance**:
- 5.7% WER on LibriSpeech test-clean with only 10 minutes of labeled fine-tuning data (BASE)
- 3.0% WER with 10 minutes for LARGE
- Transformative data efficiency for low-resource languages, accented speech, and specialized domains

**Deployment**:
- Hugging Face Transformers with unified interfaces for inference, fine-tuning, and optimization
- 20+ language-specific fine-tuned checkpoints available
- Domain adaptations (medical, legal, call center)
- ONNX export and quantization support edge deployment with modest accuracy degradation

| Best Use Cases | Limitations |
|---------------|-------------|
| Low-resource language ASR development | Requires fine-tuning for optimal performance; inferior zero-shot vs. Whisper |
| Noisy/reverberant environments (robust self-supervised features) | Fixed 16kHz input requires resampling for telephony (8kHz) or high-fidelity (48kHz) |
| Domain adaptation with limited labeled data | CTC decoding produces phonetically plausible but semantically anomalous outputs without LM rescoring |
| Research requiring interpretable intermediate representations | Real-time streaming requires careful buffer management; non-causal attention limits chunking |

---

#### 2.1.2 HuBERT (Meta)

| Attribute | Specification |
|-----------|--------------|
| Version | 1.0 (June 2021) |
| Type | Local |
| Parameters | 95M (base), 317M (large), 1B (xlarge) |
| Hosting | Hugging Face, fairseq |
| License | MIT |

HuBERT advances self-supervised learning through discrete hidden unit targets derived from offline clustering of MFCC features or intermediate representations, creating a more structured prediction objective that better captures hierarchical speech structure.

**Key Innovation**:
- Iterative refinement procedure alternates between clustering (generating targets) and prediction (learning representations)
- Later iterations benefit from improved target quality
- Discrete unit framework enables speech synthesis from discrete codes and cross-lingual transfer through shared unit inventories

**Performance**:
- State-of-the-art LibriSpeech performance (2.0% WER test-clean)
- Exceptional strength as frozen feature extractors for speaker identification, emotion recognition, and spoken language understanding

---

#### 2.1.3 WavLM (Microsoft)

| Attribute | Specification |
|-----------|--------------|
| Version | 1.0 (October 2021) |
| Type | Local |
| Parameters | 94M (BASE), 316M (LARGE) |
| Hosting | Hugging Face, UniSpeech |
| License | MIT |

WavLM unifies speaker verification and ASR within single pre-trained framework, addressing the historical tension between speaker-invariant (ASR-optimal) and speaker-discriminative (verification-optimal) representations.

**Key Innovations**:
- Denoising and dereverberation objectives during pre-training
- Produces representations robust to acoustic degradation without explicit enhancement preprocessing
- Gated relative position bias for improved temporal modeling
- Utterance-level mixing for data augmentation
- Strong SUPERB benchmark performance across diverse speech understanding tasks

**Best For**: Applications requiring simultaneous transcription and speaker diarization, where unified representations eliminate pipeline complexity and error accumulation.

---

#### 2.1.4 Whisper (OpenAI) - THE COMPREHENSIVE BREAKDOWN

> **Critical Framework**: Whisper is not one model—it's a matrix of **base weights × runtime × quantization × feature forks** yielding 20+ deployable variants.

| Attribute | Specification |
|-----------|--------------|
| Version | v3 (large-v3, September 2023) |
| Type | API and Local |
| Parameters | 39M (tiny), 74M (base), 244M (small), 769M (medium), 1.55B (large-v3) |
| Hosting | OpenAI API, Hugging Face (open weights) |
| License | MIT (model weights), proprietary (API) |
| API Pricing | $0.006/minute (standard), volume tiers available |
| Hardware (local) | 10GB+ VRAM for large-v3; CPU feasible for smaller variants |

**Capabilities**:
- 99 languages with varying performance
- Direct speech-to-English translation without intermediate text
- Word-level and segment-level timestamp alignment
- Automatic punctuation, paragraph breaks, speaker turns

##### Whisper Base Weights (The Canonical Varieties)

| Model | Params | Download Size | WER (English) | Languages | Best For |
|-------|--------|---------------|---------------|-----------|----------|
| tiny | 39M | ~75MB | ~15% | 99 | Fastest inference |
| tiny.en | 39M | ~75MB | ~12% | English-only | Better English accuracy |
| base | 74M | ~150MB | ~10% | 99 | **Current EchoPanel default** |
| base.en | 74M | ~150MB | ~8% | English-only | Recommended for English |
| small | 244M | ~500MB | ~6% | 99 | Good quality/speed balance |
| small.en | 244M | ~500MB | ~5% | English-only | Best small model |
| medium | 769M | ~1.5GB | ~4% | 99 | High accuracy |
| medium.en | 769M | ~1.5GB | ~4% | English-only | Production quality |
| large-v1 | 1.55B | ~3GB | ~3.5% | 99 | Legacy |
| large-v2 | 1.55B | ~3GB | ~3% | 99 | Previous best |
| large-v3 | 1.55B | ~3GB | ~2.5% | 99 | Current best accuracy |
| large-v3-turbo | 809M | ~1.6GB | ~3% | 99 | Speed-optimized large |

##### Whisper Production Runtimes (Each is a Separate "Product")

| Runtime | Speed Improvement | Platform Support | Best For | Notes |
|---------|-------------------|------------------|----------|-------|
| **faster-whisper** (CTranslate2) | 4× faster | Linux/macOS/Windows | Server batch processing | Supports int8 quantization, has own benchmark tables |
| **whisper.cpp** | Near-native C speed | Apple Silicon (Metal/CoreML), CPU | Edge, mobile, real-time | Supports real-time microphone streaming, multiple accelerators |
| **Transformers** | 1× (baseline) | Any | HF ecosystem integration | Easiest if already in HF land |
| **Whisper JAX** | TPU-optimized | Google Cloud | Massive batch throughput | Worth tracking for batching/TPU options |

##### Whisper Feature Forks (Add-on Capabilities)

| Fork | What It Adds | Use Case | Integration Complexity |
|------|--------------|----------|----------------------|
| **WhisperX** | Word-level timestamps + speaker diarization glue | "Who said what when" | Medium (diarization can dominate runtime) |
| **stable-ts** | Timestamp stabilization, forced alignment, post-processing | Professional subtitles that look "professional" | Low |
| **whisper-timestamped** | Alternative word timestamp approach | Subtitle generation | Low |
| **Distil-Whisper** | Smaller distilled models (mostly English-focused) | Resource-constrained deployment | Low, multiple export formats (ct2, ggml) |

**Practical Take**: In your database, "Whisper large-v3 via faster-whisper int8" and "Whisper large-v3 via whisper.cpp gguf int4" are **different products** with different latency, memory, and packaging characteristics.

| Best Use Cases | Limitations |
|---------------|-------------|
| Rapid prototyping without engineering investment | Higher compute than specialized models for single-language deployment |
| Multilingual applications where language-specific training is impractical | Occasional hallucination in challenging conditions; "fluent but wrong" outputs |
| Integration with downstream LLMs (text output feeds reasoning pipelines) | Long-form transcription requires chunking with boundary handling |
| User-generated content with unpredictable acoustic characteristics | Less competitive on clean, domain-matched benchmarks vs. fine-tuned alternatives |

---

#### 2.1.5 Mistral Voxtral Family (NEW - July 2025)

> **Key Discovery (Reported)**: Mistral AI has audio models. They launched the Voxtral family in July 2025.

**Verification status**: **Reported** (not yet verified inside this lab repo). Treat benchmark/latency/license claims below as inputs for follow-up evaluation.

| Model | Parameters | Context | Features | Latency | License |
|-------|------------|---------|----------|---------|---------|
| **Voxtral-Mini-3B** | 3B | 40 minutes | Batch transcription, multilingual | Non-streaming | Apache 2.0 |
| **Voxtral-Small-24B** | 24B | Extended | High accuracy, summarization, reasoning | Non-streaming | Apache 2.0 |
| **Voxtral-Mini-4B-Realtime** | 4B | Streaming | Real-time streaming, sub-200ms latency | <200ms configurable | Apache 2.0 |

**Key Claims**:
- Outperforms Whisper Large-v3+ on several benchmarks
- 13 language support
- Open-weight, Apache 2.0 (fully open source)
- Designed for edge device deployment

**Why This Matters for EchoPanel**:
- **Voxtral-Mini-4B-Realtime** is the top candidate for v0.3 streaming upgrade
- Apache 2.0 license means full commercial freedom
- Sub-200ms latency is competitive with cloud APIs

---

#### 2.1.6 Paraformer (Alibaba Cloud)

| Attribute | Specification |
|-----------|--------------|
| Version | 2.0 (2023) |
| Type | API and Local |
| Architecture | Non-autoregressive with CIF-based predictor |
| Hosting | Alibaba Cloud Model Studio, Hugging Face |
| API Pricing | $0.000012/second (file), $0.000035/second (real-time) |

Paraformer addresses the fundamental latency-accuracy trade-off of autoregressive ASR through **non-autoregressive parallel generation**, achieving 5-10× speedup with minimal accuracy degradation.

**Key Innovation**: Continuous Integrate-and-Fire (CIF) predictor enables accurate length modeling critical for parallel output generation, handling the inherent multimodality of multiple valid transcriptions for identical acoustics.

**Optimization**: Mandarin Chinese and multilingual Asian languages, with particular strength in tone handling, regional accents, and Chinese-English code-switching. Real-time factor below 0.1 on modern GPU enables high-throughput applications.

---

#### 2.1.7 NVIDIA Parakeet TDT

| Model Variant | Parameters | WER | RTFx (Real-Time Factor) | Notes |
|---------------|------------|-----|-------------------------|-------|
| Parakeet TDT 0.6B V2 | 600M | 6.05% | 3386× | 3-5× faster than Whisper |
| Parakeet V3 | ~1B | ~5% | — | Best for German |

**Pros**: 3-5× faster than Whisper, NeMo framework integration  
**Cons**: GPU-focused, complex integration with NeMo dependencies  
**Verdict**: Consider for v0.3 if targeting NVIDIA GPU deployments.

---

#### 2.1.8 Moonshine Tiny (Edge-Optimized)

| Attribute | Specification |
|-----------|--------------|
| Parameters | 27M |
| Download | ~50MB |
| WER | ~10% (better than Whisper Tiny) |
| Speed | 5-15× faster than Whisper |

**Best For**: Truly tiny first-run bundle, edge devices, IoT  
**Verdict**: Worth testing as smallest viable ASR for EchoPanel's initial download experience.

---

#### 2.1.9 Vosk (Lightweight Offline)

| Model | Download | Memory | WER | Notes |
|-------|----------|--------|-----|-------|
| small-en-us-0.15 | 40MB | ~300MB | ~15% | Raspberry Pi compatible |
| en-us-0.22 | 1.8GB | ~4GB | ~8% | Higher accuracy |
| spk-0.4 (speaker ID) | 13MB | — | — | Add-on module |

**Best For**: Raspberry Pi, IoT, offline without GPU, ultra-low resource devices.

---

#### 2.1.10 Shunya Labs Pingala V1

| Attribute | Specification |
|-----------|--------------|
| Type | Local |
| Architecture | Whisper-based, ONNX optimized |
| Status | Leading Open ASR Leaderboard |

Whisper-based model optimized for ONNX deployment, leading the Hugging Face Open ASR Leaderboard.

---

### 2.2 Text-to-Speech (TTS)

#### Overview Table

| Model | Size | Latency | Best For | License/API Cost |
|-------|------|---------|----------|-----------------|
| Seamless Streaming (Meta) | undisclosed | <200ms first-audio | Conversational AI, real-time translation | API: $0.01/1K chars |
| Qwen3-TTS-Flash-Realtime (Alibaba) | undisclosed | 150ms | Chinese-optimized interactive applications | API: $0.143/10K chars |
| CosyVoice2-0.5B (Open Source) | 0.5B | real-time on CPU | Edge deployment, privacy-sensitive | Apache 2.0; API: $7.15/M UTF-8 bytes |
| Fish Speech V1.5 (Open Source) | ~1.2B | real-time on GPU | Cross-lingual voice cloning | Apache 2.0; API: $15/M UTF-8 bytes |
| IndexTTS-2 (Open Source) | minimized | <100ms | Minimal hardware, IoT, embedded | Apache 2.0; API: $7.15/M UTF-8 bytes |
| Piper | Varies | Fast | Many voices, on-device | MIT |
| XTTS (Coqui) | ~1B | Moderate | Multilingual + cloning | CPML |
| Bark | ~1B | Slow | Expressive, sound effects | MIT |
| StyleTTS2 | ~100M | Fast | High-quality neural TTS | MIT |

---

#### 2.2.1 Seamless Streaming (Meta)

Achieves sub-200ms end-to-end latency through incremental text processing with speculative audio generation—producing initial audio chunks before complete utterance reception.

**Architecture**:
- Lookahead prosody prediction with state caching
- Maintains natural flow across segment boundaries
- Cross-lingual voice cloning preserves speaker characteristics across language boundaries
- 100+ languages with consistent voice identity

---

#### 2.2.2 CosyVoice2-0.5B (Open Source)

Demonstrates quality-to-size optimization enabling **CPU-based real-time inference** with natural prosody and emotional control.

| Feature | Implementation |
|---------|---------------|
| Voice cloning | Few-shot adaptation from 3-10 seconds reference |
| Emotional control | Discrete tokens (neutral, happy, sad, angry) with interpolation |
| Edge deployment | ONNX/TensorRT optimization; INT8 quantization |
| Streaming | Chunked generation with <150ms first-audio |

**Best For**: Embedded voice applications, offline accessibility tools, privacy-critical scenarios where cloud transmission is unacceptable.

---

#### 2.2.3 Fish Speech V1.5 (Open Source)

Achieves industry-leading **cross-lingual voice cloning**, synthesizing natural speech in languages absent from source speaker training data through disentangled speaker-language representations.

**Key Capability**: "Zero-shot" transfer enables consistent brand voice across 20+ languages without per-language recording.  
**Quality-to-size ratio**: ~1.2B parameters achieve perceptual quality comparable to 5B+ alternatives.  
**Strength**: Mandarin-English code-switching is particularly strong.

---

### 2.3 Voice Cloning & Conversion

#### 2.3.1 JASCO (Meta)

| Attribute | Specification |
|-----------|--------------|
| Full Name | Joint Audio and Symbolic Conditioning for Temporally Controlled Generation |
| Type | Local (inference), research (training) |
| License | MIT (inference code); weights CC-BY-NC |
| Primary Function | Controllable music generation with audio conditioning |

Extends beyond traditional voice cloning into multi-modal audio conditioning, enabling precise control through chord progressions, beat patterns, melodic contours, and reference audio.

| Conditioning Input | Control Granularity |
|-------------------|-------------------|
| Chord progression | Harmonic structure (e.g., Cmaj7-F#m7b5-B7-E7) |
| Beat pattern | Tempo, groove, rhythmic feel |
| Melodic contour | Pitch trajectory via MIDI or audio |
| Text description | Overall style, mood, instrumentation |
| Reference audio | Timbre, production aesthetic |

---

#### 2.3.2 Voice Conversion Toolchains

| Model | Type | Use Case |
|-------|------|----------|
| **RVC** (Retrieval-based Voice Conversion) | Voice Conversion | "Sound like X" conversions, very common VC baseline |
| **so-vits-svc** | Singing Voice Conversion | Music covers, singing style transfer |
| **OpenVoice** | Voice Cloning | Multi-style cloning workflows |
| **Resemblyzer** | Voice Embeddings | 256-dim embeddings for similarity, verification, TTS conditioning |

---

### 2.4 Speech Enhancement & Noise Reduction

#### 2.4.1 Demucs (Meta/Facebook Research)

| Attribute | Specification |
|-----------|--------------|
| Version | 4.0 (hybrid transformer) |
| Type | Local |
| Architecture | U-Net + bidirectional LSTM, waveform domain |
| Hosting | Hugging Face Hub, torchaudio |
| License | MIT |
| Hardware | 4-8GB VRAM (standard); CPU feasible for offline |
| Performance | 9.8 dB SDR on MUSDB18 for vocal separation |

**State-of-the-art for music source separation**, decomposing mixtures into vocals, drums, bass, and other stems with exceptional fidelity. Waveform-domain processing preserves phase information absent from spectrogram-based alternatives.

| Variant | Use Case |
|---------|----------|
| Demucs v4 (hybrid) | Maximum quality, transformer bottleneck |
| Demucs v3 (standard) | Quality-efficiency balance |
| Demucs-Light | Real-time on consumer GPU |

| Best Use Cases | Limitations |
|---------------|-------------|
| Music production: remixing, karaoke, sampling | Music-optimized; suboptimal for non-musical noise |
| Audio forensics: dialogue isolation from mixed recordings | Fixed source categories; no arbitrary sound types |
| Preprocessing for lyric transcription, chord analysis | Real-time requires GPU; 2-4× real-time typical |
| Speech enhancement in musical background | Quality degradation on speech-dominant content |

---

#### 2.4.2 DeepFilterNet (Open Source)

| Attribute | Specification |
|-----------|--------------|
| Version | 2.0 |
| Type | Local |
| Latency | 10-40ms algorithmic |
| Implementation | Python, C, WebAssembly |
| License | MIT |

Provides **real-time joint dereverberation and denoising** with perceptual loss optimization preserving speech naturalness. Complex spectral filtering with explicit phase estimation maintains speaker characteristics and prosodic nuance.

**Key Feature**: WebRTC integration and WebAssembly deployment enable browser-based real-time processing without server infrastructure.

---

#### 2.4.3 RNNoise (Mozilla/Xiph)

| Attribute | Specification |
|-----------|--------------|
| Latency | <10ms |
| CPU Usage | 1% |
| Parameters | 60K |
| License | BSD |

Achieves real-time denoising with 1% CPU usage through hybrid architecture: traditional pitch filtering preserves speech structure, while GRU neural network estimates optimal gain.

**Best For**: VoIP enhancement, hearing aid processing, ASR preprocessing.

---

## 3. Music Processing Models

### 3.1 Music Generation

| Model | Control Paradigm | Architecture | Latency | Access | License |
|-------|-----------------|--------------|---------|--------|---------|
| AudioLM (Google) | Hierarchical tokens | SoundStream + w2v-BERT + Transformer | Non-real-time | Gemini API (limited) | Proprietary |
| Lyria 2 / RealTime (Google) | Text + style reference | Diffusion-based | RealTime: interactive | Vertex AI (enterprise) | Commercial |
| JASCO (Meta) | Symbolic + audio conditioning | Latent diffusion | Non-real-time | Open: MIT inference | MIT |
| MusicGen (Meta) | Text-only (simple) | EnCodec + autoregressive | 1-50× real-time | Open: Audiocraft | MIT |
| AudioLDM | Text + style | Diffusion | Non-real-time | Open | Open |
| Stable Audio Open | Text prompt | Diffusion transformer | Non-real-time | Open | Open |

#### 3.1.1 AudioLM (Google/DeepMind)

Pioneered hierarchical neural audio generation with three-tier structure:
1. **Coarse semantic tokens** (w2v-BERT): long-term structure
2. **Mid-level tokens**: melodic/rhythmic patterns
3. **Fine acoustic tokens** (SoundStream): timbral detail

This decomposition enables minute-long coherent generation impossible with flat autoregressive models.

---

#### 3.1.2 MusicGen (Meta/Audiocraft)

Prioritizes accessibility over control granularity, with single text prompt generation and multiple size variants (300M-3.3B parameters) enabling hardware-appropriate deployment.

**Melody-conditional variant**: Accepts humming or instrumental reference for controllable melodic structure.

**Audiocraft framework** unifies MusicGen, AudioGen (sound effects), and EnCodec with consistent APIs.

---

### 3.2 Music Transcription

| Model | Target | Output | Optimization |
|-------|--------|--------|-------------|
| Basic Pitch (Spotify) | Polyphonic pitch | Note events (onset, offset, pitch, confidence) | Guitar, vocals |
| MT3 (Google) | Multi-instrument | MIDI with instrument labels | Full ensemble |
| Onsets and Frames | Piano | Note events | Classical piano |

#### 3.2.1 Basic Pitch (Spotify)

Achieves **real-time polyphonic pitch detection** through efficient deep learning, with <10M parameters enabling 100× real-time CPU processing.

**Optimization**: Guitar and vocal optimization produces superior accuracy for these sources, with handling of bends, vibrato, and timbral variation.

**Deployment**: Python and JavaScript implementations enable server-side batch processing and browser-based real-time applications.

---

### 3.3 Music Analysis & Classification

#### 3.3.1 Essentia Models (MTG/UPF)

| Model | Task | Training Data | Format |
|-------|------|--------------|--------|
| MSD-MusiCNN | Embeddings | Million Song Dataset | TensorFlow |
| Discogs-EffNet | Embeddings | 400M+ Discogs tags | TensorFlow |
| Genre/mood classifiers | High-level tags | AcousticBrainz, commercial | TensorFlow |

Provides comprehensive music information retrieval with academic-proven methodology. Discogs-EffNet embeddings (400M+ user-generated tags) offer particularly rich representations for transfer learning.

**License Diversity**: GPL/AGPL for some models, MIT for others, with commercial licensing available for restricted components.

---

## 4. Audio Classification & Tagging

### 4.1 Environmental Sound Classification

| Model | Classes | Architecture | Optimization | Deployment |
|-------|---------|--------------|--------------|------------|
| YAMNet | 521 (AudioSet) | MobileNetV1 depthwise separable | Mobile CPU, 15× real-time | TensorFlow Lite, TF.js |
| VGGish | Embedding (128-dim) | VGG-adapted | Batch processing, similarity search | TensorFlow Hub |
| PANNs | 527 (AudioSet) | CNN14/CNN10/MobileNet | Weakly-supervised, scalable | Hugging Face |
| OpenL3 | Audio-visual embeddings | Self-supervised | Cross-modal retrieval | GitHub |
| BEATs | General embeddings | Bidirectional encoder | Transfer learning backbone | GitHub |
| CLAP | Text-audio aligned | Contrastive | Zero-shot classification, retrieval | Hugging Face |

#### 4.1.1 YAMNet (Google)

Delivers broad-coverage sound event detection with mobile-optimized efficiency: **<100MB memory, 15× real-time on Pixel-class smartphones**.

**AudioSet ontology** (521 hierarchical classes) enables flexible output granularity from specific events to broad categories.

**Best For**: Always-on deployment scenarios—accessibility awareness, smart home monitoring, content moderation.

---

#### 4.1.2 CLAP (Contrastive Language-Audio Pretraining)

Enables **zero-shot audio classification through text prompts** and cross-modal retrieval by jointly training audio and text encoders with contrastive loss.

**Key Capabilities**:
- Description-based retrieval
- Compositional queries
- Open-vocabulary classification without retraining

---

### 4.2 Bioacoustics Models

| Model | Taxa | Coverage | Key Capability |
|-------|------|----------|---------------|
| BirdNET (Cornell) | Birds | 3,000+ species, global | Migration tracking, citizen science |
| DeepSqueak (UW) | Rodents | Ultrasonic calls (20-100 kHz) | Behavioral neuroscience |

#### 4.2.1 BirdNET (Cornell Lab)

Enables global bird identification with community science integration through iOS/Android applications and web submission. EfficientNet-B0 architecture balances accuracy and edge deployment feasibility.

**Research Applications**: Occurrence data for migration pattern analysis and population monitoring, with continuous model improvement through user feedback.

---

## 5. Neural Audio Codecs

| Model | Bitrate Range | Quality Target | Architecture | Access | License |
|-------|---------------|---------------|--------------|--------|---------|
| SoundStream (Google) | 3-18 kbps | Scalable, perceptual | RVQ + entropy model | API (limited) | Proprietary |
| EnCodec (Meta) | 6-24 kbps | High-fidelity, streaming | RVQ + discriminator | Open | MIT |
| Lyra (Google) | 3 kbps | Ultra-low bandwidth | WaveRNN generative | Open (limited) | Apache 2.0 |
| Descript DAC | Variable | High-fidelity | Neural audio codec | Open | Open |

#### 5.1.1 EnCodec (Meta)

Provides **open-source high-fidelity compression** with real-time streaming optimization. MIT license enables self-hosted infrastructure without API dependencies, with training code supporting domain-specific adaptation.

---

## 6. Audio Forensics & Security

### 6.1 Deepfake Detection

| Model | Approach | Robustness | License | Best For |
|-------|----------|------------|---------|----------|
| AudioSeal (Meta) | Watermarking (proactive) | Compression, filtering, noise | Commercial | Platform integrity, provenance |
| RawNet2 (Open) | Raw waveform analysis | ASVspoof benchmarks | Open | Research, forensic analysis |
| AASIST (Open) | Graph neural network | SOTA on LA/DF tracks | Open | High-security authentication |

#### 6.1.1 AudioSeal (Meta)

Addresses AI-generated speech detection through **proactive watermarking**—embedding imperceptible, robust signatures during generation that enable cryptographic certainty of synthetic origin.

**Advantage over reactive detection**: Watermarking survives quality improvements that defeat artifact-based methods.

**Limitation**: Requires embedding at generation time (cannot detect unwatermarked synthesis).

---

## 7. Multimodal Audio Models (Audio-Language)

> **Key Insight**: These models replace ASR→LLM→TTS pipelines with single-architecture solutions, enabling 10-100× latency reduction.

| Model | Architecture | Input/Output | Key Advantage | Pricing |
|-------|-------------|--------------|--------------|---------|
| gpt-audio (OpenAI) | Unified end-to-end | Audio → audio/text | Native audio, no pipeline | $2.50-5.00/M tokens (in), $10.00-20.00/M (out) |
| Qwen2-Audio (Alibaba) | Audio encoder + LLM | Audio → text | Open source, local deploy | Free (self-hosted) |
| Step-Audio 2 mini (Stepfun) | End-to-end unified | Audio → audio | Benchmark SOTA, efficient | Free (self-hosted) |
| SALMONN (Tsinghua/MS) | Multi-task unified | Audio → text | Diverse task handling | Research |
| Gemma 3n audio (Google) | Speech + translation | Audio → text | 100+ languages, local | Apache 2.0 |

### 7.1.1 gpt-audio (OpenAI)

Enables native audio input/output through Chat Completions API, eliminating latency and error accumulation of cascaded ASR→LLM→TTS pipelines. The unified architecture processes paralinguistic information—emotion, emphasis, speaking style—lost in text intermediation.

| API Endpoint | Use Case |
|-------------|----------|
| Chat Completions | Conversational agents |
| Realtime (WebSocket) | Low-latency streaming |
| Batch | Cost-effective offline processing |

**Token Efficiency**: Typical speech requires 100-200 tokens per second.

---

### 7.1.2 Step-Audio 2 mini (Stepfun-AI)

Achieves benchmark-leading performance with **reported outperformance of GPT-4o Audio** on select evaluations.

| Benchmark | Reported Performance |
|-----------|---------------------|
| MMAU | 73.2 (top open-source end-to-end) |
| URO Bench | Leading basic/professional dialogue |
| CoVoST 2 (ZH-EN translation) | 39.3 |
| Chinese ASR (CER) | 3.19% |
| English ASR (WER) | 3.50% |

**Availability**: GitHub, Hugging Face, ModelScope ensures broad accessibility with standard tooling integration.

---

### 7.1.3 Qwen2-Audio (Alibaba Cloud Intelligence)

Provides open-source audio language capabilities with **Apache 2.0 licensing** enabling unrestricted commercial deployment. 7B parameter scale achieves competitive benchmark performance with local deployment feasibility (16GB+ VRAM).

**Capabilities**: Audio question answering and reasoning—"Describe the emotions in this music and suggest similar pieces," "What environmental sounds indicate urban recording location?"

---

## 8. VAD & Diarization (The Glue Layer)

| Model | Category | Use Case | Notes |
|-------|----------|----------|-------|
| **Silero VAD** | Voice Activity Detection | Chunking for streaming | Widely used lightweight VAD |
| **pyannote.audio** | Speaker Diarization | "Who said what" | Common baseline, used by WhisperX |
| **WhisperX** | Integrated | ASR + timestamps + diarization | Convenience wrapper |

---

## 9. API Services Comparison

### 9.1 ASR API Services

| Service | Model | Latency | WER | Pricing | Best For |
|---------|-------|---------|-----|---------|----------|
| **OpenAI Whisper API** | Whisper | ~1-2s | ~3% | $0.006/min | Ease of use |
| **Deepgram Nova-3** | Proprietary | <300ms | ~5% | ~$0.0125/min | Speed, streaming |
| **AssemblyAI** | Proprietary | ~300ms | ~5% | ~$0.65/hour | Integrated features |
| **Google Cloud STT** | Chirp 2 | ~500ms | ~4% | $0.016/min | Enterprise |
| **Azure Speech** | Whisper + custom | ~500ms | ~4% | $0.016/min | Microsoft ecosystem |

### 9.2 API Feature Comparison

| Service | Streaming | Diarization | Punctuation | PII Redaction | Custom Vocabulary |
|---------|-----------|-------------|-------------|---------------|-------------------|
| OpenAI | No | No | Yes | No | No |
| Deepgram | Yes | Yes | Yes | Yes | Yes |
| AssemblyAI | Yes | Yes | Yes | Yes | Yes |
| Google Cloud | Yes | Yes | Yes | No | Yes |

---

## 10. February 2026 Twitter/X Discoveries

### 10.1 Major Foundation Model Releases with Twitter/X Traction

#### 10.1.1 xAI/Grok Ecosystem

**Grok 3 Core Model**:
- Approximately 100,000 Nvidia H100 GPUs (Colossus supercomputing infrastructure)
- 10× increase over predecessor's training compute
- Multimodal processing: text, image, audio
- Direct access to live X post streams for real-time analysis
- Competitive claims against GPT-5 series and DeepSeek

**Grok 3 Variants**:

| Variant | Primary Focus | Key Capability | Target Deployment |
|---------|--------------|----------------|-------------------|
| Grok 3 DeepSearch | Research and synthesis | Iterative multi-source retrieval with transparent reasoning chains | Academic research, competitive intelligence |
| Grok 3 Think | Reasoning transparency | Extended analysis with explicit step-by-step methodology | Educational, complex problem-solving |
| Grok-3 Mini | Efficiency optimization | Core capabilities with 70% computational reduction | Mobile, edge, high-throughput |

**Pricing**: X Premium+ at $40/month, SuperGrok at $30/month, SuperGrok Heavy at $300/month.

---

#### 10.1.2 Alibaba Qwen Family

**Qwen3-Coder-Next (Released February 3-4, 2026)**:

| Specification | Value | Competitive Significance |
|--------------|-------|-------------------------|
| Total Parameters | 80 billion | Comparable to GPT-3.5 scale |
| Active Parameters (per forward pass) | 3 billion | Matches small specialized models |
| Sparsity Ratio | 26.7:1 | Among highest in production models |
| Context Window | 256,000 tokens | Industry-leading for coding models |
| SWE-Bench Verified Performance | >70% | Matches models with 10-20× active parameters |
| Training Tasks | 800,000+ verifiable coding scenarios | Environment-interactive learning |
| Throughput Improvement | ~10× | Repository-scale task efficiency |

**Twitter/X Community Response**: Emphatic enthusiasm for "perfect size" (尺寸完美) for local execution, strong demand for portable implementations.

**Qwen3-Max-Thinking**: Flagship reasoning model achieving reported performance exceeding "Humanity's Last Exam". Benchmark claims: 92.8% GPQA Diamond, 91.5% IMO-AnswerBench, 98.0% HMMT February 2025.

---

#### 10.1.3 ByteDance/Seed Team (February 2026 Wave)

| Model | Scheduled Release | Primary Capability | Strategic Positioning |
|-------|------------------|-------------------|----------------------|
| Doubao 2.0 | Mid-February 2026 | Large language model refresh | Consumer AI assistant (163M MAU) |
| Seedream 5.0 | February 2026 | 4K image generation, 14 reference images | Professional creative workflows |
| Seeddance 2.0 | February 2026 | Video generation with temporal consistency | Short-form content creation |
| Seedream 4.5 Edit | February 2026 | High-end image editing | Precise creative control |

RMB 160 billion (~$23 billion) 2026 capital expenditure with approximately half allocated to AI semiconductors.

---

#### 10.1.4 DeepSeek V4 (Anticipated Mid-February 2026)

Specialized coding capabilities targeting competitive performance against Anthropic's Claude and OpenAI's GPT series.

**Key Claims**:
- "Handling and processing extremely long coding prompts"
- Repository-level bug resolution
- Engram conditional memory system for 1M+ token contexts

**Strategic Timing**: Mirrors DeepSeek R1's successful January 2025 approach (Lunar New Year visibility).

---

#### 10.1.5 Meituan LongCat-Flash-Thinking-2601

| Attribute | Specification |
|-----------|--------------|
| Parameter Scale | 560 billion total (MoE) |
| Expert Modules | 8 activated through learned routing |
| Training Environments | 10,000+ heterogeneous environments |
| Training Efficiency | 2-4× improvement via DORA asynchronous system |
| Release Model | Fully open-source with online demo platform |
| Online Platform | longcat.chat |

"Heavy Thinking Mode" implements parallel reasoning with 8 expert modules—"8 brains thinking in parallel"—exploring diverse solution strategies before synthesis.

---

#### 10.1.6 Incredible Small 1.0

| Feature | Implementation |
|---------|---------------|
| Core Architecture | Live-Code Model Architecture (direct code/symbolic action generation) |
| Integration Ecosystem | 200+ business applications (CRMs, ERPs, Notion, HubSpot, Slack) |
| Simultaneous Actions | 1,000+ parallel operations |
| Data Processing | Gigabyte-scale datasets |
| Hallucination Rate | Near-zero (claimed) |
| Cost Reduction vs. Alternatives | 67% |
| API Pricing | $1.15/M input tokens, $5.00/M output tokens |

**Positioning**: "If you want an LLM to write code, use Claude. If you want PhD-level intelligence, use GPT. If you want AI that takes actions in products and gets work done, use Incredible."

---

### 10.2 Anticipated Releases (Prediction Markets)

| Model | Predicted Date | Market Confidence | Information Basis |
|-------|---------------|-------------------|-------------------|
| GPT-5.3 Codex | February 5, 2026 | 87% | Pattern-based inference, potential leaks |
| MiniMax M2.2 | February 10, 2026 | 76% | Historical release pattern analysis |
| Gemini 3 Pro DeepThink API | February 2026 | 50% | Google schedule opacity |
| Claude 5 / Sonnet 5 "Fennec" | Imminent | Leaked in Vertex AI config | Model ID: claude-sonnet-5@20260203 |

---

### 10.3 Community Sentiment Themes (Twitter/X)

| Priority Theme | Manifestation | Example |
|---------------|---------------|---------|
| Parameter efficiency | Enthusiasm for sparse architectures, local deployment | Qwen3-Coder-Next "perfect size" response |
| Open-source availability | Detailed licensing analysis, community fine-tuning | Qwen family Apache 2.0 appreciation |
| Local deployment feasibility | Hardware compatibility discussion, quantization interest | Consumer GPU execution requirements |
| Comparative benchmarking | Independent evaluation, task-specific testing | SWE-Bench community verification |
| Integration friction | API compatibility, toolchain support | MCP protocol adoption |

**Key Insight**: Developer preferences increasingly favor deployable, cost-effective solutions over raw benchmark performance.

---

## 11. Speech Translation Stacks

### 11.1 Local Speech-to-Text Translation

| Model | Languages | Streaming | Use Case |
|-------|-----------|-----------|----------|
| **Gemma 3n audio** | 100+ spoken | No | Local S2TT, 16kHz tokenizer, 30s clips recommended |
| **SeamlessM4T v2** | 100+ | Yes (SeamlessStreaming) | S2TT and S2ST end-to-end |
| **TranslateGemma** | 55 (text) | Text-only | Cascaded pipelines: ASR → TranslateGemma → TTS |

### 11.2 What This Means for EchoPanel

You can offer:
- **Local STT**: Whisper/Voxtral/Gemma 3n
- **Local S2TT** (speech-to-text translation): Gemma 3n AST, SeamlessM4T
- **Local S2ST** (speech-to-speech translation): SeamlessM4T (plus TTS component)

---

## 12. Comprehensive Model Database

### 12.1 Master Table (All 60+ Models)

| Model | Category | Type | Hosting | Parameters | Hardware | License/API Cost | Tier |
|-------|----------|------|---------|------------|----------|------------------|------|
| wav2vec 2.0 | ASR | Local | Hugging Face | 95M-2B | GPU recommended (fine-tuning) | MIT / CC-BY-NC | 1 |
| HuBERT | ASR | Local | Hugging Face | 95M-1B | GPU recommended | MIT | 2 |
| WavLM | ASR | Local | Hugging Face | 94M-316M | GPU recommended | MIT | 2 |
| Whisper (all variants) | ASR | API/Local | OpenAI/HF | 39M-1.55B | 10GB+ VRAM (large-v3) | MIT (weights) / $0.006/min (API) | 1 |
| Voxtral-Mini-3B | ASR | Local | Mistral/HF | 3B | GPU | Apache 2.0 | 2 |
| Voxtral-Small-24B | ASR | Local | Mistral/HF | 24B | High GPU | Apache 2.0 | 2 |
| Voxtral-Mini-4B-Realtime | ASR | Local | Mistral/HF | 4B | GPU | Apache 2.0 | 1 |
| Paraformer | ASR | API/Local | Alibaba/HF | — | Moderate | API: $0.000012-0.000035/sec | 2 |
| Parakeet TDT | ASR | Local | NVIDIA NeMo | 600M-1B | GPU | Apache 2.0 | 2 |
| Moonshine Tiny | ASR | Local | HF | 27M | CPU | Open | 2 |
| Vosk | ASR | Local | Vosk | — | CPU | Apache 2.0 | 2 |
| Seamless Streaming | TTS | API | Meta | — | — | $0.01/1K chars | 2 |
| Qwen3-TTS-Flash | TTS | API | Alibaba | — | — | $0.143/10K chars | 2 |
| CosyVoice2-0.5B | TTS | Local | HF/GitHub | 0.5B | CPU feasible | Apache 2.0 / $7.15/M bytes | 1 |
| Fish Speech V1.5 | TTS | Local | HF/GitHub | ~1.2B | GPU | Apache 2.0 / $15/M bytes | 2 |
| IndexTTS-2 | TTS | Local | HF/GitHub | minimized | ARM/embedded | Apache 2.0 / $7.15/M bytes | 2 |
| Piper | TTS | Local | GitHub | Varies | CPU | MIT | 2 |
| XTTS (Coqui) | TTS | Local | HF | ~1B | GPU | CPML | 2 |
| Bark | TTS | Local | GitHub | ~1B | GPU | MIT | 2 |
| StyleTTS2 | TTS | Local | GitHub | ~100M | GPU | MIT | 2 |
| JASCO | Voice/Music | Local | Meta/HF | 300M-1.5B | High GPU | MIT (inference) | 2 |
| RVC | Voice Conv | Local | GitHub | — | GPU | MIT | 2 |
| so-vits-svc | Voice Conv | Local | GitHub | — | GPU | MIT | 2 |
| OpenVoice | Voice Clone | Local | GitHub | — | GPU | MIT | 2 |
| Resemblyzer | Voice Embed | Local | PyPI | — | Minimal | Apache 2.0 | 2 |
| Demucs | Enhancement | Local | Hugging Face | ~100M | 4-8GB VRAM | MIT | 1 |
| DeepFilterNet2 | Enhancement | Local | GitHub | — | CPU feasible | MIT | 2 |
| RNNoise | Enhancement | Local | Xiph | 60K | Minimal CPU | BSD | 2 |
| AudioSR | Enhancement | Local | GitHub | — | Moderate GPU | Open | 3 |
| AudioLM | Music Gen | API | Google | — | — | Gemini (limited) | 3 |
| Lyria 2/RealTime | Music Gen | API | Google | — | — | Enterprise (negotiated) | 3 |
| MusicGen | Music Gen | Local | Hugging Face | 300M-3.3B | Scalable | MIT | 2 |
| AudioLDM | Music Gen | Local | GitHub | — | GPU | Open | 2 |
| Stable Audio Open | Music Gen | Local | GitHub | — | GPU | Open | 2 |
| Basic Pitch | Transcription | Local | Spotify/GitHub | <10M | CPU, 100× real-time | Apache 2.0 | 2 |
| MT3 | Transcription | Local | Google Research | ~2B | Moderate GPU | Apache 2.0 | 2 |
| Essentia Models | Analysis | Local | MTG | Various | Minimal | Mixed (GPL/AGPL/MIT) | 2 |
| YAMNet | Classification | Local | TensorFlow | 3.5M | Mobile CPU | Apache 2.0 | 1 |
| VGGish | Classification | Local | TensorFlow Hub | — | Minimal | Apache 2.0 | 2 |
| PANNs | Classification | Local | Hugging Face | 4.8M-80M | Scalable | CC | 2 |
| OpenL3 | Event Detection | Local | Cornell | — | Moderate | MIT | 2 |
| CLAP | Embeddings | Local | Hugging Face | 150M-1B | Moderate | MIT | 2 |
| BEATs | Embeddings | Local | GitHub | — | Moderate | MIT | 2 |
| BirdNET | Bioacoustics | Local/API | Cornell | EfficientNet-B0 | Edge feasible | CC | 3 |
| DeepSqueak | Bioacoustics | Local | UW/GitHub | — | Moderate | Open | 3 |
| SoundStream | Codec | API | Google | — | — | Limited | 3 |
| EnCodec | Codec | Local | Meta/Audiocraft | — | Real-time GPU | MIT | 2 |
| Descript DAC | Codec | Local | GitHub | — | GPU | Open | 2 |
| Lyra | Codec | Local | Google | — | Minimal | Apache 2.0 | 3 |
| AudioSeal | Forensics | API/Local | Meta | — | — | Commercial | 3 |
| RawNet2 | Forensics | Local | Open | — | Moderate | Open | 3 |
| AASIST | Forensics | Local | Open | — | GPU | Open | 3 |
| gpt-audio | Audio-LM | API | OpenAI | — | — | $2.50-20.00/M tokens | 2 |
| Qwen2-Audio | Audio-LM | Local | Hugging Face | 7B | 16GB+ VRAM | Apache 2.0 | 2 |
| Step-Audio 2 mini | Audio-LM | Local | HF/GitHub/ModelScope | — | Moderate | Open | 2 |
| SALMONN | Audio-LM | Local | Tsinghua/MS | — | Substantial | Research | 3 |
| Gemma 3n audio | Audio-LM | Local | Google | — | Moderate | Apache 2.0 | 2 |
| SeamlessM4T v2 | Translation | Local | Meta | — | GPU | CC-BY-NC | 2 |
| TranslateGemma | Translation | Local | Google | — | Moderate | Apache 2.0 | 2 |
| Silero VAD | VAD | Local | GitHub | — | Minimal | MIT | 1 |
| pyannote.audio | Diarization | Local | GitHub | — | Moderate | MIT | 1 |

---

### 12.2 Database Schema for Model Tracking

```csv
model_id,family,version,local_or_api,category,tasks_supported,languages,streaming_support,realtime_latency_target_ms,base_weights,runtime,quantization,hardware_reqs,context_or_max_audio,benchmarks,license,pricing,docs_url,community_signal,integration_complexity,known_failure_modes,ideal_use_cases,avoid_when,notes
```

**Instructions**: Add one row per deployable variant (weights × runtime × quantization × fork). For example:
- `whisper-large-v3-faster-whisper-int8`
- `whisper-large-v3-whisper.cpp-gguf-int4`
- `whisper-base.en-transformers-fp16`

---

## 13. Implementation Roadmap for EchoPanel (and as a reference for Model-Lab)

This section was originally written with EchoPanel product milestones in mind (**Reported**). In Model-Lab, treat it as a prioritization checklist for what to evaluate next and what to port into the harness.

### Phase 1: Core ASR Infrastructure (Months 1-2)

| Priority | Model | Capability | Success Criteria |
|----------|-------|------------|------------------|
| P0-1 | faster-whisper base.en | Baseline ASR | <10% WER clean, <15% noisy |
| P0-2 | Silero VAD | Streaming chunking | Reliable voice detection |
| P0-3 | WhisperX (optional) | Diarization glue | Speaker labels working |
| P0-4 | pyannote.audio | Speaker diarization | "Who said what" annotations |

### Phase 2: Streaming & Quality Upgrade (Months 3-4)

| Priority | Model | Capability | Evaluation Criteria |
|----------|-------|------------|-------------------|
| P1-1 | Voxtral-Mini-4B-Realtime | Streaming ASR | <200ms latency, Apache 2.0 |
| P1-2 | faster-whisper large-v3-turbo int8 | Quality upgrade | <5% WER |
| P1-3 | Step-Audio 2 mini | End-to-end voice agent | Benchmark validation |
| P1-4 | gpt-audio (API) | Comparison baseline | Latency, accuracy, cost analysis |

### Phase 3: Full Audio Capabilities (Months 5-6)

| Priority | Model | Capability | Strategic Value |
|----------|-------|------------|-----------------|
| P2-1 | CLAP | Audio search/retrieval | Zero-shot classification |
| P2-2 | Demucs | Source separation | Speech isolation from music |
| P2-3 | DeepFilterNet2 | Noise suppression | Real-time enhancement |
| P2-4 | EnCodec | Neural compression | Bandwidth-constrained streaming |
| P2-5 | Basic Pitch | Music transcription | Polyphonic analysis |

### Phase 4: Specialized & Research (Months 7+)

| Priority | Model | Capability | Value |
|----------|-------|------------|-------|
| P3-1 | AudioSeal | Deepfake detection | Security research |
| P3-2 | BirdNET | Bioacoustics | Environmental monitoring |
| P3-3 | Qwen2-Audio | Audio reasoning | Multi-modal understanding |

---

## 14. Cost Analysis (API vs Local)

### 14.1 Break-Even Analysis

| Use Case | API Cost/Month | Local Cost (One-Time) | Break-Even Point |
|----------|---------------|----------------------|------------------|
| 100 hours/month ASR | $36 (Whisper API) | $0 (base.en already bundled) | Immediate |
| 1,000 hours/month ASR | $360 (Whisper API) | ~$50 (faster-whisper large-v3) | <1 week |
| 10,000 hours/month ASR | $3,600 (Whisper API) | ~$500 (GPU infra) | <5 days |

### 14.2 Crossover Analysis

API models become cost-disadvantageous at approximately **10,000-50,000 hours/month** for ASR, depending on pricing tier and infrastructure efficiency.

---

## 15. Sources & References

### Official Documentation
- [Mistral AI Voxtral](https://mistral.ai)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [Google Gemini Audio](https://ai.google.dev/gemini-api/docs/audio)
- [Alibaba Model Studio](https://www.alibabacloud.com/en/solutions/generative-ai/qwen)

### GitHub Repositories
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [WhisperX](https://github.com/m-bain/whisperX)
- [stable-ts](https://github.com/jianfch/stable-ts)
- [Step-Audio](https://github.com/stepfun-ai/Step-Audio)
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [AudioCraft](https://github.com/facebookresearch/audiocraft)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [CLAP](https://github.com/LAION-AI/CLAP)
- [Demucs](https://github.com/facebookresearch/demucs)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [BirdNET](https://github.com/kahst/BirdNET)
- [Piper](https://github.com/rhasspy/piper)
- [XTTS](https://huggingface.co/coqui/XTTS-v2)
- [Bark](https://github.com/suno-ai/bark)

### Hugging Face
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [AudioBench](https://huggingface.co/spaces/AudioBench)

### API Services
- [Deepgram](https://deepgram.com)
- [AssemblyAI](https://assemblyai.com)

### Twitter/X Research (February 2026)
- xAI Grok 3 announcements
- Alibaba Qwen3-Coder-Next launch discourse
- ByteDance Seed team release schedule
- DeepSeek V4 anticipation threads
- Manifold Markets prediction tracking

---

## Appendix A: Model Evaluation Criteria

### A.1 Technical Assessment Framework

| Dimension | Methodology | Key Metrics |
|-----------|-------------|-------------|
| Performance | Standardized benchmarks per category | WER/CER (ASR), MOS (TTS), SDR (separation), FAD (generation) |
| Latency | Hardware-specific measurement | First-token latency, time-to-first-audio, sustained throughput |
| Efficiency | Resource utilization analysis | FLOPs, memory footprint, energy consumption |
| Robustness | Out-of-distribution evaluation | Noise, reverberation, accent, domain shift |

### A.2 Integration & Deployment Considerations

| Factor | API Deployment | Local Deployment |
|--------|---------------|------------------|
| Time-to-value | Immediate | Weeks (infrastructure, optimization) |
| Scalability | Automatic | Requires engineering investment |
| Data privacy | External transmission | On-premise control |
| Latency | Network-dependent | Sub-100ms achievable |
| Cost structure | Variable (usage-based) | Fixed (infrastructure) |
| Customization | Limited (prompt engineering) | Extensive (fine-tuning, architecture) |

### A.3 Community & Documentation Evaluation

| Indicator | Tier 1 Threshold | Assessment Method |
|-----------|-----------------|-------------------|
| Active maintenance | Commits within 90 days | GitHub activity, release cadence |
| Documentation completeness | API reference + tutorials + examples | Coverage analysis |
| Community support | <24h median response time | Issue tracker analysis, forum activity |
| Production provenance | 3+ documented deployments | Case study verification |

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| ASR | Automatic Speech Recognition |
| TTS | Text-to-Speech |
| VAD | Voice Activity Detection |
| WER | Word Error Rate |
| CER | Character Error Rate |
| MOS | Mean Opinion Score |
| SDR | Signal-to-Distortion Ratio |
| RTFx | Real-Time Factor (multiple of real-time speed) |
| MoE | Mixture of Experts |
| S2TT | Speech-to-Text Translation |
| S2ST | Speech-to-Speech Translation |
| RVQ | Residual Vector Quantization |
