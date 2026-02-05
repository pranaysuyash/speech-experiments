# Audio Model Landscape Audit — “47 production-ready models” (chat capture)

**Date received**: 2026-02-05  
**Provenance**: Pasted by user into chat as an alternate (“or this?: …”) long-form audit writeup.  
**Evidence status**: **Reported** (not verified by Model‑Lab runs).  

Related artifact (process trace as pasted):

- `model-lab/docs/from_chat/AUDIO_MODEL_LANDSCAPE_AUDIT_TRACE_2026-02-05.md`

---

## Raw note (as shared)

TL;DR: This audit identifies 47 production-ready audio AI models across 9 categories, with Tier 1 immediate implementation recommendations including Whisper (multilingual ASR), Demucs (source separation), wav2vec 2.0 (self-supervised speech), and CosyVoice2-0.5B (edge TTS). The analysis reveals a bifurcated market: open-source models (Meta, Alibaba, community) dominate customization and privacy-sensitive deployments, while API services (OpenAI, Google) offer rapid prototyping at premium pricing. Key strategic insight: unified audio-language models (gpt-audio, Qwen2-Audio, Step-Audio 2 mini) are displacing pipeline architectures, enabling 10-100x latency reductions for voice agents.

Comprehensive Audio Model Audit for Speech Experiments Model Lab

1. Executive Summary & Implementation Framework

1.1 Audit Scope and Objectives

The audio artificial intelligence landscape has undergone transformative expansion beyond traditional speech processing, encompassing neural audio codecs, generative music systems, multimodal language models, and specialized forensic tools. This audit systematically evaluates 47 production-ready models across nine major categories: speech processing, music generation, audio classification, enhancement and restoration, neural codecs, forensics and security, multimodal integration, and specialized applications including bioacoustics and industrial monitoring. The scope deliberately transcends conventional automatic speech recognition (ASR) and text-to-speech (TTS) boundaries to capture capabilities that enable next-generation research in audio understanding, manipulation, and cross-modal reasoning.

The primary objective centers on identifying models with demonstrated production deployment, comprehensive documentation, active community support, and clear Python integration pathways. The evaluation framework prioritizes local deployment flexibility for privacy-sensitive and latency-critical applications while acknowledging the rapid-prototyping value of managed API services. Special emphasis is placed on models that expand laboratory capabilities beyond speech-centric paradigms, enabling research into music information retrieval, environmental sound analysis, audio forensics, and real-time processing under bandwidth constraints. The audit explicitly addresses the strategic inflection point in audio AI: the emergence of unified end-to-end models that replace cascaded pipelines (ASR→LLM→TTS) with single-architecture solutions offering order-of-magnitude latency improvements and superior preservation of paralinguistic information.

1.2 Implementation Priority Tiers

Tier 1: Immediate implementation (0-2 months): Whisper, wav2vec 2.0, Demucs, CosyVoice2-0.5B, YAMNet  
Tier 2: Short-term evaluation (3-4 months): Qwen2-Audio, gpt-audio, Step-Audio 2 mini, JASCO, MusicGen  
Tier 3: Long-term monitoring (6-12 months): AudioSeal, BirdNET, neural codecs (SoundStream, Lyra), industrial audio systems

2. Speech Processing Models

2.1 Automatic Speech Recognition (ASR)

2.1.1 wav2vec 2.0 (Meta/Facebook AI Research)

Version: 2.0  
Type: Local  
Parameters: 95M (BASE), 317M (LARGE), 300M-2B (XLS-R variants)  
Hosting: Hugging Face, PyTorch Hub, fairseq  
License: MIT (code), CC-BY-NC 4.0 (some pretrained models)  

2.1.4 Whisper (OpenAI)

Version: v3 (large-v3, Sep 2023)  
Type: API and Local  
Parameters: 39M (tiny) → 1.55B (large-v3)  
Hosting: OpenAI API, Hugging Face (open weights)  
License: MIT (weights), proprietary (API)

2.2 Text-to-Speech (TTS)

Seamless Streaming (Meta), Qwen3-TTS-Flash-Realtime (Alibaba), CosyVoice2-0.5B, Fish Speech V1.5, IndexTTS-2

2.4 Speech Enhancement & Noise Reduction

Demucs (Meta), DeepFilterNet (open-source)

3. Music Processing Models

AudioLM (Google), Lyria 2 / Lyria RealTime (Google), JASCO (Meta), MusicGen (Meta/Audiocraft)

4. Audio Classification & Tagging

YAMNet (Google), VGGish (Google), PANNs, OpenL3

6. Audio Codec & Compression

SoundStream (Google), EnCodec (Meta), Lyra (Google)

8. Multimodal Audio Models

gpt-audio (OpenAI), Qwen2-Audio (Alibaba), Step-Audio 2 mini (Stepfun-AI), SALMONN

11. Implementation Roadmap

Phase 1: Core Speech Infrastructure (Months 1-2): Whisper, wav2vec 2.0, Demucs, CosyVoice2-0.5B  
Phase 2: Multimodal & Generation (Months 3-4): Qwen2-Audio or gpt-audio, MusicGen or JASCO, CLAP  
Phase 3: Specialized (Months 5-6): AudioSeal, BirdNET, EnCodec or SoundStream

(The remainder of the note continues in the original chat paste; preserved as-is in the chat transcript.)

