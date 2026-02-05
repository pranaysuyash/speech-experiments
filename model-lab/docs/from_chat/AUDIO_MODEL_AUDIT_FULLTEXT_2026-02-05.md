# Audio Model Audit (Comprehensive Survey, 2026) — fulltext (chat-provided)

**Date received**: 2026-02-05  
**Provenance**: Pasted by user into chat as “this one?: Audio Model Audit (Comprehensive Survey, 2026) …”.  
**Evidence status**: **Reported** (not verified in Model‑Lab runs).

Notes:

- This file preserves the raw narrative for traceability.
- Benchmark numbers and pricing in this text should not be treated as **Observed** unless reproduced via `model-lab/runs/**`.

---

## Raw note (as shared)

Audio Model Audit (Comprehensive Survey, 2026)
Speech Processing Models
Automatic Speech Recognition (ASR)
ASR models convert spoken audio to text. Modern ASR has been revolutionized by deep learning, from end-to-end transformer models like Whisper to cloud APIs offering scalable transcription. Key ASR solutions include both open-source local models and commercial APIs:
OpenAI Whisper (v2/v3, Local): Open-source Transformer model trained on 680k hours of multilingual audio
. Supports 99 languages and even direct speech-to-English translation. Strengths: State-of-the-art accuracy (especially English), multi-lingual, and can detect language automatically
. Five model sizes (39 M to 1.5 B parameters) allow trade-offs in speed vs. accuracy
. Huge community and easy HuggingFace integration. Best Use: High-accuracy transcription when GPU resources are available; multi-language scenarios. Limitations: Tends to hallucinate text not spoken (especially latest v3)
; very compute-intensive (Whisper-large needs high-end GPU and is slow in real-time). Hardware: GPU with ≥16GB VRAM recommended for large model; smaller models can run on CPU but slowly. License: MIT (very permissive). Integration: Running locally gives full control, but scaling to enterprise use requires significant engineering for deployment
.
Google Cloud Speech-to-Text (API): Google’s proprietary ASR service powered by their Universal Speech Model (USM) – a 2 billion-parameter Conformer model trained on 12M hours of speech in 300+ languages
. Strengths: Excellent multilingual support, high accuracy for many languages, and speaker diarization, timestamps, etc. Proven at Google-scale (used in YouTube, Assistant)
. Best Use: When needing a reliable cloud solution with broad language coverage and tight integration with Google’s ecosystem. Limitations: Cost: usage is pay-as-you-go and can be expensive for large volumes (e.g. Google rounds up audio length for billing)
. Latency and dependency on internet connectivity. Limited insight into model’s internals (fully proprietary). Pricing: ~$1.44 per hour (rounded up) in latest pricing
. License: Proprietary (cloud service). Integration: Simple via API call; good documentation.
Microsoft Azure Speech-to-Text (API): Azure’s cloud ASR supporting 100+ languages
. Strengths: Enterprise-friendly with customization – you can fine-tune the model with your data or have it learn organization-specific terms
. Also offers speaker diarization, punctuation, formatting. Good integration with Azure stack and SDKs. Limitations: Proprietary and must use Azure’s cloud. Fewer languages than Google (100+ vs 300)
. Pricing is usage-based similar to Google. Best Use: Enterprise projects needing custom vocabulary (e.g. medical or legal jargon) and Azure ecosystem integration. License: Proprietary. Integration: via REST API or Azure SDK; requires Azure account.
Amazon Transcribe (API): AWS’s ASR service (updated model in 2023) supports ~100 languages now (up from 39)
. Strengths: Offers useful features like domain-specific models (e.g. call center, healthcare)
, custom vocabulary, automatic language identification, punctuation, and speaker labels
. Best Use: If already on AWS or need built-in pipeline for transcribing call audio, etc. Limitations: Also expensive (>$1.00 per audio hour)
, and can have longer processing times for long files
. Quality in under-represented languages improving but not guaranteed
. Integration: AWS SDK or console, with JSON transcripts output.
DeepSpeech (Mozilla/Coqui) (Local): An open-source end-to-end ASR (RNN + CTC) originally by Mozilla (based on Baidu’s research)
. Strengths: Fully offline, lightweight (~120 MB model for English), runs real-time on CPU for shorter utterances. Active community fork by Coqui with continued improvements. Best Use: Embedded systems or cases needing on-device speech-to-text without internet (e.g. Raspberry Pi voice assistant). Limitations: Accuracy is significantly lower than transformer models on open-domain speech (e.g. ~7-10% WER on LibriSpeech test for English). Limited language support (English, some community models for others). Technical: ~50–100M parameters. License: Mozilla Public License 2.0 (open). Integration: Python/C++ libs available (Coqui STT).
Wav2Vec 2.0 (Meta/Facebook AI, Local): A self-supervised Transformer encoder that, when fine-tuned, achieves state-of-the-art ASR for many languages. Strengths: Uses unlabeled audio pretraining to achieve high accuracy with less supervised data. Models like Wav2Vec2-Large (~317M params) approach Whisper-level quality for English
. Multi-lingual variants (XLS-R) cover 50+ languages. Best Use: Custom ASR solutions where you can fine-tune on domain-specific audio or languages with limited data. Limitations: Requires model fine-tuning for each language/domain to reach full potential. Higher memory footprint; real-time use may require smaller models or quantization. License: Apache 2.0 (open-source). Integration: Hugging Face Transformers provides ready Wav2Vec2 models.
Others (ASR): Kaldi (Toolkit, Local) – a veteran open-source C++ toolkit using older hybrid HMM/DNN approaches; very customizable and efficient for classical pipelines (needs ML expertise to adapt). Vosk (Local) – lightweight offline recognizer derived from Kaldi, good for quick integration (limited accuracy). AssemblyAI Conformer-2 (API) – AssemblyAI’s own ASR model (Conformer based) available via API
, with features like auto highlights and content filtering. Deepgram Nova (API) – end-to-end model optimized for speed
. IBM Watson Speech-to-Text (API) – older player with 10+ languages, now eclipsed by others; offers on-prem solutions. Meta MMS ASR – Meta’s Massively Multilingual Speech project released models for ~1,100 languages (many very low-resource); open-source but variable quality.
Text-to-Speech (TTS)
TTS models synthesize speech from text. Options range from cloud services with many voices to open-source models for local deployment:
ElevenLabs (API): A popular commercial TTS known for ultra-realistic, expressive voices. Type: Cloud API with voice cloning. Strengths: Very natural prosody and emotion; supports 74 languages
. Allows cloning a voice from a few samples (used for voice actors, content creation). Best Use: When highest quality and easy voice customization are needed (e.g. dialogue for media, AI voice assistants with personality). Limitations: Pricing: subscription + usage credits
; can become costly at scale. Voices may sound too expressive for monotone content. License: Proprietary (closed API). Integration: Simple API; SDKs in multiple languages.
Amazon Polly (API): AWS’s TTS with both standard and neural voices. Strengths: Dozens of voices across 36 languages
; reliable and fast (medium latency)
. Offers an inexpensive standard voice option and high-quality neural voices. Best Use: Multi-language applications on a budget, or where AWS integration is convenient. Features: Supports SSML for fine control (pronunciation, emphasis). Limitations: Voices can sound slightly robotic compared to newer services
. No built-in voice cloning. Pricing: ~$16 per 1M chars for neural voices
 (free tier available). Integration: AWS SDK/CLI or REST calls.
Google Cloud Text-to-Speech (API): Google’s TTS with WaveNet voices. Strengths: 75+ language/locale support
, many voices including WaveNet and latest Neural2 voices with excellent quality
. Offers voice tuning via SSML and Voice Adaptation (limited custom voice ability using your audio data). Best Use: Global applications needing many languages or specific Google voices. Limitations: No fully custom voice cloning for unique voices (only adaptation). Pricing per 1M chars: $16 for WaveNet voices
. Integration: Cloud API (JSON), well-documented.
Microsoft Azure TTS (API): Part of Azure Cognitive Services. Strengths: Huge selection – ~400 voices across 140 languages/dialects
. Supports Custom Neural Voice creation (you can train a new voice given ~30 minutes of recorded speech, pending Microsoft’s review). Voices are very natural and multilingual. Best Use: Enterprise projects requiring a branded voice or many languages. Limitations: Custom voice creation has an approval process (to prevent misuse)
. Slightly higher cost for custom and neural voices. Pricing: Pay-per-character, comparable to others (neural ~$16 per 1M chars). Integration: Azure SDK or REST; easily links with other Azure services.
IBM Watson TTS (API): IBM’s neural TTS service. Strengths: High-quality voices aimed at business use (IVR, virtual agents), with strong SSML support
. Can create custom voice models for a brand. Best Use: Enterprise deployments needing on-prem or IBM cloud integration. Limitations: Fewer languages than the big cloud competitors
 (IBM supports ~13 languages, focusing on major markets). Latency can be region-dependent
. Pricing: Similar pay-per-char model; IBM often negotiates enterprise pricing. Integration: API and on-prem container options.
Open-Source TTS (Local): Several local models allow running TTS without cloud:
Tacotron 2 + WaveGlow (NVIDIA): A classic two-part model (Tacotron2 synthesizes mel spectrograms, WaveGlow vocoder generates waveform). Produces natural speech given enough training data. Use: Many academic/DIY TTS projects; requires GPU for real-time.
FastSpeech 2 + HiFiGAN: FastSpeech (non-autoregressive) offers lightning-fast synthesis; HiFi-GAN vocoder for high quality. Many pretrained combos available (e.g. via Coqui TTS library).
Coqui TTS (Toolkit): Open-source platform incorporating dozens of models (Tacotron, Glow-TTS, VITS, etc.) and multi-speaker support. Community-driven (originated from Mozilla TTS). Strength: You can fine-tune voices or use pretrained multilingual models.
Tortoise-TTS: A high-quality TTS that can do zero-shot voice cloning (given a reference audio of a speaker) – known for very natural output but slow (autoregressive decoding). Good for small batches or audiobook paragraph synthesis, not real-time.
VITS (End-to-end): An advanced architecture that marries TTS and vocoder in one model (from JSDuck et al.). Some VITS variants can do zero-shot cloning too. Real-time capable if optimized.
Bark (Suno AI, Local/API): A transformer-based text-to-audio model that can produce speech (in various languages) as well as other audio (music, sound effects). It can convey emotion and voice tone, and even non-speech sounds. Limitation: Very resource-heavy to run, and output can be unpredictable.
Strengths of local TTS: Full control over voice and data (no external API or data leakage), can be run offline, no usage fees. Limitations: Typically need GPU for inference; quality may lag behind top cloud voices unless carefully tuned. Also, cloning a specific voice requires training data and ML expertise (except for few-shot models like Tortoise or Bark, which still aren’t as consistent as ElevenLabs).

(The remainder of the original chat note continues in the chat transcript; this capture preserves the portion pasted in the IDE.)

