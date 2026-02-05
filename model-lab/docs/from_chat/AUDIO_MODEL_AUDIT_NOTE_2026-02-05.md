# Audio Model Audit (Comprehensive Survey, 2026) — fulltext (chat capture)

**Date received**: 2026-02-05  
**Provenance**: Pasted by user into chat as “this one?” / “Audio Model Audit (Comprehensive Survey, 2026)”.  
**Evidence status**: **Reported** (not verified by Model‑Lab runs).  

Related artifacts:

- Citations (URLs): `model-lab/docs/from_chat/AUDIO_MODEL_AUDIT_CITATIONS_2026-02-05.md`
- Citations (CSV): `model-lab/data/from_chat/audio_model_audit_citations_2026-02-05.csv`

---

## Raw note (as shared)

Audio Model Audit (Comprehensive Survey, 2026)

Speech Processing Models

Automatic Speech Recognition (ASR)

ASR models convert spoken audio to text. Modern ASR has been revolutionized by deep learning, from end-to-end transformer models like Whisper to cloud APIs offering scalable transcription. Key ASR solutions include both open-source local models and commercial APIs:

OpenAI Whisper (v2/v3, Local): Open-source Transformer model trained on 680k hours of multilingual audio. Supports 99 languages and even direct speech-to-English translation. Strengths: State-of-the-art accuracy (especially English), multi-lingual, and can detect language automatically. Five model sizes (39 M to 1.5 B parameters) allow trade-offs in speed vs. accuracy. Huge community and easy HuggingFace integration. Best Use: High-accuracy transcription when GPU resources are available; multi-language scenarios. Limitations: Tends to hallucinate text not spoken (especially latest v3); very compute-intensive (Whisper-large needs high-end GPU and is slow in real-time). Hardware: GPU with ≥16GB VRAM recommended for large model; smaller models can run on CPU but slowly. License: MIT (very permissive). Integration: Running locally gives full control, but scaling to enterprise use requires significant engineering for deployment.

Google Cloud Speech-to-Text (API): Google’s proprietary ASR service powered by their Universal Speech Model (USM) – a 2 billion-parameter Conformer model trained on 12M hours of speech in 300+ languages. Strengths: Excellent multilingual support, high accuracy for many languages, and speaker diarization, timestamps, etc. Proven at Google-scale (used in YouTube, Assistant). Best Use: When needing a reliable cloud solution with broad language coverage and tight integration with Google’s ecosystem. Limitations: Cost: usage is pay-as-you-go and can be expensive for large volumes (e.g. Google rounds up audio length for billing). Latency and dependency on internet connectivity. Limited insight into model’s internals (fully proprietary). Pricing: ~$1.44 per hour (rounded up) in latest pricing. License: Proprietary (cloud service). Integration: Simple via API call; good documentation.

Microsoft Azure Speech-to-Text (API): Azure’s cloud ASR supporting 100+ languages. Strengths: Enterprise-friendly with customization – you can fine-tune the model with your data or have it learn organization-specific terms. Also offers speaker diarization, punctuation, formatting. Good integration with Azure stack and SDKs. Limitations: Proprietary and must use Azure’s cloud. Fewer languages than Google (100+ vs 300). Pricing is usage-based similar to Google. Best Use: Enterprise projects needing custom vocabulary (e.g. medical or legal jargon) and Azure ecosystem integration. License: Proprietary. Integration: via REST API or Azure SDK; requires Azure account.

Amazon Transcribe (API): AWS’s ASR service (updated model in 2023) supports ~100 languages now (up from 39). Strengths: Offers useful features like domain-specific models (e.g. call center, healthcare), custom vocabulary, automatic language identification, punctuation, and speaker labels. Best Use: If already on AWS or need built-in pipeline for transcribing call audio, etc. Limitations: Also expensive (>$1.00 per audio hour), and can have longer processing times for long files. Quality in under-represented languages improving but not guaranteed. Integration: AWS SDK or console, with JSON transcripts output.

DeepSpeech (Mozilla/Coqui) (Local): An open-source end-to-end ASR (RNN + CTC) originally by Mozilla (based on Baidu’s research). Strengths: Fully offline, lightweight (~120 MB model for English), runs real-time on CPU for shorter utterances. Active community fork by Coqui with continued improvements. Best Use: Embedded systems or cases needing on-device speech-to-text without internet (e.g. Raspberry Pi voice assistant). Limitations: Accuracy is significantly lower than transformer models on open-domain speech (e.g. ~7-10% WER on LibriSpeech test for English). Limited language support (English, some community models for others). Technical: ~50–100M parameters. License: Mozilla Public License 2.0 (open). Integration: Python/C++ libs available (Coqui STT).

Wav2Vec 2.0 (Meta/Facebook AI, Local): A self-supervised Transformer encoder that, when fine-tuned, achieves state-of-the-art ASR for many languages. Strengths: Uses unlabeled audio pretraining to achieve high accuracy with less supervised data. Models like Wav2Vec2-Large (~317M params) approach Whisper-level quality for English. Multi-lingual variants (XLS-R) cover 50+ languages. Best Use: Custom ASR solutions where you can fine-tune on domain-specific audio or languages with limited data. Limitations: Requires model fine-tuning for each language/domain to reach full potential. Higher memory footprint; real-time use may require smaller models or quantization. License: Apache 2.0 (open-source). Integration: Hugging Face Transformers provides ready Wav2Vec2 models.

Others (ASR): Kaldi (Toolkit, Local) – a veteran open-source C++ toolkit using older hybrid HMM/DNN approaches; very customizable and efficient for classical pipelines (needs ML expertise to adapt). Vosk (Local) – lightweight offline recognizer derived from Kaldi, good for quick integration (limited accuracy). AssemblyAI Conformer-2 (API) – AssemblyAI’s own ASR model (Conformer based) available via API, with features like auto highlights and content filtering. Deepgram Nova (API) – end-to-end model optimized for speed. IBM Watson Speech-to-Text (API) – older player with 10+ languages, now eclipsed by others; offers on-prem solutions. Meta MMS ASR – Meta’s Massively Multilingual Speech project released models for ~1,100 languages (many very low-resource); open-source but variable quality.

Text-to-Speech (TTS)

TTS models synthesize speech from text. Options range from cloud services with many voices to open-source models for local deployment:

ElevenLabs (API): A popular commercial TTS known for ultra-realistic, expressive voices. Type: Cloud API with voice cloning. Strengths: Very natural prosody and emotion; supports 74 languages. Allows cloning a voice from a few samples (used for voice actors, content creation). Best Use: When highest quality and easy voice customization are needed (e.g. dialogue for media, AI voice assistants with personality). Limitations: Pricing: subscription + usage credits; can become costly at scale. Voices may sound too expressive for monotone content. License: Proprietary (closed API). Integration: Simple API; SDKs in multiple languages.

Amazon Polly (API): AWS’s TTS with both standard and neural voices. Strengths: Dozens of voices across 36 languages; reliable and fast (medium latency). Offers an inexpensive standard voice option and high-quality neural voices. Best Use: Multi-language applications on a budget, or where AWS integration is convenient. Features: Supports SSML for fine control (pronunciation, emphasis). Limitations: Voices can sound slightly robotic compared to newer services. No built-in voice cloning. Pricing: ~$16 per 1M chars for neural voices (free tier available). Integration: AWS SDK/CLI or REST calls.

Google Cloud Text-to-Speech (API): Google’s TTS with WaveNet voices. Strengths: 75+ language/locale support, many voices including WaveNet and latest Neural2 voices with excellent quality. Offers voice tuning via SSML and Voice Adaptation (limited custom voice ability using your audio data). Best Use: Global applications needing many languages or specific Google voices. Limitations: No fully custom voice cloning for unique voices (only adaptation). Pricing per 1M chars: $16 for WaveNet voices. Integration: Cloud API (JSON), well-documented.

Microsoft Azure TTS (API): Part of Azure Cognitive Services. Strengths: Huge selection – ~400 voices across 140 languages/dialects. Supports Custom Neural Voice creation (you can train a new voice given ~30 minutes of recorded speech, pending Microsoft’s review). Voices are very natural and multilingual. Best Use: Enterprise projects requiring a branded voice or many languages. Limitations: Custom voice creation has an approval process (to prevent misuse). Slightly higher cost for custom and neural voices. Pricing: Pay-per-character, comparable to others (neural ~$16 per 1M chars). Integration: Azure SDK or REST; easily links with other Azure services.

IBM Watson TTS (API): IBM’s neural TTS service. Strengths: High-quality voices aimed at business use (IVR, virtual agents), with strong SSML support. Can create custom voice models for a brand. Best Use: Enterprise deployments needing on-prem or IBM cloud integration. Limitations: Fewer languages than the big cloud competitors (IBM supports ~13 languages, focusing on major markets). Latency can be region-dependent. Pricing: Similar pay-per-char model; IBM often negotiates enterprise pricing. Integration: API and on-prem container options.

Open-Source TTS (Local): Several local models allow running TTS without cloud:

Tacotron 2 + WaveGlow (NVIDIA): A classic two-part model (Tacotron2 synthesizes mel spectrograms, WaveGlow vocoder generates waveform). Produces natural speech given enough training data. Use: Many academic/DIY TTS projects; requires GPU for real-time.

FastSpeech 2 + HiFiGAN: FastSpeech (non-autoregressive) offers lightning-fast synthesis; HiFi-GAN vocoder for high quality. Many pretrained combos available (e.g. via Coqui TTS library).

Coqui TTS (Toolkit): Open-source platform incorporating dozens of models (Tacotron, Glow-TTS, VITS, etc.) and multi-speaker support. Community-driven (originated from Mozilla TTS). Strength: You can fine-tune voices or use pretrained multilingual models.

Tortoise-TTS: A high-quality TTS that can do zero-shot voice cloning (given a reference audio of a speaker) – known for very natural output but slow (autoregressive decoding). Good for small batches or audiobook paragraph synthesis, not real-time.

VITS (End-to-end): An advanced architecture that marries TTS and vocoder in one model (from JSDuck et al.). Some VITS variants can do zero-shot cloning too. Real-time capable if optimized.

Bark (Suno AI, Local/API): A transformer-based text-to-audio model that can produce speech (in various languages) as well as other audio (music, sound effects). It can convey emotion and voice tone, and even non-speech sounds. Limitation: Very resource-heavy to run, and output can be unpredictable.

Strengths of local TTS: Full control over voice and data (no external API or data leakage), can be run offline, no usage fees. Limitations: Typically need GPU for inference; quality may lag behind top cloud voices unless carefully tuned. Also, cloning a specific voice requires training data and ML expertise (except for few-shot models like Tortoise or Bark, which still aren’t as consistent as ElevenLabs).

Voice Cloning & Voice Conversion

These models aim to replicate or convert voices. Voice cloning generates speech in someone’s specific voice, while voice conversion transforms audio spoken by Speaker A to sound as if Speaker B said it:

ElevenLabs & Resemble AI (API): Both provide high-quality voice cloning as a service. ElevenLabs allows creating a custom voice from a few minutes of audio. Resemble AI similarly can clone voices and even offers a Speech-to-Speech mode (input one person’s speech and output another’s voice saying the same). They also have features to control emotions or mix voices. Use cases: Content creation (dubbing, audiobooks with multiple voices), personalized virtual assistants. Limitations: Cloud-based (data privacy considerations), and potential for abuse means these companies have ethics checks in place. Pricing is enterprise-level (Resemble offers on-prem licensing too).

Azure Custom Neural Voice: As noted, Microsoft’s service for custom TTS essentially is voice cloning – you provide training recordings of a target voice. It produces a synthetic voice nearly indistinguishable from the original, used in games and assistants. Limitation: Requires significant setup and Microsoft’s approval to ensure consent of voice owner.

Open Source Voice Cloning:

CorentinJ Real-Time Voice Cloning (RTVC): A popular three-stage pipeline (Speaker Encoder + Synthesizer + Vocoder) based on SV2TTS. It can clone a voice from just a ~5 second sample to generate arbitrary phrases. Strength: Quick demo-able results; runs on a GPU in real-time. Limitations: Quality is moderate (noticeable artifacts) and voice similarity isn’t perfect for very short samples; works best with a minute or more of reference audio.

YourTTS (2022, by Coqui): A multilingual zero-shot voice cloning model built on VITS. It achieved strong results in research, able to clone voices across languages (speak English text in the style of a Spanish speaker, etc.). Available on HuggingFace. Limitations: Requires GPU; some voices clone better than others depending on similarity to training data.

OpenVoice (2023, MIT): A fast voice cloning model that allows fine-grained control of output speech style (emotion, accent). It’s open-source and can clone from short samples. Still experimental but promising for real-time applications.

Voice Conversion models: e.g. AutoVC, FragmentVC, StarGAN-VC – these take an input speech and convert it to a target voice timbre. They often work by extracting content vs. speaker factors. Use: Changing a narrator’s voice without re-recording. Some are one-shot (requiring one example of target voice), others need training on paired data. Quality can be high for singing voice conversion (singer style transfer) as seen in AI cover song trends (e.g. so-vits-svc tool for song voice conversion).

Strengths: Open tools allow research and customization (e.g. building a voice clone for someone with many recordings, such as for a custom audiobook). Limitations: Ethical concerns – these models can be misused to impersonate voices. Many open models don’t have built-in safeguards. Quality often depends on having clean training data of the target voice; emotional expressiveness may be limited to what’s in that data. Real-time conversion is challenging but some models (RTVC) approach it.

Speech Enhancement & Noise Reduction

These models improve speech intelligibility by removing noise, echo, or other artifacts from audio:

RNNoise (Xiph/Mozilla, 2017): A pioneering small-footprint model combining classic DSP with a recurrent neural network for noise suppression. It’s lightweight enough to run in real-time on a CPU. Strengths: Very low resource usage, still effective at reducing constant background noise (e.g. fan noise). Limitations: No longer actively maintained, and not optimized for non-stationary noises (might struggle with speech-like background or sudden sounds).

DeepFilterNet (v2, 2022): A deep learning noise suppression model from Univ. of Erlangen-Nuremberg. It uses a two-stage filtering approach and achieves excellent noise removal on full-band (48 kHz) audio. Strengths: Open-source and shown to outperform RNNoise in quality by using a larger neural network. Hardware: Can run real-time on a decent CPU for 48 kHz mono audio. Limitations: Higher latency and compute than simpler methods; not as tiny as RNNoise.

NVIDIA Audio Effects (RTX Voice): Part of NVIDIA’s Maxine SDK, it provides high-quality noise removal using proprietary deep models (accessible via the Audio Effects SDK). Strengths: Extremely effective at removing all sorts of background noise (keyboard typing, etc.) in real-time, leveraging GPU Tensor Cores. Best Use: Desktop applications or call software where an NVIDIA GPU is available. Limitations: Requires an NVIDIA GPU (with Tensor cores for best performance); integration requires acknowledging NVIDIA and is closed-source.

Krisp SDK: A commercial SDK derived from the Krisp app, offering on-device AI noise suppression for voice calls. Strengths: Device-friendly (supports Windows, Mac, Linux, mobile) and works in real-time to cancel background voices, noise, etc. Limitations: Not publicly downloadable – one must request access. Also not customizable (black-box model).

Picovoice Koala: A cross-platform deep noise suppression engine by Picovoice. Strengths: Runs on many platforms (including web browsers and even Raspberry Pi) and is free for developers (with paid enterprise support). Good for embedding in apps where multi-platform support is needed. Limitations: As a newer solution, community size is smaller than RNNoise.

Speech Enhancement for Calls: Big tech also deploys their own models – e.g. Microsoft Teams uses a DNN noise suppression (likely derived from NSNet2 which was open-sourced in WebRTC), and Google Meet uses learned suppression. These are not directly available as standalone models but demonstrate the state-of-art when integrated.

Echo Cancellation & Room Enhancement: Specialized models address echo removal (for full-duplex calls) and dereverberation. E.g. Facebook’s denoiser library includes a deep learning dereverb. There are also speech bandwidth expansion models (to infer high frequencies lost in narrowband audio). These are niche but important for telephony and hearing aids.

Use cases: noise-canceling in video conferencing, improving ASR input quality, hearing aids (to suppress background din), cleaning up podcast audio. Strengths: Modern deep models significantly outperform traditional spectral subtraction or Wiener filters, especially on non-stationary noise. Limitations: Many are specialized to human speech frequencies and may not generalize to music or other audio. Some introduce slight artifacts or “warbling” at high suppression levels. Also, deep models can be heavy for mobile unless optimized.

Music Processing Models

Music Generation

Generative models that create music or audio from either random seeds or text prompts:

Meta AudioCraft (MusicGen & AudioGen) (Open-Source, Local): In 2023, Meta released AudioCraft, a suite of generative audio models:

MusicGen: Generates music from text descriptions (e.g. “upbeat EDM with a strong bassline”). Trained on 20k hours of licensed music, ~400k recordings. It produces coherent short instrumental music and can be steered by specifying genres or instruments. Strengths: Open model (weights available) that produces relatively high-quality music without needing a huge diffusion model. Fast for short clips (uses a Transformer decoder on compressed audio tokens). Limitations: Cannot include vocals (training vocals were removed); tends to imitate training data styles (risk of regurgitating melodies); length is limited (generates ~30 sec by default, though can be extended by looping).

AudioGen: Generates environmental sound effects from text (e.g. “footsteps on wood floor” or “rainstorm with thunder”). It’s a diffusion model for general audio (non-music). Use: Foley sound creation, background ambience generation. Limitations: As a generative model, outcomes vary and might need cherry-picking; some complex sound scenes still hard to get perfect.

EnCodec: A neural audio codec used as a foundation (it tokenizes audio into low-bitrate embeddings). Enables MusicGen/AudioGen to work in the discrete token domain. Also: EnCodec itself is useful for compression tasks (see Audio Codecs section).

Being open-source, these can be run locally on GPUs. Best Use: R&D or creative tools where on-prem generation is needed. Limitations: Require high vRAM for longer sequences; not yet as musically structured as human-composed pieces for long compositions.

OpenAI Jukebox (2020, Open-Source): A pioneering model that generates music with vocals in specific artist styles. It’s a large VAE+Transformer that can produce minute-long songs with lyrics (often nonsensical words). Strengths: Uncannily captures timbre and style of famous singers/bands. Limitations: Extremely resource-intensive (to the point that few can run it); lyrics are gibberish; lack fine control (requires a prompt or genre conditioning, but output is random). More a proof-of-concept.

Stability AI – Stable Audio (API / Open Model): Stability.ai launched Stable Audio in 2023, a latent diffusion model for audio. By 2025 they released Stable Audio 2.5, geared for enterprise sound production. Strengths: Can generate longer pieces (up to 90s or even minutes) and fast – e.g. 3-minute music in 2 seconds on powerful GPUs. Focuses on coherence and prompt adherence (e.g. following instructions like mood “uplifting” or instruments “synthesizers” better). Use: Media content creators needing quick background music or jingles. Limitations: Closed source for latest versions (offered via API or limited “Stable Audio Open” model). Quality is good for ambient and rhythmic music, but fully satisfying compositions still challenging. Also, like image models, it can inadvertently resemble training data melodies (licensed data to avoid copyright issues).

Google MusicLM (Research, experimental): A text-to-music model by Google AI (2023) that demonstrated high-quality music generation from text descriptions and hummed melodies. Google released a dataset (MusicCaps) but not the full model due to ethical concerns. However, a watered-down version or parts of it have been made available (or reproduced by others) in 2024. Strengths: Very descriptive prompt handling (e.g. “classical violin solo, 1800s style”). Limitations: Not publicly usable fully; but elements of it might appear in Google’s products eventually.

Other Notables: Riffusion – an inventive hack using Stable Diffusion (image generator) on spectrogram images to produce audio loops (often electronic or ambient music). Dance Diffusion – early diffusion models for music by Harmonai (Stability AI) focusing on beats and loops. Magenta Project (Google) – a collection of older models like MelodyRNN, PerformanceRNN, MusicVAE for generating melodies and drum beats. These are less state-of-art now but useful for certain constrained music generation (e.g. MusicVAE for interpolating between melodies). AIVA (commercial) – an AI composer known for generating classical music pieces (uses proprietary models, perhaps based on LSTMs). Mubert (API) – generates music streams via a combination of AI and human-designed loops, often used for background music in apps. Amplitude Studios’ HarmonyGPT (fictitious example for completeness) – not a real model, but some startups are exploring using GPT-like architectures to generate MIDI or compositions.

Use cases: background music for videos, game soundtracks on-the-fly, assisting composers with ideas, personalized music for users. Strengths: AI can produce infinite variations and has no licensing fees if trained on public/domain data. Limitations: Quality and structure – AI music can lack the long-term structure or novelty of human compositions, often requiring human post-editing. Also, risk of IP issues if models trained on copyrighted music (many efforts now train on fully licensed or royalty-free datasets to mitigate this).

Music Transcription

This involves converting music audio into a human-readable form (notes, MIDI, tabs):

Onsets and Frames (Google/Magenta, 2018): A deep learning model for piano transcription. It uses separate networks to detect note onsets and to track their frames (sustain). Strengths: Achieves near state-of-art for piano (on MAESTRO dataset). Open source implementation available. Limitations: Specific to piano; struggles with polyphonic instruments beyond piano.

MT3 (Multi-Task Multitrack Music Transcription) (Google, 2021): A Transformer model that can transcribe multiple instruments and even drum tracks from music audio. It treats transcription as a sequence prediction (like language translation from audio to note sequences). Strengths: More general – can output notes for different instruments given a single model. Limitations: Complex and heavy model; not as accessible as older approaches.

Spleeter/Demucs (Source Separation): While not transcription per se, these models separate music into stems (vocals, drums, bass, etc.). Once separated, one could transcribe each stem more easily. Strength: Useful pre-processing for transcription (e.g. separate a guitar track and then run a specialized model to get guitar tabs). Note: Demucs is an excellent open-source separation model by Facebook (Meta) that can isolate vocals and instruments from mixes.

Melody Extraction (CREPE): CREPE is a model for detecting pitch (frequency) of the predominant melody in real-time. Good for getting a vocal melody line or single-note sequence from polyphonic audio. Use: Transcribing a singer’s melody, or a single instrument in mix.

Academic and Commercial tools: Melodyne (commercial software) can do polyphonic pitch extraction (with some AI help) for editing audio – often used in studios. GuitarPro and others use algorithms to get guitar/bass tabs from audio (some incorporate ML now). MuseScore’s Audiveris (open-source) tries audio-to-sheet (with limited success).

Use cases: Writing sheet music from recordings, assistive tools for musicians (transcribe a hummed tune to notes), music education (analyzing student performance). Challenges: Polyphonic transcription (multiple instruments at once) remains very hard – errors increase with overlapping notes. Drum transcription is a separate subtask (detect drum hit times and types). The best results often come from hybrid methods (signal processing + ML).

Audio Classification & Tagging

Beyond music, audio classification covers environmental sounds, events, and even biological sounds:

General Audio Tagging (AudioSet models): Google’s AudioSet (a large dataset of 2+ million 10-second clips labeled with 527 sound classes) enabled training of general audio classifiers. Models like AST (Audio Spectrogram Transformer) and PANNs provide pretrained backbones. Strength: They can recognize a variety of everyday sounds (dog bark, siren, rain, baby crying, etc.) in a clip. Use: Adding sound recognition to devices (e.g. a camera that hears glass breaking), enriching media content with sound effect labels, or assisting the hearing impaired by detecting important sounds (doorbell ringing).

Environmental Sound Classification (ESC): Datasets like ESC-50 or UrbanSound8k focus on 50 common environment sounds. Many CNN models (some using pretrained ImageNet features on spectrograms) achieve >90% accuracy on ESC-50 now. Examples: A model might classify audio into categories like “thunderstorm”, “engine noise”, “fire crackling”.

Audio Event Detection (AED): This goes beyond tagging an entire clip – it detects when certain events happen (with timestamps). E.g. in a long audio, detect instances of “gunshot” or “dog bark”. Often tackled with CNN or RNN models in a sliding window, or transformer models that output sequences of event tags with time. There are yearly DCASE challenges in this domain, with top models combining convolutional layers and attention. Use: Surveillance (acoustic gunshot detection), monitoring public spaces for specific sounds (glass break alarms), or smart home devices that react to events like a smoke alarm sound.

Bioacoustics Models: Specialized classifiers for animal and biological sounds. For instance, bird call identification – models (often CNNs on spectrograms) trained on bird song datasets (e.g. Cornell BirdCLEF data) to identify species from audio. One known tool is BirdNET (Cornell) which uses a neural network to recognize ~3k bird species by sound. Similarly, marine mammal sound classifiers identify whale or dolphin calls (used in conservation to track populations). Use: Ecology research, biodiversity monitoring with autonomous recording units. Limitations: Often need retraining for each region or new species; background noise (wind, rain, other animals) can confuse models.

Audio Codec & Compression

Neural audio codecs compress audio to lower bitrates than traditional codecs while preserving quality:

Meta EnCodec: A high-fidelity neural audio compression model. It uses an encoder-decoder model with quantized latent vectors. EnCodec can compress full-band audio (48 kHz stereo) down to very low bitrates (e.g. 6 kbps) while maintaining good quality, outperforming older codecs like MP3 at similar sizes. Use: Efficient streaming or storage of music and speech. Meta uses EnCodec in MusicGen/AudioGen as a tokenizer. It’s open-sourced (as part of AudioCraft) and can run in real-time. Specs: Processes audio in frames with a convolutional encoder; the latest version uses multi-scale approaches to minimize artifacts. Limitations: At extremely low bitrates, you still get some artifacting (often a slight watery or muffled texture). Real-time encoding/decoding may require a decent CPU or a neural accelerator.

Google Lyra & SoundStream: Lyra (by Google) is a neural codec specialized for low-bitrate speech (originally ~3 kbps) targeting telephony. SoundStream is the later general model (also by Google) that can do 3 kbps speech or higher quality at 6–12 kbps for full audio. Google has deployed these in some Duo/Meet calls to save bandwidth. Strengths: In voice calls, 3 kbps neural coded speech sounds better than traditional codecs at 3 kbps. Limitations: Neural codecs can have gating artifacts if the model mis-predicts; also initial latency for a codec might be higher (need a frame of audio before decode).

Audio Forensics & Security

This area includes detecting fake or tampered audio and ensuring authenticity:

Deepfake Audio Detection: With voice cloning tech, detecting AI-generated or manipulated speech is crucial. Approaches typically involve training classifiers to distinguish real vs fake, often using spectral features or artifacts. Open source: a few HuggingFace models fine-tuned for deepfake audio detection exist (often using efficient nets or Wav2Vec as base). Challenges: As generation models improve, fakes become harder to detect – it’s an arms race. Detection models need constant updates with new types of fakes. Also, many work well in controlled settings but fail on real-world audio (phone quality, background noise).

Watermarking and Authentication: New methods propose embedding an inaudible watermark in audio generated by AI, so it can later be identified.

Multimodal Audio Models

These models combine audio with other modalities (text, vision, etc.):

Audio-Language Models: These bridge audio and natural language. A prime example is CLAP (LAION, 2023) – which learned a joint embedding for audio and text (similar to how CLIP did for image-text). With CLAP, you can do things like retrieve sounds by text description or vice versa.

Summary and Implementation Priorities

Organizing by category and priority: The above audit presented models across the audio AI landscape. In determining the next models to implement for a speech/audio lab, one should consider complementing existing capabilities and expanding into new domains. Cost-benefit for APIs vs local: Implementing an API-based model is quick integration-wise but incurs ongoing costs and depends on external service. A local open-source model may require more engineering (for optimization and scaling) but can be cheaper long-term and allow customization. Hardware considerations: If aiming to support real-time processing, some models need GPUs or specialized hardware.

Category\tModel (Type)\tPrimary Strengths\tUse Case Fit\tLimits\tPriority (Lab)
ASR\tWhisper (Local)\tSOTA accuracy, multilingual\tTranscription, captions\tCompute-heavy, hallucinations\tHigh (if not in use)
Google Cloud STT (API)\tBroad language, easy to scale\tMulti-lang, quick integration\tExpensive\tMedium (backup option)
TTS\tElevenLabs (API)\tVery natural, cloning enabled\tVoice UX, avatars\tCost, cloud-only\tHigh (for voice variety)
Coqui TTS (Local)\tOpen-source, many models\tCustom voices, offline TTS\tNeeds GPU for quality\tMedium
Voice Cloning\tResemble AI (API/on-prem)\tHigh-quality cloning, multi-lingual\tPersonalized voice bots\tCostly, must ensure consent\tHigh (if personalization needed)
RTVC (Local)\tQuick clone from sample, open-source\tDemos, low-resource cloning\tArtifacts, moderate quality\tMedium
Speech Enhancement\tDeepFilterNet2 (Local)\tStrong noise reduction (open)\tImprove call audio, ASR input\tSome latency/CPU load\tHigh (improves all speech tasks)
Music Generation\tAudioCraft MusicGen (Local)\tGenerates music from text, open\tCreative tools, assets\tLimited length, no vocals\tMedium (new domain expansion)
Stable Audio (API)\tHigh-quality, longer music gen\tContent creation for clients\tClosed, must use API\tLow-Med (depending on need)
Audio Classification\tPANNs model (Local)\tGeneral audio tagging\tSound event awareness\tNeeds adaptation to domain\tMedium
BirdNET (Local or API)\tBird species ID (state-of-art)\tWild sound monitoring\tNiche (if project needs it)\tLow (unless needed)
Audio Forensics\tFake audio detector (Local)\tDetects AI-generated speech patterns\tTrust and safety features\tNew types of fakes evolving\tMedium (important if voice security is a concern)

Sources: The information above was gathered and synthesized from a wide array of sources, including model documentation, research papers, and industry benchmarks – ensuring the audit is up-to-date as of 2025-2026.

