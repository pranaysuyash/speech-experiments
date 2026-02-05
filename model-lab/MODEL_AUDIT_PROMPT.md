# Audio Model Audit Prompt

## Objective
Conduct a comprehensive audit of ALL available audio-related models (both local and API-based) to identify the next set of models to implement in the speech experiments model lab. The audit should cover the complete audio landscape, not just speech processing models.

## Information to Collect for Each Model
- Model name and version
- Type: Local or API-based
- Category: ASR, TTS, Music Generation, Audio Classification, Audio Enhancement, Voice Cloning, Audio Forensics, Music Transcription, Audio Codecs, Multimodal Audio, Bioacoustics, Audio Restoration, etc.
- Hosting: Hugging Face, OpenAI, Google, Meta, IBM, etc.
- Primary use cases and strengths
- Best use cases (specific scenarios where it excels)
- Limitations and scenarios where it should not be used
- Technical specifications (parameters, model size, etc.)
- Hardware requirements (for local models)
- Pricing information (for API models)
- Performance benchmarks
- License information
- Community support and documentation
- Integration complexity
- Dataset requirements and compatibility

## Categories to Cover (Comprehensive Audio Landscape)
- Speech Processing Models
  - ASR (Automatic Speech Recognition)
  - TTS (Text-to-Speech)
  - Voice Cloning & Conversion
  - Speech Enhancement & Noise Reduction
- Music Processing Models
  - Music Generation
  - Music Transcription
  - Music Analysis & Classification
- Audio Classification & Tagging
  - Environmental Sound Classification
  - Audio Event Detection
  - Bioacoustics Models
- Audio Enhancement & Restoration
  - Noise Suppression
  - Audio Super Resolution
  - Audio Restoration
- Audio Codec & Compression
  - Neural Audio Codecs
  - Real-time Audio Processing
- Audio Forensics & Security
  - Deepfake Detection
  - Audio Tampering Detection
- Multimodal Audio Models
  - Audio-Language Models
  - Audio-Visual Models
- Specialized Audio Applications
  - Hearing Assistance
  - Accessibility Tools
  - Industrial Audio Monitoring

## Research Criteria
- Focus on models relevant to audio processing (not just LLMs)
- Include both open-source and commercial options
- Prioritize models with good documentation and community support
- Consider hardware compatibility (especially for local deployment)
- Include cost-benefit analysis for API models
- Emphasize models that complement existing lab capabilities
- Consider models that expand the lab's scope beyond speech-only

## Deliverable Format
Create a structured table/database with all collected information that can be used for decision-making, organized by audio category and implementation priority.