# Audio Model Audit 2026 - Specific to Speech Experiments Model Lab

## Executive Summary

This document provides a comprehensive audit of audio-focused models relevant to the speech experiments model lab. It focuses specifically on ASR (Automatic Speech Recognition), TTS (Text-to-Speech), and multimodal audio models that can be integrated into the existing model lab framework.

## Current Model Lab Status

### Existing Models
- **LFM2.5-Audio-1.5B**: Liquid AI model supporting ASR, TTS, and Chat (multi-modal)
- **Whisper-Large-V3**: OpenAI model supporting ASR only
- **Faster-Whisper**: Optimized implementation of Whisper
- **SeamlessM4T**: Facebook/Meta model for multilingual translation and transcription

### Model Lab Structure
- Models are organized in isolated folders with standardized testing notebooks
- Testing follows systematic approach: 00_smoke → 10_asr → 20_tts → 30_chat
- Shared harness ensures fair comparisons with identical metrics
- Results are automatically logged to JSON files for comparison

## Recommended Audio Models for Addition

### 1. Parler-TTS
- **Category**: TTS
- **Developer**: Meta AI
- **Parameters**: Various sizes available
- **Features**: Zero-shot TTS with detailed control over voice characteristics
- **Best For**: High-quality, customizable TTS with prosodic control
- **Integration Complexity**: Medium
- **License**: Open Source (MIT)
- **Hardware Requirements**: Consumer GPU for inference
- **Notable Advantage**: Fine-grained control over voice characteristics without training

### 2. Chatterbox
- **Category**: TTS
- **Developer**: Resemble AI
- **Parameters**: High-performance models
- **Features**: Low-latency, production-grade voice applications
- **Best For**: Production TTS applications with low latency requirements
- **Integration Complexity**: Medium
- **License**: Open Source
- **Hardware Requirements**: Consumer GPU
- **Notable Advantage**: Outperformed ElevenLabs in blind tests (63.75% preference)

### 3. Fish Audio Models
- **Category**: TTS
- **Developer**: Fish Audio
- **Features**: Ultra-realistic AI voices, scalable for brands
- **Best For**: High-quality voice generation with scalability
- **Integration Complexity**: Medium
- **License**: Varies by model
- **Hardware Requirements**: Consumer GPU
- **Notable Advantage**: 70% less expensive than ElevenLabs

### 4. AudioPaLM
- **Category**: Multimodal Audio-Language Model
- **Developer**: Google
- **Parameters**: Large-scale model
- **Features**: Speech understanding and generation, can speak and listen
- **Best For**: Advanced multimodal audio-text applications
- **Integration Complexity**: High
- **License**: Research/Commercial (depends on implementation)
- **Hardware Requirements**: High-end GPU or TPU
- **Notable Advantage**: True multimodal capabilities for speech understanding and generation

### 5. IBM Granite Speech 3.3 8B
- **Category**: ASR
- **Developer**: IBM
- **Parameters**: 8B
- **Features**: Enterprise-focused, domain-specific customization
- **Best For**: Enterprise applications, domain-specific ASR
- **Integration Complexity**: Medium
- **License**: Apache 2.0
- **Hardware Requirements**: Consumer GPU
- **Notable Advantage**: High accuracy for enterprise and domain-specific applications

### 6. Canary Qwen 2.5B
- **Category**: ASR
- **Developer**: Alibaba
- **Parameters**: 2.5B
- **Features**: Maximum English accuracy
- **Best For**: High-accuracy English ASR
- **Integration Complexity**: Medium
- **License**: Apache 2.0
- **Hardware Requirements**: Consumer GPU
- **Notable Advantage**: Best-in-class English ASR accuracy

### 7. SeamlessM4T v2 Large
- **Category**: Multimodal Translation/Transcription
- **Developer**: Meta/Facebook
- **Parameters**: Large-scale model
- **Features**: ASR, T2TT, S2TT, T2ST, S2ST (Speech-to-Text translation, Text-to-Speech translation, etc.)
- **Best For**: Multilingual applications, speech-to-speech translation
- **Integration Complexity**: Medium-High
- **License**: CC-BY-NC 4.0 (research use) or commercial license
- **Hardware Requirements**: Consumer GPU
- **Notable Advantage**: All-in-one multilingual multimodal translation model

### 8. Moonshine
- **Category**: ASR
- **Developer**: Various contributors
- **Parameters**: 27M (smallest variant)
- **Features**: Small footprint, competitive with Whisper Tiny/Small
- **Best For**: Edge deployment, IoT, mobile applications
- **Integration Complexity**: Medium
- **License**: Not specified
- **Hardware Requirements**: Edge devices, smartphones
- **Notable Advantage**: Extremely small size with competitive accuracy

### 9. Whisper.cpp
- **Category**: ASR
- **Developer**: Various contributors
- **Features**: CPU-optimized C++ implementation of Whisper
- **Best For**: CPU-only environments, embedded systems
- **Integration Complexity**: Medium
- **License**: MIT
- **Hardware Requirements**: CPU (any architecture)
- **Notable Advantage**: Runs efficiently on CPU-only systems

### 10. Distil-Whisper
- **Category**: ASR
- **Developer**: Hugging Face
- **Features**: 6x faster than original Whisper with ~1% accuracy loss
- **Best For**: Real-time transcription, resource-constrained environments
- **Integration Complexity**: Medium
- **License**: MIT
- **Hardware Requirements**: Consumer GPU (lower than original)
- **Notable Advantage**: Significant speed improvement with minimal accuracy loss

## Model Prioritization for Implementation

### Priority 1: Immediate Additions (High Impact, Low Complexity)

1. **Chatterbox** - For TTS capabilities to complement existing ASR-only models
   - High quality, open source, outperformed ElevenLabs in tests
   - Fits well with existing TTS testing framework (20_tts.ipynb)

2. **IBM Granite Speech 3.5 8B** - For enterprise-grade ASR comparison
   - High accuracy for English, good for benchmarking against Whisper/LFM2.5
   - Enterprise focus complements research models already present

### Priority 2: Medium-term Additions (Moderate Complexity)

3. **SeamlessM4T v2 Large** - For multilingual capabilities
   - Already in your model directory but needs full implementation
   - Adds multilingual translation capabilities to your lab
   - Supports ASR, TTS, and translation in one model

4. **Canary Qwen 2.5B** - For best-in-class English ASR
   - Reported as highest accuracy for English ASR
   - Good addition to ASR comparison framework

### Priority 3: Long-term Additions (Higher Complexity)

5. **AudioPaLM** - For advanced multimodal capabilities
   - Cutting-edge multimodal audio-text model
   - High complexity but represents future direction
   - Would enable advanced audio understanding tasks

6. **Moonshine** - For edge deployment testing
   - Smallest footprint model for edge/IoT scenarios
   - Good for testing resource-constrained environments
   - Expands your model lab's coverage of deployment scenarios

## Integration Strategy

### Standardized Testing Framework
Each new model should follow the existing notebook structure:
- `00_smoke.ipynb` - Quick validation (5-second audio)
- `10_asr.ipynb` - ASR evaluation (if applicable)
- `20_tts.ipynb` - TTS evaluation (if applicable)
- `30_chat.ipynb` - Conversation testing (if applicable)

### Configuration Management
- Add model-specific `config.yaml` files following existing patterns
- Ensure consistent device selection (mps/cuda/cpu)
- Maintain consistent audio processing parameters

### Metrics Consistency
- Use shared harness components for fair comparisons
- Maintain identical WER/CER calculation methods
- Preserve timing and resource monitoring consistency

## Expected Benefits

### For ASR Models
- Expanded benchmarking capabilities
- More comprehensive model comparison
- Better understanding of trade-offs (accuracy vs. speed vs. size)

### For TTS Models
- Addition of TTS capabilities to complement existing ASR
- Multi-model TTS comparison framework
- Understanding of quality vs. latency trade-offs

### For Multimodal Models
- Advanced audio-text interaction capabilities
- Multilingual support evaluation
- Future-proofing for emerging multimodal applications

## Risk Assessment

### Technical Risks
- Model integration complexity varies by model
- Hardware requirements differ significantly
- Some models may have restrictive licenses

### Mitigation Strategies
- Start with Priority 1 models to establish integration patterns
- Document hardware requirements clearly
- Verify licenses before implementation
- Use standardized testing to ensure fair comparisons

## Success Metrics

### Quantitative
- Successful integration of 3 new models within 6 weeks
- Consistent metric reporting across all models
- Performance regression testing shows no degradation

### Qualitative
- Expanded model comparison capabilities
- Better understanding of audio model landscape
- Enhanced production decision-making framework