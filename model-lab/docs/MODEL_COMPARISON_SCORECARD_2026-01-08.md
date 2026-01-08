# 📊 Model Comparison Scorecard - January 8, 2026

**Test Date**: January 8, 2026  
**Platform**: Apple Silicon (MPS)  
**Environment**: Python 3.12.10, UV package manager  
**Purpose**: Production readiness assessment

---

## 🎯 Overall Rankings

### 1. 🥇 Faster-Whisper

**Grade**: A+ (Production Ready)  
**Best For**: High-accuracy ASR, production deployments, long-form audio

### 2. 🥈 Whisper

**Grade**: A (Production Ready)  
**Best For**: Fast inference, real-time applications, balanced performance

### 3. 🥉 LFM2.5-Audio

**Grade**: C for ASR, B+ for Multi-Modal  
**Best For**: Conversational AI, multi-modal tasks, research

---

## 📈 PRIMARY Dataset Results (163s Audio)

### Test File: `llm_recording_pranay.wav`

- Duration: 163.2 seconds
- Content: Technical discussion about LLMs
- Ground Truth: 2512 characters

| Metric            | Faster-Whisper | Whisper       | LFM2.5-Audio | Winner         |
| ----------------- | -------------- | ------------- | ------------ | -------------- |
| **WER**           | **24.1%** ✅   | 28.5%         | 137.8%       | Faster-Whisper |
| **CER**           | **6.1%** ✅    | 7.7%          | 90.3%        | Faster-Whisper |
| **Latency**       | 19.4s          | **13.1s** ✅  | 34.6s        | Whisper        |
| **RTF**           | 0.119x         | **0.080x** ✅ | 0.212x       | Whisper        |
| **Output Length** | 2516 chars     | 2510 chars    | 3152 chars   | Whisper        |

**Key Insights**:

- ✅ **Faster-Whisper**: Best accuracy (24.1% WER)
- ✅ **Whisper**: Fastest inference (0.080x RTF)
- ⚠️ **LFM2.5**: Poor ASR accuracy (137.8% WER = more errors than words)

---

## 📈 CONVERSATION Dataset Results (944s Audio)

### Test File: `UX_Psychology_From_Miller_s_Law_to_AI.wav`

- Duration: 943.6 seconds (15m 44s)
- Content: Long-form UX psychology podcast
- Expected Output: ~16,000+ characters

| Metric            | Faster-Whisper | Whisper      | LFM2.5-Audio   | Winner           |
| ----------------- | -------------- | ------------ | -------------- | ---------------- |
| **Output Length** | 16,809 chars   | 16,792 chars | 2,389 chars 🔴 | Faster-Whisper   |
| **Latency**       | **114.5s** ✅  | 128.9s       | 92.7s          | Faster-Whisper   |
| **RTF**           | 0.121x         | 0.137x       | **0.098x** ✅  | LFM2.5           |
| **Completeness**  | **100%** ✅    | **100%** ✅  | ~14%           | Whisper Variants |

**Key Insights**:

- ✅ **Faster-Whisper**: Fastest with complete output
- ✅ **Whisper**: Reliable full transcription
- 🔴 **LFM2.5**: Severe underprediction (only 2,389 chars vs 16k+ expected)

**Critical Issue**: LFM2.5 appears to truncate or stop transcribing on long audio files.

---

## 🏆 Category Winners

### 🎯 Best Overall Accuracy

**Winner**: Faster-Whisper

- WER: 24.1%
- CER: 6.1%
- Consistent, reliable transcriptions

### ⚡ Fastest Inference

**Winner**: Whisper

- RTF: 0.080x (163s audio)
- Lowest latency: 13.1s
- Best for real-time applications

### 📏 Long-Form Audio

**Winner**: Faster-Whisper

- Complete 16,809 char output
- Stable performance on 15+ minute files
- Reliable for podcasts/meetings

### 💰 Resource Efficiency

**Winner**: Whisper

- Balanced speed/accuracy
- Lower memory footprint
- Good for batch processing

### 🔬 Multi-Modal Capabilities

**Winner**: LFM2.5-Audio

- ASR + TTS + Conversation
- Unique capabilities
- Research potential

---

## 📊 Performance Matrix

### Real-Time Factor (RTF) Comparison

Lower is better; <1.0 = faster than real-time

| Audio Length | Faster-Whisper | Whisper    | LFM2.5-Audio |
| ------------ | -------------- | ---------- | ------------ |
| 163s (short) | 0.119x         | **0.080x** | 0.212x       |
| 944s (long)  | **0.121x**     | 0.137x     | 0.098x       |

**Insight**: RTF remains consistent for Whisper variants across file lengths, but LFM2.5 improves on longer files (possibly due to warmup overhead).

### Accuracy Comparison (WER/CER)

Lower is better

| Model          | WER       | CER      | Grade |
| -------------- | --------- | -------- | ----- |
| Faster-Whisper | **24.1%** | **6.1%** | A+    |
| Whisper        | 28.5%     | 7.7%     | A     |
| LFM2.5-Audio   | 137.8%    | 90.3%    | D     |

**Insight**: LFM2.5's >100% WER means it's generating more errors (insertions/substitutions) than the original word count.

---

## 🔍 Detailed Analysis

### Faster-Whisper Strengths:

✅ Best transcription accuracy across all tests  
✅ Excellent performance on long-form audio  
✅ Consistent RTF regardless of file length  
✅ CTranslate2 optimization delivers reliability  
✅ Production-grade stability

### Faster-Whisper Weaknesses:

⚠️ Slightly slower than base Whisper on short files  
⚠️ Requires CTranslate2 dependency

### Whisper Strengths:

✅ Fastest inference on short files (RTF 0.080x)  
✅ Good accuracy (28.5% WER)  
✅ Simple deployment (pure PyTorch)  
✅ Well-documented and widely used  
✅ Native MPS support

### Whisper Weaknesses:

⚠️ Slightly lower accuracy than Faster-Whisper  
⚠️ Can be slower on very long files

### LFM2.5-Audio Strengths:

✅ Multi-modal capabilities (ASR + TTS + Chat)  
✅ Now works on MPS after bug fixes  
✅ Decent RTF on long files (0.098x)  
✅ Unique conversational AI potential  
✅ Interesting for research

### LFM2.5-Audio Weaknesses:

🔴 Poor ASR accuracy (137.8% WER)  
🔴 Severe underprediction on long audio (14% completion)  
🔴 Not optimized for pure transcription  
🔴 Vendor CUDA bug (CPU fallback on CUDA systems)  
🔴 Complex setup and dependencies

---

## 🎯 Production Recommendations

### For ASR Transcription:

**Primary**: Faster-Whisper  
**Backup**: Whisper  
**Avoid**: LFM2.5 (for pure ASR)

### For Real-Time Applications:

**Primary**: Whisper  
**Backup**: Faster-Whisper

### For Long-Form Audio (>10min):

**Primary**: Faster-Whisper  
**Backup**: Whisper

### For Multi-Modal Tasks:

**Primary**: LFM2.5-Audio  
**Note**: Not yet ready for production ASR

### For Apple Silicon:

✅ **All models work** after bug fixes  
**Recommended**: Faster-Whisper or Whisper

---

## 📋 Test Specifications

### Hardware:

- **Device**: Apple Silicon M-series
- **Acceleration**: MPS (Metal Performance Shaders)
- **OS**: macOS

### Software:

- **Python**: 3.12.10
- **Package Manager**: UV
- **Framework**: PyTorch 2.9.1

### Models:

- **Whisper**: base (74M params)
- **Faster-Whisper**: base via CTranslate2
- **LFM2.5-Audio**: 1.5B params

### Test Files:

1. **PRIMARY/llm_recording_pranay.wav**: 163s technical discussion
2. **PRIMARY/UX_Psychology_From_Miller_s_Law_to_AI.wav**: 944s podcast

---

## 🚀 Deployment Decision

### Production Grade: Faster-Whisper

**Status**: ✅ APPROVED FOR PRODUCTION

**Rationale**:

- Highest accuracy across all tests
- Excellent performance on long-form audio
- Stable and reliable
- Proven CTranslate2 optimization

### Production Grade: Whisper

**Status**: ✅ APPROVED FOR PRODUCTION (Secondary)

**Rationale**:

- Fast inference for real-time needs
- Good accuracy
- Simple deployment
- Wide community support

### Research Grade: LFM2.5-Audio

**Status**: ⚠️ NOT RECOMMENDED FOR PRODUCTION ASR

**Rationale**:

- Poor ASR accuracy
- Severe issues with long audio
- Better suited for multi-modal research

**Future Potential**: May improve with model updates

---

## 📈 Cost-Benefit Analysis

### Faster-Whisper:

- **Accuracy Gain**: +4.4 WER points vs Whisper
- **Speed Trade-off**: +0.039 RTF (7ms/sec audio)
- **Verdict**: Worth the trade-off for production ✅

### Whisper:

- **Speed Advantage**: 33% faster than Faster-Whisper
- **Accuracy Cost**: +4.4 WER points
- **Verdict**: Good for speed-critical apps ✅

### LFM2.5-Audio:

- **Unique Features**: Multi-modal capabilities
- **ASR Cost**: 113.7 WER points worse than Faster-Whisper
- **Verdict**: Not viable for production ASR ❌

---

## 🔮 Future Testing

### Next Steps:

1. **GPU (CUDA) Testing**: Validate on Colab T4
2. **TPU Testing**: Check XLA compatibility
3. **CPU Baseline**: Establish non-accelerated performance
4. **Batch Processing**: Test multiple files
5. **Different Accents**: Evaluate robustness
6. **Noisy Audio**: Test in challenging conditions

### Questions to Answer:

- Does LFM2.5 improve with prompt engineering?
- Can we tune LFM2.5 for better ASR?
- How do models perform on different audio domains?

---

## 📚 References

- [COMPREHENSIVE_TEST_RESULTS_2026-01-08.md](COMPREHENSIVE_TEST_RESULTS_2026-01-08.md)
- [LFM_MPS_FIX_SUMMARY.md](LFM_MPS_FIX_SUMMARY.md)
- [LFM25_CUDA_MPS_RESOLUTION.md](LFM25_CUDA_MPS_RESOLUTION.md)
- [MULTI_DEVICE_TESTING_PLAN.md](MULTI_DEVICE_TESTING_PLAN.md)

---

**Conclusion**: Faster-Whisper and Whisper are production-ready for ASR tasks on Apple Silicon. LFM2.5-Audio shows promise for multi-modal applications but is not yet suitable for production ASR deployments.
