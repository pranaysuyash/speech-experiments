# LFM-2.5-Audio Test Plan

## Test Axes (Non-Negotiable)

### Input Modality
- **Audio**: Clean speech, noisy speech, music, silence, mixed audio
- **Text**: Prompts for guided transcription/analysis
- **Mixed**: Audio + text instructions

### Output Modality  
- **Text**: Transcription, translation, analysis, summary
- **Audio**: Text-to-speech, audio transformation (future)
- **Mixed**: Audio + metadata (future)

### Constraints
- **Max Latency**: 500ms for real-time, 2000ms for batch processing
- **Memory Footprint**: <2GB GPU, <500MB CPU for 30s audio
- **Streaming vs Batch**: Both modes must be tested
- **Audio Length**: 1s to 30s segments

### Failure Modes to Detect
- **Silence**: Model behavior on silent/quiet audio
- **Hallucination**: Text output when no speech present
- **Truncation**: Premature cutoff of long audio
- **Drift**: Inconsistent outputs across identical runs
- **Memory Leaks**: Gradual resource consumption increase

### Quality Metrics
- **WER** (Word Error Rate): Primary transcription accuracy
- **CER** (Character Error Rate): Fine-grained accuracy
- **Latency Distribution**: Mean, P95, P99 response times
- **Confidence Calibration**: How well confidence scores match accuracy
- **Memory Stability**: Peak and steady-state consumption

## Canonical Test Audio

### Required Test Files
1. **clean_speech_10s.wav**: Your recorded Wikipedia reading (ground truth)
2. **noisy_speech_10s.wav**: Same speech + background noise
3. **silence_5s.wav**: Complete silence
4. **music_10s.wav**: Instrumental music (no speech)
5. **mixed_10s.wav**: Speech + background music

### Ground Truth Requirements
- Exact text of what you read saved as `clean_speech_10s.txt`
- Timestamp alignment markers for key phrases
- Speaker identification metadata (age, accent, gender optional)

## Test Execution Order

### Phase 1: Stability Testing
1. Audio → Text (clean speech)
2. Latency measurement (100 runs)
3. Memory footprint tracking
4. Run-to-run consistency check

### Phase 2: Robustness Testing  
1. Silence handling
2. Noise robustness
3. Truncation detection
4. Hallucination measurement

### Phase 3: Capability Testing
1. Different accents/speakers (if available)
2. Various audio lengths (1s, 5s, 10s, 30s)
3. Mixed language content
4. Technical/domain-specific vocabulary

## Output Schema (Canonical)

```json
{
  "model": "LFM-2.5-Audio-1.5B",
  "source": "Liquid AI",
  "test_date": "2024-01-07T15:56:49Z",
  "hardware": "M3 Max, 36GB RAM",
  "precision": "float16",
  "input": {
    "type": "audio",
    "file": "clean_speech_10s.wav",
    "duration_s": 10.0,
    "sample_rate": 16000
  },
  "output": {
    "type": "text", 
    "text": "transcribed text here",
    "confidence": 0.95,
    "language": "en"
  },
  "metrics": {
    "latency_ms": 312,
    "latency_p95_ms": 350,
    "memory_peak_mb": 487,
    "memory_steady_mb": 234,
    "wer": 0.023,
    "cer": 0.012,
    "hallucination_flag": false,
    "truncation_flag": false
  },
  "notes": "Optional contextual notes"
}
```

## Common Traps to Avoid

- ❌ Different prompts per model
- ❌ Changing sample rate per model  
- ❌ Letting SDK defaults decide decoding
- ❌ Testing quality before stability
- ❌ Trusting streaming demos
- ❌ Comparing different audio segments

## Success Criteria

- **Stability**: 100 consecutive runs without crashes
- **Consistency**: WER variance <5% across identical runs
- **Latency**: P95 <500ms for 10s audio segments
- **Memory**: No leaks over 100 runs
- **Accuracy**: WER <10% on clean speech