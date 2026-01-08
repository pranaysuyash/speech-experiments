# Recording Instructions for Canonical Test Audio

## Requirements

**What to read**: Choose any Wikipedia article (science, history, or technology recommended)
**Length**: Exactly 10-15 seconds of clear speech
**Format**: WAV, 16kHz, mono
**Environment**: Quiet room, no background noise

## Recording Process

1. **Select your text** - Pick 2-3 sentences from Wikipedia (about 30-40 words)
2. **Practice once** - Read it aloud to check timing
3. **Record** - Use your phone or computer in a quiet room
4. **Check timing** - Should be 10-15 seconds exactly
5. **Save as**: `data/audio/clean_speech_10s.wav`

## Text Documentation

**CRITICAL**: Save the EXACT text you read as `data/text/clean_speech_10s.txt`

Example format:
```
File: clean_speech_10s.txt
Recording: clean_speech_10s.wav
Date: 2024-01-07
Speaker: [Your name/identifier]

[Exact text you read goes here, word for word]
```

## Validation Checklist

- [ ] Audio is 10-15 seconds long
- [ ] No background noise or interruptions  
- [ ] Clear pronunciation at normal speaking pace
- [ ] Text file contains exact words spoken
- [ ] Files saved in `data/audio/` directory
- [ ] Audio format: WAV, 16kHz, mono

## Why This Matters

This recording becomes your **ground truth** for:
- Transcription accuracy testing (WER/CER calculation)
- Cross-model consistency comparison
- Text-to-speech quality evaluation (later)
- Baseline performance measurement

Every model will be tested on this exact same audio segment.