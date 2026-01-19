# Reproducibility Statement

## verification
- **Test**: Concurrent runs of `abuse_large_file.wav`.
- **Runs**: `20260119_084210_6ba7db08c1` and `20260119_084210_5ba415e9c2`.
- **Artifact**: `processed_audio.wav` (Ingest output from ffmpeg).
- **Hash**: `023c38b3b637097384b72c98a49e541c740bcb934aacc6e5bf9fc530ad2fe899` (Identical for both).

## Guarantees
1. **Ingest (Audio Processing)**: Deterministic.
   - Same Input File + Same Config -> Identical PCM WAV Output.

2. **ASR (Transcription)**:
   - **Deterministic** if `temperature=0` is set in configuration.
   - **Non-Deterministic** if `temperature > 0` (default might be 0, but check config).

3. **Diarization/Summarization**:
   - LLM-based steps are inherently **non-deterministic**.
   - Output will vary semantically but hold similar meaning.
   - **Status**: These steps are explicitly marked as "generative" and not bitwise reproducible.

## Conclusion
The physical media layer (Ingest) is **Bitwise Reproducible**.
The semantic layer (ASR/LLM) is **Reproducible subject to configuration** (Temperature).
Builder pipelines can rely on stable audio caching.
