# Speech Test Corpus

Regression test suite for STT/TTS with audio integrity gates and baseline tracking.

## Structure

- `golden.txt` - Single-speaker script (acronyms, numbers, hyphens, minimal pairs)
- `dialogue_2spk.txt` - Multi-speaker script (for diarization testing)
- `baselines.json` - Expected metrics per fixture (WER, glossary, numbers)
- `run_speech_gate.py` - Single-command gate runner
- `fixtures/` - Generated WAV files (TTS or recordings)
- `reports/` - Auto-generated test results (gitignored)

## Usage

### 1. Generate fixtures (one-time)

Use TTS or record yourself reading the scripts:

```bash
# Generate from golden.txt
# Output: fixtures/tts_voice1_clean.wav, fixtures/tts_voice2_clean.wav

# Generate from dialogue_2spk.txt (optional)
# Output: fixtures/dialogue_2spk_clean.wav
```

**CRITICAL:** After generating each WAV, freeze its reference text:

```bash
# Copy reference text to fixture-specific file
cp golden.txt fixtures/tts_voice1_clean.ref.txt
cp golden.txt fixtures/tts_voice2_clean.ref.txt
cp dialogue_2spk.txt fixtures/dialogue_2spk_clean.ref.txt
```

**Why:** This prevents accidental edits to `golden.txt` from invalidating baselines. Once a WAV exists, its `.ref.txt` is immutable unless you regenerate the WAV.

### 2. Establish baselines (first run)

```bash
python run_speech_gate.py --init-baseline
```

This will:
- Run ASR on all fixtures
- Measure actual WER, glossary hits, number anchors
- Auto-update `baselines.json` with measured values
- Future runs will detect regressions from these baselines

### 3. Run gate (regression detection)

```bash
python run_speech_gate.py
```

**Outcomes:**
- `PASS` - All metrics within tolerance
- `WARN` - Minor regressions (WER +2-5pp, missing 1-2 glossary terms)
- `FAIL` - Major regressions (WER +5pp+, missing 4+ glossary terms, audio integrity issues)

**Report:** `reports/latest.json`

## Gates

### Audio Integrity
- True peak clipping check
- SNR estimate (warns if < 20 dB)
- Speech coverage (VAD-based)
- Loudness normalization (auto-applied if outside [-27, -16] LUFS)
- **Fixture fingerprinting** (SHA256 + duration + sample rate)
  - Detects accidental WAV changes
  - Populated during `--init-baseline`
  - Fails if fixture modified after baseline

### ASR Quality
- Normalized WER delta vs baseline
- Glossary hit rate (acronyms: GPT, RLHF, ASR, etc.)
- Number anchor preservation (2026, 29.9, 16000, etc.)
- CER tracking (informational)

## Design Principles

1. **Stable fixtures** - WAV files don't change, agents can't game tests
2. **Frozen references** - Each WAV has its own `.ref.txt`, immune to `golden.txt` edits
3. **Fingerprinting** - SHA256 + duration + SR detect accidental WAV changes
4. **Baseline tracking** - Detect regressions, not absolute thresholds
5. **Minimal metrics** - Only gates that trigger actions
6. **Single command** - No CI theater, just local verification
7. **Per-fixture transcripts** - `reports/<timestamp>/<fixture>.hyp.txt` for easy debugging

## Next Steps

After fixtures are generated and baselines established:
- Add to pre-merge checklist
- Run before model upgrades
- Run after preprocessing changes
- Track trends over time
