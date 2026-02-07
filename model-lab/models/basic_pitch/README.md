# Basic Pitch

LCS-12: Lightweight automatic music transcription from Spotify.

## Overview

- **Architecture**: Lightweight neural network (ICASSP 2022)
- **Sample Rate**: 22.05kHz
- **Output**: Notes with onset, offset, pitch (MIDI)
- **License**: Apache 2.0

## Installation

```bash
make model-install MODEL=basic_pitch
```

**Note**: Requires TensorFlow. Use isolated venv.

## Usage

```bash
make model-info MODEL=basic_pitch
make music-transcribe MODEL=basic_pitch AUDIO=inputs/piano.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| music_transcription | transcribe | ✓ |

## Output Format

```python
{
    "notes": [
        {"onset": 0.5, "offset": 1.2, "pitch": 60, "velocity": 0.8},
        {"onset": 0.8, "offset": 1.5, "pitch": 64, "velocity": 0.7},
        ...
    ],
    "pitch_bend": [...],  # Optional pitch bend data
}
```

**Fields**:
- `onset`: Note start time in seconds
- `offset`: Note end time in seconds
- `pitch`: MIDI pitch (0-127)
- `velocity`: Note intensity (0.0-1.0)

## References

- [GitHub](https://github.com/spotify/basic-pitch)
- [Paper](https://arxiv.org/abs/2202.09048)
- [Web Demo](https://basicpitch.spotify.com)
