import struct
import wave

import pytest


@pytest.fixture
def test_wav_path(tmp_path):
    """Create a dummy WAV file for testing."""
    path = tmp_path / "test_audio.wav"

    # Generate 1 second of silence/noise
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)

        # Write silence
        data = struct.pack("<" + "h" * num_samples, *[0] * num_samples)
        f.writeframes(data)

    return path
