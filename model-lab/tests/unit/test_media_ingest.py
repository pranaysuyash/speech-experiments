"""
Tests for media ingestion (session pipeline).

These tests verify:
1. ingest_media produces a canonical processed WAV (mono, 16kHz)
2. Hash fields are populated and deterministic
3. Missing files raise a clear error
"""

from pathlib import Path

import pytest

from harness.media_ingest import IngestConfig, get_ffmpeg_version, ingest_media, sha256_file


def test_get_ffmpeg_version_returns_string():
    v = get_ffmpeg_version()
    assert isinstance(v, str)
    assert v


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")

    wav_path = tmp_path / "test.wav"
    sf.write(wav_path, audio, sr)
    return wav_path


def test_ingest_produces_canonical_wav(sample_wav: Path, tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    cfg = IngestConfig(normalize=False, trim_silence=False)

    result = ingest_media(sample_wav, artifacts_dir, cfg)

    assert result["source_media_path"].endswith("test.wav")
    assert result["source_media_hash"] == sha256_file(sample_wav)
    assert len(result["source_media_hash"]) == 64

    processed_path = Path(result["processed_audio_path"])
    assert processed_path.exists()
    assert processed_path.name == "processed_audio.wav"
    assert processed_path.parent.name == "ingest"

    assert len(result["audio_content_hash"]) == 64
    assert len(result["audio_fingerprint"]) == 64
    assert len(result["preprocess_hash"]) == 64

    import soundfile as sf

    audio, sr = sf.read(processed_path)
    assert sr == 16000
    # mono: either 1D array or Nx1
    assert len(audio.shape) in (1, 2)
    if len(audio.shape) == 2:
        assert audio.shape[1] == 1


def test_ingest_is_deterministic_for_same_input(sample_wav: Path, tmp_path: Path):
    cfg = IngestConfig(normalize=False, trim_silence=False)
    a = ingest_media(sample_wav, tmp_path / "a", cfg)
    b = ingest_media(sample_wav, tmp_path / "b", cfg)

    assert a["source_media_hash"] == b["source_media_hash"]
    assert a["audio_content_hash"] == b["audio_content_hash"]
    assert a["preprocess_hash"] == b["preprocess_hash"]
    assert a["audio_fingerprint"] == b["audio_fingerprint"]


def test_ingest_missing_file_raises(tmp_path: Path):
    cfg = IngestConfig()
    with pytest.raises(FileNotFoundError):
        ingest_media(tmp_path / "nope.wav", tmp_path / "artifacts", cfg)
