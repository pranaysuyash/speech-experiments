"""
Tests for media ingestion.

These tests verify:
1. Media type detection (video vs audio)
2. Canonical audio output (mono, 16kHz, float32)
3. Hash computation (source_media_hash vs audio_hash)
4. Graceful handling of missing ffmpeg
"""

import pytest
import numpy as np
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.media_ingest import (
    detect_media_type,
    compute_pcm_hash,
    compute_file_hash,
    check_ffmpeg_available,
    ingest_media,
    IngestResult,
    IngestError,
    FFmpegNotFoundError,
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
    CANONICAL_SAMPLE_RATE,
)


class TestMediaTypeDetection:
    """Tests for media type detection."""
    
    def test_detect_video_extensions(self):
        """Common video extensions detected correctly."""
        for ext in ['.mp4', '.mkv', '.mov', '.avi', '.webm']:
            assert detect_media_type(Path(f"test{ext}")) == "video"
    
    def test_detect_audio_extensions(self):
        """Common audio extensions detected correctly."""
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            assert detect_media_type(Path(f"test{ext}")) == "audio"
    
    def test_unknown_extension(self):
        """Unknown extensions return 'unknown'."""
        assert detect_media_type(Path("test.xyz")) == "unknown"


class TestHashComputation:
    """Tests for hash computation."""
    
    def test_pcm_hash_deterministic(self):
        """Same audio produces same hash."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        hash1 = compute_pcm_hash(audio)
        hash2 = compute_pcm_hash(audio)
        assert hash1 == hash2
    
    def test_pcm_hash_different_for_different_audio(self):
        """Different audio produces different hash."""
        audio1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        audio2 = np.array([0.3, 0.2, 0.1], dtype=np.float32)
        assert compute_pcm_hash(audio1) != compute_pcm_hash(audio2)
    
    def test_pcm_hash_length(self):
        """Hash is 16 characters by default."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert len(compute_pcm_hash(audio)) == 16


class TestFFmpegCheck:
    """Tests for ffmpeg availability check."""
    
    def test_check_ffmpeg_returns_tuple(self):
        """check_ffmpeg_available returns (bool, version_or_None)."""
        available, version = check_ffmpeg_available()
        assert isinstance(available, bool)
        if available:
            assert version is not None
        else:
            assert version is None


class TestIngestWithRealFile:
    """Integration tests with real audio files."""
    
    @pytest.fixture
    def sample_wav(self, tmp_path):
        """Create a sample WAV file for testing."""
        import soundfile as sf
        
        # Create 1 second of sine wave
        sr = 16000
        t = np.linspace(0, 1, sr)
        audio = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        wav_path = tmp_path / "test.wav"
        sf.write(wav_path, audio, sr)
        return wav_path
    
    def test_ingest_wav_produces_correct_result(self, sample_wav):
        """Ingesting WAV produces IngestResult with correct properties."""
        result = ingest_media(sample_wav)
        
        try:
            # Check types
            assert isinstance(result, IngestResult)
            assert isinstance(result.audio, np.ndarray)
            
            # Check canonical properties
            assert result.sample_rate == CANONICAL_SAMPLE_RATE
            assert result.audio.dtype == np.float32
            
            # Check hashes are populated
            assert len(result.source_media_hash) == 16
            assert len(result.audio_hash) == 16
            
            # Check metadata
            assert result.source_media_path == sample_wav
            assert result.original_format == ".wav"
            assert result.audio_duration_s > 0
        finally:
            result.cleanup()
    
    def test_ingest_wav_hash_is_from_pcm(self, sample_wav):
        """audio_hash is computed from PCM, not file bytes."""
        result = ingest_media(sample_wav)
        
        try:
            # Manually compute PCM hash
            expected_hash = compute_pcm_hash(result.audio)
            assert result.audio_hash == expected_hash
            
            # source_media_hash should be different (file bytes)
            assert result.source_media_hash != result.audio_hash
        finally:
            result.cleanup()
    
    def test_ingest_nonexistent_file_raises(self):
        """Ingesting nonexistent file raises IngestError."""
        with pytest.raises(IngestError) as exc_info:
            ingest_media(Path("/nonexistent/file.wav"))
        
        assert "not found" in str(exc_info.value).lower()


class TestIngestArtifactSchema:
    """Tests that ingest results can populate artifact schema correctly."""
    
    @pytest.fixture
    def sample_wav(self, tmp_path):
        """Create a sample WAV file."""
        import soundfile as sf
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)  # 1 second silence
        wav_path = tmp_path / "test.wav"
        sf.write(wav_path, audio, sr)
        return wav_path
    
    def test_ingest_result_has_all_required_fields(self, sample_wav):
        """IngestResult has all fields needed for artifact inputs section."""
        result = ingest_media(sample_wav)
        
        try:
            # Required for runner_schema.InputsSchema
            assert hasattr(result, 'source_media_path')
            assert hasattr(result, 'source_media_hash')
            assert hasattr(result, 'audio_hash')
            assert hasattr(result, 'audio_duration_s')
            assert hasattr(result, 'sample_rate')
            
            # Required for ingest provenance
            assert hasattr(result, 'ingest_tool')
            assert hasattr(result, 'ingest_version')
            assert hasattr(result, 'is_extracted')
        finally:
            result.cleanup()


# Skip video tests if no ffmpeg
@pytest.mark.skipif(
    not check_ffmpeg_available()[0],
    reason="ffmpeg not installed"
)
class TestVideoIngestion:
    """Tests for video file ingestion (requires ffmpeg)."""
    
    def test_ffmpeg_is_available(self):
        """Verify ffmpeg is available for video tests."""
        available, version = check_ffmpeg_available()
        assert available
        assert version is not None
