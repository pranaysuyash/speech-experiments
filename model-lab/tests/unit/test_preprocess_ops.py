"""
Tests for preprocessing operators.

Tests verify:
1. Each operator produces correct metrics
2. Hash transitions are recorded
3. Operator chain works correctly
4. Duration/content changes as expected
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.preprocess_ops import (
    OPERATORS,
    NormalizeLoudness,
    Resample,
    TrimSilence,
    compute_audio_hash,
    parse_operator_spec,
    results_to_artifact_section,
    run_preprocessing_chain,
)


class TestComputeAudioHash:
    """Tests for hash computation."""

    def test_hash_is_deterministic(self):
        """Same audio produces same hash."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert compute_audio_hash(audio) == compute_audio_hash(audio)

    def test_different_audio_different_hash(self):
        """Different audio produces different hash."""
        audio1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        audio2 = np.array([0.3, 0.2, 0.1], dtype=np.float32)
        assert compute_audio_hash(audio1) != compute_audio_hash(audio2)


class TestTrimSilence:
    """Tests for silence trimming operator."""

    @pytest.fixture
    def silence_padded_audio(self):
        """Audio with silence at start and end."""
        sr = 16000
        silence_len = int(0.5 * sr)  # 500ms
        signal_len = int(1.0 * sr)  # 1s

        silence = np.zeros(silence_len, dtype=np.float32)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, signal_len)).astype(np.float32)

        # 500ms silence + 1s signal + 500ms silence = 2s total
        audio = np.concatenate([silence, signal, silence])
        return audio, sr

    def test_trim_silence_decreases_duration(self, silence_padded_audio):
        """Trimming silence-padded audio decreases duration."""
        audio, sr = silence_padded_audio

        op = TrimSilence()
        result = op.run(audio, sr)

        # Should be shorter than original
        assert result.duration_out_s < result.duration_in_s

        # Should have trimmed from both ends
        assert result.metrics["trimmed_start_ms"] > 0
        assert result.metrics["trimmed_end_ms"] > 0
        assert result.metrics["total_trimmed_ms"] > 400  # At least 400ms

    def test_trim_silence_changes_hash(self, silence_padded_audio):
        """Trimming changes audio hash."""
        audio, sr = silence_padded_audio

        op = TrimSilence()
        result = op.run(audio, sr)

        assert result.in_audio_hash != result.out_audio_hash

    def test_trim_silence_on_no_silence(self):
        """Audio with no silence returns similar audio."""
        sr = 16000
        # Constant tone, no silence
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)

        op = TrimSilence()
        result = op.run(audio, sr)

        # Should not trim much
        assert result.metrics["total_trimmed_ms"] < 100


class TestNormalizeLoudness:
    """Tests for loudness normalization operator."""

    @pytest.fixture
    def quiet_audio(self):
        """Quiet audio that needs normalization."""
        sr = 16000
        # Very quiet sine wave
        audio = 0.01 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
        return audio, sr

    def test_normalize_changes_hash(self, quiet_audio):
        """Normalization changes audio hash."""
        audio, sr = quiet_audio

        op = NormalizeLoudness()
        result = op.run(audio, sr, target_lufs=-23.0)

        # Hash should change (unless pyloudnorm not installed)
        if "error" not in result.metrics:
            assert result.in_audio_hash != result.out_audio_hash

    def test_normalize_records_lufs(self, quiet_audio):
        """Normalization records LUFS metrics."""
        audio, sr = quiet_audio

        op = NormalizeLoudness()
        result = op.run(audio, sr, target_lufs=-23.0)

        # Should have LUFS metrics (unless error)
        if "error" not in result.metrics:
            assert "input_lufs" in result.metrics
            assert "output_lufs" in result.metrics
            assert "gain_db" in result.metrics

    @pytest.mark.skipif("pyloudnorm" not in sys.modules, reason="pyloudnorm not installed")
    def test_normalize_output_near_target(self, quiet_audio):
        """Normalized audio is near target LUFS."""
        audio, sr = quiet_audio
        target = -23.0

        op = NormalizeLoudness()
        result = op.run(audio, sr, target_lufs=target)

        if "error" not in result.metrics:
            # Output should be within 1 dB of target
            assert abs(result.metrics["output_lufs"] - target) < 1.0


class TestResample:
    """Tests for resampling operator."""

    def test_resample_no_op_when_same_rate(self):
        """No-op when already at target rate."""
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)

        op = Resample()
        result = op.run(audio, sr, target_sr=16000)

        assert result.metrics["resampled"] is False
        assert result.in_audio_hash == result.out_audio_hash

    def test_resample_changes_rate(self):
        """Resampling to different rate changes audio."""
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)

        op = Resample()
        result = op.run(audio, sr, target_sr=22050)

        assert result.metrics["resampled"] is True
        assert result.sample_rate == 22050
        assert result.in_audio_hash != result.out_audio_hash


class TestOperatorChain:
    """Tests for operator chain execution."""

    @pytest.fixture
    def test_audio(self):
        """Simple test audio."""
        sr = 16000
        silence = np.zeros(int(0.2 * sr), dtype=np.float32)
        tone = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.6, int(0.6 * sr))).astype(np.float32)
        audio = np.concatenate([silence, tone, silence])
        return audio, sr

    def test_empty_chain_returns_empty(self, test_audio):
        """Empty operator list returns empty results."""
        audio, sr = test_audio
        results = run_preprocessing_chain(audio, sr, [])
        assert results == []

    def test_chain_preserves_hash_transitions(self, test_audio):
        """Each operator's out_hash matches next operator's in_hash."""
        audio, sr = test_audio

        results = run_preprocessing_chain(audio, sr, ["trim_silence", "resample:target_sr=22050"])

        assert len(results) == 2

        # First op input should match original audio
        assert results[0].in_audio_hash == compute_audio_hash(audio)

        # First op output should match second op input
        assert results[0].out_audio_hash == results[1].in_audio_hash

    def test_chain_produces_artifact_section(self, test_audio):
        """Chain results can be converted to artifact format."""
        audio, sr = test_audio

        results = run_preprocessing_chain(audio, sr, ["trim_silence"])
        artifact_section = results_to_artifact_section(results)

        assert len(artifact_section) == 1
        assert artifact_section[0]["name"] == "trim_silence"
        assert "in_audio_hash" in artifact_section[0]
        assert "out_audio_hash" in artifact_section[0]
        assert "metrics" in artifact_section[0]
        # Audio should be excluded
        assert "audio" not in artifact_section[0]


class TestParseOperatorSpec:
    """Tests for operator specification parsing."""

    def test_simple_name(self):
        """Parse simple operator name."""
        name, params = parse_operator_spec("trim_silence")
        assert name == "trim_silence"
        assert params == {}

    def test_name_with_params(self):
        """Parse name with parameters."""
        name, params = parse_operator_spec("normalize_loudness:target_lufs=-16")
        assert name == "normalize_loudness"
        assert params == {"target_lufs": -16}

    def test_multiple_params(self):
        """Parse multiple parameters."""
        name, params = parse_operator_spec("trim_silence:threshold_db=-35,min_silence_ms=200")
        assert name == "trim_silence"
        assert params == {"threshold_db": -35, "min_silence_ms": 200}


class TestOperatorRegistry:
    """Tests for operator registry."""

    def test_all_operators_registered(self):
        """All expected operators are in registry."""
        assert "trim_silence" in OPERATORS
        assert "normalize_loudness" in OPERATORS
        assert "resample" in OPERATORS

    def test_unknown_operator_raises(self):
        """Unknown operator raises ValueError."""
        audio = np.zeros(1000, dtype=np.float32)

        with pytest.raises(ValueError) as exc_info:
            run_preprocessing_chain(audio, 16000, ["unknown_op"])

        assert "Unknown operator" in str(exc_info.value)
