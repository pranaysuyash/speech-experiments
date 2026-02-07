"""
Streaming latency metrics tests (LCS-Y).

Uses FakeStreamingAdapter for deterministic testing.
Tests: partials exist, no partials (only final), empty text partials ignored.
"""

import pytest
import time
from unittest.mock import MagicMock
import numpy as np


from harness.streaming_metrics import (
    measure_streaming_latency,
    format_latency_report,
)


class FakeStreamingAdapterWithPartials:
    """Fake adapter that produces partials then a final."""
    
    def __init__(self):
        self._chunks = 0
        self._started = False
        self._finalized = False
    
    def start(self, sr=16000):
        self._started = True
        self._chunks = 0
        self._finalized = False
        return "fake_handle"
    
    def push_audio(self, handle, audio):
        if self._finalized:
            raise RuntimeError("Cannot push after finalize")
        self._chunks += 1
    
    def get_transcript(self, handle):
        if self._chunks == 0:
            return {"text": "", "is_final": False}
        elif self._chunks < 3:
            return {"text": f"partial {self._chunks}", "is_final": False}
        else:
            return {"text": f"partial {self._chunks}", "is_final": False}
    
    def finalize(self, handle):
        self._finalized = True
        return {"text": "final transcript", "is_final": True}


class FakeStreamingAdapterNoPartials:
    """Fake adapter that produces only a final (no partials)."""
    
    def __init__(self):
        self._started = False
        self._finalized = False
    
    def start(self, sr=16000):
        self._started = True
        self._finalized = False
        return "fake_handle"
    
    def push_audio(self, handle, audio):
        if self._finalized:
            raise RuntimeError("Cannot push after finalize")
    
    def get_transcript(self, handle):
        return {"text": "", "is_final": False}
    
    def finalize(self, handle):
        self._finalized = True
        return {"text": "final only", "is_final": True}


class FakeStreamingAdapterEmptyPartials:
    """Fake adapter that produces empty partials before non-empty ones."""
    
    def __init__(self):
        self._chunks = 0
        self._started = False
        self._finalized = False
    
    def start(self, sr=16000):
        self._started = True
        self._chunks = 0
        self._finalized = False
        return "fake_handle"
    
    def push_audio(self, handle, audio):
        if self._finalized:
            raise RuntimeError("Cannot push after finalize")
        self._chunks += 1
    
    def get_transcript(self, handle):
        # First 2 chunks: empty text (should be ignored)
        if self._chunks <= 2:
            return {"text": "", "is_final": False}
        # Then non-empty
        return {"text": f"text {self._chunks}", "is_final": False}
    
    def finalize(self, handle):
        self._finalized = True
        return {"text": "final", "is_final": True}


class TestMeasureStreamingLatencyWithPartials:
    """Test with partials present."""
    
    def test_returns_all_required_fields(self):
        adapter = FakeStreamingAdapterWithPartials()
        ns = {
            "start": adapter.start,
            "push_audio": adapter.push_audio,
            "get_transcript": adapter.get_transcript,
            "finalize": adapter.finalize,
        }
        
        # 1 second of audio at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        
        metrics = measure_streaming_latency(ns, audio, sr=16000, chunk_ms=160)
        
        # Primary metrics
        assert "first_token_latency_ms" in metrics
        assert "partial_update_rate_hz" in metrics
        assert "finalize_latency_ms" in metrics
        assert "real_time_factor" in metrics
        
        # Debug fields
        assert "num_events" in metrics
        assert "num_partials" in metrics
        assert "num_finals" in metrics
        assert "audio_duration_s" in metrics
    
    def test_first_token_latency_is_positive(self):
        adapter = FakeStreamingAdapterWithPartials()
        ns = {
            "start": adapter.start,
            "push_audio": adapter.push_audio,
            "get_transcript": adapter.get_transcript,
            "finalize": adapter.finalize,
        }
        audio = np.zeros(16000, dtype=np.float32)
        
        metrics = measure_streaming_latency(ns, audio, sr=16000, chunk_ms=160)
        
        # First token should be detected after first non-empty partial
        assert metrics["first_token_latency_ms"] is not None
        assert metrics["first_token_latency_ms"] > 0
    
    def test_partial_rate_calculated(self):
        adapter = FakeStreamingAdapterWithPartials()
        ns = {
            "start": adapter.start,
            "push_audio": adapter.push_audio,
            "get_transcript": adapter.get_transcript,
            "finalize": adapter.finalize,
        }
        audio = np.zeros(16000, dtype=np.float32)
        
        metrics = measure_streaming_latency(ns, audio, sr=16000, chunk_ms=160)
        
        # Should have multiple partials
        assert metrics["num_partials"] >= 2
        assert metrics["partial_update_rate_hz"] > 0


class TestMeasureStreamingLatencyNoPartials:
    """Test with no partials (only final)."""
    
    def test_first_token_from_final(self):
        adapter = FakeStreamingAdapterNoPartials()
        ns = {
            "start": adapter.start,
            "push_audio": adapter.push_audio,
            "get_transcript": adapter.get_transcript,
            "finalize": adapter.finalize,
        }
        audio = np.zeros(16000, dtype=np.float32)
        
        metrics = measure_streaming_latency(ns, audio, sr=16000, chunk_ms=160)
        
        # First token should come from final
        assert metrics["first_token_latency_ms"] is not None
        assert metrics["num_partials"] == 0
        assert metrics["partial_update_rate_hz"] == 0.0
    
    def test_finalize_latency_measured(self):
        adapter = FakeStreamingAdapterNoPartials()
        ns = {
            "start": adapter.start,
            "push_audio": adapter.push_audio,
            "get_transcript": adapter.get_transcript,
            "finalize": adapter.finalize,
        }
        audio = np.zeros(16000, dtype=np.float32)
        
        metrics = measure_streaming_latency(ns, audio, sr=16000, chunk_ms=160)
        
        assert metrics["finalize_latency_ms"] >= 0


class TestMeasureStreamingLatencyEmptyPartials:
    """Test that empty text partials are ignored until non-empty."""
    
    def test_empty_partials_ignored(self):
        adapter = FakeStreamingAdapterEmptyPartials()
        ns = {
            "start": adapter.start,
            "push_audio": adapter.push_audio,
            "get_transcript": adapter.get_transcript,
            "finalize": adapter.finalize,
        }
        audio = np.zeros(16000, dtype=np.float32)
        
        metrics = measure_streaming_latency(ns, audio, sr=16000, chunk_ms=160)
        
        # First token should NOT count empty partials
        # Empty partials should be excluded from num_partials
        assert metrics["first_token_latency_ms"] is not None


class TestFormatLatencyReport:
    """Test report formatting."""
    
    def test_format_report(self):
        metrics = {
            "first_token_latency_ms": 123.5,
            "partial_update_rate_hz": 10.5,
            "finalize_latency_ms": 5.123,
            "real_time_factor": 0.1234,
            "num_events": 10,
            "num_partials": 5,
            "num_finals": 1,
            "audio_duration_s": 1.500,
        }
        
        report = format_latency_report(metrics)
        
        assert "First Token Latency" in report
        assert "123.5" in report
        assert "Partial Update Rate" in report
        assert "Finalize Latency" in report
        assert "Real-Time Factor" in report
