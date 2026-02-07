"""
Tests for ASR streaming contract misuse cases (LCS-03b).

Verifies that streaming lifecycle violations are caught with clear errors.
Uses a fake namespace implementation to test contract enforcement.
"""

import pytest
from typing import Any, Iterator

from harness.contracts import ASRStreamEvent, ASRResult


class FakeStreamHandle:
    """A mock stream handle that tracks state."""
    
    def __init__(self, config: dict):
        self.config = config
        self.started = True
        self.finalized = False
        self.closed = False
        self.cached_result: ASRResult | None = None
        self.events: list[str] = []


class FakeASRStreamNamespace:
    """
    Fake streaming ASR implementation for contract testing.
    
    Lifecycle rules:
    - push_audio requires started=True and finalized=False
    - finalize is idempotent (returns cached result on second call)
    - close is always safe
    """
    
    def __init__(self):
        self.handles: dict[int, FakeStreamHandle] = {}
        self._next_id = 0
    
    def start_stream(self, config: dict) -> int:
        """Start a new stream, return handle."""
        handle_id = self._next_id
        self._next_id += 1
        self.handles[handle_id] = FakeStreamHandle(config)
        return handle_id
    
    def push_audio(
        self, handle: int, pcm: bytes | Any, sr: int
    ) -> Iterator[ASRStreamEvent]:
        """Push audio to stream."""
        if handle not in self.handles:
            raise ValueError(f"Invalid handle: {handle}. Call start_stream first.")
        
        h = self.handles[handle]
        
        if not h.started:
            raise ValueError("Stream not started. Call start_stream first.")
        
        if h.finalized:
            raise RuntimeError("Stream already finalized. Cannot push more audio.")
        
        if h.closed:
            raise RuntimeError("Stream already closed. Cannot push audio.")
        
        # Emit a partial event
        h.events.append(f"push:{len(pcm) if isinstance(pcm, bytes) else 'array'}")
        yield ASRStreamEvent(
            type="partial",
            text="...",
            seq=len(h.events),
            segment_id="seg0",
        )
    
    def flush(self, handle: int) -> Iterator[ASRStreamEvent]:
        """Flush remaining audio."""
        if handle not in self.handles:
            raise ValueError(f"Invalid handle: {handle}")
        
        h = self.handles[handle]
        if h.finalized:
            return  # No events after finalize
        
        h.events.append("flush")
        yield ASRStreamEvent(type="partial", text="flushed", seq=len(h.events))
    
    def finalize(self, handle: int) -> ASRResult:
        """
        Finalize stream and return result.
        
        Idempotent: second call returns cached result.
        """
        if handle not in self.handles:
            raise ValueError(f"Invalid handle: {handle}")
        
        h = self.handles[handle]
        
        if h.closed:
            raise RuntimeError("Stream already closed. Cannot finalize.")
        
        if h.finalized:
            # Idempotent: return cached result
            return h.cached_result
        
        # First finalize: compute result
        h.finalized = True
        h.events.append("finalize")
        h.cached_result = ASRResult(
            text="Final transcription",
            segments=[],
            language="en",
        )
        return h.cached_result
    
    def close(self, handle: int) -> None:
        """
        Close stream and cleanup.
        
        Always safe, even if already finalized or closed.
        """
        if handle not in self.handles:
            return  # Already cleaned up, no-op
        
        h = self.handles[handle]
        h.closed = True
        h.events.append("close")
        # Optionally remove handle for full cleanup
        # del self.handles[handle]


class TestStreamingMisuse:
    """Test streaming lifecycle misuse cases."""
    
    @pytest.fixture
    def ns(self):
        """Create a fresh fake namespace."""
        return FakeASRStreamNamespace()
    
    def test_push_before_start_raises(self, ns):
        """push_audio before start_stream raises ValueError."""
        with pytest.raises(ValueError, match="Invalid handle"):
            # Use a handle that doesn't exist
            list(ns.push_audio(999, b"audio", 16000))
    
    def test_push_after_finalize_raises(self, ns):
        """push_audio after finalize raises RuntimeError."""
        handle = ns.start_stream({})
        list(ns.push_audio(handle, b"audio", 16000))
        ns.finalize(handle)
        
        with pytest.raises(RuntimeError, match="already finalized"):
            list(ns.push_audio(handle, b"more", 16000))
    
    def test_push_after_close_raises(self, ns):
        """push_audio after close raises RuntimeError."""
        handle = ns.start_stream({})
        ns.close(handle)
        
        with pytest.raises(RuntimeError, match="already closed"):
            list(ns.push_audio(handle, b"audio", 16000))
    
    def test_finalize_twice_is_idempotent(self, ns):
        """finalize called twice returns same result (idempotent)."""
        handle = ns.start_stream({})
        list(ns.push_audio(handle, b"audio", 16000))
        
        result1 = ns.finalize(handle)
        result2 = ns.finalize(handle)
        
        assert result1 is result2
        assert result1["text"] == "Final transcription"
    
    def test_close_after_finalize_is_safe(self, ns):
        """close is safe after finalize."""
        handle = ns.start_stream({})
        ns.finalize(handle)
        
        # Should not raise
        ns.close(handle)
    
    def test_close_twice_is_safe(self, ns):
        """close called twice is safe (no-op)."""
        handle = ns.start_stream({})
        ns.close(handle)
        
        # Second close should not raise
        ns.close(handle)
    
    def test_finalize_after_close_raises(self, ns):
        """finalize after close raises RuntimeError."""
        handle = ns.start_stream({})
        ns.close(handle)
        
        with pytest.raises(RuntimeError, match="already closed"):
            ns.finalize(handle)


class TestStreamingHappyPath:
    """Test normal streaming lifecycle."""
    
    @pytest.fixture
    def ns(self):
        return FakeASRStreamNamespace()
    
    def test_full_lifecycle(self, ns):
        """Normal lifecycle: start -> push -> flush -> finalize -> close."""
        handle = ns.start_stream({"language": "en"})
        
        # Push audio
        events = list(ns.push_audio(handle, b"chunk1", 16000))
        assert len(events) == 1
        assert events[0]["type"] == "partial"
        
        # Flush
        events = list(ns.flush(handle))
        assert len(events) == 1
        
        # Finalize
        result = ns.finalize(handle)
        assert result["text"] == "Final transcription"
        
        # Close
        ns.close(handle)
