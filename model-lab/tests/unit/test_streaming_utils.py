"""
Tests for streaming utilities (LCS-09).

CI-safe, no heavy deps. Tests chunking, adapter contract, and fake streaming.
"""

import numpy as np
import pytest

from harness.streaming import (
    # PCM conversion
    pcm_s16le_to_float32,
    float32_to_pcm_s16le,
    normalize_audio_input,
    resample_audio,
    # Chunking
    ChunkConfig,
    AudioChunker,
    # Streaming
    StreamEvent,
    StreamEventType,
    StreamHandle,
    StreamState,
    StreamingAdapter,
    StreamingAdapterError,
    # Endpointing
    EndpointConfig,
    SilenceEndpointer,
    # Test fake
    FakeStreamingAdapter,
)


# =============================================================================
# PCM Conversion Tests
# =============================================================================


class TestPCMConversion:
    """Tests for PCM format conversion."""
    
    def test_pcm_s16le_to_float32_range(self):
        """Converted audio should be in [-1, 1]."""
        # Create some pcm_s16le bytes
        samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        pcm = samples.tobytes()
        
        result = pcm_s16le_to_float32(pcm)
        
        assert result.dtype == np.float32
        assert result.min() >= -1.0
        assert result.max() <= 1.0
    
    def test_pcm_roundtrip(self):
        """PCM -> float32 -> PCM should be close to original."""
        original = np.array([0, 16384, -16384, 8192], dtype=np.int16)
        pcm = original.tobytes()
        
        float_audio = pcm_s16le_to_float32(pcm)
        roundtrip_pcm = float32_to_pcm_s16le(float_audio)
        roundtrip = np.frombuffer(roundtrip_pcm, dtype=np.int16)
        
        # Should be very close (may lose 1 LSB)
        np.testing.assert_array_almost_equal(original, roundtrip, decimal=0)
    
    def test_normalize_from_bytes(self):
        """normalize_audio_input should handle bytes."""
        samples = np.array([0, 16384], dtype=np.int16)
        pcm = samples.tobytes()
        
        result = normalize_audio_input(pcm)
        
        assert result.dtype == np.float32
        assert len(result) == 2
    
    def test_normalize_from_int16_array(self):
        """normalize_audio_input should handle int16 arrays."""
        audio = np.array([0, 16384, -16384], dtype=np.int16)
        
        result = normalize_audio_input(audio)
        
        assert result.dtype == np.float32
        assert len(result) == 3
    
    def test_normalize_from_float32_array(self):
        """normalize_audio_input should pass through float32."""
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        
        result = normalize_audio_input(audio)
        
        assert result.dtype == np.float32
        np.testing.assert_array_equal(audio, result)


# =============================================================================
# Resampling Tests
# =============================================================================


class TestResampling:
    """Tests for audio resampling."""
    
    def test_same_sr_no_change(self):
        """Same sample rate should return unchanged audio."""
        audio = np.random.randn(1000).astype(np.float32)
        
        result = resample_audio(audio, 16000, 16000)
        
        np.testing.assert_array_equal(audio, result)
    
    def test_resample_changes_length(self):
        """Resampling should change length proportionally."""
        audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz
        
        result = resample_audio(audio, 16000, 8000)
        
        # Should be roughly half the length
        assert len(result) == pytest.approx(8000, abs=10)
    
    def test_upsample(self):
        """Upsampling should increase length."""
        audio = np.random.randn(8000).astype(np.float32)
        
        result = resample_audio(audio, 8000, 16000)
        
        assert len(result) > len(audio)


# =============================================================================
# Chunking Tests
# =============================================================================


class TestChunkConfig:
    """Tests for ChunkConfig."""
    
    def test_default_config(self):
        """Default config should have sensible values."""
        config = ChunkConfig()
        
        assert config.frame_ms == 20
        assert config.chunk_ms == 160
        assert config.sample_rate == 16000
    
    def test_frame_samples(self):
        """frame_samples should be correct."""
        config = ChunkConfig(frame_ms=20, sample_rate=16000)
        
        assert config.frame_samples == 320  # 20ms * 16 samples/ms
    
    def test_chunk_samples(self):
        """chunk_samples should be correct."""
        config = ChunkConfig(chunk_ms=160, sample_rate=16000)
        
        assert config.chunk_samples == 2560  # 160ms * 16 samples/ms
    
    def test_frames_per_chunk(self):
        """frames_per_chunk should be chunk_ms / frame_ms."""
        config = ChunkConfig(frame_ms=20, chunk_ms=160)
        
        assert config.frames_per_chunk == 8


class TestAudioChunker:
    """Tests for AudioChunker."""
    
    def test_iter_chunks_complete(self):
        """iter_chunks should yield complete chunks."""
        config = ChunkConfig(chunk_ms=100, sample_rate=16000)
        # 1600 samples per chunk, 4800 samples = 3 complete chunks
        audio = np.zeros(4800, dtype=np.float32)
        chunker = AudioChunker(config, audio)
        
        chunks = list(chunker.iter_chunks(include_partial=False))
        
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) == 1600
    
    def test_iter_chunks_with_partial(self):
        """iter_chunks with partial should include remainder."""
        config = ChunkConfig(chunk_ms=100, sample_rate=16000)
        # 4000 samples = 2 complete chunks + 800 partial
        audio = np.zeros(4000, dtype=np.float32)
        chunker = AudioChunker(config, audio)
        
        chunks = list(chunker.iter_chunks(include_partial=True))
        
        assert len(chunks) == 3
        assert len(chunks[-1]) == 800  # Partial chunk
    
    def test_chunker_with_resampling(self):
        """Chunker should resample audio once."""
        config = ChunkConfig(chunk_ms=100, sample_rate=16000)
        # Audio at 8kHz (half rate)
        audio = np.zeros(800, dtype=np.float32)  # 100ms at 8kHz
        chunker = AudioChunker(config, audio, orig_sr=8000)
        
        # After resampling: 1600 samples = 1 chunk
        assert chunker.total_samples == 1600
    
    def test_get_next_chunk(self):
        """get_next_chunk should advance position."""
        config = ChunkConfig(chunk_ms=100, sample_rate=16000)
        audio = np.arange(4800, dtype=np.float32)
        chunker = AudioChunker(config, audio)
        
        chunk1 = chunker.get_next_chunk()
        chunk2 = chunker.get_next_chunk()
        
        assert chunk1 is not None
        assert chunk2 is not None
        assert not np.array_equal(chunk1, chunk2)
    
    def test_reset(self):
        """reset should restart iteration."""
        config = ChunkConfig(chunk_ms=100, sample_rate=16000)
        audio = np.arange(3200, dtype=np.float32)
        chunker = AudioChunker(config, audio)
        
        _ = chunker.get_next_chunk()
        _ = chunker.get_next_chunk()
        chunker.reset()
        
        chunk = chunker.get_next_chunk()
        assert chunk is not None
        assert chunk[0] == 0  # Back to beginning


# =============================================================================
# Stream Event Tests
# =============================================================================


class TestStreamEvent:
    """Tests for StreamEvent."""
    
    def test_to_dict(self):
        """to_dict should produce valid dictionary."""
        event = StreamEvent(
            type=StreamEventType.PARTIAL,
            text="hello",
            seq=1,
            segment_id="seg_0",
        )
        
        d = event.to_dict()
        
        assert d["type"] == "partial"
        assert d["text"] == "hello"
        assert d["seq"] == 1
        assert d["segment_id"] == "seg_0"
    
    def test_error_fields_in_dict(self):
        """Error fields should appear in dict when set."""
        event = StreamEvent(
            type=StreamEventType.ERROR,
            error_code="DECODE_ERROR",
            error_message="Failed to decode audio",
        )
        
        d = event.to_dict()
        
        assert d["error_code"] == "DECODE_ERROR"
        assert d["error_message"] == "Failed to decode audio"


# =============================================================================
# Stream Handle Tests
# =============================================================================


class TestStreamHandle:
    """Tests for StreamHandle."""
    
    def test_next_seq_monotonic(self):
        """next_seq should always increase."""
        handle = StreamHandle(stream_id="test")
        
        seqs = [handle.next_seq() for _ in range(5)]
        
        assert seqs == [0, 1, 2, 3, 4]
    
    def test_segment_id_stability(self):
        """Same segment_id should be returned for same key."""
        handle = StreamHandle(stream_id="test")
        
        id1 = handle.get_or_create_segment_id("seg_0")
        id2 = handle.get_or_create_segment_id("seg_0")
        
        assert id1 == id2 == "seg_0"
    
    def test_new_segment_id(self):
        """New segment should get new ID."""
        handle = StreamHandle(stream_id="test")
        
        id1 = handle.get_or_create_segment_id()
        id2 = handle.get_or_create_segment_id()
        
        assert id1 != id2


# =============================================================================
# Fake Streaming Adapter Tests
# =============================================================================


class TestFakeStreamingAdapter:
    """Tests for FakeStreamingAdapter contract behavior."""
    
    def test_push_before_start_raises(self):
        """push_audio before start_stream should raise."""
        adapter = FakeStreamingAdapter()
        audio = np.zeros(1600, dtype=np.float32)
        
        with pytest.raises(StreamingAdapterError, match="before start_stream"):
            list(adapter.push_audio(audio, 16000))
    
    def test_seq_monotonic(self):
        """Sequence numbers should always increase."""
        adapter = FakeStreamingAdapter(words=["a", "b", "c", "d"], emit_per_chunk=1)
        adapter.start_stream()
        
        audio = np.zeros(1600, dtype=np.float32)
        
        events = []
        for _ in range(4):
            events.extend(adapter.push_audio(audio, 16000))
        
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs)  # Monotonically increasing
        assert len(set(seqs)) == len(seqs)  # All unique
    
    def test_segment_id_stable(self):
        """segment_id should be stable across partial updates."""
        adapter = FakeStreamingAdapter(words=["hello", "world", "test"], emit_per_chunk=1)
        adapter.start_stream()
        
        audio = np.zeros(1600, dtype=np.float32)
        
        events = []
        for _ in range(3):
            events.extend(adapter.push_audio(audio, 16000))
        
        segment_ids = [e.segment_id for e in events]
        
        # All events for same segment should have same ID
        assert len(set(segment_ids)) == 1
    
    def test_flush_emits_remaining(self):
        """flush should emit remaining events."""
        adapter = FakeStreamingAdapter(words=["a", "b", "c", "d"], emit_per_chunk=1)
        adapter.start_stream()
        
        audio = np.zeros(1600, dtype=np.float32)
        list(adapter.push_audio(audio, 16000))  # Consumes 1 word
        
        flush_events = list(adapter.flush())
        
        assert len(flush_events) == 1  # Final event
        assert flush_events[0].type == StreamEventType.FINAL
        assert "a b c d" in flush_events[0].text
    
    def test_finalize_idempotent(self):
        """finalize called twice should return same result."""
        adapter = FakeStreamingAdapter(words=["test"])
        adapter.start_stream()
        list(adapter.push_audio(np.zeros(1600, dtype=np.float32), 16000))
        
        result1 = adapter.finalize()
        result2 = adapter.finalize()
        
        assert result1 == result2
    
    def test_push_after_finalize_raises(self):
        """push_audio after finalize should raise."""
        adapter = FakeStreamingAdapter(words=["test"])
        adapter.start_stream()
        adapter.finalize()
        
        with pytest.raises(StreamingAdapterError, match="after finalize"):
            list(adapter.push_audio(np.zeros(1600, dtype=np.float32), 16000))
    
    def test_close_after_finalize_safe(self):
        """close should be safe after finalize."""
        adapter = FakeStreamingAdapter(words=["test"])
        adapter.start_stream()
        adapter.finalize()
        
        # Should not raise
        adapter.close()
    
    def test_finalize_returns_asr_result(self):
        """finalize should return ASRResult-compatible dict."""
        adapter = FakeStreamingAdapter(words=["hello", "world"])
        adapter.start_stream()
        
        audio = np.zeros(1600, dtype=np.float32)
        list(adapter.push_audio(audio, 16000))
        list(adapter.push_audio(audio, 16000))
        
        result = adapter.finalize()
        
        assert "text" in result
        assert "segments" in result
        assert "language" in result


# =============================================================================
# Silence Endpointer Tests
# =============================================================================


class TestSilenceEndpointer:
    """Tests for SilenceEndpointer."""
    
    def test_no_endpoint_during_speech(self):
        """Should not trigger endpoint during speech."""
        config = EndpointConfig(
            silence_threshold_ms=500,
            energy_threshold=0.01,
        )
        endpointer = SilenceEndpointer(config)
        
        # Generate some "speech" (high energy)
        audio = np.random.randn(16000) * 0.5  # 1 second
        
        # Process in chunks
        chunk_size = 1600
        endpoints = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if endpointer.process_chunk(chunk.astype(np.float32), 16000):
                endpoints.append(i)
        
        assert len(endpoints) == 0
    
    def test_endpoint_after_silence(self):
        """Should trigger endpoint after sufficient silence."""
        config = EndpointConfig(
            silence_threshold_ms=200,  # Short threshold for test
            energy_threshold=0.01,
            min_segment_ms=100,
        )
        endpointer = SilenceEndpointer(config)
        
        # Speech followed by silence
        speech = np.random.randn(3200) * 0.5  # 200ms speech
        silence = np.zeros(4800)  # 300ms silence
        audio = np.concatenate([speech, silence]).astype(np.float32)
        
        # Process in chunks
        chunk_size = 1600
        endpoints = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if endpointer.process_chunk(chunk, 16000):
                endpoints.append(i)
        
        assert len(endpoints) >= 1
    
    def test_reset_clears_silence(self):
        """reset should clear silence tracking."""
        endpointer = SilenceEndpointer()
        
        # Process some silence
        silence = np.zeros(8000, dtype=np.float32)
        endpointer.process_chunk(silence, 16000)
        
        endpointer.reset()
        
        # Internal state should be reset
        assert endpointer._silence_start_ms is None
