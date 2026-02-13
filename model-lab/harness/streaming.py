"""
Streaming utilities for ASR and audio processing (LCS-09).

This module provides the infrastructure for streaming audio models:
1. Chunking helpers - Convert audio to frames/chunks with sample-rate normalization
2. Adapter base class - Enforce contract behavior for streaming models
3. Endpointing heuristics - Optional silence-based endpointing

Contract behavior enforced:
- seq monotonicity: sequence numbers always increase
- segment_id stability: same segment keeps same ID across updates
- Lifecycle: push_audio before start_stream raises, finalize twice is idempotent

Supported PCM formats:
- bytes: pcm_s16le (signed 16-bit little-endian)
- np.ndarray: float32 or int16

NO MODEL RUNTIME IMPORTS - this module must remain lightweight.
"""

from __future__ import annotations

import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, TypedDict

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PCM Conversion Utilities
# =============================================================================


def pcm_s16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert pcm_s16le bytes to float32 numpy array.
    
    Args:
        pcm_bytes: Raw PCM bytes in signed 16-bit little-endian format
        
    Returns:
        Float32 numpy array normalized to [-1, 1]
    """
    # Unpack as int16
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
    # Convert to float32 in [-1, 1]
    return np.array(samples, dtype=np.float32) / 32768.0


def float32_to_pcm_s16le(audio: np.ndarray) -> bytes:
    """
    Convert float32 numpy array to pcm_s16le bytes.
    
    Args:
        audio: Float32 numpy array in [-1, 1]
        
    Returns:
        Raw PCM bytes in signed 16-bit little-endian format
    """
    # Clip to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)
    # Convert to int16
    samples = (audio * 32767).astype(np.int16)
    return samples.tobytes()


def normalize_audio_input(
    audio: bytes | np.ndarray,
    target_dtype: str = "float32",
) -> np.ndarray:
    """
    Normalize audio input to consistent numpy array format.
    
    Args:
        audio: Input audio as bytes (pcm_s16le) or numpy array (float32/int16)
        target_dtype: Target dtype ("float32" or "int16")
        
    Returns:
        Numpy array in target dtype
    """
    if isinstance(audio, bytes):
        # Assume pcm_s16le
        audio = pcm_s16le_to_float32(audio)
    elif isinstance(audio, np.ndarray):
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")
    
    if target_dtype == "int16":
        return (audio * 32767).astype(np.int16)
    return audio.astype(np.float32)


# =============================================================================
# Resampling Utility
# =============================================================================


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Uses librosa if available, falls back to simple linear interpolation.
    
    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Simple linear interpolation fallback
        ratio = target_sr / orig_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


# =============================================================================
# Chunking Helpers
# =============================================================================


@dataclass
class ChunkConfig:
    """Configuration for audio chunking."""
    
    frame_ms: int = 20  # Frame size in milliseconds
    chunk_ms: int = 160  # Chunk size in milliseconds (multiple of frame_ms)
    sample_rate: int = 16000  # Target sample rate
    
    @property
    def frame_samples(self) -> int:
        """Number of samples per frame."""
        return int(self.sample_rate * self.frame_ms / 1000)
    
    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk."""
        return int(self.sample_rate * self.chunk_ms / 1000)
    
    @property
    def frames_per_chunk(self) -> int:
        """Number of frames per chunk."""
        return self.chunk_ms // self.frame_ms


class AudioChunker:
    """
    Chunk audio into frames and chunks for streaming processing.
    
    Handles sample-rate normalization once at initialization.
    
    Usage:
        chunker = AudioChunker(config, audio, orig_sr=44100)
        for chunk in chunker.iter_chunks():
            # Process chunk (numpy array of chunk_samples length)
            pass
    """
    
    def __init__(
        self,
        config: ChunkConfig,
        audio: bytes | np.ndarray,
        orig_sr: int | None = None,
    ):
        """
        Initialize chunker with audio data.
        
        Args:
            config: Chunking configuration
            audio: Input audio (bytes pcm_s16le or numpy array)
            orig_sr: Original sample rate (required if != config.sample_rate)
        """
        self.config = config
        
        # Normalize to float32
        self._audio = normalize_audio_input(audio, "float32")
        
        # Resample if needed (do this ONCE, not per chunk)
        if orig_sr is not None and orig_sr != config.sample_rate:
            self._audio = resample_audio(self._audio, orig_sr, config.sample_rate)
        
        self._position = 0
    
    @property
    def total_samples(self) -> int:
        """Total number of samples in the audio."""
        return len(self._audio)
    
    @property
    def total_chunks(self) -> int:
        """Total number of complete chunks."""
        return self.total_samples // self.config.chunk_samples
    
    @property
    def remaining_samples(self) -> int:
        """Samples remaining after all complete chunks."""
        return self.total_samples % self.config.chunk_samples
    
    def iter_frames(self) -> Iterator[np.ndarray]:
        """Iterate over audio in frame-sized pieces."""
        frame_size = self.config.frame_samples
        for i in range(0, len(self._audio), frame_size):
            yield self._audio[i:i + frame_size]
    
    def iter_chunks(self, include_partial: bool = True) -> Iterator[np.ndarray]:
        """
        Iterate over audio in chunk-sized pieces.
        
        Args:
            include_partial: Whether to include final partial chunk
        """
        chunk_size = self.config.chunk_samples
        for i in range(0, len(self._audio), chunk_size):
            chunk = self._audio[i:i + chunk_size]
            if len(chunk) == chunk_size or include_partial:
                yield chunk
    
    def get_next_chunk(self) -> np.ndarray | None:
        """Get next chunk, advancing position. Returns None when exhausted."""
        chunk_size = self.config.chunk_samples
        if self._position >= len(self._audio):
            return None
        
        chunk = self._audio[self._position:self._position + chunk_size]
        self._position += chunk_size
        return chunk
    
    def reset(self) -> None:
        """Reset position to beginning."""
        self._position = 0


# =============================================================================
# Streaming Event Types
# =============================================================================


class StreamEventType(str, Enum):
    """Types of streaming events."""
    
    PARTIAL = "partial"  # Partial/interim transcript
    FINAL = "final"  # Final transcript for segment
    ENDPOINT = "endpoint"  # Endpoint detected
    ERROR = "error"  # Error occurred
    VAD = "vad"  # Voice activity detection


@dataclass
class StreamEvent:
    """
    A streaming event from an ASR model.
    
    Matches ASRStreamEvent contract from contracts.py.
    """
    
    type: StreamEventType
    text: str = ""
    seq: int = 0  # Monotonically increasing sequence number
    segment_id: str = ""  # Stable ID for segment across updates
    
    # Timing
    t_audio_ms_start: float = 0.0
    t_audio_ms_end: float = 0.0
    t_emit_ms: float = 0.0  # Wall-clock time of emission
    
    # Optional fields
    stability: float | None = None
    is_endpoint: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)
    
    # Error fields
    error_code: str | None = None
    error_message: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching ASRStreamEvent."""
        result = {
            "type": self.type.value if isinstance(self.type, StreamEventType) else self.type,
            "text": self.text,
            "seq": self.seq,
            "segment_id": self.segment_id,
            "t_audio_ms_start": self.t_audio_ms_start,
            "t_audio_ms_end": self.t_audio_ms_end,
            "t_emit_ms": self.t_emit_ms,
        }
        if self.stability is not None:
            result["stability"] = self.stability
        if self.is_endpoint:
            result["is_endpoint"] = True
        if self.segments:
            result["segments"] = self.segments
        if self.error_code:
            result["error_code"] = self.error_code
            result["error_message"] = self.error_message
        return result


# =============================================================================
# Stream Handle
# =============================================================================


class StreamState(Enum):
    """State of a stream."""
    
    CREATED = auto()
    STARTED = auto()
    FLUSHED = auto()
    FINALIZED = auto()
    CLOSED = auto()


@dataclass
class StreamHandle:
    """
    Handle for a streaming session.
    
    Tracks state and enforces lifecycle rules.
    """
    
    stream_id: str
    state: StreamState = StreamState.CREATED
    config: dict[str, Any] = field(default_factory=dict)
    
    # Sequence tracking
    _next_seq: int = field(default=0, repr=False)
    _segments: dict[str, int] = field(default_factory=dict, repr=False)  # segment_id -> first_seq
    
    # Timing
    _audio_position_ms: float = field(default=0.0, repr=False)
    
    def next_seq(self) -> int:
        """Get next sequence number (always increasing)."""
        seq = self._next_seq
        self._next_seq += 1
        return seq
    
    def get_or_create_segment_id(self, suggested_id: str | None = None) -> str:
        """Get or create a stable segment ID."""
        if suggested_id and suggested_id in self._segments:
            return suggested_id
        
        # Create new segment ID
        segment_id = suggested_id or f"seg_{len(self._segments)}"
        self._segments[segment_id] = self._next_seq
        return segment_id
    
    def advance_audio_position(self, duration_ms: float) -> None:
        """Advance audio position by duration."""
        self._audio_position_ms += duration_ms


# =============================================================================
# Streaming Adapter Base Class
# =============================================================================


class StreamingAdapterError(Exception):
    """Error in streaming adapter."""
    pass


class StreamingAdapter(ABC):
    """
    Base class for streaming model adapters.
    
    Enforces contract behavior:
    - seq monotonicity
    - segment_id stability
    - Lifecycle rules (push after finalize raises, etc.)
    
    Subclasses implement:
    - _do_start_stream(config) -> None
    - _do_push_audio(audio, sr) -> Iterator[StreamEvent]
    - _do_flush() -> Iterator[StreamEvent]
    - _do_finalize() -> dict (final result)
    - _do_close() -> None
    """
    
    def __init__(self, model_type: str, debug: bool = False):
        """
        Initialize adapter.
        
        Args:
            model_type: Name of the model this adapts
            debug: Enable debug assertions
        """
        self.model_type = model_type
        self.debug = debug
        self._handle: StreamHandle | None = None
        self._finalize_result: dict[str, Any] | None = None
    
    @property
    def handle(self) -> StreamHandle:
        """Get current stream handle."""
        if self._handle is None:
            raise StreamingAdapterError("Stream not started. Call start_stream() first.")
        return self._handle
    
    @property
    def is_started(self) -> bool:
        """Check if stream has been started."""
        return self._handle is not None and self._handle.state != StreamState.CREATED
    
    @property
    def is_finalized(self) -> bool:
        """Check if stream has been finalized."""
        return self._handle is not None and self._handle.state == StreamState.FINALIZED
    
    def start_stream(self, config: dict[str, Any] | None = None) -> StreamHandle:
        """
        Start a new streaming session.
        
        Args:
            config: Optional configuration for the stream
            
        Returns:
            StreamHandle for this session
        """
        import uuid
        
        if self._handle is not None and self._handle.state not in (
            StreamState.CLOSED,
            StreamState.FINALIZED,
        ):
            raise StreamingAdapterError(
                f"Stream already active (state={self._handle.state}). "
                "Call close() first or finalize() before starting new stream."
            )
        
        stream_id = str(uuid.uuid4())[:8]
        self._handle = StreamHandle(
            stream_id=stream_id,
            state=StreamState.STARTED,
            config=config or {},
        )
        self._finalize_result = None
        
        logger.debug(f"[{self.model_type}] Starting stream {stream_id}")
        self._do_start_stream(config or {})
        
        return self._handle
    
    def push_audio(
        self,
        audio: bytes | np.ndarray,
        sr: int = 16000,
    ) -> Iterator[StreamEvent]:
        """
        Push audio chunk and get streaming events.
        
        Args:
            audio: Audio data (bytes pcm_s16le or numpy array)
            sr: Sample rate of the audio
            
        Yields:
            StreamEvent objects
            
        Raises:
            StreamingAdapterError: If called before start_stream or after finalize
        """
        if self._handle is None:
            raise StreamingAdapterError(
                "push_audio called before start_stream. "
                "Call start_stream() first."
            )
        
        if self._handle.state == StreamState.FINALIZED:
            raise StreamingAdapterError(
                "push_audio called after finalize. "
                "Stream is finalized. Call close() and start a new stream."
            )
        
        if self._handle.state == StreamState.CLOSED:
            raise StreamingAdapterError(
                "push_audio called after close. Start a new stream."
            )
        
        # Track last seq for monotonicity check
        last_seq = self._handle._next_seq - 1
        
        for event in self._do_push_audio(audio, sr):
            # Assign sequence number if not set
            if event.seq == 0 and last_seq >= 0:
                event.seq = self._handle.next_seq()
            elif event.seq == 0:
                event.seq = self._handle.next_seq()
            
            # Debug: verify monotonicity
            if self.debug and event.seq <= last_seq:
                raise StreamingAdapterError(
                    f"seq not monotonic: got {event.seq}, expected > {last_seq}"
                )
            last_seq = event.seq
            
            yield event
    
    def flush(self) -> Iterator[StreamEvent]:
        """
        Flush pending events.
        
        Call after sending all audio but before finalize.
        
        Yields:
            Remaining StreamEvent objects
        """
        if self._handle is None:
            return
        
        if self._handle.state == StreamState.FINALIZED:
            return
        
        self._handle.state = StreamState.FLUSHED
        yield from self._do_flush()
    
    def finalize(self) -> dict[str, Any]:
        """
        Finalize the stream and get final result.
        
        Idempotent: calling multiple times returns same result.
        
        Returns:
            Final ASRResult-compatible dictionary
        """
        if self._finalize_result is not None:
            # Idempotent: return cached result
            return self._finalize_result
        
        if self._handle is None:
            raise StreamingAdapterError("finalize called before start_stream")
        
        # Flush first if not already done
        if self._handle.state == StreamState.STARTED:
            list(self.flush())  # Consume flush events
        
        self._handle.state = StreamState.FINALIZED
        self._finalize_result = self._do_finalize()
        
        logger.debug(f"[{self.model_type}] Finalized stream {self._handle.stream_id}")
        return self._finalize_result

    def get_transcript(self) -> dict[str, Any]:
        """
        Get the current transcript snapshot.

        Default behavior returns cached final result if available.
        Subclasses may override for partial results.
        """
        if self._finalize_result is not None:
            return {
                "text": self._finalize_result.get("text", ""),
                "is_final": True,
            }

        return {"text": "", "is_final": False}
    
    def close(self) -> None:
        """
        Close the stream and release resources.
        
        Safe to call after finalize or at any time.
        """
        if self._handle is not None:
            self._handle.state = StreamState.CLOSED
            self._do_close()
            logger.debug(f"[{self.model_type}] Closed stream {self._handle.stream_id}")
    
    # Abstract methods for subclasses
    
    @abstractmethod
    def _do_start_stream(self, config: dict[str, Any]) -> None:
        """Initialize model-specific stream state."""
        pass
    
    @abstractmethod
    def _do_push_audio(
        self,
        audio: bytes | np.ndarray,
        sr: int,
    ) -> Iterator[StreamEvent]:
        """Process audio and yield events."""
        pass
    
    @abstractmethod
    def _do_flush(self) -> Iterator[StreamEvent]:
        """Flush remaining events."""
        pass
    
    @abstractmethod
    def _do_finalize(self) -> dict[str, Any]:
        """Finalize and return result."""
        pass
    
    def _do_close(self) -> None:
        """Clean up resources (optional override)."""
        pass


# =============================================================================
# Endpointing Heuristic
# =============================================================================


@dataclass
class EndpointConfig:
    """Configuration for silence-based endpointing."""
    
    silence_threshold_ms: int = 700  # Silence duration to trigger endpoint
    energy_threshold: float = 0.01  # RMS energy below this is silence
    min_segment_ms: int = 200  # Minimum segment length before endpoint


class SilenceEndpointer:
    """
    Simple silence-based endpointing utility.
    
    Uses energy-based silence detection. Default OFF.
    Only use when model does not provide its own endpoints.
    
    Usage:
        endpointer = SilenceEndpointer(config)
        for chunk in audio_chunks:
            if endpointer.process_chunk(chunk, sr):
                # Endpoint detected
                endpointer.reset()
    """
    
    def __init__(self, config: EndpointConfig | None = None):
        self.config = config or EndpointConfig()
        self._silence_start_ms: float | None = None
        self._audio_position_ms: float = 0.0
        self._last_speech_end_ms: float = 0.0
    
    def process_chunk(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> bool:
        """
        Process audio chunk and check for endpoint.
        
        Args:
            audio: Audio chunk as numpy array
            sr: Sample rate
            
        Returns:
            True if endpoint detected
        """
        chunk_duration_ms = len(audio) / sr * 1000
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        is_silence = rms < self.config.energy_threshold
        
        if is_silence:
            if self._silence_start_ms is None:
                self._silence_start_ms = self._audio_position_ms
            
            silence_duration = (
                self._audio_position_ms + chunk_duration_ms - self._silence_start_ms
            )
            
            # Check if we've had enough speech before endpoint
            segment_duration = self._audio_position_ms - self._last_speech_end_ms
            
            if (
                silence_duration >= self.config.silence_threshold_ms
                and segment_duration >= self.config.min_segment_ms
            ):
                self._audio_position_ms += chunk_duration_ms
                return True
        else:
            self._silence_start_ms = None
            self._last_speech_end_ms = self._audio_position_ms + chunk_duration_ms
        
        self._audio_position_ms += chunk_duration_ms
        return False
    
    def reset(self) -> None:
        """Reset state after endpoint."""
        self._silence_start_ms = None
        # Keep audio position, reset segment tracking
        self._last_speech_end_ms = self._audio_position_ms


# =============================================================================
# Fake Streaming Model for Testing
# =============================================================================


class FakeStreamingAdapter(StreamingAdapter):
    """
    Fake streaming adapter for testing.
    
    Emits a sequence of events: partial updates, then final.
    Demonstrates proper segment_id stability and seq monotonicity.
    """
    
    def __init__(
        self,
        words: list[str] | None = None,
        emit_per_chunk: int = 1,
        debug: bool = True,
    ):
        """
        Initialize fake adapter.
        
        Args:
            words: Words to "transcribe" (defaults to sample text)
            emit_per_chunk: How many words to add per push
            debug: Enable debug assertions
        """
        super().__init__("fake_streaming", debug=debug)
        self.words = words or ["hello", "world", "this", "is", "a", "test"]
        self.emit_per_chunk = emit_per_chunk
        
        self._word_index = 0
        self._current_text = ""
        self._current_segment_id: str = ""
    
    def _do_start_stream(self, config: dict[str, Any]) -> None:
        """Reset state for new stream."""
        self._word_index = 0
        self._current_text = ""
        self._current_segment_id = self.handle.get_or_create_segment_id("seg_0")
    
    def _do_push_audio(
        self,
        audio: bytes | np.ndarray,
        sr: int,
    ) -> Iterator[StreamEvent]:
        """Emit partial updates."""
        import time
        
        for _ in range(self.emit_per_chunk):
            if self._word_index >= len(self.words):
                return
            
            # Add next word
            word = self.words[self._word_index]
            if self._current_text:
                self._current_text += " " + word
            else:
                self._current_text = word
            self._word_index += 1
            
            # Emit partial
            yield StreamEvent(
                type=StreamEventType.PARTIAL,
                text=self._current_text,
                seq=self.handle.next_seq(),
                segment_id=self._current_segment_id,  # Stable across updates!
                t_emit_ms=time.time() * 1000,
            )
    
    def _do_flush(self) -> Iterator[StreamEvent]:
        """Emit remaining words as final."""
        import time
        
        # Add any remaining words
        while self._word_index < len(self.words):
            word = self.words[self._word_index]
            if self._current_text:
                self._current_text += " " + word
            else:
                self._current_text = word
            self._word_index += 1
        
        if self._current_text:
            yield StreamEvent(
                type=StreamEventType.FINAL,
                text=self._current_text,
                seq=self.handle.next_seq(),
                segment_id=self._current_segment_id,
                is_endpoint=True,
                t_emit_ms=time.time() * 1000,
            )
    
    def _do_finalize(self) -> dict[str, Any]:
        """Return final result."""
        return {
            "text": self._current_text,
            "segments": [
                {
                    "text": self._current_text,
                    "segment_id": self._current_segment_id,
                }
            ],
            "language": "en",
        }
