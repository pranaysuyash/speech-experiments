"""
Bundle Contract v2 - Enforceable interface for all model loaders.

Every loader MUST return a Bundle that conforms to this contract.
The runner ONLY calls bundle["asr"]["transcribe"]() etc - never raw model methods.

LCS-01: Added 6 new surfaces (classify, embed, enhance, separate, music_transcription, asr_stream)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypedDict

# Contract version - bump when signatures change
CONTRACT_VERSION = "2.0.0"


# =============================================================================
# Result Types
# =============================================================================


class ASRResult(TypedDict, total=False):
    """Result from ASR transcribe call."""

    text: str
    segments: list
    language: str
    meta: dict


class TTSResult(TypedDict, total=False):
    """Result from TTS synthesize call."""

    audio: Any  # np.ndarray
    sr: int
    duration_s: float
    meta: dict


class ChatResult(TypedDict, total=False):
    """Result from chat respond call."""

    text: str
    audio: Any  # optional audio response
    meta: dict


class ClassifyResult(TypedDict, total=False):
    """Result from audio classification."""

    labels: list[dict[str, Any]]  # [{label: str, score: float}, ...]
    embeddings: Any  # optional tensor
    meta: dict


class EmbedResult(TypedDict, total=False):
    """Result from audio embedding."""

    embedding: Any  # tensor or ndarray
    dim: int
    meta: dict


class EnhanceResult(TypedDict, total=False):
    """Result from audio enhancement."""

    audio: Any  # np.ndarray
    sr: int
    latency_ms: float
    meta: dict


class SeparateResult(TypedDict, total=False):
    """Result from source separation."""

    stems: dict[str, Any]  # {stem_name: audio_array, ...}
    sr: int
    meta: dict


class MusicTranscriptionResult(TypedDict, total=False):
    """Result from music transcription."""

    notes: list[dict[str, Any]]  # [{pitch, start, end, velocity?}, ...]
    tempo: float
    midi: bytes
    meta: dict


class ASRStreamEvent(TypedDict, total=False):
    """Event from streaming ASR."""

    type: str  # "partial" | "final" | "error" | "info"
    text: str
    segments: list
    seq: int  # monotonic ordering
    segment_id: str  # stable across partial updates
    t_audio_ms_start: int
    t_audio_ms_end: int
    t_emit_ms: int
    stability: float
    is_endpoint: bool
    meta: dict


# =============================================================================
# Capability Namespaces
# =============================================================================


class ASRNamespace(TypedDict, total=False):
    """ASR capability namespace."""

    transcribe: Callable[..., ASRResult]
    transcribe_path: Callable[..., ASRResult]  # for CLI adapters


class TTSNamespace(TypedDict, total=False):
    """TTS capability namespace."""

    synthesize: Callable[..., TTSResult]


class ChatNamespace(TypedDict, total=False):
    """Chat capability namespace."""

    respond: Callable[..., ChatResult]


class MTNamespace(TypedDict, total=False):
    """Machine Translation capability namespace."""

    translate: Callable[..., dict[str, Any]]


class VADNamespace(TypedDict, total=False):
    """Voice Activity Detection capability namespace."""

    detect: Callable[..., dict[str, Any]]  # returns {"segments": [...]}


class DiarizationNamespace(TypedDict, total=False):
    """Speaker Diarization capability namespace."""

    diarize: Callable[..., dict[str, Any]]  # returns {"turns": [...]}


class V2VNamespace(TypedDict, total=False):
    """Voice-to-Voice capability namespace."""

    run_v2v_turn: Callable[..., dict[str, Any]]  # returns {"audio": ..., "response_text": ...}


class AlignmentNamespace(TypedDict, total=False):
    """Alignment capability namespace (e.g. forced alignment)."""

    align: Callable[..., dict[str, Any]]  # returns {"segments": [...]}


# -----------------------------------------------------------------------------
# NEW SURFACES (LCS-01)
# -----------------------------------------------------------------------------


class ClassifyNamespace(TypedDict, total=False):
    """Audio classification capability namespace."""

    classify: Callable[..., ClassifyResult]


class EmbedNamespace(TypedDict, total=False):
    """Audio embedding capability namespace."""

    embed: Callable[..., EmbedResult]


class EnhanceNamespace(TypedDict, total=False):
    """Audio enhancement capability namespace."""

    enhance: Callable[..., EnhanceResult]


class SeparateNamespace(TypedDict, total=False):
    """Source separation capability namespace."""

    separate: Callable[..., SeparateResult]


class MusicTranscriptionNamespace(TypedDict, total=False):
    """Music transcription capability namespace."""

    transcribe_notes: Callable[..., MusicTranscriptionResult]


class ASRStreamNamespace(TypedDict, total=False):
    """Streaming ASR capability namespace with full lifecycle."""

    start_stream: Callable[..., Any]  # (config) -> stream_handle
    push_audio: Callable[..., Iterator[ASRStreamEvent]]  # (handle, pcm, sr) -> events
    flush: Callable[..., Iterator[ASRStreamEvent]]  # (handle) -> remaining events
    finalize: Callable[..., ASRResult]  # (handle) -> final result
    close: Callable[..., None]  # (handle) -> cleanup


# =============================================================================
# Bundle Contract
# =============================================================================


class Bundle(TypedDict, total=False):
    """
    Bundle Contract v2 - Every loader must return this shape.

    Required keys:
        model_type: str - identifier matching registry key
        device: str - actual device used ("cpu", "mps", "cuda")
        capabilities: List[str] - list of supported capabilities

    Capability namespaces (required if capability listed):
        asr: {"transcribe": callable} - for speech-to-text
        tts: {"synthesize": callable} - for text-to-speech
        chat: {"respond": callable} - for conversational AI
        mt: {"translate": callable} - for translation
        vad: {"detect": callable} - for voice activity detection
        diarization: {"diarize": callable} - for speaker diarization
        v2v: {"run_v2v_turn": callable} - for voice-to-voice
        alignment: {"align": callable} - for timestamp alignment
        classify: {"classify": callable} - for audio classification
        embed: {"embed": callable} - for audio embeddings
        enhance: {"enhance": callable} - for audio enhancement
        separate: {"separate": callable} - for source separation
        music_transcription: {"transcribe_notes": callable} - for music transcription
        asr_stream: {lifecycle methods} - for streaming ASR

    Optional:
        raw: dict - escape hatch for debugging (model, processor, etc)
        modes: List[str] - ["batch", "streaming", "cli"]
    """

    model_type: str
    device: str
    capabilities: list[str]
    modes: list[str]
    # Existing surfaces
    asr: ASRNamespace
    tts: TTSNamespace
    chat: ChatNamespace
    mt: MTNamespace
    vad: VADNamespace
    diarization: DiarizationNamespace
    v2v: V2VNamespace
    alignment: AlignmentNamespace
    # New surfaces (LCS-01)
    classify: ClassifyNamespace
    embed: EmbedNamespace
    enhance: EnhanceNamespace
    separate: SeparateNamespace
    music_transcription: MusicTranscriptionNamespace
    asr_stream: ASRStreamNamespace
    # Escape hatch
    raw: dict[str, Any]


# =============================================================================
# Validation
# =============================================================================

# Mapping of capability -> (namespace_key, required_method)
_CAPABILITY_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "asr": ("asr", "transcribe"),
    "tts": ("tts", "synthesize"),
    "chat": ("chat", "respond"),
    "mt": ("mt", "translate"),
    "vad": ("vad", "detect"),
    "diarization": ("diarization", "diarize"),
    "v2v": ("v2v", "run_v2v_turn"),
    "alignment": ("alignment", "align"),
    "classify": ("classify", "classify"),
    "embed": ("embed", "embed"),
    "enhance": ("enhance", "enhance"),
    "separate": ("separate", "separate"),
    "music_transcription": ("music_transcription", "transcribe_notes"),
}

# Streaming has multiple required methods
_ASR_STREAM_REQUIRED = ["start_stream", "push_audio", "finalize", "close"]


def validate_bundle(bundle: dict[str, Any], model_type: str) -> None:
    """
    Validate that a bundle conforms to Bundle Contract v2.
    Raises ValueError if validation fails.
    """
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_type} loader must return dict Bundle, got {type(bundle)}")

    required = ["model_type", "device", "capabilities"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"{model_type} bundle missing required keys: {missing}")

    caps = set(bundle.get("capabilities", []))

    # Validate standard capabilities
    for cap, (ns_key, method) in _CAPABILITY_REQUIREMENTS.items():
        if cap in caps:
            if ns_key not in bundle:
                raise ValueError(f"{model_type}: capability '{cap}' requires bundle['{ns_key}'] namespace")
            if method not in bundle[ns_key]:
                raise ValueError(f"{model_type}: {ns_key} namespace requires '{method}' callable")

    # Validate streaming ASR (multiple required methods)
    if "asr_stream" in caps:
        if "asr_stream" not in bundle:
            raise ValueError(f"{model_type}: capability 'asr_stream' requires bundle['asr_stream'] namespace")
        stream_ns = bundle["asr_stream"]
        for method in _ASR_STREAM_REQUIRED:
            if method not in stream_ns:
                raise ValueError(f"{model_type}: asr_stream namespace requires '{method}' callable")

