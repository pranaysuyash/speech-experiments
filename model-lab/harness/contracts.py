"""
Bundle Contract v1 - Enforceable interface for all model loaders.

Every loader MUST return a Bundle that conforms to this contract.
The runner ONLY calls bundle["asr"]["transcribe"]() etc - never raw model methods.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict


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
    translate: Callable[..., Dict[str, Any]]


class VADNamespace(TypedDict, total=False):
    """Voice Activity Detection capability namespace."""
    detect: Callable[..., Dict[str, Any]]  # returns {"segments": [...]}


class DiarizationNamespace(TypedDict, total=False):
    """Speaker Diarization capability namespace."""
    diarize: Callable[..., Dict[str, Any]]  # returns {"turns": [...]}


class V2VNamespace(TypedDict, total=False):
    """Voice-to-Voice capability namespace."""
    run_v2v_turn: Callable[..., Dict[str, Any]]  # returns {"audio": ..., "response_text": ...}


class AlignmentNamespace(TypedDict, total=False):
    """Alignment capability namespace (e.g. forced alignment)."""
    align: Callable[..., Dict[str, Any]]  # returns {"segments": [...]}


class Bundle(TypedDict, total=False):
    """
    Bundle Contract v1 - Every loader must return this shape.
    
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
    
    Optional:
        raw: dict - escape hatch for debugging (model, processor, etc)
        modes: List[str] - ["batch", "streaming", "cli"]
    """
    model_type: str
    device: str
    capabilities: List[str]
    modes: List[str]
    asr: ASRNamespace
    tts: TTSNamespace
    chat: ChatNamespace
    mt: MTNamespace
    vad: VADNamespace
    diarization: DiarizationNamespace
    v2v: V2VNamespace
    alignment: AlignmentNamespace
    raw: Dict[str, Any]


def validate_bundle(bundle: Dict[str, Any], model_type: str) -> None:
    """
    Validate that a bundle conforms to Bundle Contract v1.
    Raises ValueError if validation fails.
    """
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_type} loader must return dict Bundle, got {type(bundle)}")
    
    required = ["model_type", "device", "capabilities"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"{model_type} bundle missing required keys: {missing}")
    
    caps = set(bundle.get("capabilities", []))
    
    # Validate ASR capability - transcribe is REQUIRED
    if "asr" in caps:
        if "asr" not in bundle:
            raise ValueError(f"{model_type}: capability 'asr' requires bundle['asr'] namespace")
        asr_ns = bundle["asr"]
        if "transcribe" not in asr_ns:
            raise ValueError(f"{model_type}: asr namespace requires 'transcribe' callable")
    
    # Validate TTS capability
    if "tts" in caps:
        if "tts" not in bundle:
            raise ValueError(f"{model_type}: capability 'tts' requires bundle['tts'] namespace")
        if "synthesize" not in bundle["tts"]:
            raise ValueError(f"{model_type}: tts namespace requires 'synthesize' callable")
    
    # Validate Chat capability
    if "chat" in caps:
        if "chat" not in bundle:
            raise ValueError(f"{model_type}: capability 'chat' requires bundle['chat'] namespace")
        if "respond" not in bundle["chat"]:
            raise ValueError(f"{model_type}: chat namespace requires 'respond' callable")
    
    # Validate MT capability
    if "mt" in caps:
        if "mt" not in bundle:
            raise ValueError(f"{model_type}: capability 'mt' requires bundle['mt'] namespace")
        if "translate" not in bundle["mt"]:
            raise ValueError(f"{model_type}: mt namespace requires 'translate' callable")

    # Validate VAD capability
    if "vad" in caps:
        if "vad" not in bundle:
            raise ValueError(f"{model_type}: capability 'vad' requires bundle['vad'] namespace")
        if "detect" not in bundle["vad"]:
            raise ValueError(f"{model_type}: vad namespace requires 'detect' callable")

    # Validate Diarization capability
    if "diarization" in caps:
        if "diarization" not in bundle:
            raise ValueError(f"{model_type}: capability 'diarization' requires bundle['diarization'] namespace")
        if "diarize" not in bundle["diarization"]:
            raise ValueError(f"{model_type}: diarization namespace requires 'diarize' callable")

    # Validate V2V capability
    if "v2v" in caps:
        if "v2v" not in bundle:
            raise ValueError(f"{model_type}: capability 'v2v' requires bundle['v2v'] namespace")
        if "run_v2v_turn" not in bundle["v2v"]:
            raise ValueError(f"{model_type}: v2v namespace requires 'run_v2v_turn' callable")

    # Validate Alignment capability
    if "alignment" in caps:
        if "alignment" not in bundle:
            raise ValueError(f"{model_type}: capability 'alignment' requires bundle['alignment'] namespace")
        if "align" not in bundle["alignment"]:
            raise ValueError(f"{model_type}: alignment namespace requires 'align' callable")
