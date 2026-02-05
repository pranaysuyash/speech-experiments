"""
Streaming ASR utilities.

Ported from EchoPanel's streaming ASR pipeline so Model-Lab can evaluate
real-time chunked transcription behavior (latency/quality tradeoffs, VAD, etc.)
without coupling to any specific product UI.
"""

from .providers import ASRConfig, ASRProvider, ASRProviderRegistry, ASRSegment, AudioSource
from .stream import stream_asr

__all__ = [
    "ASRConfig",
    "ASRProvider",
    "ASRProviderRegistry",
    "ASRSegment",
    "AudioSource",
    "stream_asr",
]

