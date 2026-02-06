"""
High-level streaming ASR pipeline.

Consumes an async iterator of PCM16 audio bytes and yields dict events suitable
for UI/WS layers. Providers remain swappable via ASRProviderRegistry.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

# Import providers to trigger registration.
from . import provider_faster_whisper  # noqa: F401
from .providers import ASRConfig, ASRProviderRegistry, AudioSource

logger = logging.getLogger(__name__)


def _get_default_config() -> ASRConfig:
    return ASRConfig(
        model_name=os.getenv("MODEL_LAB_WHISPER_MODEL", "base"),
        device=os.getenv("MODEL_LAB_WHISPER_DEVICE", "auto"),
        compute_type=os.getenv("MODEL_LAB_WHISPER_COMPUTE", "int8"),
        language=os.getenv("MODEL_LAB_WHISPER_LANGUAGE") or None,
        chunk_seconds=int(os.getenv("MODEL_LAB_ASR_CHUNK_SECONDS", "4")),
        vad_enabled=os.getenv("MODEL_LAB_ASR_VAD", "0") == "1",
    )


async def stream_asr(
    pcm_stream: AsyncIterator[bytes],
    sample_rate: int = 16000,
    source: str | None = None,
) -> AsyncIterator[dict]:
    config = _get_default_config()
    provider = ASRProviderRegistry.get_provider(config=config)

    if provider is None or not provider.is_available:
        logger.warning("ASR provider unavailable")
        yield {"type": "status", "state": "no_asr_provider", "message": "ASR provider unavailable"}
        async for _ in pcm_stream:
            pass
        return

    audio_source: AudioSource | None = None
    if source == "system":
        audio_source = AudioSource.SYSTEM
    elif source == "mic":
        audio_source = AudioSource.MICROPHONE

    logger.debug("Using provider '%s', source=%s", provider.name, source)

    async for segment in provider.transcribe_stream(pcm_stream, sample_rate, audio_source):
        event = {
            "type": "asr_final" if segment.is_final else "asr_partial",
            "t0": segment.t0,
            "t1": segment.t1,
            "text": segment.text,
            "stable": segment.is_final,
            "confidence": segment.confidence,
        }
        if segment.source:
            event["source"] = segment.source.value
        if segment.language:
            event["language"] = segment.language
        if segment.speaker:
            event["speaker"] = segment.speaker
        yield event
