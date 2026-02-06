"""
ASR provider abstraction for streaming transcription.

This is intentionally minimal: it defines the interface and a registry for
swappable providers (local, API, etc.) so the streaming pipeline can stay stable.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import cast

logger = logging.getLogger(__name__)


class AudioSource(Enum):
    """Audio source identifier for multi-source capture."""

    SYSTEM = "system"
    MICROPHONE = "mic"


@dataclass(frozen=True)
class ASRSegment:
    """A transcribed segment emitted by an ASR provider."""

    text: str
    t0: float
    t1: float
    confidence: float
    is_final: bool
    source: AudioSource | None = None
    language: str | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class ASRConfig:
    """Runtime configuration for an ASR provider."""

    model_name: str = "base"
    device: str = "auto"
    compute_type: str = "int8"
    language: str | None = None  # None = auto-detect
    chunk_seconds: int = 4
    vad_enabled: bool = False


class ASRProvider(ABC):
    """Abstract base class for streaming ASR providers."""

    def __init__(self, config: ASRConfig):
        self.config = config
        self._debug = os.getenv("MODEL_LAB_DEBUG", "0") == "1"

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def transcribe_stream(
        self,
        pcm_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        source: AudioSource | None = None,
    ) -> AsyncIterator[ASRSegment]:
        if False:
            yield cast(ASRSegment, None)
        raise NotImplementedError

    def log(self, msg: str) -> None:
        logger.debug("[%s] %s", self.name, msg)


class ASRProviderRegistry:
    """Registry for managing streaming ASR providers."""

    _providers: dict[str, type[ASRProvider]] = {}
    _instances: dict[str, ASRProvider] = {}
    _lock: threading.Lock | None = None

    @classmethod
    def _get_lock(cls) -> threading.Lock:
        if cls._lock is None:
            import threading

            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def register(cls, name: str, provider_class: type[ASRProvider]) -> None:
        cls._providers[name] = provider_class

    @classmethod
    def _cfg_key(cls, name: str, cfg: ASRConfig) -> str:
        return (
            f"{name}|{cfg.model_name}|{cfg.device}|{cfg.compute_type}|"
            f"{cfg.language}|{int(cfg.vad_enabled)}|{cfg.chunk_seconds}"
        )

    @classmethod
    def get_provider(
        cls, name: str | None = None, config: ASRConfig | None = None
    ) -> ASRProvider | None:
        provider_name = name or os.getenv("MODEL_LAB_ASR_PROVIDER", "faster_whisper")
        if provider_name not in cls._providers:
            return None

        cfg = config or ASRConfig()
        key = cls._cfg_key(provider_name, cfg)

        with cls._get_lock():
            if key not in cls._instances:
                cls._instances[key] = cls._providers[provider_name](cfg)
            return cls._instances[key]
