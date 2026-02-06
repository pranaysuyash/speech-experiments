"""
Local faster-whisper provider for streaming ASR.

Notes:
- faster-whisper/CTranslate2 `model.transcribe()` is not thread-safe; calls are serialized.
- CTranslate2 does not support MPS. On macOS, "auto"/"mps" fall back to CPU.
"""

from __future__ import annotations

import asyncio
import os
import platform
import threading
from collections.abc import AsyncIterator

from .providers import ASRConfig, ASRProvider, ASRProviderRegistry, ASRSegment, AudioSource

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None


class FasterWhisperProvider(ASRProvider):
    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._model: WhisperModel | None = None
        self._infer_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "faster_whisper"

    @property
    def is_available(self) -> bool:
        return WhisperModel is not None and np is not None

    def _get_model(self) -> WhisperModel | None:
        if not self.is_available:
            return None

        if self._model is None:
            model_name = os.getenv("MODEL_LAB_WHISPER_MODEL", self.config.model_name)
            device = os.getenv("MODEL_LAB_WHISPER_DEVICE", self.config.device)

            if device == "auto" and platform.system() == "Darwin":
                device = "cpu"
            elif device == "mps":
                device = "cpu"

            compute_type = os.getenv("MODEL_LAB_WHISPER_COMPUTE", self.config.compute_type)
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"
                self.log("Forced compute_type='int8' for CPU execution (float16 unsupported)")

            self.log(f"Loading model={model_name} device={device} compute={compute_type}")
            self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

        return self._model

    async def transcribe_stream(
        self,
        pcm_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        source: AudioSource | None = None,
    ) -> AsyncIterator[ASRSegment]:
        bytes_per_sample = 2
        chunk_seconds = self.config.chunk_seconds
        chunk_bytes = int(sample_rate * chunk_seconds * bytes_per_sample)
        buffer = bytearray()
        processed_samples = 0

        self.log(f"Started streaming, chunk_bytes={chunk_bytes} ({chunk_seconds}s)")

        model = self._get_model()
        if model is None or np is None:
            yield ASRSegment(
                text="[ASR unavailable]",
                t0=0,
                t1=0,
                confidence=0,
                is_final=True,
                source=source,
            )
            async for _ in pcm_stream:
                pass
            return

        async for chunk in pcm_stream:
            buffer.extend(chunk)

            while len(buffer) >= chunk_bytes:
                audio_bytes = bytes(buffer[:chunk_bytes])
                del buffer[:chunk_bytes]

                t0 = processed_samples / sample_rate
                chunk_samples = len(audio_bytes) // bytes_per_sample
                t1 = (processed_samples + chunk_samples) / sample_rate
                processed_samples += chunk_samples

                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                def _transcribe():
                    with self._infer_lock:
                        segments, info = model.transcribe(
                            audio,
                            vad_filter=self.config.vad_enabled,
                            language=self.config.language,
                        )
                    return list(segments), info

                segments, info = await asyncio.to_thread(_transcribe)
                detected_lang = getattr(info, "language", None)

                for segment in segments:
                    text = segment.text.strip()
                    if not text:
                        continue

                    avg_logprob = getattr(segment, "avg_logprob", -0.5)
                    confidence = max(0.0, min(1.0, 1.0 + avg_logprob / 2.0))

                    yield ASRSegment(
                        text=text,
                        t0=t0 + segment.start,
                        t1=t0 + segment.end,
                        confidence=confidence,
                        is_final=True,
                        source=source,
                        language=detected_lang,
                    )

        if buffer:
            audio_bytes = bytes(buffer)
            buffer.clear()

            t0 = processed_samples / sample_rate
            chunk_samples = len(audio_bytes) // bytes_per_sample
            processed_samples += chunk_samples

            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            def _transcribe():
                with self._infer_lock:
                    segments, info = model.transcribe(
                        audio,
                        vad_filter=self.config.vad_enabled,
                        language=self.config.language,
                    )
                return list(segments), info

            segments, info = await asyncio.to_thread(_transcribe)
            detected_lang = getattr(info, "language", None)

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue

                avg_logprob = getattr(segment, "avg_logprob", -0.5)
                confidence = max(0.0, min(1.0, 1.0 + avg_logprob / 2.0))

                yield ASRSegment(
                    text=text,
                    t0=t0 + segment.start,
                    t1=t0 + segment.end,
                    confidence=confidence,
                    is_final=True,
                    source=source,
                    language=detected_lang,
                )


ASRProviderRegistry.register("faster_whisper", FasterWhisperProvider)
