import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.streaming_asr.providers import (
    ASRConfig,
    ASRProvider,
    ASRProviderRegistry,
    ASRSegment,
)
from harness.streaming_asr.stream import stream_asr


class _DummyProvider(ASRProvider):
    @property
    def name(self) -> str:  # pragma: no cover
        return "dummy"

    @property
    def is_available(self) -> bool:
        return True

    async def transcribe_stream(self, pcm_stream, sample_rate=16000, source=None):
        async for _ in pcm_stream:
            break
        yield ASRSegment(
            text="hello",
            t0=0.0,
            t1=0.5,
            confidence=0.9,
            is_final=True,
            source=source,
            language="en",
        )


@pytest.fixture(autouse=True)
def _restore_registry():
    providers_before = dict(ASRProviderRegistry._providers)
    instances_before = dict(ASRProviderRegistry._instances)
    try:
        yield
    finally:
        ASRProviderRegistry._providers = providers_before
        ASRProviderRegistry._instances = instances_before


def test_registry_caches_by_config(monkeypatch):
    ASRProviderRegistry.register("dummy", _DummyProvider)
    monkeypatch.setenv("MODEL_LAB_ASR_PROVIDER", "dummy")

    cfg1 = ASRConfig(model_name="base", chunk_seconds=4)
    cfg2 = ASRConfig(model_name="base", chunk_seconds=2)

    p1a = ASRProviderRegistry.get_provider(config=cfg1)
    p1b = ASRProviderRegistry.get_provider(config=cfg1)
    p2 = ASRProviderRegistry.get_provider(config=cfg2)

    assert p1a is p1b
    assert p1a is not p2


def test_stream_asr_emits_events(monkeypatch):
    import asyncio

    ASRProviderRegistry.register("dummy", _DummyProvider)
    monkeypatch.setenv("MODEL_LAB_ASR_PROVIDER", "dummy")

    async def _run():
        async def _pcm_iter():
            yield b"\x00\x00" * 160

        events = []
        async for event in stream_asr(_pcm_iter(), sample_rate=16000, source="mic"):
            events.append(event)
        return events

    events = asyncio.run(_run())
    assert events
    assert events[0]["type"] == "asr_final"
    assert events[0]["text"] == "hello"
    assert events[0]["source"] == "mic"
