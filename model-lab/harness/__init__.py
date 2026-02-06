"""
Shared harness for model testing.
Provides consistent interfaces for audio I/O, metrics, and model loading.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    # Audio I/O
    "AudioLoader",
    "GroundTruthLoader",
    # ASR Metrics
    "ASRMetrics",
    "ASRResult",
    "ASRBatcher",
    # TTS Metrics
    "TTSMetrics",
    "TTSResult",
    "VoiceQualityMetrics",
    # Entity Metrics
    "EntityMetrics",
    "EntityErrorResult",
    # Timing
    "PerformanceTimer",
    "LatencyProfiler",
    "MemoryProfiler",
    "TimingResult",
    # Model Registry
    "ModelRegistry",
    "load_model_from_config",
    # Text Normalization
    "TextNormalizer",
    "ComparisonNormalizer",
    # Protocol
    "RunContract",
    "NormalizationValidator",
    # Results
    "RunResult",
    "BatchResult",
    "RunMetadata",
    "PerformanceMetrics",
    "ResultsManager",
    "TaskType",
    "ResultStatus",
    "create_run_id",
    # Runner
    "GoldenTestRunner",
    "GoldenTestSet",
    "TestCase",
    "RegressionResult",
    # Gate
    "ProductionGate",
    "PromotionCriteria",
    "GateResult",
    "ModelStatus",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    # Audio I/O
    "AudioLoader": (".audio_io", "AudioLoader"),
    "GroundTruthLoader": (".audio_io", "GroundTruthLoader"),
    # ASR Metrics
    "ASRMetrics": (".metrics_asr", "ASRMetrics"),
    "ASRResult": (".metrics_asr", "ASRResult"),
    "ASRBatcher": (".metrics_asr", "ASRBatcher"),
    # TTS Metrics
    "TTSMetrics": (".metrics_tts", "TTSMetrics"),
    "TTSResult": (".metrics_tts", "TTSResult"),
    "VoiceQualityMetrics": (".metrics_tts", "VoiceQualityMetrics"),
    # Entity Metrics
    "EntityMetrics": (".metrics_entity", "EntityMetrics"),
    "EntityErrorResult": (".metrics_entity", "EntityErrorResult"),
    # Timing
    "PerformanceTimer": (".timers", "PerformanceTimer"),
    "LatencyProfiler": (".timers", "LatencyProfiler"),
    "MemoryProfiler": (".timers", "MemoryProfiler"),
    "TimingResult": (".timers", "TimingResult"),
    # Model Registry
    "ModelRegistry": (".registry", "ModelRegistry"),
    "load_model_from_config": (".registry", "load_model_from_config"),
    # Text Normalization
    "TextNormalizer": (".normalize", "TextNormalizer"),
    "ComparisonNormalizer": (".normalize", "ComparisonNormalizer"),
    # Protocol
    "RunContract": (".protocol", "RunContract"),
    "NormalizationValidator": (".protocol", "NormalizationValidator"),
    # Results
    "RunResult": (".results", "RunResult"),
    "BatchResult": (".results", "BatchResult"),
    "RunMetadata": (".results", "RunMetadata"),
    "PerformanceMetrics": (".results", "PerformanceMetrics"),
    "ResultsManager": (".results", "ResultsManager"),
    "TaskType": (".results", "TaskType"),
    "ResultStatus": (".results", "ResultStatus"),
    "create_run_id": (".results", "create_run_id"),
    # Runner
    "GoldenTestRunner": (".runner", "GoldenTestRunner"),
    "GoldenTestSet": (".runner", "GoldenTestSet"),
    "TestCase": (".runner", "TestCase"),
    "RegressionResult": (".runner", "RegressionResult"),
    # Gate
    "ProductionGate": (".gate", "ProductionGate"),
    "PromotionCriteria": (".gate", "PromotionCriteria"),
    "GateResult": (".gate", "GateResult"),
    "ModelStatus": (".gate", "ModelStatus"),
}


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = spec
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS.keys()))
