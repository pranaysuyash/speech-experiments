"""
Shared harness for model testing.
Provides consistent interfaces for audio I/O, metrics, and model loading.
"""

from .audio_io import AudioLoader, GroundTruthLoader
from .metrics_asr import ASRMetrics, ASRResult, ASRBatcher
from .metrics_tts import TTSMetrics, TTSResult, VoiceQualityMetrics
from .metrics_entity import EntityMetrics, EntityErrorResult
from .timers import PerformanceTimer, LatencyProfiler, MemoryProfiler, TimingResult
from .registry import ModelRegistry, load_model_from_config
from .normalize import TextNormalizer, ComparisonNormalizer
from .protocol import RunContract, NormalizationValidator
from .results import (
    RunResult, BatchResult, RunMetadata, PerformanceMetrics,
    ResultsManager, TaskType, ResultStatus, create_run_id
)
from .runner import GoldenTestRunner, GoldenTestSet, TestCase, RegressionResult
from .gate import ProductionGate, PromotionCriteria, GateResult, ModelStatus

__all__ = [
    # Audio I/O
    'AudioLoader',
    'GroundTruthLoader',

    # ASR Metrics
    'ASRMetrics',
    'ASRResult',
    'ASRBatcher',

    # TTS Metrics
    'TTSMetrics',
    'TTSResult',
    'VoiceQualityMetrics',

    # Entity Metrics
    'EntityMetrics',
    'EntityErrorResult',

    # Timing
    'PerformanceTimer',
    'LatencyProfiler',
    'MemoryProfiler',
    'TimingResult',

    # Model Registry
    'ModelRegistry',
    'load_model_from_config',

    # Text Normalization
    'TextNormalizer',
    'ComparisonNormalizer',

    # Protocol
    'RunContract',
    'NormalizationValidator',

    # Results
    'RunResult',
    'BatchResult',
    'RunMetadata',
    'PerformanceMetrics',
    'ResultsManager',
    'TaskType',
    'ResultStatus',
    'create_run_id',

    # Runner
    'GoldenTestRunner',
    'GoldenTestSet',
    'TestCase',
    'RegressionResult',

    # Gate
    'ProductionGate',
    'PromotionCriteria',
    'GateResult',
    'ModelStatus',
]