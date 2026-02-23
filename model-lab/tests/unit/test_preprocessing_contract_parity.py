from __future__ import annotations

from harness.pipeline_config import PREPROCESSING_REGISTRY, PipelineConfig
from harness.preprocess_ops import OPERATORS


# Registry entries intentionally implemented only in ingest/ffmpeg path.
INGEST_ONLY_OPS = {
    "compress_dynamics",
    "convert_samplerate",
    "gate_noise",
    "mono_mix",
    "normalize_peak",
}


def test_operator_registry_is_declared_in_preprocessing_registry():
    op_names = set(OPERATORS.keys())
    registry_names = set(PREPROCESSING_REGISTRY.keys())
    assert op_names.issubset(registry_names)


def test_ingest_only_set_matches_registry_minus_operator_chain():
    op_names = set(OPERATORS.keys())
    registry_names = set(PREPROCESSING_REGISTRY.keys())
    assert registry_names - op_names == INGEST_ONLY_OPS


def test_every_registry_operator_maps_to_ingest_config():
    for op_name in PREPROCESSING_REGISTRY:
        cfg = PipelineConfig(
            steps=["ingest"],
            preprocessing=[op_name],
        ).to_ingest_config()
        assert cfg is not None
