"""Tests for configurable pipeline steps.

These tests focus on the harness-level pipeline configuration logic
which doesn't require FastAPI.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone


class TestPipelineConfig:
    """Test PipelineConfig class."""

    def test_pipeline_config_creation(self):
        """Test creating a PipelineConfig."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            name="test_config",
            steps=["ingest", "asr", "diarization"],
            preprocessing=["trim_silence"],
        )

        assert cfg.name == "test_config"
        assert cfg.steps == ["ingest", "asr", "diarization"]
        assert cfg.preprocessing == ["trim_silence"]

    def test_pipeline_config_to_dict(self):
        """Test serializing PipelineConfig to dict."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            name="test_config",
            steps=["ingest", "asr", "diarization"],
            preprocessing=["trim_silence", "normalize_loudness"],
        )

        d = cfg.to_dict()
        assert d["name"] == "test_config"
        assert d["steps"] == ["ingest", "asr", "diarization"]
        assert d["preprocessing"] == ["trim_silence", "normalize_loudness"]

    def test_pipeline_config_from_dict(self):
        """Test deserializing PipelineConfig from dict."""
        from harness.pipeline_config import PipelineConfig

        d = {
            "name": "test_config",
            "steps": ["ingest", "asr"],
            "preprocessing": ["trim_silence"],
        }

        cfg = PipelineConfig.from_dict(d)
        assert cfg.name == "test_config"
        assert cfg.steps == ["ingest", "asr"]
        assert cfg.preprocessing == ["trim_silence"]

    def test_pipeline_config_roundtrip(self):
        """Test that pipeline config can be serialized and deserialized."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            name="test_config",
            steps=["ingest", "asr", "diarization"],
            preprocessing=["trim_silence"],
        )

        d = cfg.to_dict()
        cfg2 = PipelineConfig.from_dict(d)

        assert cfg2.steps == cfg.steps
        assert cfg2.preprocessing == cfg.preprocessing
        assert cfg2.name == cfg.name


class TestDependencyResolution:
    """Test step dependency resolution."""

    def test_resolve_dependencies_adds_ingest(self):
        """Test that resolving dependencies always includes ingest."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(steps=["asr"])
        resolved = cfg.resolve_dependencies()

        assert "ingest" in resolved
        assert "asr" in resolved

    def test_resolve_dependencies_adds_asr_for_alignment(self):
        """Test that alignment requires asr and diarization."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(steps=["alignment"])
        resolved = cfg.resolve_dependencies()

        # alignment depends on asr and diarization
        assert "ingest" in resolved
        assert "asr" in resolved
        assert "diarization" in resolved
        assert "alignment" in resolved

    def test_resolve_dependencies_preserves_order(self):
        """Test that resolved steps are in topological order."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(steps=["summarize_by_speaker"])
        resolved = cfg.resolve_dependencies()

        # summarize_by_speaker depends on alignment, which depends on asr and diarization
        ingest_idx = resolved.index("ingest")
        asr_idx = resolved.index("asr")
        diarization_idx = resolved.index("diarization")
        alignment_idx = resolved.index("alignment")
        summarize_idx = resolved.index("summarize_by_speaker")

        # Check ordering
        assert ingest_idx < asr_idx
        assert ingest_idx < diarization_idx
        assert asr_idx < alignment_idx
        assert diarization_idx < alignment_idx
        assert alignment_idx < summarize_idx


class TestStepRegistry:
    """Test the STEP_REGISTRY."""

    def test_step_registry_has_core_steps(self):
        """Test that STEP_REGISTRY contains all core steps."""
        from harness.pipeline_config import STEP_REGISTRY

        core_steps = ["ingest", "asr", "diarization", "alignment", "chapters", "summarize_by_speaker", "action_items_assignee"]

        for step in core_steps:
            assert step in STEP_REGISTRY, f"Missing core step: {step}"

    def test_step_registry_has_dependencies(self):
        """Test that steps have proper dependency information."""
        from harness.pipeline_config import STEP_REGISTRY

        # ingest has no dependencies
        assert STEP_REGISTRY["ingest"]["deps"] == []

        # asr depends on ingest
        assert "ingest" in STEP_REGISTRY["asr"]["deps"]

        # alignment depends on asr and diarization
        assert "asr" in STEP_REGISTRY["alignment"]["deps"]
        assert "diarization" in STEP_REGISTRY["alignment"]["deps"]


class TestPreprocessingRegistry:
    """Test the PREPROCESSING_REGISTRY."""

    def test_preprocessing_registry_has_core_ops(self):
        """Test that PREPROCESSING_REGISTRY contains core operations."""
        from harness.pipeline_config import PREPROCESSING_REGISTRY

        core_ops = ["trim_silence", "normalize_loudness"]

        for op in core_ops:
            assert op in PREPROCESSING_REGISTRY, f"Missing preprocessing op: {op}"

    def test_preprocessing_ops_have_descriptions(self):
        """Test that preprocessing ops have descriptions."""
        from harness.pipeline_config import PREPROCESSING_REGISTRY

        for name, meta in PREPROCESSING_REGISTRY.items():
            assert "description" in meta, f"Missing description for {name}"
            assert len(meta["description"]) > 0


class TestPipelineTemplates:
    """Test the PIPELINE_TEMPLATES registry."""

    def test_pipeline_templates_exist(self):
        """Test that PIPELINE_TEMPLATES is defined."""
        from harness.pipeline_config import PIPELINE_TEMPLATES

        assert isinstance(PIPELINE_TEMPLATES, dict)

    def test_templates_have_required_fields(self):
        """Test that templates have required fields."""
        from harness.pipeline_config import PIPELINE_TEMPLATES

        for name, template in PIPELINE_TEMPLATES.items():
            assert hasattr(template, "steps"), f"Template {name} missing steps"
            assert hasattr(template, "name"), f"Template {name} missing name"


class TestIngestConfigConversion:
    """Test conversion from PipelineConfig to IngestConfig."""

    def test_to_ingest_config_empty_preprocessing(self):
        """Test converting config with no preprocessing."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(steps=["ingest", "asr"])
        ingest_cfg = cfg.to_ingest_config()

        # Should return None or default config
        assert ingest_cfg is None or ingest_cfg.normalize is False

    def test_to_ingest_config_with_trim_silence(self):
        """Test converting config with trim_silence."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["trim_silence"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.trim_silence is True

    def test_to_ingest_config_with_normalize_loudness(self):
        """Test converting config with normalize_loudness."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["normalize_loudness"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.normalize is True

    def test_to_ingest_config_with_denoise(self):
        """Test converting config with denoise."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["denoise"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.denoise is True

    def test_to_ingest_config_with_speed(self):
        """Test converting config with speed adjustment."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["speed(factor=1.5)"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.speed == 1.5

    def test_to_ingest_config_with_compress_dynamics(self):
        """Test converting config with dynamic range compression."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["compress_dynamics"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.compress_dynamics is True

    def test_to_ingest_config_with_gate_noise(self):
        """Test converting config with noise gate."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["gate_noise"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.gate_noise is True

    def test_to_ingest_config_with_mono_mix(self):
        """Test converting config with mono mix."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["mono_mix"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.mono_mix is True

    def test_to_ingest_config_with_multiple_ops(self):
        """Test converting config with multiple preprocessing ops."""
        from harness.pipeline_config import PipelineConfig

        cfg = PipelineConfig(
            steps=["ingest", "asr"],
            preprocessing=["trim_silence", "normalize_loudness", "denoise"]
        )
        ingest_cfg = cfg.to_ingest_config()

        assert ingest_cfg is not None
        assert ingest_cfg.trim_silence is True
        assert ingest_cfg.normalize is True
        assert ingest_cfg.denoise is True
