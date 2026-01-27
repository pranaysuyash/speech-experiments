"""
Tests for dynamic pipeline configuration.
"""
import pytest
from pathlib import Path

from harness.pipeline_config import (
    PipelineConfig,
    STEP_REGISTRY,
    PREPROCESSING_REGISTRY,
    PIPELINE_TEMPLATES,
    list_available_steps,
    list_preprocessing_ops,
    list_pipeline_templates,
    validate_pipeline_config,
    get_pipeline_template,
)


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_default_config(self):
        """Default config has ingest and asr steps."""
        config = PipelineConfig()
        assert config.steps == ["ingest", "asr"]
        assert config.preprocessing == []
    
    def test_custom_steps(self):
        """Can create config with custom steps."""
        config = PipelineConfig(steps=["ingest", "diarization"])
        assert config.steps == ["ingest", "diarization"]
    
    def test_invalid_step_raises(self):
        """Unknown step raises ValueError."""
        with pytest.raises(ValueError, match="Unknown step"):
            PipelineConfig(steps=["ingest", "invalid_step"])
    
    def test_invalid_preprocessing_raises(self):
        """Unknown preprocessing operator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preprocessing op"):
            PipelineConfig(preprocessing=["invalid_op"])
    
    def test_parameterized_preprocessing(self):
        """Preprocessing ops with params are validated correctly."""
        config = PipelineConfig(preprocessing=["trim_silence(min_silence_ms=300)"])
        assert "trim_silence(min_silence_ms=300)" in config.preprocessing


class TestDependencyResolution:
    """Tests for step dependency resolution."""
    
    def test_ingest_always_first(self):
        """Ingest is always added as first step."""
        config = PipelineConfig(steps=["asr"])
        resolved = config.resolve_dependencies()
        assert resolved[0] == "ingest"
    
    def test_asr_adds_ingest_dep(self):
        """ASR step adds ingest dependency."""
        config = PipelineConfig(steps=["asr"])
        resolved = config.resolve_dependencies()
        assert resolved == ["ingest", "asr"]
    
    def test_diarization_only(self):
        """Diarization-only run adds ingest."""
        config = PipelineConfig(steps=["diarization"])
        resolved = config.resolve_dependencies()
        assert resolved == ["ingest", "diarization"]
    
    def test_alignment_adds_all_deps(self):
        """Alignment requires asr and diarization."""
        config = PipelineConfig(steps=["alignment"])
        resolved = config.resolve_dependencies()
        assert "ingest" in resolved
        assert "asr" in resolved
        assert "diarization" in resolved
        assert "alignment" in resolved
        # Check order
        assert resolved.index("ingest") < resolved.index("asr")
        assert resolved.index("asr") < resolved.index("alignment")
        assert resolved.index("diarization") < resolved.index("alignment")
    
    def test_chapters_full_chain(self):
        """Chapters requires full ASR+diarization+alignment chain."""
        config = PipelineConfig(steps=["chapters"])
        resolved = config.resolve_dependencies()
        expected_order = ["ingest", "asr", "diarization", "alignment", "chapters"]
        assert resolved == expected_order
    
    def test_no_duplicate_deps(self):
        """Dependencies are not duplicated."""
        config = PipelineConfig(steps=["asr", "diarization", "alignment"])
        resolved = config.resolve_dependencies()
        # ingest should appear only once
        assert resolved.count("ingest") == 1


class TestPipelineTemplates:
    """Tests for built-in pipeline templates."""
    
    def test_templates_exist(self):
        """All expected templates exist."""
        assert "ingest_only" in PIPELINE_TEMPLATES
        assert "fast_asr" in PIPELINE_TEMPLATES
        assert "full_meeting" in PIPELINE_TEMPLATES
    
    def test_get_template(self):
        """Can retrieve template by name."""
        template = get_pipeline_template("fast_asr")
        assert template is not None
        assert template.name == "fast_asr"
    
    def test_get_unknown_template(self):
        """Unknown template returns None."""
        assert get_pipeline_template("nonexistent") is None
    
    def test_list_templates(self):
        """list_pipeline_templates returns correct structure."""
        templates = list_pipeline_templates()
        assert len(templates) >= 3
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert "steps" in t


class TestRegistries:
    """Tests for step and preprocessing registries."""
    
    def test_step_registry_has_core_steps(self):
        """Step registry contains core steps."""
        assert "ingest" in STEP_REGISTRY
        assert "asr" in STEP_REGISTRY
        assert "diarization" in STEP_REGISTRY
        assert "alignment" in STEP_REGISTRY
    
    def test_step_has_required_fields(self):
        """Each step has required metadata."""
        for name, info in STEP_REGISTRY.items():
            assert "deps" in info
            assert "description" in info
    
    def test_preprocessing_registry_has_ops(self):
        """Preprocessing registry has operators."""
        assert "trim_silence" in PREPROCESSING_REGISTRY
        assert "normalize_loudness" in PREPROCESSING_REGISTRY
    
    def test_list_available_steps(self):
        """list_available_steps returns correct structure."""
        steps = list_available_steps()
        assert len(steps) >= 5
        for s in steps:
            assert "name" in s
            assert "deps" in s
            assert "description" in s
    
    def test_list_preprocessing_ops(self):
        """list_preprocessing_ops returns correct structure."""
        ops = list_preprocessing_ops()
        assert len(ops) >= 2
        for op in ops:
            assert "name" in op
            assert "description" in op


class TestValidation:
    """Tests for pipeline validation."""
    
    def test_valid_config(self):
        """Valid config returns no errors."""
        errors = validate_pipeline_config({
            "steps": ["ingest", "asr"],
            "preprocessing": ["trim_silence"],
        })
        assert errors == []
    
    def test_empty_steps_error(self):
        """Empty steps list is invalid."""
        errors = validate_pipeline_config({"steps": []})
        assert len(errors) > 0
    
    def test_unknown_step_error(self):
        """Unknown step is reported."""
        errors = validate_pipeline_config({"steps": ["ingest", "bad_step"]})
        assert any("Unknown step" in e for e in errors)
    
    def test_unknown_preprocessing_error(self):
        """Unknown preprocessing op is reported."""
        errors = validate_pipeline_config({
            "steps": ["ingest"],
            "preprocessing": ["bad_op"],
        })
        assert any("Unknown preprocessing" in e for e in errors)


class TestSerialization:
    """Tests for config serialization."""
    
    def test_to_dict(self):
        """Config converts to dict correctly."""
        config = PipelineConfig(
            name="test",
            steps=["ingest", "asr"],
            preprocessing=["normalize_loudness"],
        )
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["steps"] == ["ingest", "asr"]
        assert d["preprocessing"] == ["normalize_loudness"]
    
    def test_from_dict(self):
        """Config creates from dict correctly."""
        config = PipelineConfig.from_dict({
            "name": "test",
            "steps": ["ingest", "diarization"],
        })
        assert config.name == "test"
        assert config.steps == ["ingest", "diarization"]
    
    def test_to_yaml(self):
        """Config converts to YAML string."""
        config = PipelineConfig(steps=["ingest"])
        yaml_str = config.to_yaml()
        assert "steps:" in yaml_str
        assert "ingest" in yaml_str
