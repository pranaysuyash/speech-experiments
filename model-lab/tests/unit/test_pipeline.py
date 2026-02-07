"""
Pipeline unit tests (LCS-13).

Tests pipeline runner using fake models (no heavy deps required).
All tests should be CI-safe.
"""

import pytest
from pathlib import Path
import numpy as np

from harness.pipeline import (
    PipelineConfig,
    PipelineStep,
    PipelineRunner,
    PipelineResult,
    PipelineError,
    run_pipeline,
    create_fake_model_loader,
    create_fake_bundle,
)


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        config = PipelineConfig.from_dict({
            "steps": [
                {"id": "step1", "model": "model1", "surface": "enhance"},
            ]
        })
        
        assert config.name == "unnamed_pipeline"
        assert len(config.steps) == 1
        assert config.steps[0].id == "step1"
    
    def test_from_dict_full(self):
        """Test creating config from full dict."""
        config = PipelineConfig.from_dict({
            "name": "test_pipeline",
            "description": "A test pipeline",
            "steps": [
                {"id": "enhance", "model": "deepfilternet", "surface": "enhance", "args": {"gain": 1.0}},
                {"id": "asr", "model": "moonshine", "surface": "asr"},
            ]
        })
        
        assert config.name == "test_pipeline"
        assert config.description == "A test pipeline"
        assert len(config.steps) == 2
        assert config.steps[0].args == {"gain": 1.0}
    
    def test_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_path = Path(__file__).parent.parent.parent / "config" / "pipelines" / "enhance_asr.yaml"
        
        if yaml_path.exists():
            config = PipelineConfig.from_yaml(yaml_path)
            assert config.name == "enhance_asr"
            assert len(config.steps) == 2
    
    def test_step_with_stem(self):
        """Test step with stem field for separate surface."""
        step = PipelineStep.from_dict({
            "id": "separate",
            "model": "demucs",
            "surface": "separate",
            "stem": "vocals",
        })
        
        assert step.stem == "vocals"


class TestPipelineRunnerFakeModels:
    """Tests for PipelineRunner with fake models."""
    
    @pytest.fixture
    def fake_runner(self):
        """Create runner with fake model loader."""
        return PipelineRunner(model_loader=create_fake_model_loader())
    
    @pytest.fixture
    def test_audio(self):
        """Create test audio."""
        return np.random.randn(16000).astype(np.float32) * 0.5
    
    def test_single_step_enhance(self, fake_runner, test_audio):
        """Test single enhance step."""
        config = PipelineConfig.from_dict({
            "name": "single_enhance",
            "steps": [
                {"id": "enhance", "model": "fake_enhance", "surface": "enhance"},
            ]
        })
        
        result = fake_runner.run(config, test_audio, 16000)
        
        assert isinstance(result, PipelineResult)
        assert "enhance" in result.artifacts
        assert result.steps_executed == ["enhance"]
        assert result.artifacts["enhance"]["audio"] is not None
    
    def test_single_step_asr(self, fake_runner, test_audio):
        """Test single ASR step."""
        config = PipelineConfig.from_dict({
            "name": "single_asr",
            "steps": [
                {"id": "asr", "model": "fake_asr", "surface": "asr"},
            ]
        })
        
        result = fake_runner.run(config, test_audio, 16000)
        
        assert "asr" in result.artifacts
        assert "text" in result.final
        assert result.artifacts["asr"]["audio"] is None  # ASR doesn't output audio
    
    def test_two_step_enhance_asr(self, fake_runner, test_audio):
        """Test enhance → asr pipeline."""
        config = PipelineConfig.from_dict({
            "name": "enhance_asr",
            "steps": [
                {"id": "enhance", "model": "fake_enhance", "surface": "enhance"},
                {"id": "asr", "model": "fake_asr", "surface": "asr"},
            ]
        })
        
        result = fake_runner.run(config, test_audio, 16000)
        
        assert result.steps_executed == ["enhance", "asr"]
        assert "enhance" in result.artifacts
        assert "asr" in result.artifacts
        assert "text" in result.final
    
    def test_separate_asr(self, fake_runner, test_audio):
        """Test separate(vocals) → asr pipeline."""
        config = PipelineConfig.from_dict({
            "name": "separate_asr",
            "steps": [
                {"id": "separate", "model": "fake_separate", "surface": "separate", "stem": "vocals"},
                {"id": "asr", "model": "fake_asr", "surface": "asr"},
            ]
        })
        
        result = fake_runner.run(config, test_audio, 16000)
        
        assert result.steps_executed == ["separate", "asr"]
        assert result.artifacts["separate"]["metadata"]["stem_extracted"] == "vocals"
        assert "text" in result.final
    
    def test_separate_music_transcription(self, fake_runner, test_audio):
        """Test separate(vocals) → music_transcription pipeline."""
        config = PipelineConfig.from_dict({
            "name": "separate_transcribe",
            "steps": [
                {"id": "separate", "model": "fake_separate", "surface": "separate", "stem": "vocals"},
                {"id": "transcribe", "model": "fake_music_transcription", "surface": "music_transcription"},
            ]
        })
        
        result = fake_runner.run(config, test_audio, 16000)
        
        assert result.steps_executed == ["separate", "transcribe"]
        assert "notes" in result.final
    
    def test_artifacts_stored_correctly(self, fake_runner, test_audio):
        """Test that all artifacts are stored and accessible."""
        config = PipelineConfig.from_dict({
            "name": "multi_step",
            "steps": [
                {"id": "step1", "model": "fake_enhance", "surface": "enhance"},
                {"id": "step2", "model": "fake_asr", "surface": "asr"},
            ]
        })
        
        result = fake_runner.run(config, test_audio, 16000)
        
        # All steps should have artifacts
        for step_id in result.steps_executed:
            assert step_id in result.artifacts
            artifact = result.get_artifact(step_id)
            assert "audio" in artifact
            assert "sr" in artifact
            assert "metadata" in artifact
    
    def test_sr_preserved(self, fake_runner, test_audio):
        """Test that sample rate is preserved through pipeline."""
        config = PipelineConfig.from_dict({
            "name": "sr_test",
            "steps": [
                {"id": "enhance", "model": "fake_enhance", "surface": "enhance"},
            ]
        })
        
        result = fake_runner.run(config, test_audio, 44100)
        
        assert result.sample_rate == 44100
        assert result.artifacts["enhance"]["sr"] == 44100
    
    def test_invalid_surface_raises_error(self, fake_runner, test_audio):
        """Test that invalid surface raises descriptive error."""
        config = PipelineConfig.from_dict({
            "name": "invalid",
            "steps": [
                {"id": "bad", "model": "fake_enhance", "surface": "nonexistent"},
            ]
        })
        
        with pytest.raises(PipelineError) as exc_info:
            fake_runner.run(config, test_audio, 16000)
        
        error = exc_info.value
        assert error.step_id == "bad"
        assert "nonexistent" in str(error)
    
    def test_invalid_model_raises_error(self, fake_runner, test_audio):
        """Test that invalid model raises descriptive error."""
        config = PipelineConfig.from_dict({
            "name": "invalid",
            "steps": [
                {"id": "bad", "model": "nonexistent_model", "surface": "enhance"},
            ]
        })
        
        with pytest.raises(PipelineError) as exc_info:
            fake_runner.run(config, test_audio, 16000)
        
        error = exc_info.value
        assert error.step_id == "bad"
        assert error.model_id == "nonexistent_model"
    
    def test_invalid_stem_raises_error(self, fake_runner, test_audio):
        """Test that invalid stem raises descriptive error."""
        config = PipelineConfig.from_dict({
            "name": "invalid",
            "steps": [
                {"id": "sep", "model": "fake_separate", "surface": "separate", "stem": "piano"},
            ]
        })
        
        with pytest.raises(PipelineError) as exc_info:
            fake_runner.run(config, test_audio, 16000)
        
        error = exc_info.value
        assert error.step_id == "sep"
        assert "piano" in str(error.original_error)


class TestRunPipelineConvenience:
    """Tests for run_pipeline convenience function."""
    
    def test_run_pipeline_with_dict(self):
        """Test run_pipeline with dict config."""
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        
        # Can't use convenience function with fake loader directly
        # but can test dict parsing
        config = {
            "name": "test",
            "steps": [
                {"id": "enhance", "model": "fake_enhance", "surface": "enhance"},
            ]
        }
        
        runner = PipelineRunner(model_loader=create_fake_model_loader())
        result = runner.run(config, audio, 16000)
        
        assert result.steps_executed == ["enhance"]


class TestFakeBundles:
    """Tests for fake bundles."""
    
    def test_create_fake_enhance_bundle(self):
        """Test fake enhance bundle."""
        bundle = create_fake_bundle("enhance")
        
        assert bundle["model_type"] == "fake_enhance"
        assert "enhance" in bundle
        
        audio = np.random.randn(1000).astype(np.float32)
        result = bundle["enhance"]["process"](audio)
        
        assert len(result) == len(audio)
    
    def test_create_fake_separate_bundle(self):
        """Test fake separate bundle."""
        bundle = create_fake_bundle("separate")
        
        audio = np.random.randn(1000).astype(np.float32)
        result = bundle["separate"]["separate"](audio)
        
        assert "stems" in result
        assert "vocals" in result["stems"]
        assert "drums" in result["stems"]
    
    def test_create_fake_asr_bundle(self):
        """Test fake ASR bundle."""
        bundle = create_fake_bundle("asr")
        
        audio = np.random.randn(1000).astype(np.float32)
        result = bundle["asr"]["transcribe"](audio)
        
        assert "text" in result
    
    def test_create_fake_music_transcription_bundle(self):
        """Test fake music transcription bundle."""
        bundle = create_fake_bundle("music_transcription")
        
        audio = np.random.randn(1000).astype(np.float32)
        result = bundle["music_transcription"]["transcribe"](audio)
        
        assert "notes" in result
        assert len(result["notes"]) > 0


class TestPipelineConfigFiles:
    """Tests for pipeline config files."""
    
    def test_enhance_asr_config_exists(self):
        """Test enhance_asr.yaml exists and is valid."""
        config_path = Path(__file__).parent.parent.parent / "config" / "pipelines" / "enhance_asr.yaml"
        assert config_path.exists()
        
        config = PipelineConfig.from_yaml(config_path)
        assert len(config.steps) == 2
    
    def test_separate_vocals_asr_config_exists(self):
        """Test separate_vocals_asr.yaml exists and is valid."""
        config_path = Path(__file__).parent.parent.parent / "config" / "pipelines" / "separate_vocals_asr.yaml"
        assert config_path.exists()
        
        config = PipelineConfig.from_yaml(config_path)
        assert len(config.steps) == 2
        assert config.steps[0].stem == "vocals"
    
    def test_separate_vocals_transcribe_config_exists(self):
        """Test separate_vocals_transcribe.yaml exists and is valid."""
        config_path = Path(__file__).parent.parent.parent / "config" / "pipelines" / "separate_vocals_transcribe.yaml"
        assert config_path.exists()
        
        config = PipelineConfig.from_yaml(config_path)
        assert len(config.steps) == 2
