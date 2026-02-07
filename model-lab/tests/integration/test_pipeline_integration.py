"""
Pipeline integration tests (LCS-13).

Tests pipeline with real models if available.
Skips if dependencies not installed.
"""

import pytest
from pathlib import Path
import numpy as np

from harness.pipeline import (
    PipelineRunner,
    PipelineConfig,
    create_fake_model_loader,
    FakeASRBundle,
)


# Check if RNNoise is available
try:
    import pyrnnoise
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False


# Check if registry can be imported
try:
    from harness.registry import ModelRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    ModelRegistry = None
    REGISTRY_AVAILABLE = False


needs_rnnoise = pytest.mark.skipif(
    not RNNOISE_AVAILABLE,
    reason="pyrnnoise not installed"
)

needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)


class TestPipelineIntegrationFakeASR:
    """Integration tests with real enhance model + fake ASR."""
    
    @needs_rnnoise
    @needs_registry
    def test_rnnoise_to_fake_asr(self):
        """Test real RNNoise enhancement → fake ASR."""
        # Create a hybrid model loader
        def hybrid_loader(model_id: str, config: dict) -> dict:
            if model_id == "rnnoise":
                # Use real RNNoise
                return ModelRegistry.load_model("rnnoise", config, device="cpu")
            elif model_id == "fake_asr":
                # Use fake ASR
                return {
                    "model_type": "fake_asr",
                    "device": "cpu",
                    "capabilities": ["asr"],
                    "modes": ["batch"],
                    "asr": {"transcribe": FakeASRBundle.transcribe},
                }
            else:
                raise ValueError(f"Unknown model: {model_id}")
        
        runner = PipelineRunner(model_loader=hybrid_loader)
        
        # Create noisy test audio (1 second at 48kHz - RNNoise native)
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        
        config = PipelineConfig.from_dict({
            "name": "rnnoise_fake_asr",
            "steps": [
                {"id": "enhance", "model": "rnnoise", "surface": "enhance"},
                {"id": "asr", "model": "fake_asr", "surface": "asr"},
            ]
        })
        
        result = runner.run(config, audio, 48000)
        
        # Verify pipeline executed
        assert result.steps_executed == ["enhance", "asr"]
        
        # Verify enhancement step produced audio
        assert result.artifacts["enhance"]["audio"] is not None
        assert len(result.artifacts["enhance"]["audio"]) > 0
        
        # Verify ASR step produced text
        assert "text" in result.final


class TestPipelineConfigLoading:
    """Test loading and running pipeline configs from files."""
    
    def test_load_enhance_asr_config(self):
        """Test loading enhance_asr.yaml config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "pipelines" / "enhance_asr.yaml"
        config = PipelineConfig.from_yaml(config_path)
        
        assert config.name == "enhance_asr"
        assert len(config.steps) == 2
        assert config.steps[0].surface == "enhance"
        assert config.steps[1].surface == "asr"
    
    def test_load_separate_asr_config(self):
        """Test loading separate_vocals_asr.yaml config."""
        config_path = Path(__file__).parent.parent.parent / "config" / "pipelines" / "separate_vocals_asr.yaml"
        config = PipelineConfig.from_yaml(config_path)
        
        assert config.name == "separate_vocals_asr"
        assert config.steps[0].stem == "vocals"


class TestPipelineErrorMessages:
    """Test that error messages are descriptive."""
    
    def test_error_includes_step_id(self):
        """Test that errors include the step ID."""
        runner = PipelineRunner(model_loader=create_fake_model_loader())
        audio = np.random.randn(16000).astype(np.float32)
        
        config = PipelineConfig.from_dict({
            "name": "error_test",
            "steps": [
                {"id": "my_step", "model": "nonexistent", "surface": "enhance"},
            ]
        })
        
        with pytest.raises(Exception) as exc_info:
            runner.run(config, audio, 16000)
        
        error_str = str(exc_info.value)
        assert "my_step" in error_str
    
    def test_error_includes_model_id(self):
        """Test that errors include the model ID."""
        runner = PipelineRunner(model_loader=create_fake_model_loader())
        audio = np.random.randn(16000).astype(np.float32)
        
        config = PipelineConfig.from_dict({
            "name": "error_test",
            "steps": [
                {"id": "step", "model": "my_bad_model", "surface": "enhance"},
            ]
        })
        
        with pytest.raises(Exception) as exc_info:
            runner.run(config, audio, 16000)
        
        error_str = str(exc_info.value)
        assert "my_bad_model" in error_str
