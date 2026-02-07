"""
RNNoise model smoke test (LCS-06).

Tests that RNNoise loads and produces valid enhanced audio output.
Skips if pyrnnoise is not installed.
"""

import pytest
from pathlib import Path


# Check if pyrnnoise is available
try:
    from pyrnnoise import RNNoise
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False


needs_rnnoise = pytest.mark.skipif(not RNNOISE_AVAILABLE, reason="pyrnnoise not installed")


@needs_rnnoise
class TestRNNoiseSmokeTest:
    """Smoke tests for RNNoise enhancement model."""

    def test_rnnoise_registered(self):
        """Verify rnnoise is registered in the model registry."""
        from harness.registry import ModelRegistry
        
        models = ModelRegistry.list_models()
        assert "rnnoise" in models
    
    def test_rnnoise_metadata(self):
        """Verify rnnoise metadata is correctly set."""
        from harness.registry import ModelRegistry
        
        meta = ModelRegistry.get_model_metadata("rnnoise")
        assert "enhance" in meta["capabilities"]
        assert "cpu" in meta["hardware"]
        assert "streaming" in meta["modes"]
    
    def test_rnnoise_loads(self):
        """Verify rnnoise loads without error."""
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("rnnoise", {}, device="cpu")
        
        assert bundle["model_type"] == "rnnoise"
        assert "enhance" in bundle["capabilities"]
        assert "process" in bundle["enhance"]
    
    def test_rnnoise_enhance_produces_output(self):
        """Verify rnnoise produces enhanced audio."""
        import numpy as np
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("rnnoise", {}, device="cpu")
        enhance = bundle["enhance"]["process"]
        
        # Create test audio (1 second of noise at 48kHz)
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        
        result = enhance(audio, sr=48000)
        
        # Structural checks
        assert "audio" in result
        assert isinstance(result["audio"], np.ndarray)
        assert len(result["audio"]) == len(audio)
        
        assert "sample_rate" in result
        assert result["sample_rate"] == 48000
        
        assert "vad_probs" in result
        assert isinstance(result["vad_probs"], list)


class TestRNNoiseStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for rnnoise."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "rnnoise" / "claims.yaml"
        assert claims_path.exists(), f"Claims file not found: {claims_path}"
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for rnnoise."""
        config_path = Path(__file__).parent.parent.parent / "models" / "rnnoise" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for rnnoise."""
        req_path = Path(__file__).parent.parent.parent / "models" / "rnnoise" / "requirements.txt"
        assert req_path.exists()
    
    def test_native_runtime(self):
        """Verify runtime is set to native."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "rnnoise" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        assert claims.get("runtime") == "native"
        assert claims.get("streaming") is True
