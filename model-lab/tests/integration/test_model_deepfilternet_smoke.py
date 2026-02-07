"""
DeepFilterNet model smoke test (LCS-07).

Tests that DeepFilterNet loads and produces valid enhanced audio output.
Skips if deepfilternet package is not installed.
"""

import pytest
from pathlib import Path


# Check if deepfilternet is available
try:
    from df.enhance import init_df
    DF_AVAILABLE = True
except ImportError:
    DF_AVAILABLE = False


needs_df = pytest.mark.skipif(not DF_AVAILABLE, reason="deepfilternet not installed")


@needs_df
class TestDeepFilterNetSmokeTest:
    """Smoke tests for DeepFilterNet enhancement model."""

    def test_deepfilternet_registered(self):
        """Verify deepfilternet is registered in the model registry."""
        from harness.registry import ModelRegistry
        
        models = ModelRegistry.list_models()
        assert "deepfilternet" in models
    
    def test_deepfilternet_metadata(self):
        """Verify deepfilternet metadata is correctly set."""
        from harness.registry import ModelRegistry
        
        meta = ModelRegistry.get_model_metadata("deepfilternet")
        assert "enhance" in meta["capabilities"]
        assert "cpu" in meta["hardware"]
    
    def test_deepfilternet_loads(self):
        """Verify deepfilternet loads without error."""
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("deepfilternet", {}, device="cpu")
        
        assert bundle["model_type"] == "deepfilternet"
        assert "enhance" in bundle["capabilities"]
        assert "process" in bundle["enhance"]
    
    def test_deepfilternet_enhance_preserves_length(self):
        """Verify deepfilternet output matches input length."""
        import numpy as np
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("deepfilternet", {}, device="cpu")
        enhance = bundle["enhance"]["process"]
        
        # Create test audio (1 second at 48kHz)
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        
        result = enhance(audio, sr=48000)
        
        # Structural checks
        assert "audio" in result
        assert isinstance(result["audio"], np.ndarray)
        
        # Length must match (alignment preservation)
        assert len(result["audio"]) == len(audio)
        
        # Sample rate must match
        assert result["sample_rate"] == 48000
    
    def test_deepfilternet_output_is_numeric(self):
        """Verify output is numeric ndarray."""
        import numpy as np
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("deepfilternet", {}, device="cpu")
        enhance = bundle["enhance"]["process"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.3
        result = enhance(audio, sr=48000)
        
        assert np.issubdtype(result["audio"].dtype, np.floating)


class TestDeepFilterNetStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for deepfilternet."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "deepfilternet" / "claims.yaml"
        assert claims_path.exists(), f"Claims file not found: {claims_path}"
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for deepfilternet."""
        config_path = Path(__file__).parent.parent.parent / "models" / "deepfilternet" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for deepfilternet."""
        req_path = Path(__file__).parent.parent.parent / "models" / "deepfilternet" / "requirements.txt"
        assert req_path.exists()
    
    def test_ci_flag_is_false(self):
        """Verify ci=false for heavy PyTorch dependency."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "deepfilternet" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        assert claims.get("ci") is False
        assert "PyTorch" in claims.get("ci_reason", "")
    
    def test_sample_rate_is_48k(self):
        """Verify native sample rate is 48kHz."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "deepfilternet" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        assert claims.get("io", {}).get("sample_rate") == 48000
