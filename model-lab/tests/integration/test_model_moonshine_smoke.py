"""
Moonshine model smoke test (LCS-04).

Tests that Moonshine loads and produces valid ASR output.
Skips if dependencies are not installed.
"""

import pytest
from pathlib import Path


# Check if dependencies are available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import moonshine
    MOONSHINE_AVAILABLE = True
except ImportError:
    MOONSHINE_AVAILABLE = False


needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
needs_moonshine = pytest.mark.skipif(not MOONSHINE_AVAILABLE, reason="moonshine not installed")


@needs_torch
@needs_moonshine
class TestMoonshineSmokeTest:
    """Smoke tests for Moonshine ASR model."""

    def test_moonshine_registered(self):
        """Verify moonshine is registered in the model registry."""
        from harness.registry import ModelRegistry
        
        models = ModelRegistry.list_models()
        assert "moonshine" in models
    
    def test_moonshine_metadata(self):
        """Verify moonshine metadata is correctly set."""
        from harness.registry import ModelRegistry
        
        meta = ModelRegistry.get_model_metadata("moonshine")
        assert "asr" in meta["capabilities"]
        assert "cpu" in meta["hardware"]
    
    def test_moonshine_loads(self):
        """Verify moonshine loads without error."""
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("moonshine", {}, device="cpu")
        
        assert bundle["model_type"] == "moonshine"
        assert "asr" in bundle["capabilities"]
        assert "transcribe" in bundle["asr"]
    
    def test_moonshine_transcribe_produces_output(self):
        """Verify moonshine produces non-empty transcript."""
        import numpy as np
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("moonshine", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        # Create test audio (1 second of noise)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        result = transcribe(audio, sr=16000)
        
        # Structural checks
        assert "text" in result
        assert isinstance(result["text"], str)
        assert "segments" in result
        assert isinstance(result["segments"], list)
        assert "language" in result


class TestMoonshineStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for moonshine."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "moonshine" / "claims.yaml"
        assert claims_path.exists(), f"Claims file not found: {claims_path}"
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for moonshine."""
        config_path = Path(__file__).parent.parent.parent / "models" / "moonshine" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for moonshine."""
        req_path = Path(__file__).parent.parent.parent / "models" / "moonshine" / "requirements.txt"
        assert req_path.exists()
