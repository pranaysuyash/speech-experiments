"""
YAMNet model smoke test (LCS-05).

Tests that YAMNet loads and produces valid classification output.
Skips if TensorFlow is not installed.
"""

import pytest
from pathlib import Path


# Check if TensorFlow is available
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


needs_tf = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


@needs_tf
class TestYAMNetSmokeTest:
    """Smoke tests for YAMNet classification model."""

    def test_yamnet_registered(self):
        """Verify yamnet is registered in the model registry."""
        from harness.registry import ModelRegistry
        
        models = ModelRegistry.list_models()
        assert "yamnet" in models
    
    def test_yamnet_metadata(self):
        """Verify yamnet metadata is correctly set."""
        from harness.registry import ModelRegistry
        
        meta = ModelRegistry.get_model_metadata("yamnet")
        assert "classify" in meta["capabilities"]
        assert "cpu" in meta["hardware"]
    
    def test_yamnet_loads(self):
        """Verify yamnet loads without error."""
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("yamnet", {}, device="cpu")
        
        assert bundle["model_type"] == "yamnet"
        assert "classify" in bundle["capabilities"]
        assert "predict" in bundle["classify"]
    
    def test_yamnet_classify_produces_output(self):
        """Verify yamnet produces valid classification output."""
        import numpy as np
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("yamnet", {}, device="cpu")
        classify = bundle["classify"]["predict"]
        
        # Create test audio (1 second of noise at 16kHz)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        result = classify(audio, sr=16000, top_k=5)
        
        # Structural checks
        assert "labels" in result
        assert isinstance(result["labels"], list)
        assert len(result["labels"]) == 5
        
        assert "scores" in result
        assert isinstance(result["scores"], list)
        assert len(result["scores"]) == 5
        
        # Scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in result["scores"])


class TestYAMNetStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for yamnet."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "yamnet" / "claims.yaml"
        assert claims_path.exists(), f"Claims file not found: {claims_path}"
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for yamnet."""
        config_path = Path(__file__).parent.parent.parent / "models" / "yamnet" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for yamnet."""
        req_path = Path(__file__).parent.parent.parent / "models" / "yamnet" / "requirements.txt"
        assert req_path.exists()
    
    def test_ci_flag_is_false(self):
        """Verify ci=false for heavy TensorFlow dependency."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "yamnet" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        assert claims.get("ci") is False
        assert "TensorFlow" in claims.get("ci_reason", "")
