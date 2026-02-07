"""
CLAP model smoke test (LCS-08).

Tests that CLAP loads and produces valid embeddings and classifications.
First multi-surface model: embed + classify.
Skips if laion-clap package is not installed.
"""

import pytest
from pathlib import Path


# Check if laion-clap is available
try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False


needs_clap = pytest.mark.skipif(not CLAP_AVAILABLE, reason="laion-clap not installed")


@needs_clap
class TestCLAPSmokeTest:
    """Smoke tests for CLAP embed/classify model."""

    def test_clap_registered(self):
        """Verify clap is registered in the model registry."""
        from harness.registry import ModelRegistry
        
        models = ModelRegistry.list_models()
        assert "clap" in models
    
    def test_clap_metadata_multi_surface(self):
        """Verify clap has both embed and classify capabilities."""
        from harness.registry import ModelRegistry
        
        meta = ModelRegistry.get_model_metadata("clap")
        assert "embed" in meta["capabilities"]
        assert "classify" in meta["capabilities"]
    
    def test_clap_loads_with_both_surfaces(self):
        """Verify clap bundle contains both embed and classify."""
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("clap", {}, device="cpu")
        
        assert bundle["model_type"] == "clap"
        assert "embed" in bundle["capabilities"]
        assert "classify" in bundle["capabilities"]
        
        # Verify surface namespaces exist
        assert "embed" in bundle
        assert "encode" in bundle["embed"]
        assert "classify" in bundle
        assert "predict" in bundle["classify"]
    
    def test_clap_embed_produces_512d_vector(self):
        """Verify clap produces 512-d embeddings."""
        import numpy as np
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("clap", {}, device="cpu")
        encode = bundle["embed"]["encode"]
        
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = encode(audio, sr=48000)
        
        assert "embedding" in result
        assert "dim" in result
        assert result["dim"] == 512
        assert len(result["embedding"]) == 512
    
    def test_clap_classify_with_text_prompts(self):
        """Verify clap zero-shot classification works."""
        import numpy as np
        from harness.registry import ModelRegistry
        
        bundle = ModelRegistry.load_model("clap", {}, device="cpu")
        classify = bundle["classify"]["predict"]
        
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        labels = ["speech", "music", "noise"]
        result = classify(audio, sr=48000, labels=labels, top_k=3)
        
        assert "labels" in result
        assert "scores" in result
        assert len(result["labels"]) == 3
        assert len(result["scores"]) == 3
        
        # Scores should sum close to 1 (softmax)
        assert 0.99 <= sum(result["scores"]) <= 1.01


class TestCLAPStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for clap."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "clap" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for clap."""
        config_path = Path(__file__).parent.parent.parent / "models" / "clap" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for clap."""
        req_path = Path(__file__).parent.parent.parent / "models" / "clap" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_has_both_surfaces(self):
        """Verify claims define both embed and classify."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "clap" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        tasks = [c["task"] for c in claims.get("claims", [])]
        assert "embed" in tasks
        assert "classify" in tasks
