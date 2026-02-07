"""
Kyutai Streaming ASR smoke test (LCS-19).

Tests Kyutai streaming with StreamingAdapter lifecycle.
First streaming model in Batch 3.
"""

import pytest
from pathlib import Path
import numpy as np


try:
    from harness.registry import ModelRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    ModelRegistry = None
    REGISTRY_AVAILABLE = False


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)

needs_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch not installed"
)


@needs_registry
@needs_torch
class TestKyutaiStreamingSmokeTest:
    """Smoke tests for Kyutai Streaming ASR."""

    def test_model_registered(self):
        """Verify model is registered."""
        models = ModelRegistry.list_models()
        assert "kyutai_streaming" in models
    
    def test_model_metadata(self):
        """Verify model metadata has asr_stream capability."""
        meta = ModelRegistry.get_model_metadata("kyutai_streaming")
        assert "asr_stream" in meta["capabilities"]
    
    def test_model_loads(self):
        """Verify model loads without error."""
        bundle = ModelRegistry.load_model("kyutai_streaming", {}, device="cpu")
        
        assert bundle["model_type"] == "kyutai_streaming"
        assert "asr_stream" in bundle["capabilities"]
    
    def test_streaming_lifecycle(self):
        """Verify streaming lifecycle works."""
        bundle = ModelRegistry.load_model("kyutai_streaming", {}, device="cpu")
        stream = bundle["asr_stream"]
        
        # Start
        handle = stream["start"](sr=16000)
        assert isinstance(handle, str)
        
        # Push audio
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        stream["push_audio"](handle, audio)
        
        # Get transcript
        result = stream["get_transcript"](handle)
        assert isinstance(result, dict)
        
        # Finalize
        final = stream["finalize"](handle)
        assert "text" in final
    
    def test_finalize_idempotent(self):
        """Verify finalize is idempotent."""
        bundle = ModelRegistry.load_model("kyutai_streaming", {}, device="cpu")
        stream = bundle["asr_stream"]
        
        handle = stream["start"](sr=16000)
        final1 = stream["finalize"](handle)
        final2 = stream["finalize"](handle)  # Should not raise
        
        assert final1 == final2


class TestKyutaiStreamingStructural:
    """Structural tests that don't require model loading."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "kyutai_streaming" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "kyutai_streaming" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "kyutai_streaming" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_streaming_true(self):
        """Verify streaming=true in claims."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "kyutai_streaming" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("streaming") is True
    
    def test_claims_has_seq_monotonic(self):
        """Verify claims include seq_monotonic threshold."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "kyutai_streaming" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        for claim in claims.get("claims", []):
            if claim.get("id") == "kyutai_streaming_asr_stream_structural":
                assert claim.get("thresholds", {}).get("seq_monotonic") is True
                break
