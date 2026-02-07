"""
GLM-ASR-Nano-2512 smoke test (LCS-16).

Tests THUDM GLM-4 voice decoder (non-Whisper ASR).
PyTorch runtime.
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
class TestGLMASRSmokeTest:
    """Smoke tests for GLM-ASR-Nano-2512."""

    def test_model_registered(self):
        """Verify model is registered."""
        models = ModelRegistry.list_models()
        assert "glm_asr_nano_2512" in models
    
    def test_model_metadata(self):
        """Verify model metadata has asr capability."""
        meta = ModelRegistry.get_model_metadata("glm_asr_nano_2512")
        assert "asr" in meta["capabilities"]
    
    def test_model_loads(self):
        """Verify model loads without error."""
        bundle = ModelRegistry.load_model("glm_asr_nano_2512", {}, device="cpu")
        
        assert bundle["model_type"] == "glm_asr_nano_2512"
        assert "asr" in bundle["capabilities"]
    
    def test_transcribe_returns_text(self):
        """Verify transcription returns text field."""
        bundle = ModelRegistry.load_model("glm_asr_nano_2512", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = transcribe(audio, sr=16000)
        
        assert "text" in result


class TestGLMASRStructural:
    """Structural tests that don't require model loading."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "glm_asr_nano_2512" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "glm_asr_nano_2512" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "glm_asr_nano_2512" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_pytorch_runtime(self):
        """Verify PyTorch runtime."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "glm_asr_nano_2512" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("runtime") == "pytorch"
    
    def test_claims_non_whisper(self):
        """Verify non-Whisper architecture claim."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "glm_asr_nano_2512" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        claim_ids = [c["id"] for c in claims.get("claims", [])]
        assert "glm_asr_nano_non_whisper" in claim_ids
