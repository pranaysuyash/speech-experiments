"""
GLM-TTS smoke test (LCS-21).

Tests THUDM GLM-4 Voice TTS.
Broadens TTS coverage beyond lfm2_5_audio.
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
class TestGLMTTSSmokeTest:
    """Smoke tests for GLM-TTS."""

    def test_model_registered(self):
        """Verify model is registered."""
        models = ModelRegistry.list_models()
        assert "glm_tts" in models
    
    def test_model_metadata(self):
        """Verify model metadata has tts capability."""
        meta = ModelRegistry.get_model_metadata("glm_tts")
        assert "tts" in meta["capabilities"]
    
    def test_model_loads(self):
        """Verify model loads without error."""
        bundle = ModelRegistry.load_model("glm_tts", {}, device="cpu")
        
        assert bundle["model_type"] == "glm_tts"
        assert "tts" in bundle["capabilities"]
    
    def test_synthesize_returns_audio(self):
        """Verify synthesis returns audio array and sample rate."""
        bundle = ModelRegistry.load_model("glm_tts", {}, device="cpu")
        synthesize = bundle["tts"]["synthesize"]
        
        audio, sr = synthesize("Hello world")
        
        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int)
        assert sr > 0


class TestGLMTTSStructural:
    """Structural tests that don't require model loading."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "glm_tts" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "glm_tts" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "glm_tts" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_tts_surface(self):
        """Verify claims define tts surface."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "glm_tts" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        tasks = [c["task"] for c in claims.get("claims", [])]
        assert "tts" in tasks
