"""
NB-Whisper-Small-ONNX smoke test (LCS-17).

Tests Norwegian Whisper with ONNX runtime.
Cross-platform inference without PyTorch.
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
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)

needs_onnx = pytest.mark.skipif(
    not ONNX_AVAILABLE,
    reason="onnxruntime not installed"
)


@needs_registry
@needs_onnx
class TestNBWhisperONNXSmokeTest:
    """Smoke tests for NB-Whisper-Small-ONNX."""

    def test_model_registered(self):
        """Verify model is registered."""
        models = ModelRegistry.list_models()
        assert "nb_whisper_small_onnx" in models
    
    def test_model_metadata(self):
        """Verify model metadata has asr capability."""
        meta = ModelRegistry.get_model_metadata("nb_whisper_small_onnx")
        assert "asr" in meta["capabilities"]
    
    def test_model_loads(self):
        """Verify model loads without error."""
        bundle = ModelRegistry.load_model("nb_whisper_small_onnx", {}, device="cpu")
        
        assert bundle["model_type"] == "nb_whisper_small_onnx"
        assert "asr" in bundle["capabilities"]
    
    def test_transcribe_returns_text(self):
        """Verify transcription returns text field."""
        bundle = ModelRegistry.load_model("nb_whisper_small_onnx", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = transcribe(audio, sr=16000)
        
        assert "text" in result


class TestNBWhisperONNXStructural:
    """Structural tests that don't require model loading."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "nb_whisper_small_onnx" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "nb_whisper_small_onnx" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "nb_whisper_small_onnx" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_onnx_runtime(self):
        """Verify ONNX runtime."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "nb_whisper_small_onnx" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("runtime") == "onnxruntime"
