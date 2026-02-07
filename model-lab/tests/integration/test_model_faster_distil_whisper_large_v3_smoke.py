"""
Faster-Distil-Whisper Large V3 smoke test (LCS-15).

Tests Systran's distilled whisper with CTranslate2.
Same runtime as LCS-14 but 2-3x faster inference.
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
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)

needs_faster_whisper = pytest.mark.skipif(
    not FASTER_WHISPER_AVAILABLE,
    reason="faster-whisper not installed"
)


@needs_registry
@needs_faster_whisper
class TestFasterDistilWhisperSmokeTest:
    """Smoke tests for Faster-Distil-Whisper Large V3."""

    def test_model_registered(self):
        """Verify model is registered."""
        models = ModelRegistry.list_models()
        assert "faster_distil_whisper_large_v3" in models
    
    def test_model_metadata(self):
        """Verify model metadata has asr capability."""
        meta = ModelRegistry.get_model_metadata("faster_distil_whisper_large_v3")
        assert "asr" in meta["capabilities"]
    
    def test_model_loads(self):
        """Verify model loads without error."""
        bundle = ModelRegistry.load_model("faster_distil_whisper_large_v3", {}, device="cpu")
        
        assert bundle["model_type"] == "faster_distil_whisper_large_v3"
        assert "asr" in bundle["capabilities"]
    
    def test_transcribe_returns_text(self):
        """Verify transcription returns text."""
        bundle = ModelRegistry.load_model("faster_distil_whisper_large_v3", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = transcribe(audio, sr=16000)
        
        assert "text" in result
    
    def test_transcribe_returns_segments(self):
        """Verify transcription returns segments."""
        bundle = ModelRegistry.load_model("faster_distil_whisper_large_v3", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = transcribe(audio, sr=16000)
        
        assert "segments" in result


class TestFasterDistilWhisperStructural:
    """Structural tests that don't require model loading."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "faster_distil_whisper_large_v3" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "faster_distil_whisper_large_v3" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "faster_distil_whisper_large_v3" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_cta2_runtime(self):
        """Verify CTranslate2 runtime."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "faster_distil_whisper_large_v3" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("runtime") == "ctranslate2"
