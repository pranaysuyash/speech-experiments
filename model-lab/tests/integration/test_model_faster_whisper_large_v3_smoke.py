"""
Faster-Whisper Large V3 smoke test (LCS-14).

Tests Systran's faster-whisper-large-v3 CTranslate2 model.
Skips if faster-whisper not installed.
"""

import pytest
from pathlib import Path
import numpy as np


# Check if registry can be imported
try:
    from harness.registry import ModelRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    ModelRegistry = None
    REGISTRY_AVAILABLE = False


# Check if faster-whisper is available
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
class TestFasterWhisperLargeV3SmokeTest:
    """Smoke tests for Faster-Whisper Large V3."""

    def test_model_registered(self):
        """Verify faster_whisper_large_v3 is registered."""
        models = ModelRegistry.list_models()
        assert "faster_whisper_large_v3" in models
    
    def test_model_metadata(self):
        """Verify model metadata has asr capability."""
        meta = ModelRegistry.get_model_metadata("faster_whisper_large_v3")
        assert "asr" in meta["capabilities"]
    
    def test_model_loads(self):
        """Verify model loads without error."""
        bundle = ModelRegistry.load_model("faster_whisper_large_v3", {}, device="cpu")
        
        assert bundle["model_type"] == "faster_whisper_large_v3"
        assert "asr" in bundle["capabilities"]
        assert "transcribe" in bundle["asr"]
    
    def test_transcribe_returns_text(self):
        """Verify transcription returns text field."""
        bundle = ModelRegistry.load_model("faster_whisper_large_v3", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        # Simple sine wave (won't transcribe to meaningful text but tests pipeline)
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        result = transcribe(audio, sr=16000)
        
        assert "text" in result
        assert isinstance(result["text"], str)
    
    def test_transcribe_returns_segments(self):
        """Verify transcription returns segments list."""
        bundle = ModelRegistry.load_model("faster_whisper_large_v3", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = transcribe(audio, sr=16000)
        
        assert "segments" in result
        assert isinstance(result["segments"], list)
    
    def test_transcribe_returns_language(self):
        """Verify transcription returns language field."""
        bundle = ModelRegistry.load_model("faster_whisper_large_v3", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = transcribe(audio, sr=16000)
        
        assert "language" in result


class TestFasterWhisperLargeV3Structural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "faster_whisper_large_v3" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        """Verify config.yaml exists."""
        config_path = Path(__file__).parent.parent.parent / "models" / "faster_whisper_large_v3" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists."""
        req_path = Path(__file__).parent.parent.parent / "models" / "faster_whisper_large_v3" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_has_asr_surface(self):
        """Verify claims define asr surface."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "faster_whisper_large_v3" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        tasks = [c["task"] for c in claims.get("claims", [])]
        assert "asr" in tasks
    
    def test_claims_cta2_runtime(self):
        """Verify claims specify CTranslate2 runtime."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "faster_whisper_large_v3" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        assert claims.get("runtime") == "ctranslate2"
