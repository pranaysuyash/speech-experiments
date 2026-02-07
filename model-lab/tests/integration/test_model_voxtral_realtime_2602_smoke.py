"""
Voxtral Realtime 2602 smoke test (LCS-22).

Tests real-time streaming ASR with configurable transcription_delay_ms.
Reuses StreamingAdapter for lifecycle.
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
    from harness.streaming import StreamingAdapter
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)

needs_streaming = pytest.mark.skipif(
    not STREAMING_AVAILABLE,
    reason="streaming utilities not available"
)


@needs_registry
@needs_streaming
class TestVoxtralRealtimeSmokeTest:
    """Smoke tests for Voxtral Realtime 2602."""

    def test_model_registered(self):
        """Verify model is registered."""
        models = ModelRegistry.list_models()
        assert "voxtral_realtime_2602" in models
    
    def test_model_metadata(self):
        """Verify model metadata has asr_stream capability."""
        meta = ModelRegistry.get_model_metadata("voxtral_realtime_2602")
        assert "asr_stream" in meta["capabilities"]
    
    def test_model_loads_with_default_delay(self):
        """Verify model loads with default delay."""
        bundle = ModelRegistry.load_model("voxtral_realtime_2602", {}, device="cpu")
        
        assert bundle["model_type"] == "voxtral_realtime_2602"
        assert bundle["config"]["transcription_delay_ms"] == 200
    
    def test_model_loads_with_custom_delay(self):
        """Verify model loads with custom delay."""
        bundle = ModelRegistry.load_model("voxtral_realtime_2602", {
            "transcription_delay_ms": 150
        }, device="cpu")
        
        assert bundle["config"]["transcription_delay_ms"] == 150
    
    def test_delay_clamped_to_range(self):
        """Verify delay is clamped to 80-2400ms range."""
        # Too low
        bundle = ModelRegistry.load_model("voxtral_realtime_2602", {
            "transcription_delay_ms": 50
        }, device="cpu")
        assert bundle["config"]["transcription_delay_ms"] == 80
        
        # Too high
        bundle = ModelRegistry.load_model("voxtral_realtime_2602", {
            "transcription_delay_ms": 5000
        }, device="cpu")
        assert bundle["config"]["transcription_delay_ms"] == 2400
    
    def test_streaming_lifecycle(self):
        """Verify streaming lifecycle works."""
        bundle = ModelRegistry.load_model("voxtral_realtime_2602", {}, device="cpu")
        stream = bundle["asr_stream"]
        
        handle = stream["start"](sr=16000)
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        stream["push_audio"](handle, audio)
        result = stream["get_transcript"](handle)
        assert "delay_ms" in result
        final = stream["finalize"](handle)
        assert "is_final" in final


class TestVoxtralRealtimeStructural:
    """Structural tests that don't require model loading."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "voxtral_realtime_2602" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "voxtral_realtime_2602" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "voxtral_realtime_2602" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_streaming_true(self):
        """Verify streaming=true in claims."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "voxtral_realtime_2602" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("streaming") is True
    
    def test_claims_has_delay_configurable(self):
        """Verify claims include transcription_delay_ms configurable claim."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "voxtral_realtime_2602" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        claim_ids = [c["id"] for c in claims.get("claims", [])]
        assert "voxtral_realtime_delay_configurable" in claim_ids
    
    def test_claims_delay_range(self):
        """Verify claims document 80-2400ms delay range."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "voxtral_realtime_2602" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        for claim in claims.get("claims", []):
            if claim.get("id") == "voxtral_realtime_delay_configurable":
                thresholds = claim.get("thresholds", {})
                assert thresholds.get("transcription_delay_ms_min") == 80
                assert thresholds.get("transcription_delay_ms_max") == 2400
                break
