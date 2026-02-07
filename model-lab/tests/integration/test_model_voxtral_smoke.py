"""
Voxtral model smoke test (LCS-10).

Tests that Voxtral loads and streaming contract is honored.
Uses mock mode (no API key required for tests).
"""

import pytest
from pathlib import Path
import numpy as np


# Check if registry can be imported (requires torch for some loaders)
try:
    from harness.registry import ModelRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    ModelRegistry = None
    REGISTRY_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)


@needs_registry
class TestVoxtralSmokeTest:
    """Smoke tests for Voxtral streaming ASR (mock mode)."""

    def test_voxtral_registered(self):
        """Verify voxtral is registered in the model registry."""
        models = ModelRegistry.list_models()
        assert "voxtral" in models
    
    def test_voxtral_metadata(self):
        """Verify voxtral metadata has asr_stream capability."""
        meta = ModelRegistry.get_model_metadata("voxtral")
        assert "asr_stream" in meta["capabilities"]
        assert "asr" in meta["capabilities"]
        assert "streaming" in meta["modes"]
    
    def test_voxtral_loads_mock_mode(self):
        """Verify voxtral loads in mock mode (no API key)."""
        bundle = ModelRegistry.load_model("voxtral", {}, device="cpu")
        
        assert bundle["model_type"] == "voxtral"
        assert "asr_stream" in bundle["capabilities"]
        assert "asr" in bundle["capabilities"]
    
    def test_voxtral_has_streaming_namespace(self):
        """Verify voxtral bundle has asr_stream namespace."""
        bundle = ModelRegistry.load_model("voxtral", {}, device="cpu")
        
        assert "asr_stream" in bundle
        asr_stream = bundle["asr_stream"]
        
        # Verify all lifecycle methods present
        assert "start_stream" in asr_stream
        assert "push_audio" in asr_stream
        assert "flush" in asr_stream
        assert "finalize" in asr_stream
        assert "close" in asr_stream
    
    def test_voxtral_streaming_lifecycle(self):
        """Verify voxtral streaming lifecycle works."""
        bundle = ModelRegistry.load_model("voxtral", {}, device="cpu")
        asr_stream = bundle["asr_stream"]
        
        # Start stream
        handle = asr_stream["start_stream"]()
        assert handle is not None
        
        # Push some audio
        audio = np.random.randn(1600).astype(np.float32) * 0.5
        events = list(asr_stream["push_audio"](audio, 16000))
        
        # Flush
        flush_events = list(asr_stream["flush"]())
        
        # Finalize
        result = asr_stream["finalize"]()
        assert "text" in result
        
        # Close
        asr_stream["close"]()
    
    def test_voxtral_seq_monotonic(self):
        """Verify sequence numbers are monotonically increasing."""
        bundle = ModelRegistry.load_model("voxtral", {}, device="cpu")
        asr_stream = bundle["asr_stream"]
        
        asr_stream["start_stream"]()
        
        all_events = []
        for _ in range(3):
            audio = np.random.randn(1600).astype(np.float32) * 0.5
            events = list(asr_stream["push_audio"](audio, 16000))
            all_events.extend(events)
        
        flush_events = list(asr_stream["flush"]())
        all_events.extend(flush_events)
        
        if all_events:
            seqs = [e.seq for e in all_events]
            assert seqs == sorted(seqs), "seq not monotonic"
            assert len(seqs) == len(set(seqs)), "seq has duplicates"
        
        asr_stream["close"]()
    
    def test_voxtral_segment_id_stable(self):
        """Verify segment_id is stable across partial updates."""
        bundle = ModelRegistry.load_model("voxtral", {}, device="cpu")
        asr_stream = bundle["asr_stream"]
        
        asr_stream["start_stream"]()
        
        all_events = []
        for _ in range(3):
            audio = np.random.randn(1600).astype(np.float32) * 0.5
            events = list(asr_stream["push_audio"](audio, 16000))
            all_events.extend(events)
        
        if all_events:
            segment_ids = [e.segment_id for e in all_events]
            # Within a segment, all IDs should be the same
            assert len(set(segment_ids)) == 1, "segment_id not stable"
        
        asr_stream["close"]()
    
    def test_voxtral_batch_transcribe(self):
        """Verify batch transcribe works via streaming internally."""
        bundle = ModelRegistry.load_model("voxtral", {}, device="cpu")
        transcribe = bundle["asr"]["transcribe"]
        
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = transcribe(audio, sr=16000)
        
        assert "text" in result
        assert "segments" in result
        assert "language" in result


class TestVoxtralStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for voxtral."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "voxtral" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for voxtral."""
        config_path = Path(__file__).parent.parent.parent / "models" / "voxtral" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for voxtral."""
        req_path = Path(__file__).parent.parent.parent / "models" / "voxtral" / "requirements.txt"
        assert req_path.exists()
    
    def test_streaming_claim_true(self):
        """Verify claims has streaming=true."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "voxtral" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        assert claims.get("streaming") is True
    
    def test_asr_stream_claim_exists(self):
        """Verify asr_stream claim exists."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "voxtral" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        tasks = [c["task"] for c in claims.get("claims", [])]
        assert "asr_stream" in tasks
