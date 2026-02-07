"""
Basic Pitch model smoke test (LCS-12).

Tests that Basic Pitch loads and produces valid note transcription.
First music_transcription surface implementation.
Skips model tests if basic-pitch package is not installed.
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


# Check if basic-pitch is available
try:
    from basic_pitch.inference import predict
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)

needs_basic_pitch = pytest.mark.skipif(
    not BASIC_PITCH_AVAILABLE,
    reason="basic-pitch not installed"
)


@needs_registry
@needs_basic_pitch
class TestBasicPitchSmokeTest:
    """Smoke tests for Basic Pitch music transcription."""

    def test_basic_pitch_registered(self):
        """Verify basic_pitch is registered in the model registry."""
        models = ModelRegistry.list_models()
        assert "basic_pitch" in models
    
    def test_basic_pitch_metadata(self):
        """Verify basic_pitch metadata has music_transcription capability."""
        meta = ModelRegistry.get_model_metadata("basic_pitch")
        assert "music_transcription" in meta["capabilities"]
    
    def test_basic_pitch_loads(self):
        """Verify basic_pitch loads without error."""
        bundle = ModelRegistry.load_model("basic_pitch", {}, device="cpu")
        
        assert bundle["model_type"] == "basic_pitch"
        assert "music_transcription" in bundle["capabilities"]
        assert "transcribe" in bundle["music_transcription"]
    
    def test_basic_pitch_notes_list_present(self):
        """Verify output contains notes list."""
        bundle = ModelRegistry.load_model("basic_pitch", {}, device="cpu")
        transcribe = bundle["music_transcription"]["transcribe"]
        
        # Create simple sine wave (A4 = 440Hz)
        sr = 22050
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        
        result = transcribe(audio, sr=sr)
        
        assert "notes" in result
        assert isinstance(result["notes"], list)
    
    def test_basic_pitch_notes_have_required_fields(self):
        """Verify each note has onset, offset, pitch."""
        bundle = ModelRegistry.load_model("basic_pitch", {}, device="cpu")
        transcribe = bundle["music_transcription"]["transcribe"]
        
        # Simple sine wave
        sr = 22050
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        
        result = transcribe(audio, sr=sr)
        
        # If notes detected, verify structure
        for note in result["notes"]:
            assert "onset" in note, "Note missing onset"
            assert "offset" in note, "Note missing offset"
            assert "pitch" in note, "Note missing pitch"
            assert isinstance(note["onset"], (int, float))
            assert isinstance(note["offset"], (int, float))
            assert isinstance(note["pitch"], int)
            assert note["offset"] >= note["onset"], "offset < onset"
    
    def test_basic_pitch_output_structure_deterministic(self):
        """Verify output structure is consistent."""
        bundle = ModelRegistry.load_model("basic_pitch", {}, device="cpu")
        transcribe = bundle["music_transcription"]["transcribe"]
        
        # Random audio (unlikely to produce notes, but structure should be valid)
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        
        result = transcribe(audio, sr=22050)
        
        # Even with no notes, structure should be present
        assert "notes" in result
        assert isinstance(result["notes"], list)


class TestBasicPitchStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for basic_pitch."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "basic_pitch" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for basic_pitch."""
        config_path = Path(__file__).parent.parent.parent / "models" / "basic_pitch" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for basic_pitch."""
        req_path = Path(__file__).parent.parent.parent / "models" / "basic_pitch" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_has_music_transcription_surface(self):
        """Verify claims define music_transcription surface."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "basic_pitch" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        tasks = [c["task"] for c in claims.get("claims", [])]
        assert "music_transcription" in tasks
