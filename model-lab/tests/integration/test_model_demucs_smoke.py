"""
Demucs model smoke test (LCS-11).

Tests that Demucs loads and produces valid separated stems.
First separate surface implementation.
Skips model tests if demucs package is not installed.
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


# Check if demucs is available
try:
    import demucs.api
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires torch"
)

needs_demucs = pytest.mark.skipif(
    not DEMUCS_AVAILABLE,
    reason="demucs not installed"
)


@needs_registry
@needs_demucs
class TestDemucsSmokeTest:
    """Smoke tests for Demucs source separation."""

    def test_demucs_registered(self):
        """Verify demucs is registered in the model registry."""
        models = ModelRegistry.list_models()
        assert "demucs" in models
    
    def test_demucs_metadata(self):
        """Verify demucs metadata has separate capability."""
        meta = ModelRegistry.get_model_metadata("demucs")
        assert "separate" in meta["capabilities"]
    
    def test_demucs_loads(self):
        """Verify demucs loads without error."""
        bundle = ModelRegistry.load_model("demucs", {}, device="cpu")
        
        assert bundle["model_type"] == "demucs"
        assert "separate" in bundle["capabilities"]
        assert "separate" in bundle["separate"]
    
    def test_demucs_all_stems_present(self):
        """Verify demucs produces all expected stems."""
        bundle = ModelRegistry.load_model("demucs", {}, device="cpu")
        separate = bundle["separate"]["separate"]
        
        # Create test audio (1 second at 44.1kHz)
        audio = np.random.randn(44100).astype(np.float32) * 0.3
        
        result = separate(audio, sr=44100)
        
        # All expected stems must be present
        expected_stems = {"vocals", "drums", "bass", "other"}
        assert "stems" in result
        assert set(result["stems"].keys()) >= expected_stems
    
    def test_demucs_stems_are_arrays(self):
        """Verify all stems are numpy arrays."""
        bundle = ModelRegistry.load_model("demucs", {}, device="cpu")
        separate = bundle["separate"]["separate"]
        
        audio = np.random.randn(44100).astype(np.float32) * 0.3
        result = separate(audio, sr=44100)
        
        for name, stem in result["stems"].items():
            assert isinstance(stem, np.ndarray), f"Stem {name} not ndarray"
            assert np.issubdtype(stem.dtype, np.floating), f"Stem {name} not float"
    
    def test_demucs_length_alignment(self):
        """Verify all stems match input length."""
        bundle = ModelRegistry.load_model("demucs", {}, device="cpu")
        separate = bundle["separate"]["separate"]
        
        input_length = 44100  # 1 second
        audio = np.random.randn(input_length).astype(np.float32) * 0.3
        
        result = separate(audio, sr=44100)
        
        for name, stem in result["stems"].items():
            assert len(stem) == input_length, (
                f"Stem {name} length {len(stem)} != input {input_length}"
            )
    
    def test_demucs_sr_matches(self):
        """Verify returned sr matches expected."""
        bundle = ModelRegistry.load_model("demucs", {}, device="cpu")
        separate = bundle["separate"]["separate"]
        
        audio = np.random.randn(44100).astype(np.float32) * 0.3
        result = separate(audio, sr=44100)
        
        assert result["sr"] == 44100


class TestDemucsStructural:
    """Structural tests that don't require loading the model."""

    def test_claims_file_exists(self):
        """Verify claims.yaml exists for demucs."""
        claims_path = Path(__file__).parent.parent.parent / "models" / "demucs" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        """Verify config.yaml exists for demucs."""
        config_path = Path(__file__).parent.parent.parent / "models" / "demucs" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        """Verify requirements.txt exists for demucs."""
        req_path = Path(__file__).parent.parent.parent / "models" / "demucs" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_has_separate_surface(self):
        """Verify claims define separate surface."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "demucs" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        tasks = [c["task"] for c in claims.get("claims", [])]
        assert "separate" in tasks
    
    def test_config_has_stems_list(self):
        """Verify config defines expected stems."""
        import yaml
        config_path = Path(__file__).parent.parent.parent / "models" / "demucs" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        stems = config.get("config", {}).get("stems", [])
        assert "vocals" in stems
        assert "drums" in stems
        assert "bass" in stems
        assert "other" in stems


class TestDemucsSyntheticMix:
    """Test with synthetic audio mix (no model required)."""
    
    def test_synthetic_mix_structure(self):
        """Create synthetic mix to validate test infrastructure."""
        # Two sine waves at different frequencies
        sr = 44100
        t = np.linspace(0, 1, sr)
        
        # "Vocals" - higher frequency
        vocals = np.sin(2 * np.pi * 440 * t) * 0.5
        
        # "Bass" - lower frequency
        bass = np.sin(2 * np.pi * 100 * t) * 0.5
        
        # Mix
        mix = vocals + bass
        
        # Verify mix is valid audio
        assert len(mix) == sr
        assert mix.dtype == np.float64
        assert np.max(np.abs(mix)) <= 1.0
