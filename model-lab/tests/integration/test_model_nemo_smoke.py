"""
Nemotron Streaming and Parakeet Multitalker smoke tests (LCS-18, LCS-20).

Tests NeMo-based models. Both require dedicated venvs.
"""

import pytest
from pathlib import Path


# Structural tests only - NeMo too heavy for CI

class TestNemotronStreamingStructural:
    """Structural tests for Nemotron Streaming."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "nemotron_streaming" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "nemotron_streaming" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "nemotron_streaming" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_streaming_true(self):
        """Verify streaming=true."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "nemotron_streaming" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("streaming") is True
    
    def test_claims_nemo_runtime(self):
        """Verify NeMo runtime."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "nemotron_streaming" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("runtime") == "nemo"


class TestParakeetMultitalkerStructural:
    """Structural tests for Parakeet Multitalker."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "parakeet_multitalker" / "claims.yaml"
        assert claims_path.exists()
    
    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "parakeet_multitalker" / "config.yaml"
        assert config_path.exists()
    
    def test_requirements_file_exists(self):
        req_path = Path(__file__).parent.parent.parent / "models" / "parakeet_multitalker" / "requirements.txt"
        assert req_path.exists()
    
    def test_claims_nemo_runtime(self):
        """Verify NeMo runtime."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "parakeet_multitalker" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        assert claims.get("runtime") == "nemo"
    
    def test_claims_has_diarization(self):
        """Verify diarization claim."""
        import yaml
        claims_path = Path(__file__).parent.parent.parent / "models" / "parakeet_multitalker" / "claims.yaml"
        with open(claims_path) as f:
            claims = yaml.safe_load(f)
        
        claim_ids = [c["id"] for c in claims.get("claims", [])]
        assert "parakeet_multitalker_diarization" in claim_ids
