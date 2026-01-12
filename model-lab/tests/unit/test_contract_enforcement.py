"""
Tests for Bundle Contract v1 enforcement.
These tests verify that the contract is correctly implemented and enforced.
"""

import pytest
from pathlib import Path


class TestBundleContract:
    """Test Bundle Contract v1 enforcement."""
    
    def test_contracts_import(self):
        """Verify contracts module imports correctly."""
        from harness.contracts import Bundle, validate_bundle, ASRResult
        assert Bundle is not None
        assert validate_bundle is not None
    
    def test_registry_returns_valid_bundles(self):
        """Verify all registered loaders return valid Bundle shape."""
        from harness.registry import ModelRegistry
        from harness.contracts import validate_bundle
        
        for model_id in ModelRegistry.list_models():
            meta = ModelRegistry.get_model_metadata(model_id)
            assert 'capabilities' in meta
            assert 'status' in meta
            assert 'hash' in meta
            # Hash should be stable (8 chars, no datetime component)
            assert len(meta['hash']) == 8
    
    def test_asr_capability_requires_transcribe(self):
        """ASR capability must have transcribe() callable."""
        from harness.contracts import validate_bundle
        
        # Valid bundle
        valid_bundle = {
            "model_type": "test",
            "device": "cpu",
            "capabilities": ["asr"],
            "asr": {"transcribe": lambda x, sr: {"text": "test"}}
        }
        # Should not raise
        validate_bundle(valid_bundle, "test")
        
        # Invalid: missing transcribe
        invalid_bundle = {
            "model_type": "test",
            "device": "cpu", 
            "capabilities": ["asr"],
            "asr": {"transcribe_path": lambda x: {"text": "test"}}  # Only path, no transcribe
        }
        with pytest.raises(ValueError, match="requires 'transcribe' callable"):
            validate_bundle(invalid_bundle, "test")
    
    def test_bundle_missing_keys_raises(self):
        """Bundle must have required keys."""
        from harness.contracts import validate_bundle
        
        incomplete = {"model_type": "test"}
        with pytest.raises(ValueError, match="missing required keys"):
            validate_bundle(incomplete, "test")
    
    def test_bundle_wrong_type_raises(self):
        """Bundle must be a dict."""
        from harness.contracts import validate_bundle
        
        with pytest.raises(TypeError, match="must return dict"):
            validate_bundle("not a dict", "test")


class TestRunnerContractCompliance:
    """Test that runner scripts use bundle contract correctly."""
    
    def test_run_asr_has_no_per_model_functions(self):
        """run_asr.py must not have per-model transcribe functions."""
        run_asr_path = Path(__file__).parent.parent.parent / "scripts" / "run_asr.py"
        
        if not run_asr_path.exists():
            pytest.skip("run_asr.py not found")
        
        content = run_asr_path.read_text()
        
        # These patterns indicate per-model special casing (bad)
        forbidden_patterns = [
            "def transcribe_whisper",
            "def transcribe_faster_whisper",
            "def transcribe_seamless",
            "def transcribe_lfm",
            'model_wrapper["model"]',
            "model_wrapper['model']",
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in content, f"Found forbidden pattern: {pattern}"
    
    def test_run_asr_uses_bundle_contract(self):
        """run_asr.py must use bundle contract."""
        run_asr_path = Path(__file__).parent.parent.parent / "scripts" / "run_asr.py"
        
        if not run_asr_path.exists():
            pytest.skip("run_asr.py not found")
        
        content = run_asr_path.read_text()
        
        # Must have bundle contract usage
        assert 'bundle["asr"]["transcribe"]' in content or "bundle['asr']['transcribe']" in content


class TestAudioIOMetadata:
    """Test audio_io.py metadata handling."""
    
    def test_original_sample_rate_captured_before_resample(self):
        """original_sample_rate must reflect source file, not resampled rate."""
        import numpy as np
        import tempfile
        import soundfile as sf
        from harness.audio_io import AudioLoader
        
        # Create test audio at 44100 Hz
        original_sr = 44100
        duration = 0.5
        audio = np.random.randn(int(original_sr * duration)).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, original_sr)
            temp_path = Path(f.name)
        
        try:
            # Load with resampling to 16000
            loader = AudioLoader(target_sample_rate=16000)
            loaded_audio, sr, metadata = loader.load_audio(temp_path, "whisper")
            
            # Returned sr should be resampled rate
            assert sr == 16000
            
            # Metadata should capture ORIGINAL rate
            assert metadata['original_sample_rate'] == original_sr
            assert metadata['sample_rate'] == 16000
        finally:
            temp_path.unlink()
