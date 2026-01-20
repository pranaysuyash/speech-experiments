import pytest
from harness.asr import resolve_asr_config

class TestDeviceSelection:
    
    def test_basic_preference(self):
        """Should pick first available"""
        config = {"model_type": "distil_whisper", "device_preference": ["cuda", "cpu"]}
        res = resolve_asr_config(config)
        assert res.device == "cuda"
        assert res.reason == "preference_cuda"

    def test_faster_whisper_skips_mps(self):
        """faster_whisper should skip mps and fallback"""
        config = {"model_type": "faster_whisper", "device_preference": ["mps", "cpu"]}
        res = resolve_asr_config(config)
        assert res.device == "cpu"
        assert res.reason == "preference_cpu"
        
    def test_faster_whisper_allows_cuda(self):
        config = {"model_type": "faster_whisper", "device_preference": ["cuda", "cpu"]}
        res = resolve_asr_config(config)
        assert res.device == "cuda"
        assert res.reason == "preference_cuda"

    def test_lfm_skips_cuda(self):
        """lfm should skip cuda"""
        config = {"model_type": "lfm2_5_audio", "device_preference": ["cuda", "mps", "cpu"]}
        res = resolve_asr_config(config)
        assert res.device == "mps"
        assert res.reason == "preference_mps"

    def test_legacy_device_string(self):
        """Should handle legacy 'device' key"""
        config = {"model_type": "distil_whisper", "device": "mps"}
        res = resolve_asr_config(config)
        assert res.device == "mps"
        assert res.reason == "preference_mps"

    def test_legacy_device_fallback(self):
        """Should still apply checks to legacy key"""
        config = {"model_type": "faster_whisper", "device": "mps"}
        res = resolve_asr_config(config)
        assert res.device == "cpu"
        # Since I convert legacy "device" to "device_preference": [device], the loop logic applies.
        # It skips mps, then falls back to default logic at end: "exhausted_preference"
        assert res.reason == "exhausted_preference"

    def test_default_fallback(self):
        """Should default to cpu if empty"""
        config = {"model_type": "any_model"}
        res = resolve_asr_config(config)
        assert res.device == "cpu"
        assert res.reason == "preference_cpu"
