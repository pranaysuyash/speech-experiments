"""Tests for ASR configuration resolution."""



class TestResolveASRConfig:
    """Test resolve_asr_config function."""

    def test_default_config(self):
        """Test default configuration."""
        from harness.asr import resolve_asr_config

        resolved = resolve_asr_config()

        assert resolved.model_id == "faster_whisper:large-v3"
        assert resolved.source == "hf"
        assert resolved.device == "cpu"  # Default fallback
        assert resolved.language == "auto"

    def test_model_size_maps_to_model_name(self):
        """Test that model_size is accepted and used as model_name."""
        from harness.asr import resolve_asr_config

        # UI sends model_size
        resolved = resolve_asr_config({"model_size": "base"})

        assert "base" in resolved.model_id
        assert resolved.model_id == "faster_whisper:base"

    def test_model_name_takes_precedence(self):
        """Test that model_name takes precedence over model_size."""
        from harness.asr import resolve_asr_config

        resolved = resolve_asr_config(
            {
                "model_name": "medium",
                "model_size": "base",  # Should be ignored
            }
        )

        assert "medium" in resolved.model_id
        assert resolved.model_id == "faster_whisper:medium"

    def test_device_preference_list(self):
        """Test device preference list handling."""
        from harness.asr import resolve_asr_config

        # MPS preference with faster_whisper should fall back to cpu
        resolved = resolve_asr_config(
            {"model_type": "faster_whisper", "device_preference": ["mps", "cpu"]}
        )

        # faster_whisper doesn't support MPS, should use cpu
        assert resolved.device == "cpu"

    def test_device_preference_cuda(self):
        """Test CUDA device selection."""
        from harness.asr import resolve_asr_config

        resolved = resolve_asr_config(
            {"model_type": "whisper", "device_preference": ["cuda", "cpu"]}
        )

        # Regular whisper supports CUDA
        assert resolved.device == "cuda"

    def test_language_override(self):
        """Test language override."""
        from harness.asr import resolve_asr_config

        resolved = resolve_asr_config({"language": "en"})

        assert resolved.language == "en"

    def test_model_type_selection(self):
        """Test different model types."""
        from harness.asr import resolve_asr_config

        # Whisper
        resolved = resolve_asr_config({"model_type": "whisper"})
        assert resolved.model_id.startswith("whisper:")
        assert resolved.source == "hf"

        # Distil-Whisper
        resolved = resolve_asr_config({"model_type": "distil_whisper"})
        assert resolved.model_id.startswith("distil_whisper:")
        assert resolved.source == "hf"

    def test_to_dict(self):
        """Test serialization to dict."""
        from harness.asr import resolve_asr_config

        resolved = resolve_asr_config(
            {"model_size": "small", "language": "es", "device_preference": ["cpu"]}
        )

        d = resolved.to_dict()

        assert d["model_id"] == "faster_whisper:small"
        assert d["language"] == "es"
        assert d["device"] == "cpu"
        assert "source" in d
        assert "reason" in d


class TestASRConfigIntegration:
    """Test ASR config integration with session runner."""

    def test_config_passthrough_from_session(self):
        """Test that session extra_config is passed to ASR."""
        # This is more of a documentation test showing the expected flow
        #
        # WorkbenchPage sends:
        #   { asr: { model_size: "base", language: "en" }, device_preference: ["mps", "cpu"] }
        #
        # SessionRunner extracts:
        #   self.extra_config.get("asr", {}) -> { model_size: "base", language: "en" }
        #
        # run_asr receives that config and resolve_asr_config handles it
        #
        # This test verifies the config shape is compatible

        ui_config = {
            "asr": {"model_size": "base", "language": "en"},
            "diarization": {"model_name": "pyannote_diarization"},
            "device_preference": ["mps", "cpu"],
        }

        # Extract ASR config as SessionRunner would
        asr_config = ui_config.get("asr", {})

        # Verify it can be resolved
        from harness.asr import resolve_asr_config

        resolved = resolve_asr_config(asr_config)

        assert resolved.model_id == "faster_whisper:base"
        assert resolved.language == "en"
