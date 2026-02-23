"""
Kokoro-TTS smoke test.

By default, heavy runtime tests are skipped unless RUN_KOKORO_SMOKE=1.
"""

import os
from pathlib import Path

import pytest

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from harness.registry import ModelRegistry

    REGISTRY_AVAILABLE = True
except ImportError:
    ModelRegistry = None
    REGISTRY_AVAILABLE = False


needs_registry = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="registry requires runtime dependencies",
)

run_heavy = pytest.mark.skipif(
    os.environ.get("RUN_KOKORO_SMOKE") != "1",
    reason="set RUN_KOKORO_SMOKE=1 to run live Kokoro model loading",
)


class TestKokoroTTSStructural:
    """Structural checks that should run in CI."""

    def test_claims_file_exists(self):
        claims_path = Path(__file__).parent.parent.parent / "models" / "kokoro_tts" / "claims.yaml"
        assert claims_path.exists()

    def test_config_file_exists(self):
        config_path = Path(__file__).parent.parent.parent / "models" / "kokoro_tts" / "config.yaml"
        assert config_path.exists()

    def test_requirements_file_exists(self):
        req_path = (
            Path(__file__).parent.parent.parent / "models" / "kokoro_tts" / "requirements.txt"
        )
        assert req_path.exists()


@needs_registry
class TestKokoroTTSRegistry:
    def test_model_registered(self):
        models = ModelRegistry.list_models()
        assert "kokoro_tts" in models

    def test_model_metadata_has_tts_capability(self):
        meta = ModelRegistry.get_model_metadata("kokoro_tts")
        assert "tts" in meta["capabilities"]


@needs_registry
@run_heavy
class TestKokoroTTSRuntime:
    def test_model_loads(self):
        bundle = ModelRegistry.load_model("kokoro_tts", {}, device="cpu")
        assert bundle["model_type"] == "kokoro_tts"
        assert "tts" in bundle["capabilities"]

    def test_synthesize_returns_audio(self):
        bundle = ModelRegistry.load_model("kokoro_tts", {}, device="cpu")
        result = bundle["tts"]["synthesize"]("Hello from Kokoro")

        assert isinstance(result, dict)
        assert "audio" in result
        assert "sample_rate" in result
        assert isinstance(result["audio"], np.ndarray)
        assert result["sample_rate"] > 0
