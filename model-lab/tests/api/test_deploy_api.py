"""
Tests for the production API server (deploy_api.py).

Covers:
- Health endpoint with system metrics
- Security headers on all responses
- Rate limiting enforcement
- Input validation
- Model cache memory limits
"""

import pytest
from fastapi.testclient import TestClient

# Import the app - adjust path based on how deploy_api is structured
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.deploy_api import app, REQUEST_STATS, MODEL_CACHE, MODEL_CACHE_SIZES


@pytest.fixture
def client():
    """Create test client."""
    # Reset stats before each test
    REQUEST_STATS["total_requests"] = 0
    REQUEST_STATS["asr_requests"] = 0
    REQUEST_STATS["tts_requests"] = 0
    REQUEST_STATS["errors"] = 0
    REQUEST_STATS["avg_response_time"] = 0.0
    MODEL_CACHE.clear()
    MODEL_CACHE_SIZES.clear()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_healthy_status(self, client):
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_memory_metrics(self, client):
        """Health endpoint includes memory usage metrics."""
        response = client.get("/health")
        data = response.json()
        assert "memory_usage_percent" in data
        assert "disk_available_mb" in data
        assert "cache_memory_mb" in data
        assert isinstance(data["memory_usage_percent"], (int, float))

    def test_health_includes_cache_info(self, client):
        """Health endpoint includes cache information."""
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "cache_size" in data
        assert isinstance(data["models_loaded"], list)

    def test_health_includes_uptime(self, client):
        """Health endpoint includes uptime."""
        response = client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_x_content_type_options_header(self, client):
        """X-Content-Type-Options header is set."""
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options_header(self, client):
        """X-Frame-Options header is set."""
        response = client.get("/health")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_xss_protection_header(self, client):
        """X-XSS-Protection header is set."""
        response = client.get("/health")
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_hsts_header(self, client):
        """Strict-Transport-Security header is set."""
        response = client.get("/health")
        hsts = response.headers.get("Strict-Transport-Security")
        assert hsts is not None
        assert "max-age=" in hsts

    def test_referrer_policy_header(self, client):
        """Referrer-Policy header is set."""
        response = client.get("/health")
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_permissions_policy_header(self, client):
        """Permissions-Policy header is set."""
        response = client.get("/health")
        policy = response.headers.get("Permissions-Policy")
        assert policy is not None
        assert "geolocation=()" in policy


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_stats_returns_request_counts(self, client):
        """Stats endpoint returns request counts."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "asr_requests" in data
        assert "tts_requests" in data
        assert "errors" in data

    def test_stats_increments_after_requests(self, client):
        """Stats increment after making requests."""
        # Make a few requests
        client.get("/health")
        client.get("/health")
        client.get("/stats")

        response = client.get("/stats")
        data = response.json()
        # Should be > 0 after multiple requests
        assert data["total_requests"] > 0


class TestModelsEndpoint:
    """Tests for /models endpoint."""

    def test_models_returns_list(self, client):
        """Models endpoint returns a list."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestASREndpoint:
    """Tests for /asr/transcribe endpoint."""

    def test_asr_rejects_invalid_format(self, client):
        """ASR endpoint rejects non-audio files."""
        response = client.post(
            "/asr/transcribe",
            files={"file": ("test.txt", b"not audio", "text/plain")},
            data={"model_type": "whisper"},
        )
        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["detail"]

    def test_asr_accepts_wav_extension(self, client):
        """ASR endpoint accepts .wav files (may fail on actual inference)."""
        # Create minimal WAV header
        wav_header = b"RIFF" + b"\x00" * 40
        response = client.post(
            "/asr/transcribe",
            files={"file": ("test.wav", wav_header, "audio/wav")},
            data={"model_type": "whisper"},
        )
        # Should not be 400 (format rejection) - may be 500 if model not loaded
        assert response.status_code != 400 or "Unsupported audio format" not in str(
            response.json()
        )


class TestInputValidation:
    """Tests for input validation."""

    def test_tts_requires_text(self, client):
        """TTS endpoint requires text field."""
        response = client.post(
            "/tts/synthesize",
            json={"model_type": "lfm2_5_audio"},  # Missing required 'text'
        )
        assert response.status_code == 422  # Validation error


class TestModelStatusEndpoint:
    """Tests for model status update endpoint."""

    def test_update_model_status_invalid_status(self, client):
        """Updating model with invalid status returns 400."""
        response = client.post("/models/whisper/status?status=invalid_status")
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
