"""Tests for structured error handling."""


class TestErrorClassification:
    """Test error classification logic."""

    def test_classify_oom_error(self):
        """Test OOM error classification."""
        from harness.errors import E_MODEL_OOM, classify_error

        # CUDA OOM
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        code, recoverable = classify_error(exc)
        assert code == E_MODEL_OOM
        assert recoverable is False

    def test_classify_memory_error(self):
        """Test MemoryError classification."""
        from harness.errors import E_MODEL_OOM, classify_error

        exc = MemoryError("Unable to allocate array")
        code, recoverable = classify_error(exc)
        assert code == E_MODEL_OOM
        assert recoverable is False

    def test_classify_model_not_found(self):
        """Test model not found classification."""
        from harness.errors import E_MODEL_NOT_FOUND, classify_error

        exc = FileNotFoundError("Model checkpoint not found: /path/to/model.pt")
        code, recoverable = classify_error(exc)
        assert code == E_MODEL_NOT_FOUND
        assert recoverable is False

    def test_classify_audio_corrupt(self):
        """Test audio corruption classification."""
        from harness.errors import E_AUDIO_CORRUPT, classify_error

        exc = RuntimeError("Failed to decode audio: corrupt wav header")
        code, recoverable = classify_error(exc)
        assert code == E_AUDIO_CORRUPT
        assert recoverable is False

    def test_classify_timeout(self):
        """Test timeout classification."""
        from harness.errors import E_STEP_TIMEOUT, classify_error

        exc = TimeoutError("Step exceeded time limit")
        code, recoverable = classify_error(exc)
        assert code == E_STEP_TIMEOUT
        assert recoverable is True  # Timeouts may succeed on retry

    def test_classify_disk_full(self):
        """Test disk full classification."""
        from harness.errors import E_DISK_FULL, classify_error

        exc = OSError("No space left on device")
        code, recoverable = classify_error(exc)
        assert code == E_DISK_FULL
        assert recoverable is False

    def test_classify_permission_denied(self):
        """Test permission denied classification."""
        from harness.errors import E_PERMISSION_DENIED, classify_error

        exc = PermissionError("Permission denied: /path/to/file")
        code, recoverable = classify_error(exc)
        assert code == E_PERMISSION_DENIED
        assert recoverable is False

    def test_classify_network_error(self):
        """Test network error classification."""
        from harness.errors import E_NETWORK_ERROR, classify_error

        exc = ConnectionError("Connection refused")
        code, recoverable = classify_error(exc)
        assert code == E_NETWORK_ERROR
        assert recoverable is True  # Network errors may be transient

    def test_classify_unknown_error(self):
        """Test unknown error classification."""
        from harness.errors import E_UNKNOWN, classify_error

        exc = ValueError("Some random error")
        code, recoverable = classify_error(exc)
        assert code == E_UNKNOWN
        assert recoverable is False


class TestStepError:
    """Test StepError dataclass."""

    def test_step_error_creation(self):
        """Test creating a StepError."""
        from harness.errors import E_MODEL_OOM, StepError

        error = StepError(
            code=E_MODEL_OOM,
            message="CUDA out of memory",
            step="asr",
            traceback_path="logs/asr_traceback.txt",
            recoverable=False,
        )

        assert error.code == E_MODEL_OOM
        assert error.step == "asr"
        assert error.recoverable is False

    def test_step_error_to_dict(self):
        """Test StepError serialization."""
        from harness.errors import E_STEP_TIMEOUT, StepError

        error = StepError(
            code=E_STEP_TIMEOUT,
            message="Step exceeded time limit",
            step="diarization",
            recoverable=True,
        )

        d = error.to_dict()
        assert d["code"] == E_STEP_TIMEOUT
        assert d["message"] == "Step exceeded time limit"
        assert d["step"] == "diarization"
        assert d["recoverable"] is True
        # None values should be excluded
        assert "traceback_path" not in d
        assert "details" not in d


class TestCreateStepError:
    """Test create_step_error utility."""

    def test_create_step_error_from_exception(self):
        """Test creating StepError from exception."""
        from harness.errors import E_MODEL_OOM, create_step_error

        exc = RuntimeError("CUDA out of memory")
        error = create_step_error(exc, step="asr")

        assert error.code == E_MODEL_OOM
        assert error.step == "asr"
        assert "CUDA out of memory" in error.message
        assert error.details["exception_type"] == "RuntimeError"

    def test_create_step_error_with_traceback_path(self):
        """Test creating StepError with traceback path."""
        from harness.errors import create_step_error

        exc = ValueError("Test error")
        error = create_step_error(exc, step="ingest", traceback_path="logs/ingest_traceback.txt")

        assert error.traceback_path == "logs/ingest_traceback.txt"

    def test_message_truncation(self):
        """Test that long messages are truncated."""
        from harness.errors import create_step_error

        long_message = "x" * 1000
        exc = ValueError(long_message)
        error = create_step_error(exc, step="test")

        assert len(error.message) <= 500


class TestFormatTraceback:
    """Test traceback formatting."""

    def test_format_traceback(self):
        """Test formatting exception traceback."""
        from harness.errors import format_traceback

        try:
            raise ValueError("Test error")
        except ValueError as e:
            tb = format_traceback(e)

        assert "ValueError" in tb
        assert "Test error" in tb
        assert "Traceback" in tb
