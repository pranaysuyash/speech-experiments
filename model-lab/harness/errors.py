"""
Structured error types for pipeline execution.

Provides:
- Error codes for common failure modes
- StepError dataclass for structured error reporting
- Error classification utilities
"""

import traceback
from dataclasses import asdict, dataclass
from typing import Any

# ============================================================================
# ERROR CODES
# ============================================================================

# Model loading errors
E_MODEL_NOT_FOUND = "E_MODEL_NOT_FOUND"
E_MODEL_OOM = "E_MODEL_OOM"
E_MODEL_LOAD_FAILED = "E_MODEL_LOAD_FAILED"

# Audio errors
E_AUDIO_CORRUPT = "E_AUDIO_CORRUPT"
E_AUDIO_TOO_SHORT = "E_AUDIO_TOO_SHORT"
E_AUDIO_TOO_LONG = "E_AUDIO_TOO_LONG"
E_AUDIO_FORMAT_UNSUPPORTED = "E_AUDIO_FORMAT_UNSUPPORTED"

# Pipeline errors
E_STEP_TIMEOUT = "E_STEP_TIMEOUT"
E_DEPENDENCY_FAILED = "E_DEPENDENCY_FAILED"
E_STEP_SKIPPED = "E_STEP_SKIPPED"

# System errors
E_DISK_FULL = "E_DISK_FULL"
E_PERMISSION_DENIED = "E_PERMISSION_DENIED"
E_NETWORK_ERROR = "E_NETWORK_ERROR"

# Catch-all
E_UNKNOWN = "E_UNKNOWN"


# ============================================================================
# STRUCTURED ERROR TYPE
# ============================================================================


@dataclass
class StepError:
    """
    Structured error information for a failed step.

    Attributes:
        code: Error code (E_* constant)
        message: Human-readable error message
        step: Name of the step that failed
        traceback_path: Relative path to traceback file (if saved)
        recoverable: Whether the error might succeed on retry
        details: Additional error context
    """

    code: str
    message: str
    step: str
    traceback_path: str | None = None
    recoverable: bool = False
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None}


# ============================================================================
# ERROR CLASSIFICATION
# ============================================================================


def classify_error(exception: Exception) -> tuple[str, bool]:
    """
    Classify an exception into an error code and recoverable flag.

    Args:
        exception: The exception to classify

    Returns:
        (error_code, is_recoverable)
    """
    exc_type = type(exception).__name__
    exc_msg = str(exception).lower()

    # Memory errors - potentially recoverable with smaller model
    if (
        exc_type in ("OutOfMemoryError", "MemoryError")
        or "cuda out of memory" in exc_msg
        or "oom" in exc_msg
    ):
        return E_MODEL_OOM, False  # OOM usually not recoverable without config change

    # Model not found
    if exc_type == "FileNotFoundError" and ("model" in exc_msg or "checkpoint" in exc_msg):
        return E_MODEL_NOT_FOUND, False

    # Audio errors
    if "corrupt" in exc_msg or "decode" in exc_msg or "invalid" in exc_msg:
        if "audio" in exc_msg or "wav" in exc_msg or "mp3" in exc_msg:
            return E_AUDIO_CORRUPT, False

    if "too short" in exc_msg:
        return E_AUDIO_TOO_SHORT, False

    # Timeout
    if exc_type == "TimeoutError" or "timeout" in exc_msg:
        return E_STEP_TIMEOUT, True  # Timeout might succeed on retry

    # Disk errors
    if exc_type == "OSError" and ("no space" in exc_msg or "disk full" in exc_msg):
        return E_DISK_FULL, False

    if exc_type == "PermissionError":
        return E_PERMISSION_DENIED, False

    # Network errors - often recoverable
    if "connection" in exc_msg or "network" in exc_msg or "timeout" in exc_msg:
        if exc_type in ("ConnectionError", "TimeoutError", "URLError"):
            return E_NETWORK_ERROR, True

    # Model loading failures
    if "load" in exc_msg and "model" in exc_msg:
        return E_MODEL_LOAD_FAILED, True  # Might be transient

    # Default: unknown, not recoverable
    return E_UNKNOWN, False


def format_traceback(exception: Exception) -> str:
    """Format exception with full traceback."""
    return "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))


def create_step_error(
    exception: Exception,
    step: str,
    traceback_path: str | None = None,
) -> StepError:
    """
    Create a StepError from an exception.

    Args:
        exception: The exception that occurred
        step: Name of the step that failed
        traceback_path: Path where traceback was saved (if any)

    Returns:
        StepError instance
    """
    code, recoverable = classify_error(exception)

    return StepError(
        code=code,
        message=str(exception)[:500],  # Truncate long messages
        step=step,
        traceback_path=traceback_path,
        recoverable=recoverable,
        details={
            "exception_type": type(exception).__name__,
        },
    )
