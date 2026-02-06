"""
Event streaming for real-time pipeline progress.

Provides:
- EventWriter: Writes events to a JSONL file for streaming
- Event types for run lifecycle and step progress
"""

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ============================================================================
# EVENT TYPES
# ============================================================================

EVENT_RUN_STARTED = "run_started"
EVENT_RUN_COMPLETED = "run_completed"
EVENT_RUN_FAILED = "run_failed"
EVENT_RUN_CANCELLED = "run_cancelled"

EVENT_STEP_STARTED = "step_started"
EVENT_STEP_PROGRESS = "step_progress"
EVENT_STEP_COMPLETED = "step_completed"
EVENT_STEP_FAILED = "step_failed"
EVENT_STEP_SKIPPED = "step_skipped"

EVENT_HEARTBEAT = "heartbeat"


# ============================================================================
# EVENT DATA CLASSES
# ============================================================================


@dataclass
class RunEvent:
    """Base event structure."""

    ts: str
    type: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class StepProgressEvent(RunEvent):
    """Event for step progress updates."""

    step: str
    progress: float  # 0.0 to 1.0
    message: str | None = None
    estimated_remaining_s: int | None = None


@dataclass
class StepCompletedEvent(RunEvent):
    """Event for step completion."""

    step: str
    duration_ms: int
    artifacts: list | None = None


@dataclass
class StepFailedEvent(RunEvent):
    """Event for step failure."""

    step: str
    error_code: str
    error_message: str
    duration_ms: int


@dataclass
class RunCompletedEvent(RunEvent):
    """Event for run completion."""

    status: str  # COMPLETED | FAILED | CANCELLED
    total_duration_ms: int
    steps_completed: int
    steps_failed: int


# ============================================================================
# EVENT WRITER
# ============================================================================


class EventWriter:
    """
    Writes events to a JSONL file for streaming.

    Events are appended atomically and can be tailed by the WebSocket endpoint.
    Thread-safe for concurrent writes.
    """

    def __init__(self, run_dir: Path, run_id: str):
        """
        Initialize event writer.

        Args:
            run_dir: Directory for the run
            run_id: Unique run identifier
        """
        self.run_dir = run_dir
        self.run_id = run_id
        self.events_file = run_dir / "events.jsonl"
        self._lock = threading.Lock()

        # Ensure directory exists
        run_dir.mkdir(parents=True, exist_ok=True)

    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def emit(self, event_type: str, **payload) -> None:
        """
        Emit an event to the events file.

        Args:
            event_type: Type of event (use EVENT_* constants)
            **payload: Additional event data
        """
        event = {"ts": self._now_iso(), "type": event_type, "run_id": self.run_id, **payload}

        with self._lock:
            with self.events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
                f.flush()  # Ensure immediate write for tailing

    def emit_run_started(self, steps: list[str]) -> None:
        """Emit run started event."""
        self.emit(EVENT_RUN_STARTED, steps=steps)

    def emit_run_completed(
        self, status: str, total_duration_ms: int, steps_completed: int = 0, steps_failed: int = 0
    ) -> None:
        """Emit run completed event."""
        self.emit(
            EVENT_RUN_COMPLETED,
            status=status,
            total_duration_ms=total_duration_ms,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
        )

    def emit_step_started(self, step: str) -> None:
        """Emit step started event."""
        self.emit(EVENT_STEP_STARTED, step=step, progress=0.0)

    def emit_step_progress(
        self,
        step: str,
        progress: float,
        message: str | None = None,
        estimated_remaining_s: int | None = None,
    ) -> None:
        """
        Emit step progress event.

        Args:
            step: Step name
            progress: Progress 0.0 to 1.0
            message: Optional progress message
            estimated_remaining_s: Optional estimated seconds remaining
        """
        payload = {"step": step, "progress": progress}
        if message:
            payload["message"] = message
        if estimated_remaining_s is not None:
            payload["estimated_remaining_s"] = estimated_remaining_s
        self.emit(EVENT_STEP_PROGRESS, **payload)

    def emit_step_completed(
        self, step: str, duration_ms: int, artifacts: list | None = None
    ) -> None:
        """Emit step completed event."""
        payload = {"step": step, "duration_ms": duration_ms, "progress": 1.0}
        if artifacts:
            payload["artifacts"] = artifacts
        self.emit(EVENT_STEP_COMPLETED, **payload)

    def emit_step_failed(
        self, step: str, error_code: str, error_message: str, duration_ms: int
    ) -> None:
        """Emit step failed event."""
        self.emit(
            EVENT_STEP_FAILED,
            step=step,
            error_code=error_code,
            error_message=error_message[:200],  # Truncate
            duration_ms=duration_ms,
        )

    def emit_heartbeat(self, current_step: str | None = None) -> None:
        """Emit heartbeat event to keep connection alive."""
        payload = {}
        if current_step:
            payload["current_step"] = current_step
        self.emit(EVENT_HEARTBEAT, **payload)


def get_events_file(run_dir: Path) -> Path:
    """Get the events file path for a run directory."""
    return run_dir / "events.jsonl"


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    """
    Read all events from a run's events file.

    Args:
        run_dir: Run directory

    Returns:
        List of event dictionaries
    """
    events_file = get_events_file(run_dir)
    if not events_file.exists():
        return []

    events = []
    with events_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Skip malformed lines
    return events
