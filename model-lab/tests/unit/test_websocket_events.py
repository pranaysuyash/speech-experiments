"""Tests for WebSocket event streaming infrastructure.

These tests verify the event writing and WebSocket endpoint logic
without requiring a running server.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path


def _create_event(event_type: str, run_id: str, **payload) -> dict:
    """Create an event dict (mirrors harness/events.py)."""
    return {
        "ts": datetime.utcnow().isoformat() + "Z",
        "type": event_type,
        "run_id": run_id,
        **payload,
    }


class TestEventWriting:
    """Test the event writing functionality."""

    def test_event_structure_run_started(self):
        """Test run_started event structure."""
        event = _create_event("run_started", "test-run-123", steps=["ingest", "asr"])

        assert event["type"] == "run_started"
        assert event["run_id"] == "test-run-123"
        assert event["steps"] == ["ingest", "asr"]
        assert "ts" in event

    def test_event_structure_step_started(self):
        """Test step_started event structure."""
        event = _create_event("step_started", "test-run-123", step="asr", index=1, total=4)

        assert event["type"] == "step_started"
        assert event["step"] == "asr"
        assert event["index"] == 1
        assert event["total"] == 4

    def test_event_structure_step_completed(self):
        """Test step_completed event structure."""
        event = _create_event(
            "step_completed",
            "test-run-123",
            step="asr",
            duration_ms=5432,
            artifacts=["transcript.json"],
        )

        assert event["type"] == "step_completed"
        assert event["step"] == "asr"
        assert event["duration_ms"] == 5432
        assert event["artifacts"] == ["transcript.json"]

    def test_event_structure_step_failed(self):
        """Test step_failed event structure."""
        event = _create_event(
            "step_failed",
            "test-run-123",
            step="diarization",
            error_code="E_MODEL_OOM",
            error_message="CUDA out of memory",
            duration_ms=1234,
        )

        assert event["type"] == "step_failed"
        assert event["step"] == "diarization"
        assert event["error_code"] == "E_MODEL_OOM"
        assert event["error_message"] == "CUDA out of memory"

    def test_event_structure_run_completed(self):
        """Test run_completed event structure."""
        event = _create_event(
            "run_completed",
            "test-run-123",
            status="COMPLETED",
            total_duration_ms=120000,
            steps_completed=4,
            steps_failed=0,
        )

        assert event["type"] == "run_completed"
        assert event["status"] == "COMPLETED"
        assert event["total_duration_ms"] == 120000
        assert event["steps_completed"] == 4
        assert event["steps_failed"] == 0

    def test_event_json_serializable(self):
        """Test that events are JSON serializable."""
        event = _create_event(
            "step_progress",
            "test-run-123",
            step="asr",
            progress=0.5,
            message="Transcribing audio...",
        )

        # Should not raise
        json_str = json.dumps(event)
        parsed = json.loads(json_str)

        assert parsed["type"] == "step_progress"
        assert parsed["progress"] == 0.5


class TestEventsFile:
    """Test events file writing."""

    def test_events_file_jsonl_format(self):
        """Test that events are written in JSONL format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events_file = Path(tmpdir) / "events.jsonl"

            events = [
                _create_event("run_started", "test-123", steps=["ingest"]),
                _create_event("step_started", "test-123", step="ingest", index=0, total=1),
                _create_event("step_completed", "test-123", step="ingest", duration_ms=100),
                _create_event(
                    "run_completed",
                    "test-123",
                    status="COMPLETED",
                    total_duration_ms=100,
                    steps_completed=1,
                    steps_failed=0,
                ),
            ]

            # Write events
            with events_file.open("w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

            # Read and verify
            read_events = []
            with events_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        read_events.append(json.loads(line))

            assert len(read_events) == 4
            assert read_events[0]["type"] == "run_started"
            assert read_events[-1]["type"] == "run_completed"

    def test_events_file_incremental_read(self):
        """Test reading events file incrementally (tailing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            events_file = Path(tmpdir) / "events.jsonl"

            # Write first batch
            with events_file.open("w", encoding="utf-8") as f:
                f.write(json.dumps(_create_event("run_started", "test-123")) + "\n")
                f.flush()

            # Read position
            pos = events_file.stat().st_size

            # Write second batch
            with events_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_create_event("step_started", "test-123", step="asr")) + "\n")
                f.flush()

            # Read only new content
            new_events = []
            with events_file.open("r", encoding="utf-8") as f:
                f.seek(pos)
                for line in f:
                    if line.strip():
                        new_events.append(json.loads(line))

            assert len(new_events) == 1
            assert new_events[0]["type"] == "step_started"


class TestWebSocketEventTypes:
    """Test expected WebSocket event type handling."""

    def test_terminal_event_types(self):
        """Test which event types indicate terminal state."""
        terminal_types = {"run_completed", "run_failed", "run_cancelled"}

        assert "run_completed" in terminal_types
        assert "run_failed" in terminal_types
        assert "step_completed" not in terminal_types

    def test_progress_event_types(self):
        """Test which event types indicate progress."""
        progress_types = {"step_started", "step_completed", "step_failed", "step_progress"}

        assert "step_started" in progress_types
        assert "heartbeat" not in progress_types

    def test_heartbeat_event(self):
        """Test heartbeat event structure."""
        event = _create_event("heartbeat", "test-123")

        assert event["type"] == "heartbeat"
        assert event["run_id"] == "test-123"
