"""Tests for pipeline progress tracking."""

import json
import time

import pytest

from harness.session import SessionRunner, StepProgress, now_iso


class TestStepProgressDataclass:
    """Tests for the StepProgress dataclass."""

    def test_minimal_fields(self):
        """Test StepProgress with only required fields."""
        progress = StepProgress(
            step_name="asr",
            status="RUNNING",
            progress_pct=50,
        )
        assert progress.step_name == "asr"
        assert progress.status == "RUNNING"
        assert progress.progress_pct == 50
        assert progress.message is None
        assert progress.started_at is None
        assert progress.estimated_remaining_s is None

    def test_all_fields(self):
        """Test StepProgress with all fields populated."""
        progress = StepProgress(
            step_name="diarization",
            status="RUNNING",
            progress_pct=75,
            message="Processing speakers...",
            started_at="2025-01-08T10:00:00Z",
            estimated_remaining_s=30,
        )
        assert progress.step_name == "diarization"
        assert progress.status == "RUNNING"
        assert progress.progress_pct == 75
        assert progress.message == "Processing speakers..."
        assert progress.started_at == "2025-01-08T10:00:00Z"
        assert progress.estimated_remaining_s == 30

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        progress = StepProgress(
            step_name="ingest",
            status="COMPLETED",
            progress_pct=100,
        )
        d = progress.to_dict()
        assert d == {
            "step_name": "ingest",
            "status": "COMPLETED",
            "progress_pct": 100,
        }

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        progress = StepProgress(
            step_name="asr",
            status="RUNNING",
            progress_pct=45,
            message="Transcribing...",
            started_at="2025-01-08T10:00:00Z",
            estimated_remaining_s=60,
        )
        d = progress.to_dict()
        assert d == {
            "step_name": "asr",
            "status": "RUNNING",
            "progress_pct": 45,
            "message": "Transcribing...",
            "started_at": "2025-01-08T10:00:00Z",
            "estimated_remaining_s": 60,
        }

    def test_status_values(self):
        """Test various status values."""
        for status in ["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"]:
            progress = StepProgress(step_name="test", status=status, progress_pct=0)
            assert progress.status == status

    def test_progress_pct_boundary_values(self):
        """Test progress percentage boundary values."""
        progress_0 = StepProgress(step_name="test", status="PENDING", progress_pct=0)
        assert progress_0.progress_pct == 0

        progress_100 = StepProgress(step_name="test", status="COMPLETED", progress_pct=100)
        assert progress_100.progress_pct == 100


class TestProgressDebouncing:
    """Tests for progress update debouncing in SessionRunner."""

    @pytest.fixture
    def temp_session_dir(self, tmp_path):
        """Create a temporary session directory with manifest."""
        session_dir = tmp_path / "sessions" / "testhash" / "test_run_id"
        session_dir.mkdir(parents=True)

        manifest = {
            "run_id": "test_run_id",
            "status": "RUNNING",
            "steps": {
                "asr": {
                    "status": "RUNNING",
                    "started_at": now_iso(),
                }
            },
            "updated_at": now_iso(),
        }
        (session_dir / "manifest.json").write_text(json.dumps(manifest))
        return session_dir

    @pytest.fixture
    def mock_runner(self, temp_session_dir, tmp_path):
        """Create a mock SessionRunner with session_dir set."""
        # Create a dummy input file
        input_file = tmp_path / "test.wav"
        input_file.write_bytes(b"dummy audio content for testing" * 100)

        runner = SessionRunner(
            input_path=str(input_file),
            output_dir=str(tmp_path),
            config={"resume_from": str(temp_session_dir)},
        )
        return runner

    def test_first_update_writes_immediately(self, mock_runner, temp_session_dir):
        """First progress update should write immediately."""
        mock_runner.update_step_progress("asr", 10, "Starting...")

        manifest = json.loads((temp_session_dir / "manifest.json").read_text())
        assert manifest["steps"]["asr"]["progress_pct"] == 10
        assert manifest["steps"]["asr"]["progress_message"] == "Starting..."

    def test_rapid_updates_debounced(self, mock_runner, temp_session_dir):
        """Rapid updates within debounce window should be skipped."""
        # First update
        mock_runner.update_step_progress("asr", 10, "10%")

        # Immediate second update (should be debounced)
        mock_runner.update_step_progress("asr", 20, "20%")

        manifest = json.loads((temp_session_dir / "manifest.json").read_text())
        # Should still be 10% because second update was debounced
        assert manifest["steps"]["asr"]["progress_pct"] == 10
        assert manifest["steps"]["asr"]["progress_message"] == "10%"

    def test_update_after_debounce_window(self, mock_runner, temp_session_dir):
        """Updates after debounce window should write."""
        # First update
        mock_runner.update_step_progress("asr", 10, "10%")

        # Wait past debounce window
        time.sleep(1.1)

        # Second update (should write)
        mock_runner.update_step_progress("asr", 50, "50%")

        manifest = json.loads((temp_session_dir / "manifest.json").read_text())
        assert manifest["steps"]["asr"]["progress_pct"] == 50
        assert manifest["steps"]["asr"]["progress_message"] == "50%"

    def test_progress_clamps_to_valid_range(self, mock_runner, temp_session_dir):
        """Progress should be clamped between 0 and 100."""
        mock_runner.update_step_progress("asr", -10, "negative")
        manifest = json.loads((temp_session_dir / "manifest.json").read_text())
        assert manifest["steps"]["asr"]["progress_pct"] == 0

        # Wait for debounce
        time.sleep(1.1)

        mock_runner.update_step_progress("asr", 150, "over 100")
        manifest = json.loads((temp_session_dir / "manifest.json").read_text())
        assert manifest["steps"]["asr"]["progress_pct"] == 100

    def test_estimated_remaining_s_written(self, mock_runner, temp_session_dir):
        """Estimated remaining time should be written to manifest."""
        mock_runner.update_step_progress("asr", 50, "Halfway", estimated_remaining_s=30)

        manifest = json.loads((temp_session_dir / "manifest.json").read_text())
        assert manifest["steps"]["asr"]["estimated_remaining_s"] == 30

    def test_nonexistent_step_ignored(self, mock_runner, temp_session_dir):
        """Updates for nonexistent steps should be silently ignored."""
        # Should not raise
        mock_runner.update_step_progress("nonexistent_step", 50, "test")

        manifest = json.loads((temp_session_dir / "manifest.json").read_text())
        assert "nonexistent_step" not in manifest["steps"]


class TestManifestProgressFields:
    """Tests for progress fields in manifest structure."""

    def test_manifest_contains_progress_fields(self, tmp_path):
        """Test that manifest steps can contain progress fields."""
        session_dir = tmp_path / "sessions" / "hash" / "run_id"
        session_dir.mkdir(parents=True)

        manifest = {
            "run_id": "run_id",
            "status": "RUNNING",
            "current_step": "asr",
            "steps": {
                "ingest": {
                    "status": "COMPLETED",
                    "progress_pct": 100,
                    "duration_ms": 1234,
                },
                "asr": {
                    "status": "RUNNING",
                    "progress_pct": 45,
                    "progress_message": "Transcribing...",
                    "estimated_remaining_s": 30,
                    "started_at": "2025-01-08T10:00:00Z",
                },
                "diarization": {
                    "status": "PENDING",
                    "progress_pct": 0,
                },
            },
        }

        manifest_path = session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        loaded = json.loads(manifest_path.read_text())

        # Verify all fields are preserved
        assert loaded["steps"]["ingest"]["progress_pct"] == 100
        assert loaded["steps"]["asr"]["progress_pct"] == 45
        assert loaded["steps"]["asr"]["progress_message"] == "Transcribing..."
        assert loaded["steps"]["asr"]["estimated_remaining_s"] == 30
        assert loaded["steps"]["diarization"]["progress_pct"] == 0

    def test_steps_progress_api_structure(self):
        """Test the expected structure for steps_progress API response."""
        # This tests the expected structure that the API should return
        steps_progress = [
            {
                "name": "ingest",
                "status": "COMPLETED",
                "progress_pct": 100,
                "duration_ms": 1234,
            },
            {
                "name": "asr",
                "status": "RUNNING",
                "progress_pct": 45,
                "message": "Transcribing...",
                "estimated_remaining_s": 30,
            },
            {
                "name": "diarization",
                "status": "PENDING",
                "progress_pct": 0,
            },
        ]

        # Verify structure matches expected API contract
        assert len(steps_progress) == 3
        assert steps_progress[0]["name"] == "ingest"
        assert steps_progress[0]["status"] == "COMPLETED"
        assert steps_progress[1]["name"] == "asr"
        assert steps_progress[1]["status"] == "RUNNING"
        assert "message" in steps_progress[1]
        assert steps_progress[2]["status"] == "PENDING"
