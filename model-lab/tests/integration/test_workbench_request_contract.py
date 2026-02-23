"""
Test run_request.json contract for workbench reproducibility.
"""

import hashlib
import json
import os
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch


def _create_tiny_wav(path: Path, duration_s: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Create a minimal WAV file and return its bytes."""
    nframes = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        silence = struct.pack("<h", 0)
        wf.writeframes(silence * nframes)

    return path.read_bytes()


def test_run_request_json_written_with_correct_schema():
    """Verify run_request.json is written with all required fields."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from fastapi.testclient import TestClient

    import server.api.workbench as workbench
    import server.main
    from harness.session import SessionRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        os.environ["MODEL_LAB_INPUTS_ROOT"] = str(tmpdir_path / "inputs")

        # Create test WAV
        wav_path = tmpdir_path / "test.wav"
        wav_bytes = _create_tiny_wav(wav_path)
        wav_sha256 = hashlib.sha256(wav_bytes).hexdigest()

        # Mock SessionRunner
        mock_runner = MagicMock(spec=SessionRunner)
        mock_runner.run_id = "test_run_123"
        mock_runner.session_dir = tmpdir_path / "runs" / "sessions" / "test" / "test_run_123"
        mock_runner.manifest_path = mock_runner.session_dir / "manifest.json"
        mock_runner.input_path = tmpdir_path / "inputs" / "test.wav"

        # Create session dir and manifest
        mock_runner.session_dir.mkdir(parents=True, exist_ok=True)
        mock_runner.manifest_path.write_text(
            json.dumps({"run_id": "test_run_123", "status": "RUNNING"})
        )

        # Capture run_request_data passed to launch_run_worker
        captured_request_data = {}

        def mock_launch(runner, run_request_data, background=True):
            captured_request_data.update(run_request_data)
            # Write run_request.json like the real function does
            request_path = runner.session_dir / "run_request.json"
            run_request_data["run_id"] = runner.run_id
            run_request_data["input_path"] = str(runner.input_path)
            request_path.write_text(json.dumps(run_request_data, indent=2))
            return {"worker_pid": 12345}

        with (
            patch.object(workbench, "SessionRunner", return_value=mock_runner),
            patch.object(workbench, "launch_run_worker", side_effect=mock_launch),
        ):
            client = TestClient(server.main.app)

            response = client.post(
                "/api/workbench/runs",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={
                    "use_case_id": "test_case",
                    "steps_preset": "ingest",
                    "reference_text": "hello world",
                },
            )

            assert response.status_code == 200

            # Verify run_request.json exists
            request_path = mock_runner.session_dir / "run_request.json"
            assert request_path.exists(), "run_request.json should exist"

            # Load and verify schema
            request_data = json.loads(request_path.read_text())

            assert request_data["schema_version"] == "1"
            assert request_data["source"] == "workbench"
            assert request_data["use_case_id"] == "test_case"
            assert request_data["steps_preset"] == "ingest"
            assert request_data["filename_original"] == "test.wav"
            assert request_data["content_type"] == "audio/wav"
            assert request_data["bytes_uploaded"] == len(wav_bytes)
            assert request_data["sha256"] == wav_sha256, (
                f"SHA256 mismatch: {request_data['sha256']} != {wav_sha256}"
            )
            assert request_data["reference_text"] == "hello world"
            assert "requested_at" in request_data


def test_run_request_sha256_matches_upload():
    """Verify SHA256 in run_request.json matches actual uploaded bytes."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from fastapi.testclient import TestClient

    import server.api.workbench as workbench
    import server.main
    from harness.session import SessionRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        os.environ["MODEL_LAB_INPUTS_ROOT"] = str(tmpdir_path / "inputs")

        # Create a slightly larger WAV to ensure streaming works
        wav_path = tmpdir_path / "larger.wav"
        wav_bytes = _create_tiny_wav(wav_path, duration_s=0.5)
        expected_sha = hashlib.sha256(wav_bytes).hexdigest()

        mock_runner = MagicMock(spec=SessionRunner)
        mock_runner.run_id = "test_sha_456"
        mock_runner.session_dir = tmpdir_path / "runs" / "sessions" / "test" / "test_sha_456"
        mock_runner.manifest_path = mock_runner.session_dir / "manifest.json"
        mock_runner.input_path = tmpdir_path / "inputs" / "larger.wav"

        mock_runner.session_dir.mkdir(parents=True, exist_ok=True)
        mock_runner.manifest_path.write_text(
            json.dumps({"run_id": "test_sha_456", "status": "RUNNING"})
        )

        def mock_launch(runner, run_request_data, background=True):
            request_path = runner.session_dir / "run_request.json"
            run_request_data["run_id"] = runner.run_id
            run_request_data["input_path"] = str(runner.input_path)
            request_path.write_text(json.dumps(run_request_data, indent=2))
            return {"worker_pid": 12345}

        with (
            patch.object(workbench, "SessionRunner", return_value=mock_runner),
            patch.object(workbench, "launch_run_worker", side_effect=mock_launch),
        ):
            client = TestClient(server.main.app)

            response = client.post(
                "/api/workbench/runs",
                files={"file": ("larger.wav", wav_bytes, "audio/wav")},
                data={"use_case_id": "sha_test", "steps_preset": "full"},
            )

            assert response.status_code == 200

            request_path = mock_runner.session_dir / "run_request.json"
            request_data = json.loads(request_path.read_text())

            assert request_data["sha256"] == expected_sha
            assert request_data["bytes_uploaded"] == len(wav_bytes)
