"""
Hermetic tests for Candidate Library V1.

Tests:
- test_list_use_cases_shape
- test_list_candidates_for_use_case
- test_experiment_create_accepts_candidate_ids_and_persists_snapshot
"""

import json
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path


def _create_tiny_wav(path: Path, duration_s: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Create a minimal WAV file and return its bytes."""
    nframes = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silence = struct.pack("<h", 0) * nframes
        wf.writeframes(silence)
    return path.read_bytes()


def test_list_use_cases_shape():
    """Verify GET /api/use-cases returns list with expected shape."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from fastapi.testclient import TestClient

    import server.main

    client = TestClient(server.main.app)
    response = client.get("/api/use-cases")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    # Check shape of first use case
    uc = data[0]
    assert "use_case_id" in uc
    assert "title" in uc
    assert "description" in uc
    assert "supported_steps_presets" in uc
    assert isinstance(uc["supported_steps_presets"], list)


def test_list_candidates_for_use_case():
    """Verify GET /api/use-cases/{id}/candidates returns candidates."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from fastapi.testclient import TestClient

    import server.main

    client = TestClient(server.main.app)
    response = client.get("/api/use-cases/meeting_smoke/candidates")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2  # Should have at least 2 candidates

    # Check shape of first candidate
    c = data[0]
    assert "candidate_id" in c
    assert "label" in c
    assert "use_case_id" in c
    assert "steps_preset" in c
    assert "params" in c
    assert "expected_artifacts" in c
    assert c["use_case_id"] == "meeting_smoke"


def test_experiment_create_accepts_candidate_ids_and_persists_snapshot():
    """Verify experiment creation accepts candidate_ids and stores snapshots."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from fastapi.testclient import TestClient

    import server.main

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")

        wav_path = tmpdir_path / "test.wav"
        wav_bytes = _create_tiny_wav(wav_path)

        client = TestClient(server.main.app)

        # Create experiment with specific candidate_ids
        response = client.post(
            "/api/experiments",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={
                "use_case_id": "meeting_smoke",
                "candidate_ids": "meeting_ingest_fast,meeting_full_default",
            },
        )

        assert response.status_code == 201
        data = response.json()
        exp_id = data["experiment_id"]

        # Verify candidates are returned
        assert len(data["candidates"]) == 2
        assert data["candidates"][0]["candidate_id"] == "A"
        assert data["candidates"][1]["candidate_id"] == "B"
        assert data["candidates"][0]["candidate_ref"] == "meeting_ingest_fast"
        assert data["candidates"][1]["candidate_ref"] == "meeting_full_default"

        # Verify experiment_request.json contains candidate snapshots
        exp_dir = tmpdir_path / "runs" / "experiments" / exp_id
        request_path = exp_dir / "experiment_request.json"
        assert request_path.exists()

        request = json.loads(request_path.read_text())
        assert request["schema_version"] == "2"
        assert len(request["candidates"]) == 2

        # Check candidate snapshot exists
        assert "candidate_snapshot" in request["candidates"][0]
        snapshot = request["candidates"][0]["candidate_snapshot"]
        assert "candidate_id" in snapshot
        assert "label" in snapshot
        assert "steps_preset" in snapshot
        assert "params" in snapshot
