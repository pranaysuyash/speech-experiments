import pytest
from fastapi.testclient import TestClient
from unittest import mock
import json
import os
from pathlib import Path
from dataclasses import asdict

from server.main import app
from server.api.candidates import Candidate, CANDIDATES

client = TestClient(app)

# Test Data
GOLDEN_ID = "test_golden_candidate"
GOLDEN_CANDIDATE = Candidate(
    candidate_id=GOLDEN_ID,
    label="Original Label",
    use_case_id="meeting_smoke",
    steps_preset="ingest",
    params={"alpha": 1},
    expected_artifacts=[],
    description="For golden replay test"
)

@pytest.fixture
def mock_experiments_root(tmp_path):
    """Isolate experiments to a temp directory."""
    # We patch the _experiments_root function directly for robust isolation
    with mock.patch("server.api.experiments._experiments_root") as mock_root:
        mock_root.return_value = tmp_path
        yield tmp_path

def test_experiment_snapshot_is_immutable_and_detects_tampering(mock_experiments_root):
    """
    A2: Golden Replay Test.
    
    1. Create experiment with a candidate.
    2. Modifies the candidate definition in the system (Simulate code drift).
    3. Reload experiment -> Assert it still shows original definition (Snapshot check).
    4. Tamper with the file on disk.
    5. Reload experiment -> Assert provenance mismatch (Tamper check).
    """
    
    # 1. Setup Phase: Inject Golden Candidate
    with mock.patch.dict(CANDIDATES, {GOLDEN_ID: GOLDEN_CANDIDATE}):
        
        # Create experiment
        # We need a dummy file
        files = {'file': ('test.wav', b'fake audio data', 'audio/wav')}
        data = {
            'use_case_id': 'meeting_smoke',
            'candidate_ids': f"{GOLDEN_ID},{GOLDEN_ID}" # Need 2 for valid experiment usually, or 1?
            # create_experiment supports 1 if we pass it, it dupes to A/B? 
            # Or we can just pass two of the same for simplicity.
            # Actually create_experiment with 1 candidate ID: "parsed_candidate_ids" list.
            # Logic: if len >= 1: candidates_list = [get(cid)...]
            # And Slot A/B?
            # If 1 candidate passed, we get 1 config? 
            # Wait, api/experiments.py lines 193: for i, c in enumerate(candidates_list).
            # If we pass 1 candidate, we get 1 run (Slot A).
            # The system supports single-run experiments now (Verified in Phase 1).
            # But let's verify if create_experiment enforces 2? 
            # Logic lines 129: "if len(candidates_list) < 2: ... Fall back to presets ... error INVALID_CANDIDATE_COUNT"
            # That logic is ONLY if "else" (user didn't specify candidates).
            # If user specifies candidates (if parsed_candidate_ids and len >= 1), we use exactly those.
            # So 1 is fine if explicit.
        }
        
        # Just use 1 for clarity
        response = client.post(
            "/api/experiments",
            data={'use_case_id': 'meeting_smoke', 'candidate_ids': GOLDEN_ID},
            files=files
        )
        assert response.status_code == 201, f"Creation failed: {response.text}"
        exp_id = response.json()["experiment_id"]
        
        # Verify initial snapshot
        created = response.json()
        cand_config = created["candidates"][0]
        assert cand_config["label"] == "Original Label"
        assert cand_config["candidate_snapshot"]["params"]["alpha"] == 1
    
    # 2. Drift Phase: Mutate the Candidate in the System
    # We exit the first patch context, so CANDIDATES is back to normal (missing our ID).
    # But let's explicitly mock it with CHANGED values to simulate "Update"
    
    MODIFIED_CANDIDATE = Candidate(
        candidate_id=GOLDEN_ID,
        label="Modified Label", # Changed
        use_case_id="meeting_smoke",
        steps_preset="ingest",
        params={"alpha": 999}, # Changed
        expected_artifacts=[],
        description="Drifted version"
    )
    
    with mock.patch.dict(CANDIDATES, {GOLDEN_ID: MODIFIED_CANDIDATE}):
        # 3. Replay Phase: Reload Experiment
        # The system should NOT look at CANDIDATES dict, but read from disk.
        
        response = client.get(f"/api/experiments/{exp_id}")
        assert response.status_code == 200
        
        data = response.json()
        loaded_cand = data["candidates"][0]
        
        # Assert Immutability (Snapshot Correctness)
        assert loaded_cand["label"] == "Original Label", "Snapshot label drifted!"
        assert loaded_cand["candidate_snapshot"]["params"]["alpha"] == 1, "Snapshot params drifted!"
        assert loaded_cand["label"] != "Modified Label"
        
        # Assert Provenance is Valid
        assert data.get("provenance_status") == "VERIFIED", f"Provenance status was: {data.get('provenance_status')}"

    # 4. Tamper Phase: Modify file on disk
    exp_dir = mock_experiments_root / exp_id
    req_path = exp_dir / "experiment_request.json"
    
    # Read, Tamper, Write
    content = json.loads(req_path.read_text())
    content["candidates"][0]["label"] = "Tampered Label"
    req_path.write_text(json.dumps(content)) # Saving non-canonically might also trigger hash fail, which is good. 
    # But even if we save strictly, the content changed so hash MUST change.
    
    # 5. Verify Detection
    response = client.get(f"/api/experiments/{exp_id}")
    assert response.status_code == 200
    
    tampered_data = response.json()
    
    # Should show the tampered data (we don't hide corruption)
    assert tampered_data["candidates"][0]["label"] == "Tampered Label"
    
    # BUT must flag as CORRUPTED
    assert tampered_data.get("provenance_status") == "CORRUPTED", "Failed to detect tampering!"

