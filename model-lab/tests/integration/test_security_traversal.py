import pytest
import os
import shutil
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from server.main import app

class TestSecurityTraversal:
    
    @pytest.fixture
    def env_setup(self, monkeypatch):
        # Temp dir for runs
        tmp = Path(tempfile.mkdtemp())
        runs_dir = tmp / "runs"
        runs_dir.mkdir()
        monkeypatch.setenv("MODEL_LAB_RUNS_ROOT", str(runs_dir))
        
        # Secret file outside runs
        secret_file = tmp / "secret.txt"
        secret_file.write_text("SUPER_SECRET_DATA")
        
        yield tmp, runs_dir, secret_file
        shutil.rmtree(tmp)

    def test_artifact_traversal(self, env_setup):
        tmp, runs_dir, secret_file = env_setup
        client = TestClient(app)
        
        # Setup Run
        run_id = "attack_run"
        session_dir = runs_dir / "sessions" / "hash" / run_id
        session_dir.mkdir(parents=True)
        (session_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "status": "COMPLETED",
            "steps": {
                "step1": {
                    "artifacts": [
                        {
                            "id": "rel_attack",
                            "path": "../../../secret.txt",
                            "downloadable": True,
                        },
                        {
                            "id": "abs_attack",
                            "path": str(secret_file),
                            "downloadable": True,
                        }
                    ]
                }
            }
        }))
        
        # Mock Index
        with patch("server.services.runs_index.get_index") as mock_idx:
            mock_idx.return_value.get_run.return_value = {
                "manifest_path": str(session_dir / "manifest.json")
            }
            
            # 1. Relative Attack
            resp = client.get(f"/api/runs/{run_id}/artifacts/rel_attack")
            # Should be 403 Forbidden (Blocked) or 404 if file not found logic triggers first
            # Current naive logic might resolve it and serve it if check fails!
            # If current logic uses startswith((session / ..).resolve()), it handles valid resolution
            # but standard '..' might bypass regex checks if any.
            # pathlib.resolve() handles .. so startswith IS somewhat safe IF exact match
            # but is_relative_to is safer.
            assert resp.status_code in [403, 404], f"Leaked content: {resp.text}"
            assert "SUPER_SECRET_DATA" not in resp.text

            # 2. Absolute Attack
            resp = client.get(f"/api/runs/{run_id}/artifacts/abs_attack")
            assert resp.status_code in [403, 404]
            assert "SUPER_SECRET_DATA" not in resp.text
            
    def test_symlink_escape(self, env_setup):
        tmp, runs_dir, secret_file = env_setup
        client = TestClient(app)
        
        run_id = "symlink_run"
        session_dir = runs_dir / "sessions" / "hash" / run_id
        session_dir.mkdir(parents=True)
        
        # Create Symlink
        symlink_path = session_dir / "link_to_secret.txt"
        os.symlink(secret_file, symlink_path)
        
        (session_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "steps": {
                "step1": {
                    "artifacts": [
                        {
                            "id": "symlink_art",
                            "path": "link_to_secret.txt", 
                            "downloadable": True
                        }
                    ]
                }
            }
        }))
        
        with patch("server.services.runs_index.get_index") as mock_idx:
            mock_idx.return_value.get_run.return_value = {
                "manifest_path": str(session_dir / "manifest.json")
            }
            
            # 3. Symlink Attack
            resp = client.get(f"/api/runs/{run_id}/artifacts/symlink_art")
            # Should be 403 because it resolves to outside file
            assert resp.status_code == 403
            assert "SUPER_SECRET_DATA" not in resp.text
