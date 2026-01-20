import json
import os
from pathlib import Path

import pytest


def test_artifact_download_path_traversal(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_LAB_RUNS_ROOT", str(tmp_path / "runs"))

    import server.services.runs_index as runs_index
    runs_index.RunsIndex._instance = None

    run_id = "artifact_traversal_test"
    input_hash = "hash_for_tests"
    run_dir = tmp_path / "runs" / "sessions" / input_hash / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # File outside run_dir that must never be served
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("TOP_SECRET", encoding="utf-8")

    good_rel = "artifacts/good.txt"
    (run_dir / good_rel).write_text("ok", encoding="utf-8")

    evil_rel = "../secret.txt"

    manifest = {
        "run_id": run_id,
        "status": "COMPLETED",
        "steps": {
            "asr": {
                "status": "COMPLETED",
                "artifacts": [
                    {
                        "id": "good",
                        "path": good_rel,
                        "filename": "good.txt",
                        "content_type": "text/plain",
                        "downloadable": True,
                    },
                    {
                        "id": "evil",
                        "path": evil_rel,
                        "filename": "secret.txt",
                        "content_type": "text/plain",
                        "downloadable": True,
                    },
                ],
            }
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    from fastapi.testclient import TestClient
    import server.main

    client = TestClient(server.main.app)

    # Baseline: normal artifact works.
    r_good = client.get(f"/api/runs/{run_id}/artifacts/good")
    assert r_good.status_code == 200
    assert r_good.text == "ok"

    # Path traversal via manifest path is blocked (no file leakage).
    r_evil = client.get(f"/api/runs/{run_id}/artifacts/evil")
    assert r_evil.status_code in (403, 404)
    assert "TOP_SECRET" not in r_evil.text

    # Path traversal via user-controlled artifact_id must not probe filesystem.
    r_id = client.get(f"/api/runs/{run_id}/artifacts/../secret.txt")
    assert r_id.status_code == 404

    # Symlink escape: if a symlink inside the run points outside, it must not be served.
    link_rel = "artifacts/link_out"
    link_path = run_dir / link_rel
    try:
        os.symlink(str(secret_path), str(link_path))
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    manifest["steps"]["asr"]["artifacts"].append(
        {
            "id": "symlink",
            "path": link_rel,
            "filename": "link_out",
            "content_type": "text/plain",
            "downloadable": True,
        }
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Clear index cache so updated manifest is picked up reliably.
    runs_index.RunsIndex._instance = None

    r_link = client.get(f"/api/runs/{run_id}/artifacts/symlink")
    assert r_link.status_code in (403, 404)
    assert "TOP_SECRET" not in r_link.text
