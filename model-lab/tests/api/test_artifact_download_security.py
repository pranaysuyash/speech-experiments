import json
from pathlib import Path


def test_artifact_download_path_traversal(monkeypatch, tmp_path):
    """
    Ensure artifact download cannot escape the run directory via:
    - ../ traversal in manifest path
    - absolute paths in manifest path
    - symlink escape inside the run directory
    - tricky artifact_id values (should not be treated as paths)
    """
    monkeypatch.setenv("MODEL_LAB_RUNS_ROOT", str(tmp_path / "runs"))

    import server.services.runs_index as runs_index
    runs_index.RunsIndex._instance = None

    run_id = "run_artifact_traversal_test"
    input_hash = "hash_for_tests"
    run_dir = tmp_path / "runs" / "sessions" / input_hash / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    outside_file = run_dir.parent / "outside.txt"
    outside_file.write_text("TOP_SECRET", encoding="utf-8")

    safe_file = run_dir / "artifacts" / "safe.txt"
    safe_file.write_text("SAFE", encoding="utf-8")

    symlink_path = run_dir / "artifacts" / "symlink.txt"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(outside_file)

    manifest = {
        "run_id": run_id,
        "status": "FAILED",
        "steps": {
            "asr": {
                "status": "COMPLETED",
                "artifacts": [
                    {
                        "id": "safe",
                        "downloadable": True,
                        "path": "artifacts/safe.txt",
                        "filename": "safe.txt",
                        "content_type": "text/plain",
                    },
                    {
                        "id": "traversal_rel",
                        "downloadable": True,
                        "path": "../outside.txt",
                        "filename": "outside.txt",
                        "content_type": "text/plain",
                    },
                    {
                        "id": "traversal_abs",
                        "downloadable": True,
                        "path": str(outside_file.resolve()),
                        "filename": "outside.txt",
                        "content_type": "text/plain",
                    },
                    {
                        "id": "traversal_symlink",
                        "downloadable": True,
                        "path": "artifacts/symlink.txt",
                        "filename": "symlink.txt",
                        "content_type": "text/plain",
                    },
                ],
            }
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    from fastapi.testclient import TestClient
    import server.main

    client = TestClient(server.main.app)

    ok = client.get(f"/api/runs/{run_id}/artifacts/safe")
    assert ok.status_code == 200
    assert ok.text == "SAFE"

    rel = client.get(f"/api/runs/{run_id}/artifacts/traversal_rel")
    assert rel.status_code == 403
    assert "TOP_SECRET" not in rel.text

    abs_ = client.get(f"/api/runs/{run_id}/artifacts/traversal_abs")
    assert abs_.status_code == 403
    assert "TOP_SECRET" not in abs_.text

    sym = client.get(f"/api/runs/{run_id}/artifacts/traversal_symlink")
    assert sym.status_code == 403
    assert "TOP_SECRET" not in sym.text

    # artifact_id is opaque; httpx normalizes literal "..", so URL-encode.
    tricky = client.get(
        f"/api/runs/{run_id}/artifacts/%2e%2e%2f",
        follow_redirects=False,
    )
    assert tricky.status_code in (307, 404)
