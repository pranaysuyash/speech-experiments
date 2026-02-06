import importlib
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def test_bundle_endpoint_lists_and_downloads(tmp_path: Path, monkeypatch):
    run_id = "20260101_000000_deadbeef99"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "sessions" / "hash" / run_id
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "manifest.json", {"run_id": run_id, "status": "COMPLETED", "steps": {}})

    (bundle_dir / "summary.md").write_text("# Summary\n- Hello\n", encoding="utf-8")
    _write_json(bundle_dir / "transcript.json", {"segments": []})
    (bundle_dir / "action_items.csv").write_text(
        "assignee,priority,text\nA,high,Do X\n", encoding="utf-8"
    )

    bundle_manifest = {
        "schema_version": "meeting_pack_bundle_manifest.v0.1",
        "run_id": run_id,
        "generated_at": "2026-01-01T00:00:00Z",
        "artifacts": [
            {
                "name": "summary.md",
                "rel_path": "bundle/summary.md",
                "bytes": (bundle_dir / "summary.md").stat().st_size,
                "sha256": "x",
                "content_type": "text/markdown",
            },
            {
                "name": "transcript.json",
                "rel_path": "bundle/transcript.json",
                "bytes": (bundle_dir / "transcript.json").stat().st_size,
                "sha256": "x",
                "content_type": "application/json",
            },
            {
                "name": "action_items.csv",
                "rel_path": "bundle/action_items.csv",
                "bytes": (bundle_dir / "action_items.csv").stat().st_size,
                "sha256": "x",
                "content_type": "text/csv",
            },
        ],
        "absent": [{"name": "decisions.md", "reason": "missing"}],
    }
    _write_json(bundle_dir / "bundle_manifest.json", bundle_manifest)

    monkeypatch.setenv("MODEL_LAB_RUNS_ROOT", str(runs_root))

    # Reload modules that may have captured env at import time.
    import server.main as server_main
    import server.services.runs_index as runs_index
    import server.services.safe_files as safe_files

    importlib.reload(safe_files)
    importlib.reload(runs_index)
    importlib.reload(server_main)

    client = TestClient(server_main.app)

    r = client.get(f"/api/runs/{run_id}/bundle")
    assert r.status_code == 200
    assert r.json()["run_id"] == run_id
    assert any(a["name"] == "summary.md" for a in r.json()["artifacts"])

    r = client.get(f"/api/runs/{run_id}/bundle/summary.md")
    assert r.status_code == 200
    assert "# Summary" in r.text
    assert r.headers["content-type"].startswith("text/markdown")

    # Preview cap: refuse when too small
    r = client.get(f"/api/runs/{run_id}/bundle/summary.md?max_bytes=1")
    assert r.status_code == 413

    # Path traversal attempts should not succeed
    r = client.get(f"/api/runs/{run_id}/bundle/%2e%2e")
    assert r.status_code in (400, 404)
    r = client.get(f"/api/runs/{run_id}/bundle/%2e%2e%2fsummary.md")
    assert r.status_code in (400, 404)

    r = client.get(f"/api/runs/{run_id}/bundle.zip")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/zip")

    zip_path = tmp_path / "out.zip"
    zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "bundle_manifest.json" in names
        assert "summary.md" in names
        assert "transcript.json" in names
        assert "action_items.csv" in names
        for n in names:
            assert not n.startswith("/")
            assert ".." not in n
            assert "/" not in n  # zip contains only flat, relative names

    # Cached zip should be stable for the same manifest
    r2 = client.get(f"/api/runs/{run_id}/bundle.zip")
    assert r2.status_code == 200
    assert r.content == r2.content
