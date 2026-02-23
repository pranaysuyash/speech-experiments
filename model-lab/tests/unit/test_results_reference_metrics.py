from __future__ import annotations

import json
from pathlib import Path

from server.services import results_v1


def test_compute_results_includes_reference_metrics(monkeypatch, tmp_path: Path):
    run_id = "run_ref_metrics"

    manifest = {
        "run_id": run_id,
        "status": "COMPLETED",
        "started_at": "2026-02-21T00:00:00+00:00",
        "ended_at": "2026-02-21T00:00:10+00:00",
        "steps": {"asr": {"status": "COMPLETED"}},
    }
    transcript = {
        "segments": [
            {"text": "hello world"},
            {"text": "this is a test"},
        ]
    }
    run_request = {
        "reference_text": "hello world this is a test",
    }

    files = {
        "manifest.json": tmp_path / "manifest.json",
        "bundle/transcript.json": tmp_path / "bundle" / "transcript.json",
        "run_request.json": tmp_path / "run_request.json",
    }
    files["manifest.json"].parent.mkdir(parents=True, exist_ok=True)
    files["bundle/transcript.json"].parent.mkdir(parents=True, exist_ok=True)
    files["run_request.json"].parent.mkdir(parents=True, exist_ok=True)

    files["manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
    files["bundle/transcript.json"].write_text(json.dumps(transcript), encoding="utf-8")
    files["run_request.json"].write_text(json.dumps(run_request), encoding="utf-8")

    class _Index:
        def get_run(self, rid: str):
            assert rid == run_id
            return {"run_id": run_id, "status": "COMPLETED"}

    monkeypatch.setattr(results_v1, "get_index", lambda: _Index())

    def _safe_file_path(_: str, rel: str) -> Path:
        return files[rel]

    monkeypatch.setattr(results_v1, "safe_file_path", _safe_file_path)

    out = results_v1.compute_result_v1(run_id)
    assert out is not None
    assert out["metrics"]["wer"] == 0.0
    assert out["metrics"]["cer"] == 0.0
    assert out["metrics"]["reference_word_count"] == 6
