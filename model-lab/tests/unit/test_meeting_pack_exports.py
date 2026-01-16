import json
from pathlib import Path

import pytest

from harness.meeting_pack import build_meeting_pack, MEETING_PACK_SCHEMA_VERSION


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def test_bundle_manifest_schema(tmp_path: Path):
    run_id = "20260101_000000_deadbeef00"
    run_dir = tmp_path / "runs" / "sessions" / "hash" / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "status": "COMPLETED",
            "steps": {},
        },
    )

    result = build_meeting_pack(run_dir)
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert m["schema_version"] == MEETING_PACK_SCHEMA_VERSION
    assert m["run_id"] == run_id
    assert isinstance(m["generated_at"], str)
    assert isinstance(m["artifacts"], list)
    assert isinstance(m["absent"], list)

    absent_names = {a["name"] for a in m["absent"]}
    assert "transcript.json" in absent_names
    assert "summary.md" in absent_names
    assert "action_items.csv" in absent_names
    assert "decisions.md" in absent_names


def test_bundle_contains_expected_files_when_inputs_exist(tmp_path: Path):
    run_id = "20260101_000000_deadbeef01"
    run_dir = tmp_path / "runs" / "sessions" / "hash" / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = artifacts_dir / "alignment_test.json"
    _write_json(
        alignment_path,
        {
            "output": {
                "segments": [
                    {"start_s": 0.0, "end_s": 1.0, "text": "Hello", "speaker_id": "SPEAKER_00"},
                    {"start_s": 1.0, "end_s": 2.0, "text": "World", "speaker_id": "SPEAKER_01"},
                ]
            }
        },
    )

    summary_path = artifacts_dir / "summary_by_speaker_test.json"
    _write_json(
        summary_path,
        {
            "output": {
                "speaker_summaries": {
                    "SPEAKER_00": ["Did X", "Asked Y"],
                    "SPEAKER_01": ["Confirmed Z"],
                }
            }
        },
    )

    action_items_path = artifacts_dir / "action_items_test.json"
    _write_json(
        action_items_path,
        {
            "output": {
                "action_items": [
                    {"text": "Follow up", "assignee": "SPEAKER_00", "priority": "high"},
                    {"text": "Send doc", "assignee": "SPEAKER_01", "priority": "low"},
                ]
            }
        },
    )

    decisions_src = artifacts_dir / "decisions.md"
    decisions_src.write_text("# Decisions\n- We will do A\n", encoding="utf-8")

    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "status": "COMPLETED",
            "steps": {
                "alignment": {"status": "COMPLETED", "artifacts": [{"path": str(alignment_path)}]},
                "summarize_by_speaker": {"status": "COMPLETED", "artifacts": [{"path": str(summary_path)}]},
                "action_items_assignee": {"status": "COMPLETED", "artifacts": [{"path": str(action_items_path)}]},
            },
        },
    )

    build_meeting_pack(run_dir)

    bundle_dir = run_dir / "bundle"
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert (bundle_dir / "transcript.json").exists()
    assert (bundle_dir / "summary.md").exists()
    assert (bundle_dir / "action_items.csv").exists()
    assert (bundle_dir / "decisions.md").exists()

    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    present = {a["name"] for a in manifest["artifacts"]}
    assert {"transcript.json", "summary.md", "action_items.csv", "decisions.md"}.issubset(present)

    absent = {a["name"] for a in manifest["absent"]}
    assert "transcript.json" not in absent
    assert "summary.md" not in absent
    assert "action_items.csv" not in absent
    assert "decisions.md" not in absent

