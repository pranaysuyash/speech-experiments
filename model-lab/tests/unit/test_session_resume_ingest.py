from __future__ import annotations

import logging
from pathlib import Path

from harness.media_ingest import IngestConfig
from harness.session import SessionRunner

# Disable root logger for tests to avoid spam
logging.getLogger().setLevel(logging.CRITICAL)


def test_resume_ingest_skips_when_hashes_match(tmp_path: Path, test_wav_path: Path) -> None:
    out_dir = tmp_path / "runs"
    cfg = IngestConfig(normalize=False, trim_silence=False)

    # First Run
    r1 = SessionRunner(test_wav_path, out_dir, preprocessing=cfg, resume=True, steps=["ingest"])
    m1 = r1.run()
    assert m1["status"] == "COMPLETED"
    run_id_1 = m1["run_id"]

    # Check ingest artifact exists
    ingest_step = m1["steps"]["ingest"]
    assert ingest_step["status"] == "COMPLETED"

    # Second Run - SAME output dir (implies resuming same session ID usually?)
    # But SessionRunner current implementation generates NEW ID based on timestamp.
    # To truly test resume, we must pass `resume_from` pointing to the previous run dir.

    prev_run_dir = out_dir / "sessions" / m1["input"]["input_hash"] / run_id_1

    r2 = SessionRunner(
        test_wav_path,
        out_dir,
        preprocessing=cfg,
        resume=True,
        config={"resume_from": prev_run_dir},
        steps=["ingest"],
    )

    # We expect it to modify the SAME manifest?
    # Yes, if we point to the dir.

    # Mocking internal logger to verify "Skipping"
    # But simplest is checking the started_at of the step didn't change?
    # Or duration is 0?
    # Or just check logs?

    # Let's check that the step duration is suspiciously low or timestamps match.
    start_time_1 = ingest_step["started_at"]

    m2 = r2.run()
    assert m2["status"] == "COMPLETED"
    assert m2["run_id"] == run_id_1  # Should be same

    ingest_step_2 = m2["steps"]["ingest"]
    # If skipped, it might not update started_at? Or explicitly updates it?
    # Logic: if skipped, we print "Skipping" and return. We do NOT update start/end times in manifest step entry usually if skipped?
    # Wait, `_run_step` implementation:
    # if valid_for_resume: return None.
    # It does NOT update manifest step entry if skipped.

    assert ingest_step_2["started_at"] == start_time_1
