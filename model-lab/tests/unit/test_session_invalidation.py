from __future__ import annotations

from pathlib import Path
import pytest
import logging

from harness.media_ingest import IngestConfig
from harness.session import SessionRunner

# Disable root logger
logging.getLogger().setLevel(logging.CRITICAL)

def test_invalidation_when_ingest_config_changes(tmp_path: Path, test_wav_path: Path) -> None:
    out_dir = tmp_path / "runs"

    # 1. Run with Config A
    r1 = SessionRunner(test_wav_path, out_dir, preprocessing=IngestConfig(normalize=False), resume=True, steps=["ingest"])
    m1 = r1.run()
    assert m1["status"] == "COMPLETED"
    run_dir = r1.session_dir
    
    # 2. Run with Config B (Resuming same dir)
    # This should trigger Ingest rerunning because config mismatches manifest.
    # And because ingest reruns, if there were downstream steps, they would be invalidated.
    # Currently we only have ingest active fully. 
    # But we can verify ingest DID re-run (started_at changes).
    
    t1 = m1["steps"]["ingest"]["started_at"]
    
    r2 = SessionRunner(test_wav_path, out_dir, preprocessing=IngestConfig(normalize=True), resume=True, config={"resume_from": run_dir}, steps=["ingest"])
    m2 = r2.run()
    
    # Fast tests might result in same timestamp if resolution is 1s
    # Better to check if the RESULT reflects the new config.
    # If skipped, it would retain old result (normalize=False).
    # If invalidation worked, it should have new result (normalize=True).
    
    res_cfg = m2["steps"]["ingest"]["result"]["preprocessing_config"]
    assert res_cfg["normalize"] is True, "Ingest should have re-run with new config (normalize=True)"
    
    assert m2["steps"]["ingest"]["status"] == "COMPLETED"
