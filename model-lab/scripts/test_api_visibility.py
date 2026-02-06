#!/usr/bin/env python3
"""
Test API Visibility
-------------------
Verifies that runs created by `run_session.py` (which uses the Unified RUNS_ROOT)
are automatically visible to the `runs_index` service without copying.
"""

import logging
import shutil
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.services.runs_index import RunsIndex, _runs_root


def test_visibility():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_visibility")

    # 1. Simulate a run creation in the default runs location
    root = _runs_root()

    # create a unique run
    run_id = f"test_vis_{uuid.uuid4().hex[:8]}"
    session_hash = "manual_test_hash"

    run_dir = root / "sessions" / session_hash / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest_content = f'''
    {{
        "run_id": "{run_id}",
        "status": "COMPLETED",
        "started_at": "2026-01-21T10:00:00.000000",
        "steps": {{
            "ingest": {{"status": "COMPLETED"}}
        }}
    }}
    '''

    try:
        manifest_path.write_text(manifest_content)

        # 2. Verify Indexer sees it
        indexer = RunsIndex()
        runs = indexer.refresh()

        found = any(r["run_id"] == run_id for r in runs)

        if found:
            logger.info("✅ SUCCESS: Run found in API index via common root.")
        else:
            logger.error("❌ FAILURE: Run NOT found in API index.")
            sys.exit(1)

    finally:
        # Cleanup
        if run_dir.exists():
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    test_visibility()
