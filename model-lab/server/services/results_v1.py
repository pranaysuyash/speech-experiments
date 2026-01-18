"""
Results Semantics Layer (v1)

Pure projection of Run Artifacts into Meaningful Results.
See docs/results_contract.md for schema and rules.
"""

from typing import Dict, Any, Optional, List
import json
import logging
from pathlib import Path
from datetime import datetime
import hashlib
from fastapi import HTTPException

from server.services.runs_index import get_index
from server.services.safe_files import safe_file_path
from server.api.experiments import _load_experiment

logger = logging.getLogger("server.results")

def compute_result_v1(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Computes the v1 ResultSummary for a run.
    
    This function is a PURE PROJECTION.
     It reads artifacts.
     It returns JSON.
     It never executes code or mutates state.
    """
    # 1. Load Run Index / Manifest
    run_summary = get_index().get_run(run_id)
    if not run_summary:
        return None
        
    try:
        manifest_path = safe_file_path(run_id, "manifest.json")
        manifest = json.loads(manifest_path.read_text())
        
        # 2. Load Experiment Snapshot (for Label)
        # We need identifying info from manifest to find experiment
        # Assuming run directory structure: .../<exp_id>/runs/<run_id>/...
        # Or manifest contains experiment_id?
        # Current manifest schema: { "run_id": ..., "experiment_id": ... } ?
        # Let's check manifest content from previous tools.
        # If not in manifest, we might have to infer from path or parent dir.
        # Runs are stored in sessions/hash/run_id usually? No, experiments/exp_id/runs/run_id?
        # Actually structure: runs/sessions/<input_hash>/<run_id> OR experiments/<exp_id>/runs/<run_id>?
        # Wait, the experiment `create_experiment` logic separates them.
        # Experiment -> experiment_state.json -> list of runs.
        # But where do runs live on disk? 
        # Looking at `runs.py`, `_runs_root().glob("sessions/*/*/manifest.json")`.
        # So runs are independent entities?
        # Ah, Experiment State links to Run IDs.
        # But Run Manifest might not link back to Experiment explicitly in v1?
        # If so, we can't easily get the snapshot label without passing experiment_id.
        # But the contract said "GET /api/runs/{id}/results".
        # If run doesn't know its experiment, we can't look up the snapshot.
        
        # Let's check if manifest has experiment_id.
        experiment_id = manifest.get("experiment_id")
        candidate_label = "Unknown"
        
        if experiment_id:
            exp_data = _load_experiment(experiment_id)
            if exp_data:
                # Find this run in the experiment to get the snapshot label
                # Match by run_id
                found_run = next((r for r in exp_data.get("runs", []) if r.get("run_id") == run_id), None)
                if found_run:
                    # We need the candidate config. 
                    # experiment_request inputs: "candidates" list.
                    # experiment_state "runs" list matches by index or candidate_id?
                    # `state["runs"]` has `candidate_id` and `candidate_ref`.
                    # `request["candidates"]` has `candidate_id` and `label`.
                    cid = found_run.get("candidate_id")
                    candidate_config = next((c for c in exp_data.get("candidates", []) if c.get("candidate_id") == cid), None)
                    if candidate_config:
                        candidate_label = candidate_config.get("label", candidate_label)

        # 3. Compute Metrics
        started_at = manifest.get("started_at")
        ended_at = manifest.get("ended_at")
        duration_s = None
        if started_at and ended_at:
             try:
                 start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                 end_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
                 duration_s = (end_dt - start_dt).total_seconds()
             except Exception:
                 pass

        # Word Count & Confidence
        word_count = 0
        confidence_sum = 0.0
        segment_count = 0
        
        # Try to load transcript
        try:
            transcript_path = safe_file_path(run_id, "bundle/transcript.json")
            transcript = json.loads(transcript_path.read_text())
            segments = transcript.get("segments", [])
            segment_count = len(segments)
            for seg in segments:
                text = seg.get("text", "")
                word_count += len(text.split())
                # Confidence if available?
                # Example format might not have it, but if it did:
                # confidence_sum += seg.get("avg_logprob", 0) # Placeholder
        except (Exception, HTTPException):
            # Transcript missing or unparseable - expected for new or failed runs
            pass

        confidence_avg = None # Not yet standard in our transcripts
        
        # 4. Input Duration
        audio_duration_s = manifest.get("input_duration_s") # If available

        # 5. Quality Flags
        status = run_summary.get("status", "UNKNOWN")
        is_partial = False
        is_empty = False
        warnings = []

        if status == "FAILED":
            # If we have a transcript (word count > 0) but status FAILED, it's partial
            if word_count > 0:
                is_partial = True
                warnings.append("Partial Result")
        
        if status == "COMPLETED" and word_count == 0:
            is_empty = True
            warnings.append("Empty Transcript")
            
        # 6. Provenance
        manifest_bytes = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

        return {
            "schema_version": "v1",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "candidate_label": candidate_label,
            "status": status,
            "executed_steps": list(manifest.get("steps", {}).keys()),
            "metrics": {
                "duration_s": duration_s,
                "audio_duration_s": audio_duration_s,
                "word_count": word_count,
                "segment_count": segment_count,
                "confidence_avg": confidence_avg,
            },
            "quality_flags": {
                "is_partial": is_partial,
                "is_empty": is_empty,
                "warnings": warnings,
            },
            "provenance": {
                "manifest_hash": manifest_hash,
                "computed_at": datetime.now().isoformat(),
                "semantics_version": "v1"
            }
        }

    except Exception as e:
        logger.error(f"Failed to compute results for {run_id}: {e}")
        # Return minimal failure structure or raise?
        # Contract says: 200 OK provided run exists.
        # If computation fails completely, maybe return empty/error result?
        # But we shouldn't fail hard.
        raise e
