import json
from datetime import UTC
from pathlib import Path
from typing import Any

from server.services.results_v1 import compute_result_v1
from server.services.runs_index import get_index


class RunsService:
    @staticmethod
    def get_run_results(run_id: str) -> dict[str, Any]:
        """Get semantic results for a run."""
        if not get_index().get_run(run_id):
            raise ValueError("Run not found")

        result = compute_result_v1(run_id)
        if not result:
            raise ValueError("Run not found or unreadable")
        return result

    @staticmethod
    def list_runs(refresh: bool = False) -> list:
        """List all runs."""
        return get_index().list_runs(force_refresh=refresh)

    @staticmethod
    def refresh_runs() -> list:
        """Refresh runs index."""
        return get_index().refresh()

    @staticmethod
    def get_runs_by_input(input_hash: str) -> list:
        """Get runs by input hash."""
        runs = get_index().list_runs()
        matching = [r for r in runs if r.get("input_hash") == input_hash]
        return sorted(matching, key=lambda r: r.get("created_at", ""), reverse=True)

    @staticmethod
    def compare_runs(run_a: str, run_b: str) -> dict[str, Any]:
        """Compare two runs."""
        index = get_index()
        run_a_data = index.get_run(run_a)
        run_b_data = index.get_run(run_b)

        if not run_a_data:
            raise ValueError(f"Run A not found: {run_a}")
        if not run_b_data:
            raise ValueError(f"Run B not found: {run_b}")

        # Load manifests
        manifest_a = RunsService._load_manifest(run_a_data.get("manifest_path"))
        manifest_b = RunsService._load_manifest(run_b_data.get("manifest_path"))

        # Metrics
        metrics_a = RunsService._get_run_metrics(run_a)
        metrics_b = RunsService._get_run_metrics(run_b)
        metrics_comparison = RunsService._build_metrics_comparison(metrics_a, metrics_b)

        return {
            "run_a": run_a_data,
            "run_b": run_b_data,
            "manifest_a": manifest_a,
            "manifest_b": manifest_b,
            "metrics_comparison": metrics_comparison,
        }

    @staticmethod
    def _load_manifest(manifest_path: str | None) -> dict[str, Any]:
        """Load manifest from path."""
        if not manifest_path:
            return {}
        try:
            return json.loads(Path(manifest_path).read_text())
        except Exception:
            return {}

    @staticmethod
    def _get_run_metrics(run_id: str) -> dict[str, Any]:
        """Extract key metrics from a run."""
        try:
            result = compute_result_v1(run_id)
            if result and "metrics" in result:
                return {
                    "transcript_word_count": result["metrics"].get("word_count"),
                    "segment_count": result["metrics"].get("segment_count"),
                    "duration_s": result["metrics"].get("duration_s"),
                }
        except Exception:
            pass
        return {}

    @staticmethod
    def _build_metrics_comparison(
        metrics_a: dict[str, Any], metrics_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Build comparison dict for metrics."""
        comparison = {}
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            diff = None
            if val_a is not None and val_b is not None:
                try:
                    diff = val_b - val_a
                except (TypeError, ValueError):
                    pass
            comparison[key] = {"a": val_a, "b": val_b, "diff": diff}
        return comparison

    @staticmethod
    def get_run_status(run_id: str) -> dict[str, Any]:
        """Get lightweight status for a run with stale detection."""
        from datetime import datetime

        run = get_index().get_run(run_id)
        if not run:
            raise ValueError("Run not found")

        # Load manifest with fallback
        manifest, snapshot_source, snapshot_reason, manifest_mtime = (
            RunsService._load_manifest_with_fallback(run)
        )

        status = manifest.get("status", run["status"])
        current_step = manifest.get("current_step")
        updated_at = manifest.get("updated_at")

        # Stale detection
        STALE_THRESHOLD_SECONDS = 90
        if status == "RUNNING" and updated_at:
            try:
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(UTC)
                elapsed = (now - last_update).total_seconds()

                if elapsed > STALE_THRESHOLD_SECONDS:
                    status = "STALE"
                    snapshot_reason = f"no_heartbeat_{int(elapsed)}s"
                    if "error" not in manifest:
                        manifest["error"] = {
                            "type": "StaleRun",
                            "message": f"No heartbeat in {int(elapsed)}s",
                        }
            except Exception:
                pass

        # Build steps progress and config
        steps_completed = RunsService._get_steps_completed(manifest, run)
        steps_progress = RunsService._build_steps_progress(manifest)
        status_config = RunsService._derive_status_config(manifest)

        return {
            "run_id": run_id,
            "status": status,
            "current_step": current_step,
            "updated_at": updated_at,
            "steps_completed": steps_completed,
            "steps_progress": steps_progress,
            "failure_step": manifest.get("failure_step"),
            "error_message": manifest.get("error", {}).get("message"),
            "input_metadata": manifest.get("input_metadata", {}),
            "input_hash": run.get("input_hash"),
            "config": status_config,
            "artifacts": manifest.get("artifacts_by_type", {}),
            "resolved_device": run.get("config", {}).get("resolved_device"),
            "meta": {
                "snapshot_source": snapshot_source,
                "snapshot_reason": snapshot_reason,
                "manifest_mtime": manifest_mtime,
            },
        }

    @staticmethod
    def rerun_pipeline(
        run_id: str, config_overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Re-run a pipeline with optional config changes."""
        import os
        from datetime import datetime

        from harness.session import SessionRunner
        from server.services.lifecycle import (
            RunnerBusyError,
            launch_run_worker,
            release_worker,
            try_acquire_worker,
        )

        index = get_index()
        run_data = index.get_run(run_id)

        if not run_data:
            raise ValueError("Run not found")

        # Load run_request.json to get original input file
        run_request_path = Path(run_data["root_path"]) / "run_request.json"
        if not run_request_path.exists():
            raise ValueError("Cannot rerun: run_request.json missing")

        try:
            run_request = json.loads(run_request_path.read_text())
        except Exception as e:
            raise ValueError(f"Failed to read run request: {e}")

        # Find original input file
        manifest_path = Path(run_data["manifest_path"])
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            raise ValueError("Failed to read manifest")

        input_path_str = manifest.get("input_path")
        if not input_path_str:
            raise ValueError("Cannot rerun: original input path not found")

        input_path = Path(input_path_str)
        if not input_path.exists():
            raise ValueError(f"Cannot rerun: input file missing: {input_path}")

        # Get original configuration
        original_steps = run_request.get("steps_requested")
        original_config = run_request.get("config", {})
        preprocessing = run_request.get("preprocessing", [])

        # Merge config overrides
        merged_config = {**original_config, **(config_overrides or {})}

        # Acquire worker
        if not try_acquire_worker():
            raise RunnerBusyError("Runner is busy with another job")

        try:
            runs_root = Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()
            runner = SessionRunner(
                input_path,
                runs_root,
                steps=original_steps,
                config=merged_config,
            )

            now = datetime.now(UTC)

            # Create new run_request with parent reference
            new_run_request = {
                "schema_version": "1",
                "requested_at": now.isoformat(),
                "source": "rerun",
                "parent_run_id": run_id,
                "use_case_id": run_request.get("use_case_id", "rerun"),
                "steps_preset": run_request.get("steps_preset", "custom"),
                "steps_requested": original_steps,
                "filename_original": input_path.name,
                "sha256": run_request.get("sha256"),
                "config": merged_config,
                "preprocessing": preprocessing,
            }

            result = launch_run_worker(runner, new_run_request, background=True)

            return {
                "run_id": runner.run_id,
                "parent_run_id": run_id,
                "console_url": f"/runs/{runner.run_id}",
                "worker_pid": result.get("worker_pid"),
            }
        except Exception:
            release_worker()
            raise

    @staticmethod
    def _load_manifest_with_fallback(run: dict[str, Any]) -> tuple:
        """Load manifest with fallback logic. Returns (manifest, source, reason, mtime)."""
        snapshot_source = "manifest"
        snapshot_reason = None
        manifest_mtime = 0.0

        try:
            manifest_path = Path(run["manifest_path"])
            manifest = json.loads(manifest_path.read_text())
            manifest_mtime = manifest_path.stat().st_mtime
        except FileNotFoundError:
            snapshot_source = "index"
            snapshot_reason = "manifest_missing"
            manifest = {
                "status": run["status"],
                "steps": {k: {"status": "COMPLETED"} for k in run.get("steps_completed", [])},
                "error": {"type": "SnapshotError", "message": "Run manifest missing on disk"},
            }
        except json.JSONDecodeError:
            snapshot_source = "index"
            snapshot_reason = "manifest_corrupt"
            manifest = {
                "status": run["status"],
                "steps": {k: {"status": "COMPLETED"} for k in run.get("steps_completed", [])},
                "error": {"type": "SnapshotError", "message": "Run manifest corrupt"},
            }

        return manifest, snapshot_source, snapshot_reason, manifest_mtime

    @staticmethod
    def _get_steps_completed(manifest: dict[str, Any], run: dict[str, Any]) -> list:
        """Get steps completed from manifest or fallback to run data."""
        if "steps" in manifest:
            return [
                k
                for k, v in manifest["steps"].items()
                if v.get("status") in ("COMPLETED", "SKIPPED")
            ]
        return run.get("steps_completed", [])

    @staticmethod
    def _build_steps_progress(manifest: dict[str, Any]) -> list:
        """Build ordered steps progress array."""
        PIPELINE_STEP_ORDER = [
            "ingest",
            "asr",
            "diarization",
            "alignment",
            "chapters",
            "summarize_by_speaker",
            "action_items_assignee",
            "bundle",
        ]

        manifest_steps = manifest.get("steps", {})
        steps_requested = manifest.get("steps_requested", [])

        if steps_requested:
            ordered_steps = steps_requested
        else:
            ordered_steps = [s for s in PIPELINE_STEP_ORDER if s in manifest_steps]
            for s in sorted(manifest_steps.keys()):
                if s not in ordered_steps:
                    ordered_steps.append(s)

        result = []
        for step_name in ordered_steps:
            step_data = manifest_steps.get(step_name, {})
            step_status = step_data.get("status", "PENDING")

            progress_entry = {
                "name": step_name,
                "status": step_status,
                "progress_pct": step_data.get(
                    "progress_pct",
                    1
                    if step_status == "RUNNING"
                    else 0
                    if step_status == "PENDING"
                    else 100
                    if step_status in ("COMPLETED", "SKIPPED")
                    else 0,
                ),
            }

            if step_data.get("progress_message"):
                progress_entry["message"] = step_data["progress_message"]

            if step_data.get("duration_ms") is not None:
                progress_entry["duration_ms"] = step_data["duration_ms"]

            if step_data.get("estimated_remaining_s") is not None:
                progress_entry["estimated_remaining_s"] = step_data["estimated_remaining_s"]

            if step_data.get("started_at"):
                progress_entry["started_at"] = step_data["started_at"]

            if step_data.get("ended_at"):
                progress_entry["ended_at"] = step_data["ended_at"]

            result.append(progress_entry)

        return result

    @staticmethod
    def _derive_status_config(manifest: dict[str, Any]) -> dict[str, Any]:
        """Return status-friendly config."""
        config_source = manifest.get("config") or {}
        config = dict(config_source)
        steps = manifest.get("steps", {})

        asr_step = steps.get("asr", {})
        resolved_asr = asr_step.get("resolved_config") or {}
        asr_config = dict(config.get("asr") or {})

        if resolved_asr:
            if resolved_asr.get("model_id"):
                asr_config.setdefault("model_id", resolved_asr["model_id"])
            if resolved_asr.get("model_name"):
                asr_config.setdefault("model_name", resolved_asr["model_name"])
            elif resolved_asr.get("model_id"):
                asr_config.setdefault("model_name", resolved_asr["model_id"])

            for field in ("source", "device", "language"):
                if resolved_asr.get(field):
                    asr_config[field] = resolved_asr[field]

        config["asr"] = asr_config

        diarization_config = dict(config.get("diarization") or {})
        if diarization_config.get("enabled") is None:
            diarization_config["enabled"] = "diarization" in steps
        config["diarization"] = diarization_config

        return config

    @staticmethod
    def get_run_details(run_id: str) -> dict[str, Any]:
        """Get full manifest for a specific run, with enhanced details."""
        from datetime import datetime

        run = get_index().get_run(run_id)
        if not run:
            raise ValueError("Run not found in index (try refresh?)")

        # Load manifest with fallback
        manifest = RunsService._load_manifest(run.get("manifest_path"))
        if not manifest:
            # Fallback to index data
            manifest = {
                "status": run["status"],
                "steps": {k: {"status": "COMPLETED"} for k in run.get("steps_completed", [])},
                "error": {"type": "SnapshotError", "message": "Run manifest missing or corrupt"},
            }

        status = manifest.get("status", run["status"])
        current_step = manifest.get("current_step")
        updated_at = manifest.get("updated_at")
        last_semantic_progress_at = manifest.get("last_semantic_progress_at")

        # Stale detection
        STALE_THRESHOLD_SECONDS = 90
        is_stalled = False
        if status == "RUNNING" and updated_at:
            try:
                last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                now = datetime.now(UTC)
                elapsed = (now - last_update).total_seconds()

                if elapsed > STALE_THRESHOLD_SECONDS:
                    status = "STALE"
                    is_stalled = True
            except Exception:
                pass

        # Input metadata
        input_meta = manifest.get("input_metadata", {})
        if not input_meta and run.get("input_metadata"):
            input_meta = run["input_metadata"]

        # Artifacts and steps
        artifacts = manifest.get("artifacts_by_type", {})
        steps = RunsService._build_detailed_steps(manifest)

        return {
            "run_id": run_id,
            "status": status,
            "started_at": manifest.get("started_at"),
            "steps_completed": run.get("steps_completed", []),
            "current_step": current_step,
            "updated_at": updated_at,
            "last_semantic_progress_at": last_semantic_progress_at,
            "is_stalled": is_stalled,
            "failure_step": manifest.get("failure_step"),
            "input_metadata": input_meta,
            "config": manifest.get("config", {}),
            "artifacts_availability": artifacts,
            "steps": steps,
        }

    @staticmethod
    def _build_detailed_steps(manifest: dict[str, Any]) -> list:
        """Build detailed steps array with artifacts and errors."""
        steps = []
        if "steps" in manifest:
            for step_name, step_data in manifest["steps"].items():
                step_info = {
                    "name": step_name,
                    "status": step_data.get("status", "PENDING"),
                    "started_at": step_data.get("started_at"),
                    "ended_at": step_data.get("ended_at"),
                    "duration_ms": step_data.get("duration_ms"),
                    "resolved_config": step_data.get("resolved_config"),
                }

                # Add artifacts
                raw_artifacts = step_data.get("artifacts", [])
                api_artifacts = []
                for art in raw_artifacts:
                    if art.get("id"):
                        api_artifacts.append(
                            {
                                "id": art.get("id"),
                                "filename": art.get("filename"),
                                "role": art.get("role"),
                                "produced_by": art.get("produced_by"),
                                "size_bytes": art.get("size_bytes"),
                                "downloadable": art.get("downloadable", False),
                            }
                        )

                if api_artifacts:
                    step_info["artifacts"] = api_artifacts

                # Add error details
                if step_info["status"] == "FAILED":
                    error_info = {}
                    if "error" in step_data:
                        error_info["type"] = step_data["error"].get("type", "UnknownError")
                        error_info["message"] = step_data["error"].get(
                            "message", "No error message available"
                        )
                    else:
                        error_info["type"] = "UnknownError"
                        error_info["message"] = "Step failed without error details"
                    step_info["error"] = error_info

                steps.append(step_info)

        return steps

    @staticmethod
    def get_run_manifest(run_id: str) -> dict[str, Any]:
        """Get run summary and full manifest."""
        run = get_index().get_run(run_id)
        if not run:
            raise ValueError("Run not found in index (try refresh?)")

        manifest_path = Path(run["manifest_path"])
        try:
            data = json.loads(manifest_path.read_text())
            return {"summary": run, "manifest": data}
        except Exception as e:
            raise RuntimeError(f"Failed to read manifest: {str(e)}")
