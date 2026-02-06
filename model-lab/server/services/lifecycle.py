import json
import os
import signal
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.session import SessionRunner

# Concurrency State
_WORKER_LOCK = threading.Lock()
_ACTIVE_RUNS_COUNT = 0
_MAX_CONCURRENT_RUNS = 3
_ACTIVE_RUN_ID: str | None = None


class RunnerBusyError(Exception):
    """Raised when runner is busy and cannot accept new runs."""

    pass


def _runs_root() -> Path:
    return Path(os.environ.get("MODEL_LAB_RUNS_ROOT", "runs")).resolve()


def try_acquire_worker() -> bool:
    global _ACTIVE_RUNS_COUNT
    with _WORKER_LOCK:
        if _ACTIVE_RUNS_COUNT >= _MAX_CONCURRENT_RUNS:
            return False
        _ACTIVE_RUNS_COUNT += 1
        return True


def release_worker() -> None:
    global _ACTIVE_RUNS_COUNT
    with _WORKER_LOCK:
        if _ACTIVE_RUNS_COUNT > 0:
            _ACTIVE_RUNS_COUNT -= 1


def launch_run_worker(
    runner: SessionRunner, run_request_data: dict[str, Any], background: bool = True
) -> dict[str, Any]:
    """
    Launch a run worker subprocess.
    """
    global _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = runner.run_id

    # 1. Write run_request.json
    run_request_path = runner.session_dir / "run_request.json"

    # If run_request.json already exists (retry), load it and merge
    if run_request_path.exists():
        existing_request = json.loads(run_request_path.read_text(encoding="utf-8"))
        existing_request.update(run_request_data)
        run_request_data = existing_request

    # Ensure run_id / input path are set
    run_request_data["run_id"] = runner.run_id
    run_request_data["input_path"] = str(runner.input_path)

    # Atomic write
    run_request_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = run_request_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(run_request_data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(run_request_path)

    # 2. Write initial manifest if not exists (or update status if retry)
    # For retry, we assume manifest exists and caller adjusted it.
    # For new run, we create it.
    manifest_path = runner.manifest_path
    if not manifest_path.exists():
        initial_manifest = {
            "schema_version": run_request_data.get("schema_version", "1"),
            "run_id": runner.run_id,
            "status": "RUNNING",
            "requested_at": run_request_data.get("requested_at"),
            "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": run_request_data.get("source"),
            "use_case_id": run_request_data.get("use_case_id"),
            "pipeline": {
                "steps_preset": run_request_data.get("steps_preset"),
                "steps_requested": run_request_data.get("steps_requested", []),
                "steps_custom": run_request_data.get("steps_custom"),
                "pipeline_template": run_request_data.get("pipeline_template"),
                "preprocessing": run_request_data.get("preprocessing"),
                "pipeline_config": run_request_data.get("pipeline_config"),
                "config_overrides": run_request_data.get("config"),
            },
            "input_metadata": {
                "filename": run_request_data.get("filename_original", "upload"),
                "size_bytes": run_request_data.get("bytes_uploaded"),
                "content_type": run_request_data.get("content_type"),
                "sha256": run_request_data.get("sha256"),
            },
            "steps_completed": [],
            "steps": {},
            "warnings": [],
        }
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tmp_manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp_manifest.write_text(
            json.dumps(initial_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp_manifest.replace(manifest_path)

    # 3. Spawn subprocess
    log_file = runner.session_dir / "worker.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Locate project root for python -m harness
    # Assuming this file is in server/services/lifecycle.py -> ../../../
    project_root = Path(__file__).parent.parent.parent

    # We open file handle for stdout/stderr redirect
    # Note: caller is responsible for ensuring worker eventually exits or we depend on OS cleanup?
    # Actually, subprocess.Popen with start_new_session=True detaches it.

    # Helper for atomic write (imported/redefined to avoid circular imports? No, defined in this file context? No, run_worker has it. We need it here.)
    def atomic_write_json(path: Path, data: Any) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    log_f = open(log_file, "a")  # Append mode for retries
    try:
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "python",
                "-m",
                "harness.run_worker",
                "--run-dir",
                str(runner.session_dir),
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach
            cwd=str(project_root),
        )
    except Exception as e:
        log_f.close()
        raise e

    # 4. Store PID
    pid_file = runner.session_dir / "worker.pid"
    pid_file.write_text(str(proc.pid))

    # 4b. Also store PID in manifest for resilience
    try:
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            m["worker_pid"] = proc.pid
            atomic_write_json(manifest_path, m)
    except Exception:
        pass

    # 5. Handle Background Wait
    if background:

        def _wait_and_release() -> None:
            try:
                proc.wait()
            finally:
                release_worker()
                log_f.close()

        thread = threading.Thread(target=_wait_and_release, daemon=True)
        thread.start()
    else:
        # If foreground, we don't wait here, but caller must handle release?
        # Typically we always use background for API simplicity
        pass

    return {"run_id": runner.run_id, "run_dir": str(runner.session_dir), "worker_pid": proc.pid}


def update_manifest_pid(manifest_path: Path, pid: int) -> None:
    """Persist PID to manifest for robust lifecycle management."""
    try:
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            m["worker_pid"] = pid
            atomic_write_json(manifest_path, m)
    except Exception:
        pass


def update_manifest_status(
    manifest_path: Path, status: str, error: dict[str, str] | None = None
) -> None:
    """Helper to update status atomically."""
    try:
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            # Don't overwrite terminal states unless forcing
            current = m.get("status")
            if current in ["COMPLETED", "FAILED", "CANCELLED"] and status != "CANCELLED":
                return

            m["status"] = status
            if error:
                m["error"] = error
            m["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            atomic_write_json(manifest_path, m)
    except Exception:
        pass


def kill_run(run_id: str) -> tuple[bool, str]:
    """
    Kill a running job.
    Returns (success, outcome_code).
    """
    run_dir = _runs_root() / "sessions" / "unknown" / run_id
    from server.services.runs_index import get_index

    run = get_index().get_run(run_id)
    if not run:
        return False, "not_found"

    manifest_path = Path(run["manifest_path"])
    run_dir = manifest_path.parent
    pid_file = run_dir / "worker.pid"

    # Check if already terminal
    status = run.get("status")
    if status in ["COMPLETED", "FAILED", "CANCELLED", "STALE"]:
        return True, "already_terminal"

    pid = None
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            pass

    # Try looking in manifest if pid_file missing/bad
    if pid is None:
        try:
            m = json.loads(manifest_path.read_text())
            pid = m.get("worker_pid")
        except Exception:
            pass

    if pid is None:
        # Run says running but no PID found -> Forced cleanup
        kill_manifest_update(manifest_path)
        return True, "forced_cancel"

    try:
        os.kill(pid, signal.SIGTERM)
        # We rely on worker to catch SIGTERM and exit cleanly, usually
        # But we force status update just in case
        kill_manifest_update(manifest_path)
        return True, "killed"
    except ProcessLookupError:
        # Already dead
        kill_manifest_update(manifest_path)
        return True, "already_dead"
    except Exception:
        return False, "error"


def kill_manifest_update(manifest_path: Path):
    try:
        m = json.loads(manifest_path.read_text())
        if m.get("status") in ["RUNNING", "STALE"]:
            m["status"] = "CANCELLED"
            m["error"] = {"type": "UserCancelled", "message": "Run cancelled by user."}
            m["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(m, indent=2, sort_keys=True))
            os.replace(tmp, manifest_path)
    except Exception:
        pass


def retry_run(run_id: str, from_step: str | None = None) -> dict[str, Any]:
    """
    Retry a run.
    """

    def _iso_now() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Pure-manifest invalidation: keep this lightweight (no model imports).
    def _reset_step_entry(entry: dict[str, Any]) -> None:
        entry["status"] = "PENDING"
        entry["artifacts"] = []
        entry["warnings"] = []
        entry["ended_at"] = None
        entry.pop("started_at", None)
        entry.pop("duration_ms", None)
        entry.pop("result", None)
        entry.pop("metrics", None)
        entry.pop("error", None)

    # Step dependency graph (names only). Mirrors harness.session.SessionRunner._register_steps.
    _DEPS: dict[str, list[str]] = {
        "ingest": [],
        "asr": ["ingest"],
        "diarization": ["ingest"],
        "alignment": ["asr", "diarization"],
        "chapters": ["alignment"],
        "summarize_by_speaker": ["alignment"],
        "action_items_assignee": ["alignment"],
        "bundle": [
            "ingest",
            "asr",
            "diarization",
            "alignment",
            "chapters",
            "summarize_by_speaker",
            "action_items_assignee",
        ],
    }

    def _collect_dependents(step_name: str) -> list[str]:
        out: list[str] = []
        visited: set[str] = set()

        def dfs(s: str) -> None:
            for name, deps in _DEPS.items():
                if s in deps and name not in visited:
                    visited.add(name)
                    out.append(name)
                    dfs(name)

        dfs(step_name)
        return out

    def _invalidate_step_and_downstream(m: dict[str, Any], step_name: str) -> None:
        steps = m.setdefault("steps", {})

        entry = steps.get(step_name)
        if entry and isinstance(entry, dict):
            _reset_step_entry(entry)

        for dep in _collect_dependents(step_name):
            d_entry = steps.get(dep)
            if d_entry and isinstance(d_entry, dict):
                _reset_step_entry(d_entry)

    from server.services.runs_index import get_index

    run = get_index().get_run(run_id)
    if not run:
        raise ValueError("Run not found")

    manifest_path = Path(run["manifest_path"])
    run_dir = manifest_path.parent

    if not manifest_path.exists():
        raise ValueError("Manifest not found")

    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Idempotence: if already RUNNING with a known PID, do not spawn again.
    if m.get("status") == "RUNNING":
        pid = None
        pid_file = run_dir / "worker.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
            except Exception:
                pid = None
        if pid is None:
            pid = m.get("worker_pid")
        if pid is not None:
            return {"run_id": run_id, "run_dir": str(run_dir), "worker_pid": pid}

    # 1. Acquire lock (only when launching a new worker)
    if not try_acquire_worker():
        raise RunnerBusyError("System busy")

    try:
        # 2. Update manifest for retry
        prior_failure_step = m.get("failure_step")

        m["status"] = "RUNNING"
        m["updated_at"] = _iso_now()
        m["ended_at"] = None
        m["duration_ms"] = None
        m["current_step"] = None
        m.pop("worker_pid", None)

        for k in ["failure_step", "error", "error_message", "error_code", "traceback"]:
            m.pop(k, None)

        step_to_retry = from_step or prior_failure_step
        if step_to_retry:
            _invalidate_step_and_downstream(m, step_to_retry)

        # Write manifest
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, manifest_path)

        # 3. Launch
        runner = SessionRunner(
            input_path=run_dir / "input_file_placeholder",  # overwritten below
            output_dir=_runs_root(),
            resume=True,
            config={"resume_from": str(run_dir)},
        )
        # Correct the input path from manifest/request
        req_path = run_dir / "run_request.json"
        if req_path.exists():
            req = json.loads(req_path.read_text(encoding="utf-8"))
            if "input_path" in req:
                input_candidate = Path(req["input_path"])
                if input_candidate.exists():
                    runner.input_path = input_candidate

        return launch_run_worker(runner, {}, background=True)
    except Exception:
        release_worker()
        raise
