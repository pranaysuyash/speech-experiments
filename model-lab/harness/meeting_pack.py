from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


MEETING_PACK_SCHEMA_VERSION = "meeting_pack_bundle_manifest.v0.1"


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes_if_changed(path: Path, data: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_bytes() == data:
                return False
        except Exception:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _find_manifest_artifact_paths(manifest: Dict[str, Any], step_name: str) -> List[Path]:
    step = (manifest.get("steps") or {}).get(step_name) or {}
    artifacts = step.get("artifacts") or []
    out: List[Path] = []
    for a in artifacts:
        if not isinstance(a, dict):
            continue
        p = a.get("path")
        if isinstance(p, str) and p:
            out.append(Path(p))
    return out


def _discover_run_dir_artifact(run_dir: Path, patterns: List[str]) -> Optional[Path]:
    artifacts_dir = run_dir / "artifacts"
    for pat in patterns:
        hit = _pick_first_existing(sorted(artifacts_dir.glob(pat)))
        if hit:
            return hit
    return None


def _normalize_segments(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = raw.get("output", raw)
    segments = output.get("segments") or []
    normalized: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start_s = seg.get("start_s", seg.get("start"))
        end_s = seg.get("end_s", seg.get("end"))
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker_id", seg.get("speaker"))
        if start_s is None or end_s is None:
            continue
        normalized.append(
            {
                "speaker": speaker,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "text": text,
            }
        )
    return normalized


def _render_summary_md(summary_artifact: Dict[str, Any]) -> str:
    output = summary_artifact.get("output", summary_artifact)
    speaker_summaries = output.get("speaker_summaries") or {}
    lines: List[str] = ["# Summary", ""]
    if not speaker_summaries:
        return "\n".join(lines).rstrip() + "\n"

    lines.append("## Summary by speaker")
    lines.append("")
    for speaker in sorted(speaker_summaries.keys()):
        bullets = speaker_summaries.get(speaker) or []
        if not bullets:
            continue
        lines.append(f"### {speaker}")
        for b in bullets:
            if isinstance(b, str) and b.strip():
                lines.append(f"- {b.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extract_action_items_rows(action_items_artifact: Dict[str, Any]) -> List[Dict[str, str]]:
    output = action_items_artifact.get("output", action_items_artifact)
    items = output.get("action_items") or []
    rows: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "assignee": str(item.get("assignee") or ""),
                "priority": str(item.get("priority") or ""),
                "text": str(item.get("text") or ""),
            }
        )
    return rows


@dataclass(frozen=True)
class BundleArtifact:
    name: str
    rel_path: str
    content_type: str


_EXPECTED_ARTIFACTS: List[BundleArtifact] = [
    BundleArtifact("transcript.json", "bundle/transcript.json", "application/json"),
    BundleArtifact("summary.md", "bundle/summary.md", "text/markdown"),
    BundleArtifact("action_items.csv", "bundle/action_items.csv", "text/csv"),
    BundleArtifact("decisions.md", "bundle/decisions.md", "text/markdown"),
]

def _canonicalize_iso_utc(ts: Optional[str]) -> Optional[str]:
    if not ts or not isinstance(ts, str):
        return None
    t = ts.strip()
    if not t:
        return None
    # SessionRunner timestamps are UTC-like but may omit trailing "Z".
    if t.endswith("Z"):
        return t
    return t + "Z"


def _deterministic_generated_at(run_manifest: Dict[str, Any]) -> str:
    # Prefer stable lifecycle timestamps to make bundle idempotent for the same run state.
    for k in ("ended_at", "started_at", "updated_at"):
        v = _canonicalize_iso_utc(run_manifest.get(k))
        if v:
            return v
    return _now_iso_utc()


def build_meeting_pack(run_dir: Path) -> Dict[str, Any]:
    """
    Creates/updates a deterministic bundle folder in a run directory:
      - bundle/bundle_manifest.json (always)
      - bundle/transcript.json (if ASR/alignment exists)
      - bundle/summary.md (if summary exists)
      - bundle/action_items.csv (if action items exist)
      - bundle/decisions.md (only if already exists as an artifact file; no synthesis)

    Returns a dict with `written_paths` and `manifest_path`.
    """
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    manifest = _read_json(manifest_path)
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    absent: List[Dict[str, str]] = []

    # --- transcript.json ---
    transcript_out = bundle_dir / "transcript.json"
    alignment_path = _pick_first_existing(_find_manifest_artifact_paths(manifest, "alignment")) or _discover_run_dir_artifact(
        run_dir, ["alignment*.json"]
    )
    asr_path = _pick_first_existing(_find_manifest_artifact_paths(manifest, "asr")) or _discover_run_dir_artifact(
        run_dir, ["asr*.json"]
    )

    transcript_src = alignment_path or asr_path
    if transcript_src:
        segments = _normalize_segments(_read_json(transcript_src))
        payload = (json.dumps({"segments": segments}, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if _atomic_write_bytes_if_changed(transcript_out, payload):
            written.append(transcript_out)
    else:
        absent.append({"name": "transcript.json", "reason": "No alignment/asr artifact found"})

    # --- summary.md ---
    summary_out = bundle_dir / "summary.md"
    summary_path = _pick_first_existing(_find_manifest_artifact_paths(manifest, "summarize_by_speaker")) or _discover_run_dir_artifact(
        run_dir, ["summary_by_speaker_*.json", "summary*.json"]
    )
    if summary_path:
        md = _render_summary_md(_read_json(summary_path))
        if _atomic_write_bytes_if_changed(summary_out, md.encode("utf-8")):
            written.append(summary_out)
    else:
        absent.append({"name": "summary.md", "reason": "No summary artifact found"})

    # --- action_items.csv ---
    action_items_out = bundle_dir / "action_items.csv"
    action_items_path = _pick_first_existing(_find_manifest_artifact_paths(manifest, "action_items_assignee")) or _discover_run_dir_artifact(
        run_dir, ["action_items_*.json"]
    )
    if action_items_path:
        rows = _extract_action_items_rows(_read_json(action_items_path))
        s = io.StringIO()
        writer = csv.DictWriter(s, fieldnames=["assignee", "priority", "text"], lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        if _atomic_write_bytes_if_changed(action_items_out, s.getvalue().encode("utf-8")):
            written.append(action_items_out)
    else:
        absent.append({"name": "action_items.csv", "reason": "No action items artifact found"})

    # --- decisions.md (only pass-through if it exists; no synthesis) ---
    decisions_out = bundle_dir / "decisions.md"
    decisions_src = _discover_run_dir_artifact(run_dir, ["decisions.md", "decisions_*.md"])
    if decisions_src:
        if _atomic_write_bytes_if_changed(decisions_out, decisions_src.read_bytes()):
            written.append(decisions_out)
    else:
        absent.append({"name": "decisions.md", "reason": "No decisions artifact found"})

    # Build manifest of what exists
    artifacts: List[Dict[str, Any]] = []
    for a in _EXPECTED_ARTIFACTS:
        p = run_dir / a.rel_path
        if not p.exists():
            continue
        stat = p.stat()
        artifacts.append(
            {
                "name": a.name,
                "rel_path": a.rel_path.replace("\\", "/"),
                "bytes": int(stat.st_size),
                "sha256": _sha256_file(p),
                "content_type": a.content_type,
            }
        )

    bundle_manifest = {
        "schema_version": MEETING_PACK_SCHEMA_VERSION,
        "run_id": manifest.get("run_id", run_dir.name),
        "generated_at": _deterministic_generated_at(manifest),
        "artifacts": sorted(artifacts, key=lambda x: x.get("name", "")),
        "absent": sorted(absent, key=lambda x: x.get("name", "")),
    }

    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    manifest_bytes = (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if _atomic_write_bytes_if_changed(bundle_manifest_path, manifest_bytes):
        written.append(bundle_manifest_path)

    return {
        "manifest_path": str(bundle_manifest_path),
        "written_paths": [str(p) for p in written],
    }
