#!/usr/bin/env python3
"""
Cross-runtime benchmark + repo-check matrix.

What it does:
1) Checks official vs community model repos on Hugging Face.
2) Benchmarks local runtime backends (transformers/mlx/onnx/faster-whisper/cpp/streaming).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_RE = re.compile(r"ARTIFACT_PATH:(?P<path>.+)")


@dataclass
class RepoCheck:
    family: str
    lane: str
    repo_id: str
    exists: bool
    private: bool | None
    downloads: int | None
    likes: int | None
    last_modified: str | None
    error: str | None


@dataclass
class BenchCase:
    lane: str
    model_id: str
    command: list[str]
    env: dict[str, str]


def run_repo_checks() -> list[RepoCheck]:
    api = HfApi()
    repos = [
        ("voxtral_realtime", "official_transformers", "mistralai/Voxtral-Mini-4B-Realtime-2602"),
        ("voxtral_realtime", "mlx_community", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"),
        ("voxtral_realtime", "gguf_community", "andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF"),
        ("whisper_small", "official_openai", "openai/whisper-small"),
        ("whisper_small", "mlx_community", "mlx-community/whisper-small.en-asr-fp16"),
        ("whisper_small", "onnx_community", "notebook-community/whisper-small-onnx"),
        ("whisper_large_turbo", "gguf_community", "xkeyC/whisper-large-v3-turbo-gguf"),
    ]

    rows: list[RepoCheck] = []
    for family, lane, repo_id in repos:
        try:
            info = api.model_info(repo_id)
            rows.append(
                RepoCheck(
                    family=family,
                    lane=lane,
                    repo_id=repo_id,
                    exists=True,
                    private=getattr(info, "private", None),
                    downloads=getattr(info, "downloads", None),
                    likes=getattr(info, "likes", None),
                    last_modified=info.last_modified.isoformat() if getattr(info, "last_modified", None) else None,
                    error=None,
                )
            )
        except Exception as exc:
            rows.append(
                RepoCheck(
                    family=family,
                    lane=lane,
                    repo_id=repo_id,
                    exists=False,
                    private=None,
                    downloads=None,
                    likes=None,
                    last_modified=None,
                    error=str(exc),
                )
            )
    return rows


def build_bench_cases(dataset: str, stream_dataset: str) -> list[BenchCase]:
    return [
        BenchCase(
            lane="official_transformers",
            model_id="whisper",
            command=[sys.executable, "scripts/run_asr.py", "--model", "whisper", "--dataset", dataset, "--device", "cpu"],
            env={},
        ),
        BenchCase(
            lane="transformers_distilled",
            model_id="distil_whisper",
            command=[sys.executable, "scripts/run_asr.py", "--model", "distil_whisper", "--dataset", dataset, "--device", "cpu"],
            env={},
        ),
        BenchCase(
            lane="faster_whisper",
            model_id="faster_whisper",
            command=[sys.executable, "scripts/run_asr.py", "--model", "faster_whisper", "--dataset", dataset, "--device", "cpu"],
            env={},
        ),
        BenchCase(
            lane="onnx",
            model_id="nb_whisper_small_onnx",
            command=[sys.executable, "scripts/run_asr.py", "--model", "nb_whisper_small_onnx", "--dataset", dataset, "--device", "cpu"],
            env={},
        ),
        BenchCase(
            lane="mlx",
            model_id="mlx_whisper",
            command=[sys.executable, "scripts/run_asr.py", "--model", "mlx_whisper", "--dataset", dataset, "--device", "mps"],
            env={},
        ),
        BenchCase(
            lane="cpp",
            model_id="whisper_cpp",
            command=[sys.executable, "scripts/run_asr.py", "--model", "whisper_cpp", "--dataset", dataset, "--device", "cpu"],
            env={},
        ),
        BenchCase(
            lane="realtime_voxtral",
            model_id="voxtral",
            command=[sys.executable, "scripts/run_asr_stream.py", "--model", "voxtral", "--dataset", stream_dataset, "--device", "cpu", "--chunk-ms", "160"],
            env={"VOXTRAL_BACKEND": "mock"},
        ),
        BenchCase(
            lane="realtime_voxtral_rt2602",
            model_id="voxtral_realtime_2602",
            command=[sys.executable, "scripts/run_asr_stream.py", "--model", "voxtral_realtime_2602", "--dataset", stream_dataset, "--device", "cpu", "--chunk-ms", "160"],
            env={"VOXTRAL_REALTIME_BACKEND": "mock"},
        ),
    ]


def _extract_artifact(stdout: str) -> str | None:
    m = ARTIFACT_RE.search(stdout)
    return m.group("path").strip() if m else None


def run_case(case: BenchCase, timeout_sec: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(case.env)
    started = datetime.now(UTC).isoformat()
    try:
        proc = subprocess.run(
            case.command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        status = "ok" if proc.returncode == 0 else "failed"
        return {
            "lane": case.lane,
            "model_id": case.model_id,
            "status": status,
            "exit_code": proc.returncode,
            "artifact_path": _extract_artifact(proc.stdout),
            "started_at": started,
            "stdout_tail": (proc.stdout or "")[-800:],
            "stderr_tail": (proc.stderr or "")[-800:],
            "command": " ".join(case.command),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "lane": case.lane,
            "model_id": case.model_id,
            "status": "timeout",
            "exit_code": None,
            "artifact_path": None,
            "started_at": started,
            "stdout_tail": ((exc.stdout or "") if isinstance(exc.stdout, str) else "")[-800:],
            "stderr_tail": ((exc.stderr or "") if isinstance(exc.stderr, str) else "")[-800:],
            "command": " ".join(case.command),
        }


def write_markdown(path: Path, repos: list[RepoCheck], bench: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Runtime Matrix Benchmark")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append("## Repo Check (Official vs Community)")
    lines.append("")
    lines.append("| family | lane | repo | exists | downloads | likes | last_modified |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for r in repos:
        lines.append(
            f"| {r.family} | {r.lane} | `{r.repo_id}` | {r.exists} | {r.downloads or 0} | {r.likes or 0} | {r.last_modified or '-'} |"
        )
    lines.append("")
    lines.append("## Local Benchmark")
    lines.append("")
    lines.append("| lane | model_id | status | exit_code | artifact |")
    lines.append("|---|---|---|---:|---|")
    for b in bench:
        lines.append(
            f"| {b['lane']} | `{b['model_id']}` | {b['status']} | {b['exit_code'] if b['exit_code'] is not None else '-'} | {b['artifact_path'] or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official/community check + runtime benchmarks")
    parser.add_argument("--dataset", default="asr_smoke_v1", help="Batch ASR dataset")
    parser.add_argument("--stream-dataset", default="asr_smoke_v1", help="Streaming ASR dataset")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-case timeout")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/frontier/runtime_matrix.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/frontier/runtime_matrix.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/frontier/runtime_matrix.md"),
    )
    args = parser.parse_args()

    repos = run_repo_checks()
    cases = build_bench_cases(args.dataset, args.stream_dataset)
    bench_rows = [run_case(case, args.timeout_sec) for case in cases]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "stream_dataset": args.stream_dataset,
        "repo_checks": [asdict(r) for r in repos],
        "benchmarks": bench_rows,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lane", "model_id", "status", "exit_code", "artifact_path", "command"],
        )
        writer.writeheader()
        for row in bench_rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    write_markdown(args.out_md, repos, bench_rows)

    ok = sum(1 for r in bench_rows if r["status"] == "ok")
    print(f"✓ Runtime matrix complete: ok={ok}/{len(bench_rows)}")
    print(f"JSON: {args.out_json}")
    print(f"CSV: {args.out_csv}")
    print(f"MD: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
