#!/usr/bin/env python3
"""
Run ASR preprocessing matrix on a fixed audio + transcript pair.

Executes scripts/run_asr.py in adhoc mode, then scores output text against
ground-truth with project normalization + WER/CER utilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from harness.metrics_asr import ASRMetrics
from harness.protocol import NormalizationValidator


@dataclass
class Row:
    model: str
    preprocess: str
    status: str
    artifact_path: str | None
    wer: float | None
    cer: float | None
    latency_ms: float | None
    rtf: float | None
    edit_s: int | None
    edit_d: int | None
    edit_i: int | None
    error: str | None


ARTIFACT_RE = re.compile(r"ARTIFACT_PATH:(?P<path>.+)")


def run_case(audio: Path, model: str, device: str, pre: str | None) -> tuple[str, str | None, str]:
    cmd = [
        sys.executable,
        "scripts/run_asr.py",
        "--model",
        model,
        "--audio",
        str(audio),
        "--device",
        device,
    ]
    if pre:
        cmd.extend(["--pre", pre])

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return "failed", None, combined[-2000:]

    m = ARTIFACT_RE.search(combined)
    if not m:
        return "failed", None, "Missing ARTIFACT_PATH in output"
    return "ok", m.group("path").strip(), combined[-1200:]


def score_artifact(artifact_path: Path, reference_text: str) -> dict[str, float | int]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    hyp = payload.get("output", {}).get("text", "") or ""

    normalized_ref = NormalizationValidator.normalize_text(reference_text)
    normalized_hyp = NormalizationValidator.normalize_text(hyp)
    wer, s, d, i = ASRMetrics.calculate_wer(normalized_ref, normalized_hyp)
    cer = ASRMetrics.calculate_cer(normalized_ref, normalized_hyp)

    structural = payload.get("metrics_structural", {})
    return {
        "wer": float(wer),
        "cer": float(cer),
        "latency_ms": float(structural.get("latency_ms", 0.0)),
        "rtf": float(structural.get("rtf", 0.0)),
        "edit_s": int(s),
        "edit_d": int(d),
        "edit_i": int(i),
    }


def write_markdown(path: Path, rows: list[Row]) -> None:
    lines = [
        "# ASR Preprocessing Matrix",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "| model | preprocess | status | WER | CER | latency_ms | RTF | artifact |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.model}` | `{r.preprocess or 'none'}` | {r.status} | "
            f"{'' if r.wer is None else f'{r.wer:.3f}'} | "
            f"{'' if r.cer is None else f'{r.cer:.3f}'} | "
            f"{'' if r.latency_ms is None else f'{r.latency_ms:.1f}'} | "
            f"{'' if r.rtf is None else f'{r.rtf:.3f}'} | "
            f"{r.artifact_path or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ASR preprocessing matrix")
    parser.add_argument("--audio", type=Path, required=True, help="Audio input path")
    parser.add_argument("--text", type=Path, required=True, help="Ground truth text path")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["faster_whisper", "faster_distil_whisper_large_v3"],
        help="Model IDs",
    )
    parser.add_argument(
        "--pre",
        action="append",
        default=[],
        help="Preprocess chain spec (repeatable). Example: trim_silence,normalize_loudness",
    )
    parser.add_argument("--device", default="mps", help="Device to pass to run_asr")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/hf_sprint_2026q1/preprocess_matrix"),
        help="Output directory",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio}")
    if not args.text.exists():
        raise FileNotFoundError(f"Text not found: {args.text}")

    preprocesses = [""] + args.pre
    reference_text = args.text.read_text(encoding="utf-8").strip()
    rows: list[Row] = []

    for model in args.models:
        for pre in preprocesses:
            status, artifact_path, err = run_case(args.audio, model, args.device, pre or None)
            if status != "ok" or artifact_path is None:
                rows.append(
                    Row(
                        model=model,
                        preprocess=pre,
                        status="failed",
                        artifact_path=None,
                        wer=None,
                        cer=None,
                        latency_ms=None,
                        rtf=None,
                        edit_s=None,
                        edit_d=None,
                        edit_i=None,
                        error=err,
                    )
                )
                continue

            metrics = score_artifact(PROJECT_ROOT / artifact_path, reference_text)
            rows.append(
                Row(
                    model=model,
                    preprocess=pre,
                    status="ok",
                    artifact_path=artifact_path,
                    wer=metrics["wer"],
                    cer=metrics["cer"],
                    latency_ms=metrics["latency_ms"],
                    rtf=metrics["rtf"],
                    edit_s=metrics["edit_s"],
                    edit_d=metrics["edit_d"],
                    edit_i=metrics["edit_i"],
                    error=None,
                )
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_dir / f"matrix_{timestamp}.json"
    out_csv = args.out_dir / f"matrix_{timestamp}.csv"
    out_md = args.out_dir / f"matrix_{timestamp}.md"

    out_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "audio": str(args.audio),
                "text": str(args.text),
                "models": args.models,
                "preprocesses": preprocesses,
                "rows": [r.__dict__ for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)

    write_markdown(out_md, rows)

    print(f"JSON: {out_json}")
    print(f"CSV: {out_csv}")
    print(f"MD: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

