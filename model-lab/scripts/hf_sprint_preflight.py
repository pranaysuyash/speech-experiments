#!/usr/bin/env python3
"""
Preflight checks for the HF Pro sprint.

What this catches early:
- Missing HF auth env vars (needed for gated models like pyannote)
- Missing python modules for known "heavy" models (TF, NeMo, onnxruntime, etc.)
- Missing local dataset/audio files referenced by golden datasets
- Missing external tools (ffmpeg)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.env import load_dotenv_if_present


def _can_import(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    name: str
    detail: str


def _check_ffmpeg() -> CheckResult:
    if shutil.which("ffmpeg") is None:
        return CheckResult(False, "ffmpeg", "Missing ffmpeg (brew install ffmpeg)")
    return CheckResult(True, "ffmpeg", "ok")


def _check_hf_auth() -> CheckResult:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return CheckResult(True, "hf_auth", "ok")
    return CheckResult(
        False,
        "hf_auth",
        "Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN (required for gated HF assets; rotate token if leaked)",
    )


def _check_model_prereqs(model_id: str) -> list[CheckResult]:
    results: list[CheckResult] = []

    # Map of model_id -> python module(s) required to even attempt loading.
    required_modules: dict[str, list[str]] = {
        "pyannote_diarization": ["pyannote.audio"],
        "yamnet": ["tensorflow", "tensorflow_hub"],
        "clap": ["laion_clap"],
        "demucs": ["demucs"],
        "deepfilternet": ["deepfilternet"],
        "nb_whisper_small_onnx": ["onnxruntime"],
        "nemotron_streaming": ["nemo.collections.asr"],
        "parakeet_multitalker": ["nemo.collections.asr"],
    }

    for mod in required_modules.get(model_id, []):
        if not _can_import(mod):
            results.append(CheckResult(False, f"{model_id}:{mod}", f"Missing python module: {mod}"))
        else:
            results.append(CheckResult(True, f"{model_id}:{mod}", "ok"))

    # Special-case: whisper.cpp needs a binary + local model path configured.
    if model_id == "whisper_cpp":
        if shutil.which("whisper-cli") is None:
            results.append(
                CheckResult(False, "whisper_cpp:binary", "Missing whisper-cli in PATH")
            )
        else:
            results.append(CheckResult(True, "whisper_cpp:binary", "ok"))

        cfg_path = PROJECT_ROOT / "models" / "whisper_cpp" / "config.yaml"
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        whisper_cfg = cfg.get("whisper_cpp") or {}
        model_path = whisper_cfg.get("model_path")
        if not model_path:
            results.append(
                CheckResult(
                    False,
                    "whisper_cpp:model_path",
                    "Missing models/whisper_cpp/config.yaml whisper_cpp.model_path",
                )
            )
        else:
            results.append(CheckResult(True, "whisper_cpp:model_path", "ok"))

    return results


def _check_dataset_files(dataset_id: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    dataset_path = PROJECT_ROOT / "data" / "golden" / f"{dataset_id}.yaml"
    if not dataset_path.exists():
        return [CheckResult(False, f"dataset:{dataset_id}", f"Missing {dataset_path}")]

    try:
        dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return [
            CheckResult(False, f"dataset:{dataset_id}", f"Failed to parse YAML: {exc}"),
        ]

    cases = dataset.get("cases") or []
    if not isinstance(cases, list) or not cases:
        results.append(CheckResult(True, f"dataset:{dataset_id}", "No cases to check"))
        return results

    for case in cases:
        if not isinstance(case, dict):
            continue
        audio_rel = case.get("audio_path")
        truth_rel = case.get("ground_truth_path")
        if audio_rel:
            audio_path = (dataset_path.parent / audio_rel).resolve()
            results.append(
                CheckResult(audio_path.exists(), f"{dataset_id}:audio", str(audio_path))
            )
        if truth_rel:
            truth_path = (dataset_path.parent / truth_rel).resolve()
            results.append(
                CheckResult(truth_path.exists(), f"{dataset_id}:truth", str(truth_path))
            )

    return results


def main() -> int:
    load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="HF Sprint preflight checks")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/hf_sprint_2026q1.yaml"),
        help="Sprint config YAML",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero if any check fails",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    agents = cfg.get("agents") or {}
    datasets = cfg.get("datasets") or {}
    audio_inputs = cfg.get("audio_inputs") or {}

    checks: list[CheckResult] = []

    checks.append(_check_ffmpeg())

    # HF auth is only strictly required for some models, but we surface it early.
    checks.append(_check_hf_auth())

    # Local audio inputs used by run_bench tasks.
    for key, rel in (audio_inputs or {}).items():
        p = (PROJECT_ROOT / str(rel)).resolve()
        checks.append(CheckResult(p.exists(), f"audio_inputs:{key}", str(p)))

    # Dataset checks:
    # - ASR + streaming ASR datasets are resolved by the runner scripts (not YAML files).
    # - VAD/diarization/TTS/V2V use golden YAML datasets.
    try:
        from scripts.run_asr import get_dataset_files as _asr_get_files
    except Exception:
        _asr_get_files = None
    try:
        from scripts.run_asr_stream import resolve_dataset as _asr_stream_resolve
    except Exception:
        _asr_stream_resolve = None

    def _check_asr_dataset(dataset_id: str) -> None:
        if _asr_get_files is None:
            checks.append(
                CheckResult(
                    False,
                    f"asr_dataset:{dataset_id}",
                    "Unable to import scripts.run_asr.get_dataset_files",
                )
            )
            return
        try:
            audio_path, text_path = _asr_get_files(dataset_id)
            checks.append(CheckResult(Path(audio_path).exists(), f"asr:{dataset_id}:audio", str(audio_path)))
            if text_path is not None:
                checks.append(CheckResult(Path(text_path).exists(), f"asr:{dataset_id}:text", str(text_path)))
        except Exception as exc:
            checks.append(CheckResult(False, f"asr_dataset:{dataset_id}", f"Unknown dataset: {exc}"))

    def _check_asr_stream_dataset(dataset_id: str) -> None:
        if _asr_stream_resolve is None:
            checks.append(
                CheckResult(
                    False,
                    f"asr_stream_dataset:{dataset_id}",
                    "Unable to import scripts.run_asr_stream.resolve_dataset",
                )
            )
            return
        try:
            audio_path, _, _ = _asr_stream_resolve(dataset_id)
            checks.append(CheckResult(Path(audio_path).exists(), f"asr_stream:{dataset_id}:audio", str(audio_path)))
        except Exception as exc:
            checks.append(CheckResult(False, f"asr_stream_dataset:{dataset_id}", f"Unknown dataset: {exc}"))

    # ASR datasets (batch + streaming share dataset IDs in config)
    for key in ("asr_smoke", "asr_primary", "asr_secondary"):
        dataset_id = datasets.get(key)
        if dataset_id:
            _check_asr_dataset(str(dataset_id))
            _check_asr_stream_dataset(str(dataset_id))

    # Golden YAML datasets for other capabilities
    for key in ("diarization_smoke", "vad_smoke", "tts_smoke", "v2v_smoke"):
        dataset_id = datasets.get(key)
        if dataset_id:
            checks.extend(_check_dataset_files(str(dataset_id)))

    # Model prereqs for all models in the sprint config
    model_ids: set[str] = set()
    for agent_cfg in agents.values():
        for mid in (agent_cfg or {}).get("include_models", []) or []:
            model_ids.add(str(mid))
    for model_id in sorted(model_ids):
        checks.extend(_check_model_prereqs(model_id))

    failed = [c for c in checks if not c.ok]
    ok = [c for c in checks if c.ok]

    print("HF Sprint Preflight")
    print(f"Config: {args.config}")
    print(f"Checks: {len(checks)} (ok={len(ok)} failed={len(failed)})")
    if failed:
        print("\nFailed:")
        for c in failed:
            print(f"- {c.name}: {c.detail}")
    else:
        print("\nAll checks passed.")

    return 1 if (args.fail_on_warn and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
