#!/usr/bin/env python3
"""
Generate a multi-agent HF sprint execution plan from registry metadata.

Outputs:
  - <output>/plan.json
  - <output>/assignment_matrix.csv
  - <output>/agent_queues/<agent>.json
  - <output>/agent_queues/<agent>.md
  - <output>/dispatch_commands.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATUS_ORDER = {
    "production": 0,
    "candidate": 1,
    "experimental": 2,
    "deprecated": 3,
}

CAPABILITY_ORDER = {
    "asr": 0,
    "asr_stream": 0,
    "diarization": 1,
    "vad": 1,
    "tts": 1,
    "v2v": 1,
    "enhance": 2,
    "separate": 2,
    "classify": 2,
    "embed": 2,
    "music_transcription": 3,
    "chat": 3,
    "mt": 3,
}

DATASET_ORDER = {
    "asr_smoke_v1": 0,
    "vad_smoke_v1": 0,
    "diar_smoke_v1": 0,
    "tts_smoke_v1": 0,
    "v2v_smoke_v1": 0,
    "primary": 1,
    "ux_primary": 2,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "agents" not in data or not data["agents"]:
        raise ValueError("Config must define non-empty 'agents'")
    if "defaults" not in data:
        raise ValueError("Config must define 'defaults'")
    if "datasets" not in data:
        raise ValueError("Config must define 'datasets'")
    return data


def load_registry_snapshot() -> dict[str, dict[str, Any]]:
    from harness.registry import ModelRegistry

    models: dict[str, dict[str, Any]] = {}
    for model_id in ModelRegistry.list_models():
        meta = ModelRegistry.get_model_metadata(model_id) or {}
        models[model_id] = {
            "model_id": model_id,
            "status": str(meta.get("status") or "experimental"),
            "capabilities": list(meta.get("capabilities") or []),
            "hardware": list(meta.get("hardware") or []),
            "version": str(meta.get("version") or "unknown"),
        }
    return models


def assign_models_to_agents(
    registry_models: dict[str, dict[str, Any]],
    agent_config: dict[str, Any],
    fallback_agent: str,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    seen: dict[str, str] = {}

    for agent_id, cfg in agent_config.items():
        include = cfg.get("include_models") or []
        for model_id in include:
            if model_id in seen:
                prev = seen[model_id]
                raise ValueError(f"Model '{model_id}' assigned twice: {prev}, {agent_id}")
            seen[model_id] = agent_id
            if model_id in registry_models:
                assignments[model_id] = agent_id

    if fallback_agent not in agent_config:
        raise ValueError(f"Fallback agent '{fallback_agent}' is not defined in agents")

    for model_id in registry_models:
        assignments.setdefault(model_id, fallback_agent)

    return assignments


def _task_id(model_id: str, capability: str, dataset: str | None) -> str:
    dataset_key = dataset or "na"
    return f"{model_id}__{capability}__{dataset_key}".replace("/", "_").replace(".", "_")


def _priority_tuple(status: str, capability: str, dataset: str | None) -> tuple[int, int, int]:
    s = STATUS_ORDER.get(status, 9)
    c = CAPABILITY_ORDER.get(capability, 9)
    d = DATASET_ORDER.get(dataset or "", 9)
    return (s, c, d)


def _add_ready_task(
    tasks: list[dict[str, Any]],
    *,
    model_id: str,
    status: str,
    capability: str,
    dataset: str | None,
    command: str,
    artifact_hint: str,
    notes: str = "",
) -> None:
    tasks.append(
        {
            "task_id": _task_id(model_id, capability, dataset),
            "model_id": model_id,
            "status": status,
            "capability": capability,
            "dataset": dataset,
            "mode": "ready",
            "command": command,
            "artifact_hint": artifact_hint,
            "notes": notes,
            "priority_rank": list(_priority_tuple(status, capability, dataset)),
        }
    )


def _add_manual_task(
    tasks: list[dict[str, Any]],
    *,
    model_id: str,
    status: str,
    capability: str,
    notes: str,
) -> None:
    tasks.append(
        {
            "task_id": _task_id(model_id, capability, "manual"),
            "model_id": model_id,
            "status": status,
            "capability": capability,
            "dataset": None,
            "mode": "manual",
            "command": None,
            "artifact_hint": f"runs/{model_id}/{capability}",
            "notes": notes,
            "priority_rank": list(_priority_tuple(status, capability, None)),
        }
    )


def build_tasks_for_model(
    model_id: str,
    model_meta: dict[str, Any],
    *,
    datasets: dict[str, str],
    audio_inputs: dict[str, str],
    device: str,
    chunk_ms: int,
) -> list[dict[str, Any]]:
    status = str(model_meta.get("status") or "experimental")
    capabilities = list(model_meta.get("capabilities") or [])

    tasks: list[dict[str, Any]] = []

    for cap in capabilities:
        if cap == "asr":
            for ds_key in ("asr_smoke", "asr_primary", "asr_secondary"):
                dataset = datasets.get(ds_key)
                if not dataset:
                    continue
                cmd = (
                    "uv run python scripts/run_asr.py "
                    f"--model {model_id} --dataset {dataset} --device {device}"
                )
                _add_ready_task(
                    tasks,
                    model_id=model_id,
                    status=status,
                    capability=cap,
                    dataset=dataset,
                    command=cmd,
                    artifact_hint=f"runs/{model_id}/asr",
                )
            continue

        if cap == "asr_stream":
            for ds_key in ("asr_smoke", "asr_primary"):
                dataset = datasets.get(ds_key)
                if not dataset:
                    continue
                cmd = (
                    "uv run python scripts/run_asr_stream.py "
                    f"--model {model_id} --dataset {dataset} --device {device} "
                    f"--chunk-ms {chunk_ms}"
                )
                _add_ready_task(
                    tasks,
                    model_id=model_id,
                    status=status,
                    capability=cap,
                    dataset=dataset,
                    command=cmd,
                    artifact_hint=f"runs/{model_id}/asr_stream",
                )
            continue

        if cap == "tts":
            dataset = datasets.get("tts_smoke")
            if dataset:
                cmd = (
                    "uv run python scripts/run_tts.py "
                    f"--model {model_id} --dataset {dataset} --device {device}"
                )
                _add_ready_task(
                    tasks,
                    model_id=model_id,
                    status=status,
                    capability=cap,
                    dataset=dataset,
                    command=cmd,
                    artifact_hint=f"runs/{model_id}/tts",
                )
            continue

        if cap == "vad":
            dataset = datasets.get("vad_smoke")
            if dataset:
                cmd = (
                    "uv run python scripts/run_vad.py "
                    f"--model {model_id} --dataset {dataset} --device {device}"
                )
                _add_ready_task(
                    tasks,
                    model_id=model_id,
                    status=status,
                    capability=cap,
                    dataset=dataset,
                    command=cmd,
                    artifact_hint=f"runs/{model_id}/vad",
                )
            continue

        if cap == "diarization":
            dataset = datasets.get("diarization_smoke")
            if dataset:
                cmd = (
                    "uv run python scripts/run_diarization.py "
                    f"--model {model_id} --dataset {dataset} --device {device}"
                )
                _add_ready_task(
                    tasks,
                    model_id=model_id,
                    status=status,
                    capability=cap,
                    dataset=dataset,
                    command=cmd,
                    artifact_hint=f"runs/{model_id}/diarization",
                )
            continue

        if cap == "v2v":
            dataset = datasets.get("v2v_smoke")
            if dataset:
                cmd = (
                    "uv run python scripts/run_v2v.py "
                    f"--model {model_id} --dataset {dataset} --device {device}"
                )
                _add_ready_task(
                    tasks,
                    model_id=model_id,
                    status=status,
                    capability=cap,
                    dataset=dataset,
                    command=cmd,
                    artifact_hint=f"runs/{model_id}/v2v",
                )
            continue

        if cap in {"enhance", "classify", "embed", "separate"}:
            surface = cap
            audio = audio_inputs.get("bench_audio", "data/audio/clean_speech_10s.wav")
            if cap == "separate":
                audio = audio_inputs.get("separate_audio", audio)

            cmd = (
                "uv run python scripts/run_bench.py "
                f"{surface} --model {model_id} --audio {audio} --device {device} --output"
            )
            if cap == "enhance" and audio_inputs.get("enhance_clean_audio"):
                clean_audio = audio_inputs["enhance_clean_audio"]
                cmd += f" --clean {clean_audio}"

            _add_ready_task(
                tasks,
                model_id=model_id,
                status=status,
                capability=cap,
                dataset="bench_audio",
                command=cmd,
                artifact_hint="bench/results",
            )
            continue

        if cap in {"music_transcription", "chat", "mt"}:
            _add_manual_task(
                tasks,
                model_id=model_id,
                status=status,
                capability=cap,
                notes=f"No dedicated CLI runner configured for capability '{cap}'.",
            )
            continue

        _add_manual_task(
            tasks,
            model_id=model_id,
            status=status,
            capability=cap,
            notes=f"Unknown capability '{cap}' requires manual handling.",
        )

    return tasks


def generate_plan(
    config: dict[str, Any],
    registry_models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    defaults = config["defaults"]
    datasets = config["datasets"]
    audio_inputs = config.get("audio_inputs", {})
    agents = config["agents"]

    assignments = assign_models_to_agents(
        registry_models=registry_models,
        agent_config=agents,
        fallback_agent=defaults["unassigned_agent"],
    )

    queues: dict[str, dict[str, Any]] = {}
    for agent_id, agent_cfg in agents.items():
        queues[agent_id] = {
            "agent_id": agent_id,
            "description": str(agent_cfg.get("description") or ""),
            "models": [],
            "tasks": [],
        }

    for model_id, model_meta in sorted(registry_models.items()):
        agent_id = assignments[model_id]
        model_tasks = build_tasks_for_model(
            model_id=model_id,
            model_meta=model_meta,
            datasets=datasets,
            audio_inputs=audio_inputs,
            device=str(defaults.get("device") or "cpu"),
            chunk_ms=int(defaults.get("chunk_ms") or 160),
        )

        model_entry = {
            "model_id": model_id,
            "status": model_meta["status"],
            "capabilities": model_meta["capabilities"],
            "hardware": model_meta["hardware"],
            "version": model_meta["version"],
            "task_count": len(model_tasks),
        }
        queues[agent_id]["models"].append(model_entry)
        queues[agent_id]["tasks"].extend(model_tasks)

    for agent_queue in queues.values():
        agent_queue["tasks"].sort(
            key=lambda t: (
                tuple(t["priority_rank"]),
                t["model_id"],
                t["capability"],
                t.get("dataset") or "",
            )
        )
        agent_queue["models"].sort(key=lambda m: m["model_id"])
        agent_queue["summary"] = {
            "model_count": len(agent_queue["models"]),
            "task_count": len(agent_queue["tasks"]),
            "ready_tasks": sum(1 for t in agent_queue["tasks"] if t["mode"] == "ready"),
            "manual_tasks": sum(1 for t in agent_queue["tasks"] if t["mode"] == "manual"),
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "sprint": config.get("sprint", {}),
        "defaults": defaults,
        "assignments": assignments,
        "queues": queues,
    }


def _write_agent_markdown(path: Path, queue: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Agent Queue: {queue['agent_id']}")
    lines.append("")
    if queue.get("description"):
        lines.append(queue["description"])
        lines.append("")

    summary = queue.get("summary", {})
    lines.append(
        "Summary: "
        f"{summary.get('model_count', 0)} models, "
        f"{summary.get('task_count', 0)} tasks "
        f"({summary.get('ready_tasks', 0)} ready / {summary.get('manual_tasks', 0)} manual)."
    )
    lines.append("")
    lines.append("| task_id | model | capability | dataset | mode | command |")
    lines.append("|---|---|---|---|---|---|")

    for task in queue.get("tasks", []):
        cmd = task.get("command") or "(manual)"
        cmd = cmd.replace("|", "\\|")
        dataset = task.get("dataset") or "-"
        lines.append(
            f"| {task['task_id']} | {task['model_id']} | {task['capability']} | "
            f"{dataset} | {task['mode']} | `{cmd}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(plan: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_dir = output_dir / "agent_queues"
    queue_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, default=str),
        encoding="utf-8",
    )

    with (output_dir / "assignment_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "model_id", "status", "capabilities", "hardware", "task_count"])
        for agent_id, queue in sorted(plan["queues"].items()):
            by_model = {m["model_id"]: m for m in queue.get("models", [])}
            for model_id in sorted(by_model):
                model = by_model[model_id]
                writer.writerow(
                    [
                        agent_id,
                        model["model_id"],
                        model["status"],
                        ",".join(model["capabilities"]),
                        ",".join(model["hardware"]),
                        model["task_count"],
                    ]
                )

    for agent_id, queue in sorted(plan["queues"].items()):
        queue_json = queue_dir / f"{agent_id}.json"
        queue_md = queue_dir / f"{agent_id}.md"
        queue_json.write_text(json.dumps(queue, indent=2, default=str), encoding="utf-8")
        _write_agent_markdown(queue_md, queue)

    dispatch_path = output_dir / "dispatch_commands.sh"
    dispatch_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated at {plan['generated_at']}",
        "# Run one command per worker machine/session.",
        "",
    ]
    for agent_id in sorted(plan["queues"]):
        queue_path = f"{output_dir}/agent_queues/{agent_id}.json"
        cmd = (
            "uv run python scripts/hf_sprint_worker.py "
            f"--queue {queue_path} "
            f"--execution-root {output_dir}/execution "
            "--continue-on-error"
        )
        dispatch_lines.append(cmd)
    dispatch_path.write_text("\n".join(dispatch_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HF sprint multi-agent plan")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/hf_sprint_2026q1.yaml"),
        help="Path to sprint config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hf_sprint_2026q1"),
        help="Directory to write plan artifacts",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    registry_models = load_registry_snapshot()
    plan = generate_plan(config, registry_models)
    write_outputs(plan, args.output_dir)

    total_models = sum(q["summary"]["model_count"] for q in plan["queues"].values())
    total_tasks = sum(q["summary"]["task_count"] for q in plan["queues"].values())

    print(f"Generated HF sprint plan at: {args.output_dir}")
    print(f"Agents: {len(plan['queues'])}, Models: {total_models}, Tasks: {total_tasks}")
    for agent_id, queue in sorted(plan["queues"].items()):
        s = queue["summary"]
        print(
            f"  - {agent_id}: {s['model_count']} models, {s['task_count']} tasks "
            f"({s['ready_tasks']} ready / {s['manual_tasks']} manual)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
