from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

hf_sprint_plan = importlib.import_module("scripts.hf_sprint_plan")
assign_models_to_agents = hf_sprint_plan.assign_models_to_agents
build_tasks_for_model = hf_sprint_plan.build_tasks_for_model
generate_plan = hf_sprint_plan.generate_plan


def _config() -> dict:
    return {
        "sprint": {"id": "hf_test"},
        "defaults": {"device": "cpu", "chunk_ms": 160, "unassigned_agent": "general"},
        "datasets": {
            "asr_smoke": "asr_smoke_v1",
            "asr_primary": "primary",
            "asr_secondary": "ux_primary",
            "diarization_smoke": "diar_smoke_v1",
            "vad_smoke": "vad_smoke_v1",
            "tts_smoke": "tts_smoke_v1",
            "v2v_smoke": "v2v_smoke_v1",
        },
        "audio_inputs": {
            "bench_audio": "data/audio/clean_speech_10s.wav",
            "separate_audio": "data/audio/conversation_2ppl_30s.wav",
        },
        "agents": {
            "edge": {"include_models": ["moonshine"]},
            "general": {"include_models": []},
        },
    }


def test_assign_models_to_agents_with_fallback():
    registry_models = {
        "moonshine": {},
        "whisper": {},
        "kyutai_streaming": {},
    }
    agent_cfg = {"edge": {"include_models": ["moonshine"]}, "general": {"include_models": []}}
    got = assign_models_to_agents(registry_models, agent_cfg, fallback_agent="general")

    assert got["moonshine"] == "edge"
    assert got["whisper"] == "general"
    assert got["kyutai_streaming"] == "general"


def test_assign_models_to_agents_duplicate_raises():
    registry_models = {"moonshine": {}, "whisper": {}}
    agent_cfg = {
        "edge": {"include_models": ["moonshine"]},
        "general": {"include_models": ["moonshine"]},
    }
    with pytest.raises(ValueError, match="assigned twice"):
        assign_models_to_agents(registry_models, agent_cfg, fallback_agent="general")


def test_build_tasks_for_streaming_model():
    tasks = build_tasks_for_model(
        "kyutai_streaming",
        {"status": "experimental", "capabilities": ["asr_stream"]},
        datasets=_config()["datasets"],
        audio_inputs=_config()["audio_inputs"],
        device="cpu",
        chunk_ms=160,
    )

    assert len(tasks) == 2
    assert all(t["capability"] == "asr_stream" for t in tasks)
    assert all(t["mode"] == "ready" for t in tasks)
    assert all("scripts/run_asr_stream.py" in (t["command"] or "") for t in tasks)


def test_build_tasks_for_music_transcription_is_manual():
    tasks = build_tasks_for_model(
        "basic_pitch",
        {"status": "experimental", "capabilities": ["music_transcription"]},
        datasets=_config()["datasets"],
        audio_inputs=_config()["audio_inputs"],
        device="cpu",
        chunk_ms=160,
    )

    assert len(tasks) == 1
    assert tasks[0]["mode"] == "manual"
    assert tasks[0]["command"] is None
    assert "No dedicated CLI runner" in tasks[0]["notes"]


def test_generate_plan_includes_models_and_summary():
    config = _config()
    registry_models = {
        "moonshine": {
            "model_id": "moonshine",
            "status": "experimental",
            "capabilities": ["asr"],
            "hardware": ["cpu"],
            "version": "1.0.0",
        },
        "whisper": {
            "model_id": "whisper",
            "status": "production",
            "capabilities": ["asr"],
            "hardware": ["cpu", "mps"],
            "version": "3.0.0",
        },
    }

    plan = generate_plan(config, registry_models)

    assert "queues" in plan
    assert "edge" in plan["queues"]
    assert "general" in plan["queues"]

    edge_models = [m["model_id"] for m in plan["queues"]["edge"]["models"]]
    general_models = [m["model_id"] for m in plan["queues"]["general"]["models"]]
    assert edge_models == ["moonshine"]
    assert general_models == ["whisper"]

    edge_summary = plan["queues"]["edge"]["summary"]
    assert edge_summary["model_count"] == 1
    assert edge_summary["task_count"] >= 1
