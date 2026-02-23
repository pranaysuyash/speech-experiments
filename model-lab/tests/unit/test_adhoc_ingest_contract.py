from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from scripts import run_diarization, run_vad


def _write_tone(path: Path, sr: int = 16000, seconds: float = 0.5) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, audio, sr)


def test_run_vad_adhoc_uses_current_ingest_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_wav = tmp_path / "input.wav"
    processed_wav = tmp_path / "processed.wav"
    _write_tone(input_wav)
    _write_tone(processed_wav)

    calls: dict[str, object] = {}

    def fake_ingest_media(input_path, artifacts_dir, cfg):
        calls["input_path"] = input_path
        calls["artifacts_dir"] = artifacts_dir
        calls["cfg_type"] = type(cfg).__name__
        return {
            "source_media_path": str(input_path),
            "source_media_hash": "a" * 64,
            "preprocessing_config": {},
            "ffmpeg_version": "ffmpeg test",
            "ffmpeg_argv": ["ffmpeg"],
            "processed_audio_path": str(processed_wav),
            "audio_content_hash": "b" * 64,
            "preprocess_hash": "c" * 64,
            "audio_fingerprint": "d" * 64,
            "duration_s": 0.5,
        }

    def fake_loader(_config_path, device="cpu"):
        return {
            "vad": {
                "detect": lambda audio, sr=16000: {
                    "segments": [{"start": 0, "end": min(len(audio), sr // 4)}]
                }
            }
        }

    monkeypatch.setattr(run_vad, "ingest_media", fake_ingest_media)
    monkeypatch.setattr(run_vad, "load_model_from_config", fake_loader)

    result, artifact_path = run_vad.run_vad_adhoc("fake_vad_model", str(input_wav), device="cpu")
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    assert calls["cfg_type"] == "IngestConfig"
    assert Path(str(calls["artifacts_dir"])).as_posix().endswith("runs/fake_vad_model/vad/_adhoc_ingest")
    assert result["provenance"]["ingest_version"] == "ffmpeg test"
    assert artifact["inputs"]["audio_hash"] == "b" * 64


def test_run_diarization_adhoc_uses_current_ingest_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_wav = tmp_path / "input.wav"
    processed_wav = tmp_path / "processed.wav"
    _write_tone(input_wav)
    _write_tone(processed_wav)

    model_dir = tmp_path / "models" / "fake_diar_model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"model_type": "fake_diar"}),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    def fake_ingest_media(input_path, artifacts_dir, cfg):
        calls["input_path"] = input_path
        calls["artifacts_dir"] = artifacts_dir
        calls["cfg_type"] = type(cfg).__name__
        return {
            "source_media_path": str(input_path),
            "source_media_hash": "e" * 64,
            "preprocessing_config": {},
            "ffmpeg_version": "ffmpeg test",
            "ffmpeg_argv": ["ffmpeg"],
            "processed_audio_path": str(processed_wav),
            "audio_content_hash": "f" * 64,
            "preprocess_hash": "1" * 64,
            "audio_fingerprint": "2" * 64,
            "duration_s": 0.5,
        }

    def fake_load_model(_model_type, _config, device="cpu"):
        return {
            "diarization": {
                "diarize": lambda audio, sr=16000: {
                    "segments": [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]
                }
            }
        }

    monkeypatch.setattr(run_diarization, "ingest_media", fake_ingest_media)
    monkeypatch.setattr(run_diarization.ModelRegistry, "load_model", fake_load_model)

    result, artifact_path = run_diarization.run_diarization_adhoc(
        "fake_diar_model",
        str(input_wav),
        device="cpu",
    )
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    assert calls["cfg_type"] == "IngestConfig"
    assert Path(str(calls["artifacts_dir"])).as_posix().endswith(
        "runs/fake_diar_model/diarization/_adhoc_ingest"
    )
    assert result["provenance"]["ingest_version"] == "ffmpeg test"
    assert artifact["inputs"]["audio_hash"] == "f" * 64
