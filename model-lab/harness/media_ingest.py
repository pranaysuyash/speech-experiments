from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class IngestConfig:
    normalize: bool = False
    trim_silence: bool = False
    # Explicitly encode choices to avoid future silent behavior changes
    loudnorm_mode: str = "single_pass"
    # Conservative silenceremove thresholds
    silence_threshold_db: int = -35
    silence_duration_s: float = 0.3
    # Canonical audio format
    sample_rate: int = 16000
    channels: int = 1
    sample_fmt: str = "pcm_s16le"

    def to_json_obj(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def json_sha256(obj: Any) -> str:
    # Strict, stable JSON encoding
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


def get_ffmpeg_version() -> str:
    # Keep this deterministic and cheap
    try:
        p = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        first = (p.stdout or "").splitlines()[0].strip()
        return first or "unknown"
    except Exception:
        return "unknown"


def build_ffmpeg_filter_chain(cfg: IngestConfig) -> Optional[str]:
    filters: List[str] = []

    if cfg.trim_silence:
        th = f"{cfg.silence_threshold_db}dB"
        d = str(cfg.silence_duration_s)
        # Conservative start+stop trimming
        filters.append(
            "silenceremove="
            f"start_periods=1:start_duration={d}:start_threshold={th}:"
            f"stop_periods=1:stop_duration={d}:stop_threshold={th}"
        )

    if cfg.normalize:
        if cfg.loudnorm_mode != "single_pass":
            raise ValueError(f"Unsupported loudnorm_mode={cfg.loudnorm_mode} (expected single_pass)")
        # Single pass loudnorm. Documented trade-off.
        filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")

    if not filters:
        return None

    return ",".join(filters)


def atomic_replace(src_tmp: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src_tmp), str(dst))


def extract_audio_ffmpeg(
    input_media: Path,
    output_wav: Path,
    cfg: IngestConfig,
) -> Tuple[List[str], str]:
    """
    Writes output_wav atomically via temp file.
    Returns (ffmpeg_argv, ffmpeg_version).
    """
    ffmpeg_version = get_ffmpeg_version()

    filter_chain = build_ffmpeg_filter_chain(cfg)

    # temp file in same directory to keep rename atomic across filesystems
    tmp_dir = output_wav.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="processed_audio.", suffix=".wav.tmp", dir=str(tmp_dir), delete=False) as tf:
        tmp_path = Path(tf.name)

    try:
        argv: List[str] = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_media),
            "-vn",
        ]

        if filter_chain:
            argv += ["-af", filter_chain]

        # Canonical audio output format
        argv += [
            "-ac",
            str(cfg.channels),
            "-ar",
            str(cfg.sample_rate),
            "-c:a",
            cfg.sample_fmt,
            "-f", "wav",
            str(tmp_path),
        ]

        logger.info(f"Running ffmpeg: {' '.join(argv)}")
        subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        atomic_replace(tmp_path, output_wav)
        return argv, ffmpeg_version
    except subprocess.CalledProcessError as e:
        # Keep stderr for debugging
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, (bytes, bytearray)) else str(e.stderr)
        raise RuntimeError(f"ffmpeg failed: {stderr}") from e
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def compute_audio_fingerprint(
    processed_wav: Path,
    cfg: IngestConfig,
    ffmpeg_version: str,
) -> Tuple[str, str, str]:
    audio_content_hash = sha256_file(processed_wav)
    preprocess_hash = json_sha256(
        {
            "preprocessing_config": cfg.to_json_obj(),
            "ffmpeg_version": ffmpeg_version,
        }
    )
    audio_fingerprint = sha256_bytes((audio_content_hash + preprocess_hash).encode("utf-8"))
    return audio_content_hash, preprocess_hash, audio_fingerprint


@dataclass
class IngestResult:
    source_media_path: str
    source_media_hash: str
    preprocessing_config: Dict[str, Any]
    ffmpeg_version: str
    ffmpeg_argv: List[str]
    processed_audio_path: str
    audio_content_hash: str
    preprocess_hash: str
    audio_fingerprint: str
    
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def ingest_media(
    input_path: Path,
    artifacts_dir: Path,
    cfg: IngestConfig,
) -> Dict[str, Any]:
    """
    Produces:
      artifacts/ingest/processed_audio.wav
      returns dict suitable for manifest step entry
    """
    artifacts_dir = artifacts_dir / "ingest"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    processed_audio = artifacts_dir / "processed_audio.wav"

    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input media not found: {input_path}")

    # Always recompute hashes from real files. Determinism over cleverness.
    source_media_hash = sha256_file(input_path)

    ffmpeg_argv, ffmpeg_version = extract_audio_ffmpeg(
        input_media=input_path,
        output_wav=processed_audio,
        cfg=cfg,
    )

    audio_content_hash, preprocess_hash, audio_fingerprint = compute_audio_fingerprint(
        processed_wav=processed_audio,
        cfg=cfg,
        ffmpeg_version=ffmpeg_version,
    )

    return IngestResult(
        source_media_path=str(input_path),
        source_media_hash=source_media_hash,
        preprocessing_config=cfg.to_json_obj(),
        ffmpeg_version=ffmpeg_version,
        ffmpeg_argv=ffmpeg_argv,
        processed_audio_path=str(processed_audio),
        audio_content_hash=audio_content_hash,
        preprocess_hash=preprocess_hash,
        audio_fingerprint=audio_fingerprint,
    ).to_dict()
