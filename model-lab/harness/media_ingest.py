from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    # Denoise settings
    denoise: bool = False
    denoise_strength: float = 0.5  # 0.0-1.0
    # Speed adjustment
    speed: float = 1.0  # 1.0 = normal, 0.5 = half speed, 2.0 = double speed
    # Peak normalization
    peak_normalize: bool = False
    peak_target_db: float = -1.0
    # Dynamic range compression
    compress_dynamics: bool = False
    compress_threshold_db: float = -20.0
    compress_ratio: float = 4.0
    # Noise gate
    gate_noise: bool = False
    gate_threshold_db: float = -40.0
    # Mono mix
    mono_mix: bool = False

    def to_json_obj(self) -> dict[str, Any]:
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
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return sha256_bytes(payload)


def get_ffmpeg_version() -> str:
    # Keep this deterministic and cheap
    try:
        p = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            text=True,
        )
        first = (p.stdout or "").splitlines()[0].strip()
        return first or "unknown"
    except Exception:
        return "unknown"


def build_ffmpeg_filter_chain(cfg: IngestConfig) -> str | None:
    filters: list[str] = []

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
            raise ValueError(
                f"Unsupported loudnorm_mode={cfg.loudnorm_mode} (expected single_pass)"
            )
        # Single pass loudnorm. Documented trade-off.
        filters.append("loudnorm=I=-16:LRA=11:TP=-1.5")

    # Denoise using highpass + lowpass filter chain (basic noise reduction)
    if getattr(cfg, "denoise", False):
        strength = getattr(cfg, "denoise_strength", 0.5)
        # Use highpass to remove low-frequency rumble and lowpass for high-frequency hiss
        # Strength 0-1 maps to cutoff frequencies
        highpass_freq = int(80 + (strength * 120))  # 80-200Hz
        lowpass_freq = int(12000 - (strength * 4000))  # 8000-12000Hz
        filters.append(f"highpass=f={highpass_freq}")
        filters.append(f"lowpass=f={lowpass_freq}")

    # Speed adjustment using atempo filter
    if getattr(cfg, "speed", None) and getattr(cfg, "speed", 1.0) != 1.0:
        speed = cfg.speed
        # atempo only supports 0.5-2.0, chain multiple for extreme values
        if 0.5 <= speed <= 2.0:
            filters.append(f"atempo={speed}")
        elif speed < 0.5:
            # Chain multiple atempo filters for very slow speeds
            while speed < 0.5:
                filters.append("atempo=0.5")
                speed = speed / 0.5
            if speed != 1.0:
                filters.append(f"atempo={speed}")
        else:
            # Chain multiple atempo filters for very fast speeds
            while speed > 2.0:
                filters.append("atempo=2.0")
                speed = speed / 2.0
            if speed != 1.0:
                filters.append(f"atempo={speed}")

    # Peak normalization using dynaudnorm
    if getattr(cfg, "peak_normalize", False):
        target_db = getattr(cfg, "peak_target_db", -1.0)
        filters.append(f"dynaudnorm=p={10 ** (target_db / 20):.4f}")

    # Stereo to mono downmix
    if getattr(cfg, "mono_mix", False):
        filters.append("pan=mono|c0=0.5*c0+0.5*c1")

    # Dynamic range compression
    if getattr(cfg, "compress_dynamics", False):
        threshold_db = getattr(cfg, "compress_threshold_db", -20.0)
        ratio = getattr(cfg, "compress_ratio", 4.0)
        filters.append(f"acompressor=threshold={threshold_db}dB:ratio={ratio}")

    # Noise gate
    if getattr(cfg, "gate_noise", False):
        threshold_db = getattr(cfg, "gate_threshold_db", -40.0)
        filters.append(f"agate=threshold={threshold_db}dB")

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
) -> tuple[list[str], str]:
    """
    Writes output_wav atomically via temp file.
    Returns (ffmpeg_argv, ffmpeg_version).
    """
    ffmpeg_version = get_ffmpeg_version()

    filter_chain = build_ffmpeg_filter_chain(cfg)

    # temp file in same directory to keep rename atomic across filesystems
    tmp_dir = output_wav.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="processed_audio.", suffix=".wav.tmp", dir=str(tmp_dir), delete=False
    ) as tf:
        tmp_path = Path(tf.name)

    try:
        argv: list[str] = [
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
            "-f",
            "wav",
            str(tmp_path),
        ]

        logger.debug(f"Running ffmpeg: {' '.join(argv)}")
        subprocess.run(argv, check=True, capture_output=True)
        atomic_replace(tmp_path, output_wav)
        return argv, ffmpeg_version
    except subprocess.CalledProcessError as e:
        # Keep stderr for debugging
        stderr = (
            (e.stderr or b"").decode("utf-8", errors="replace")
            if isinstance(e.stderr, (bytes, bytearray))
            else str(e.stderr)
        )
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
) -> tuple[str, str, str]:
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
    preprocessing_config: dict[str, Any]
    ffmpeg_version: str
    ffmpeg_argv: list[str]
    processed_audio_path: str
    audio_content_hash: str
    preprocess_hash: str
    audio_fingerprint: str

    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def ingest_media(
    input_path: Path,
    artifacts_dir: Path,
    cfg: IngestConfig,
) -> dict[str, Any]:
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

    # Compute duration
    import wave

    with wave.open(str(processed_audio), "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration_s = frames / float(rate)

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
        duration_s=duration_s,
    ).to_dict()
