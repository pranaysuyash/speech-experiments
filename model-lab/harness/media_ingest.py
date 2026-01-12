"""
Media Ingestion - Handle video/audio input with canonical audio output.

Canonical Audio Contract (frozen):
- Mono channel
- 16 kHz sample rate
- Float32 PCM
- audio_hash computed from decoded PCM bytes
- source_media_hash computed from original file bytes

Usage:
    from harness.media_ingest import ingest_media, IngestResult
    
    result = ingest_media(Path("meeting.mp4"))
    # result.audio is the decoded PCM
    # result.audio_hash is the hash of the PCM
    # result.source_media_hash is the hash of the original file
"""

import hashlib
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Canonical audio parameters (frozen)
CANONICAL_SAMPLE_RATE = 16000
CANONICAL_CHANNELS = 1  # Mono
CANONICAL_DTYPE = np.float32

# Supported file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
ALL_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class IngestError(Exception):
    """Raised when media ingestion fails."""
    pass


class FFmpegNotFoundError(IngestError):
    """Raised when ffmpeg is not installed."""
    pass


@dataclass
class IngestResult:
    """Result of media ingestion with full provenance."""
    # Paths
    source_media_path: Path         # Original file
    audio_path: Optional[Path]      # Temp WAV if extracted, else None
    
    # Hashes
    source_media_hash: str          # SHA256 of original file bytes (16 chars)
    audio_hash: str                 # SHA256 of decoded PCM bytes (16 chars)
    
    # Audio properties
    audio: np.ndarray               # Decoded PCM (mono, 16kHz, float32)
    sample_rate: int                # Always 16000
    audio_duration_s: float         # Duration in seconds
    
    # Ingest metadata
    ingest_tool: str                # "ffmpeg" or "native"
    ingest_version: str             # Tool version string
    is_extracted: bool              # True if audio was extracted from video
    original_format: str            # Original file extension
    
    def cleanup(self):
        """Remove temp audio file if it exists."""
        if self.audio_path and self.audio_path.exists():
            try:
                self.audio_path.unlink()
            except Exception:
                pass


def get_ffmpeg_version() -> Optional[str]:
    """Get ffmpeg version string, or None if not installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse first line: "ffmpeg version X.X.X ..."
            first_line = result.stdout.split('\n')[0]
            parts = first_line.split()
            if len(parts) >= 3:
                return parts[2]  # version number
            return first_line[:50]
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def check_ffmpeg_available() -> Tuple[bool, Optional[str]]:
    """Check if ffmpeg is available. Returns (available, version)."""
    version = get_ffmpeg_version()
    return (version is not None, version)


def compute_file_hash(path: Path, length: int = 16) -> str:
    """Compute SHA256 hash of file bytes."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:length]


def compute_pcm_hash(audio: np.ndarray, length: int = 16) -> str:
    """Compute SHA256 hash of decoded PCM bytes."""
    audio_bytes = audio.astype(np.float32).tobytes()
    return hashlib.sha256(audio_bytes).hexdigest()[:length]


def detect_media_type(path: Path) -> str:
    """
    Detect if file is video, audio, or unknown.
    
    Returns: "video", "audio", or "unknown"
    """
    ext = path.suffix.lower()
    
    if ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    else:
        # Fallback: try ffprobe if available
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                streams = result.stdout.strip().split('\n')
                if "video" in streams:
                    return "video"
                elif "audio" in streams:
                    return "audio"
        except Exception:
            pass
        
        return "unknown"


def extract_audio_ffmpeg(
    source_path: Path,
    output_path: Path,
    target_sr: int = CANONICAL_SAMPLE_RATE
) -> None:
    """
    Extract audio from video or transcode audio to canonical format.
    
    Outputs: mono, 16kHz, PCM WAV
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_path),
        "-vn",                      # No video
        "-acodec", "pcm_f32le",     # Float32 PCM
        "-ar", str(target_sr),      # Sample rate
        "-ac", "1",                 # Mono
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise IngestError(f"ffmpeg failed: {result.stderr[-500:]}")
    except FileNotFoundError:
        raise FFmpegNotFoundError(
            "ffmpeg not found. Install ffmpeg to process video files."
        )
    except subprocess.TimeoutExpired:
        raise IngestError("ffmpeg timed out after 5 minutes")


def load_audio_native(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile (for WAV/FLAC)."""
    import soundfile as sf
    
    audio, sr = sf.read(str(path), dtype='float32')
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    return audio.astype(np.float32), sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        # Fallback: use scipy if librosa not available
        from scipy import signal
        num_samples = int(len(audio) * target_sr / orig_sr)
        return signal.resample(audio, num_samples).astype(np.float32)


def ingest_media(
    path: Path,
    target_sr: int = CANONICAL_SAMPLE_RATE,
    cleanup_on_error: bool = True
) -> IngestResult:
    """
    Ingest media file (video or audio) and produce canonical audio.
    
    Args:
        path: Path to media file
        target_sr: Target sample rate (default 16000)
        cleanup_on_error: Remove temp files on error
        
    Returns:
        IngestResult with decoded audio and full provenance
        
    Raises:
        IngestError: If ingestion fails
        FFmpegNotFoundError: If ffmpeg needed but not installed
    """
    path = Path(path).resolve()
    
    if not path.exists():
        raise IngestError(f"File not found: {path}")
    
    # Detect media type
    media_type = detect_media_type(path)
    if media_type == "unknown":
        raise IngestError(
            f"Unknown media type: {path.suffix}. "
            f"Supported: {', '.join(sorted(ALL_MEDIA_EXTENSIONS))}"
        )
    
    # Compute source hash
    source_hash = compute_file_hash(path)
    logger.info(f"Ingesting {path.name} (type={media_type}, hash={source_hash[:8]})")
    
    temp_path: Optional[Path] = None
    audio: np.ndarray
    sample_rate: int
    ingest_tool: str
    ingest_version: str
    is_extracted: bool
    
    try:
        if media_type == "video":
            # Video: always extract via ffmpeg
            ffmpeg_available, ffmpeg_version = check_ffmpeg_available()
            if not ffmpeg_available:
                raise FFmpegNotFoundError(
                    "ffmpeg required for video files. Install: brew install ffmpeg"
                )
            
            # Create temp file for extracted audio
            temp_fd, temp_path_str = tempfile.mkstemp(suffix=".wav")
            temp_path = Path(temp_path_str)
            
            # Extract audio
            extract_audio_ffmpeg(path, temp_path, target_sr)
            
            # Load extracted audio
            audio, sample_rate = load_audio_native(temp_path)
            
            ingest_tool = "ffmpeg"
            ingest_version = ffmpeg_version or "unknown"
            is_extracted = True
            
        else:
            # Audio file
            ext = path.suffix.lower()
            
            if ext == ".wav":
                # Try native loading first
                try:
                    audio, sample_rate = load_audio_native(path)
                    
                    # Resample if needed
                    if sample_rate != target_sr:
                        audio = resample_audio(audio, sample_rate, target_sr)
                        sample_rate = target_sr
                    
                    ingest_tool = "native"
                    ingest_version = "soundfile"
                    is_extracted = False
                    
                except Exception as e:
                    logger.warning(f"Native load failed, trying ffmpeg: {e}")
                    # Fallback to ffmpeg
                    ffmpeg_available, ffmpeg_version = check_ffmpeg_available()
                    if not ffmpeg_available:
                        raise
                    
                    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".wav")
                    temp_path = Path(temp_path_str)
                    extract_audio_ffmpeg(path, temp_path, target_sr)
                    audio, sample_rate = load_audio_native(temp_path)
                    
                    ingest_tool = "ffmpeg"
                    ingest_version = ffmpeg_version or "unknown"
                    is_extracted = True
            else:
                # Non-WAV audio (mp3, flac, etc): use ffmpeg
                ffmpeg_available, ffmpeg_version = check_ffmpeg_available()
                if not ffmpeg_available:
                    raise FFmpegNotFoundError(
                        f"ffmpeg required for {ext} files. Install: brew install ffmpeg"
                    )
                
                temp_fd, temp_path_str = tempfile.mkstemp(suffix=".wav")
                temp_path = Path(temp_path_str)
                extract_audio_ffmpeg(path, temp_path, target_sr)
                audio, sample_rate = load_audio_native(temp_path)
                
                ingest_tool = "ffmpeg"
                ingest_version = ffmpeg_version or "unknown"
                is_extracted = True
        
        # Ensure canonical format
        audio = audio.astype(np.float32)
        
        # Compute audio hash from PCM
        audio_hash = compute_pcm_hash(audio)
        
        # Calculate duration
        duration_s = len(audio) / sample_rate
        
        logger.info(f"Ingested: {duration_s:.1f}s, sr={sample_rate}, hash={audio_hash[:8]}")
        
        return IngestResult(
            source_media_path=path,
            audio_path=temp_path,
            source_media_hash=source_hash,
            audio_hash=audio_hash,
            audio=audio,
            sample_rate=sample_rate,
            audio_duration_s=duration_s,
            ingest_tool=ingest_tool,
            ingest_version=ingest_version,
            is_extracted=is_extracted,
            original_format=path.suffix.lower(),
        )
        
    except Exception as e:
        # Cleanup temp file on error
        if cleanup_on_error and temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise
