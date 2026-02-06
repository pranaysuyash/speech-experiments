"""
Arsenal Taxonomy
Strict vocabulary for Tasks, Roles, and Evidence Grades.
This file is the single source of truth for these constants.
"""

from enum import Enum, unique


@unique
class TaskType(str, Enum):
    ASR = "asr"
    TTS = "tts"
    VAD = "vad"
    DIARIZATION = "diarization"
    V2V = "v2v"
    ALIGNMENT = "alignment"
    MT = "mt"
    CHAT = "chat"


@unique
class TaskRole(str, Enum):
    PRIMARY = "primary"  # Critical for the model's value prop. Failure = Rejection.
    SECONDARY = "secondary"  # Functional/Auxiliary. Failure = Warning.
    UNDECLARED = "undeclared"  # Not claimed. Informational only.


@unique
class EvidenceGrade(str, Enum):
    GOLDEN_BATCH = "golden_batch"  # Paired ground truth, standard dataset
    SMOKE = "smoke"  # Basic sanity/structural check
    ADHOC = "adhoc"  # Manual or unpaired run
    COMPUTED = "computed"  # Analytically derived (no direct run)
    UNKNOWN = "unknown"
