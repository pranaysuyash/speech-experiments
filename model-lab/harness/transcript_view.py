"""
Transcript View - Canonical representation of ASR output for NLP tasks.

Provides a unified view regardless of whether ASR produced segments or just text.
This enables chunking, time-aligned extraction, and caching.

Usage:
    from harness.transcript_view import TranscriptView, from_asr_artifact

    view = from_asr_artifact(artifact_path)
    print(view.full_text)
    for seg in view.segments:
        print(f"[{seg.start_s:.1f}-{seg.end_s:.1f}] {seg.text}")
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.nlp_schema import compute_text_hash


@dataclass
class Segment:
    """A time-aligned segment of transcript."""

    start_s: float
    end_s: float
    text: str
    speaker: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass
class TranscriptView:
    """Canonical view of a transcript with optional time alignment."""

    full_text: str
    segments: list[Segment]
    duration_s: float
    text_hash: str
    word_count: int
    has_timestamps: bool
    source_artifact_path: str | None = None
    source_artifact_hash: str | None = None
    asr_model_id: str | None = None

    @classmethod
    def from_text_only(cls, text: str, duration_s: float = 0) -> "TranscriptView":
        """Create view from plain text (no segments)."""
        # Create synthetic segments by splitting on paragraph boundaries
        segments = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            paragraphs = [text]

        # Distribute duration evenly if available
        if duration_s > 0 and len(paragraphs) > 0:
            seg_duration = duration_s / len(paragraphs)
            for i, para in enumerate(paragraphs):
                segments.append(
                    Segment(
                        start_s=i * seg_duration,
                        end_s=(i + 1) * seg_duration,
                        text=para,
                    )
                )
        else:
            # No duration info - single segment
            segments.append(
                Segment(
                    start_s=0,
                    end_s=duration_s,
                    text=text,
                )
            )

        return cls(
            full_text=text,
            segments=segments,
            duration_s=duration_s,
            text_hash=compute_text_hash(text),
            word_count=len(text.split()),
            has_timestamps=False,
        )

    @classmethod
    def from_segments(
        cls, segments: list[Segment], duration_s: float | None = None
    ) -> "TranscriptView":
        """Create view from time-aligned segments."""
        full_text = " ".join(seg.text for seg in segments)

        if duration_s is None and segments:
            duration_s = max(seg.end_s for seg in segments)

        return cls(
            full_text=full_text,
            segments=segments,
            duration_s=duration_s or 0,
            text_hash=compute_text_hash(full_text),
            word_count=len(full_text.split()),
            has_timestamps=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_text": self.full_text,
            "segments": [s.to_dict() for s in self.segments],
            "duration_s": round(self.duration_s, 2),
            "text_hash": self.text_hash,
            "word_count": self.word_count,
            "has_timestamps": self.has_timestamps,
            "source_artifact_path": self.source_artifact_path,
            "asr_model_id": self.asr_model_id,
        }


def from_asr_artifact(artifact_path: Path) -> TranscriptView:
    """
    Load TranscriptView from an ASR artifact.

    Handles both segment-aware and text-only artifacts.
    """
    with open(artifact_path) as f:
        artifact = json.load(f)

    output = artifact.get("output", {})
    run_context = artifact.get("run_context", {})
    inputs = artifact.get("inputs", {})

    # Get basic info
    full_text = output.get("text", "")
    duration_s = inputs.get("audio_duration_s", 0)
    asr_model_id = run_context.get("model_id")

    # Check for segments
    raw_segments = output.get("segments", [])

    if raw_segments and isinstance(raw_segments, list):
        # Convert to Segment objects
        segments = []
        for seg in raw_segments:
            if isinstance(seg, dict):
                segments.append(
                    Segment(
                        start_s=float(seg.get("start") or seg.get("start_s") or 0),
                        end_s=float(seg.get("end") or seg.get("end_s") or 0),
                        text=seg.get("text", ""),
                        speaker=seg.get("speaker"),
                    )
                )

        if segments:
            view = TranscriptView.from_segments(segments, duration_s)
            view.source_artifact_path = str(artifact_path)
            view.asr_model_id = asr_model_id
            return view

    # Fallback to text-only
    view = TranscriptView.from_text_only(full_text, duration_s)
    view.source_artifact_path = str(artifact_path)
    view.asr_model_id = asr_model_id
    return view


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences for chunking."""
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]
