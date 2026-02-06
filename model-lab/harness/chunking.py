"""
Chunking Module - Deterministic chunking for long transcripts.

Provides time-based and text-based chunking with overlap for boundary loss reduction.
All chunks are hash-identified for caching.

Usage:
    from harness.chunking import chunk_transcript, ChunkingPolicy

    chunks = chunk_transcript(transcript_view, ChunkingPolicy())
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}: {len(chunk.text)} chars, {chunk.text_hash[:8]}")
"""

import re
from dataclasses import dataclass
from typing import Any

from harness.nlp_schema import compute_text_hash
from harness.transcript_view import Segment, TranscriptView


@dataclass
class ChunkingPolicy:
    """Configuration for chunking behavior."""

    max_chunk_seconds: float = 60.0  # Preferred max duration per chunk
    max_chunk_chars: int = 4000  # Hard cap on characters
    overlap_seconds: float = 5.0  # Overlap between chunks
    max_chunks: int = 50  # Operability limit
    min_chunk_chars: int = 100  # Don't create tiny chunks

    # Policy identification for provenance
    policy_version: str = "v1"

    @property
    def policy_id(self) -> str:
        return f"time_or_text_{self.policy_version}"

    @property
    def policy_hash(self) -> str:
        """Hash of policy parameters for reproducibility."""
        params = f"{self.max_chunk_seconds}_{self.max_chunk_chars}_{self.overlap_seconds}_{self.max_chunks}"
        return compute_text_hash(params, length=12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "max_chunk_seconds": self.max_chunk_seconds,
            "max_chunk_chars": self.max_chunk_chars,
            "overlap_seconds": self.overlap_seconds,
            "max_chunks": self.max_chunks,
        }


@dataclass
class Chunk:
    """A chunk of transcript with metadata for caching."""

    chunk_id: int
    start_s: float
    end_s: float
    text: str
    text_hash: str
    word_count: int
    is_overlap: bool = False  # True if this chunk overlaps with previous

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "text": self.text,
            "text_hash": self.text_hash,
            "word_count": self.word_count,
        }


@dataclass
class ChunkingResult:
    """Result of chunking operation."""

    chunks: list[Chunk]
    policy: ChunkingPolicy
    total_chunks: int
    total_chars: int
    chunking_required: bool  # True if input was split
    error: str | None = None

    @property
    def chunk_hashes(self) -> list[str]:
        return [c.text_hash for c in self.chunks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "policy": self.policy.to_dict(),
            "total_chunks": self.total_chunks,
            "total_chars": self.total_chars,
            "chunking_required": self.chunking_required,
            "chunk_hashes": self.chunk_hashes,
            "error": self.error,
        }


def chunk_by_time(view: TranscriptView, policy: ChunkingPolicy) -> list[Chunk]:
    """
    Chunk transcript by time boundaries using segments.

    Respects segment boundaries - won't split mid-segment.
    """
    if not view.segments or not view.has_timestamps:
        return []

    chunks = []
    current_segments: list[Segment] = []
    current_start = 0.0
    chunk_id = 0

    for seg in view.segments:
        # Check if adding this segment exceeds limits
        current_text = " ".join(s.text for s in current_segments)
        new_text = current_text + " " + seg.text if current_text else seg.text
        current_duration = (seg.end_s - current_start) if current_segments else 0

        # Should we start a new chunk?
        should_split = current_segments and (
            current_duration + seg.duration_s > policy.max_chunk_seconds
            or len(new_text) > policy.max_chunk_chars
        )

        if should_split:
            # Emit current chunk
            chunk_text = " ".join(s.text for s in current_segments)
            if len(chunk_text) >= policy.min_chunk_chars:
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        start_s=current_start,
                        end_s=current_segments[-1].end_s,
                        text=chunk_text,
                        text_hash=compute_text_hash(chunk_text),
                        word_count=len(chunk_text.split()),
                    )
                )
                chunk_id += 1

            # Start new chunk, potentially with overlap
            overlap_segments: list[Segment] = []
            if policy.overlap_seconds > 0:
                # Include segments from end of previous chunk
                overlap_time = 0
                for s in reversed(current_segments):
                    overlap_time += s.duration_s
                    if overlap_time <= policy.overlap_seconds:
                        overlap_segments.insert(0, s)
                    else:
                        break

            current_segments = overlap_segments + [seg]
            current_start = current_segments[0].start_s if overlap_segments else seg.start_s
        else:
            current_segments.append(seg)
            if not current_segments[:-1]:
                current_start = seg.start_s

    # Emit final chunk
    if current_segments:
        chunk_text = " ".join(s.text for s in current_segments)
        if len(chunk_text) >= policy.min_chunk_chars or not chunks:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    start_s=current_start,
                    end_s=current_segments[-1].end_s,
                    text=chunk_text,
                    text_hash=compute_text_hash(chunk_text),
                    word_count=len(chunk_text.split()),
                )
            )

    return chunks


def chunk_by_text(text: str, policy: ChunkingPolicy, duration_s: float = 0) -> list[Chunk]:
    """
    Chunk transcript by text boundaries (sentences/paragraphs).

    Used when timestamps are not available.
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [
            Chunk(
                chunk_id=0,
                start_s=0,
                end_s=duration_s,
                text=text,
                text_hash=compute_text_hash(text),
                word_count=len(text.split()),
            )
        ]

    chunks: list[TranscriptChunk] = []
    current_sentences: list[str] = []
    chunk_id = 0

    for sent in sentences:
        current_text = " ".join(current_sentences)
        new_text = current_text + " " + sent if current_text else sent

        if len(new_text) > policy.max_chunk_chars and current_sentences:
            # Emit chunk
            chunk_text = current_text
            if len(chunk_text) >= policy.min_chunk_chars:
                # Estimate time proportionally
                char_ratio = len(chunk_text) / len(text) if text else 0
                start_ratio = (
                    sum(len(" ".join(current_sentences[:i])) for i in range(len(chunks)))
                    / len(text)
                    if text
                    else 0
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        start_s=start_ratio * duration_s,
                        end_s=(start_ratio + char_ratio) * duration_s,
                        text=chunk_text,
                        text_hash=compute_text_hash(chunk_text),
                        word_count=len(chunk_text.split()),
                    )
                )
                chunk_id += 1

            current_sentences = [sent]
        else:
            current_sentences.append(sent)

    # Final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        char_start = len(text) - len(chunk_text)
        start_ratio = char_start / len(text) if text else 0

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                start_s=start_ratio * duration_s,
                end_s=duration_s,
                text=chunk_text,
                text_hash=compute_text_hash(chunk_text),
                word_count=len(chunk_text.split()),
            )
        )

    return chunks


def chunk_transcript(
    view: TranscriptView, policy: ChunkingPolicy | None = None
) -> ChunkingResult:
    """
    Chunk a transcript using the best available method.

    Prefers time-based chunking if timestamps available,
    falls back to text-based.
    """
    if policy is None:
        policy = ChunkingPolicy()

    # Check if chunking is needed
    if (
        len(view.full_text) <= policy.max_chunk_chars
        and view.duration_s <= policy.max_chunk_seconds
    ):
        # No chunking needed - return single chunk
        single_chunk = Chunk(
            chunk_id=0,
            start_s=0,
            end_s=view.duration_s,
            text=view.full_text,
            text_hash=view.text_hash,
            word_count=view.word_count,
        )
        return ChunkingResult(
            chunks=[single_chunk],
            policy=policy,
            total_chunks=1,
            total_chars=len(view.full_text),
            chunking_required=False,
        )

    # Try time-based first
    if view.has_timestamps and view.segments:
        chunks = chunk_by_time(view, policy)
    else:
        chunks = chunk_by_text(view.full_text, policy, view.duration_s)

    # Check operability limits
    error = None
    if len(chunks) > policy.max_chunks:
        error = f"NLP_INPUT_TOO_LARGE: {len(chunks)} chunks exceeds max {policy.max_chunks}"
        # Truncate to limit
        chunks = chunks[: policy.max_chunks]

    return ChunkingResult(
        chunks=chunks,
        policy=policy,
        total_chunks=len(chunks),
        total_chars=sum(len(c.text) for c in chunks),
        chunking_required=len(chunks) > 1,
        error=error,
    )


def dedupe_items(items: list[dict], text_key: str = "text", threshold: float = 0.8) -> list[dict]:
    """
    Deduplicate items using Jaccard similarity on text.

    Keeps first occurrence of similar items.
    """

    def normalize(text: str) -> set:
        """Normalize text to token set for comparison."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        return set(tokens)

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    result = []
    seen_tokens: list[list[str]] = []

    for item in items:
        text = item.get(text_key, "")
        if not text:
            continue

        tokens = normalize(text)

        # Check against all seen items
        is_dupe = False
        for seen in seen_tokens:
            if jaccard(tokens, seen) >= threshold:
                is_dupe = True
                break

        if not is_dupe:
            result.append(item)
            seen_tokens.append(tokens)

    return result
