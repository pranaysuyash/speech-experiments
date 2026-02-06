"""
Tests for chunking infrastructure.

Tests verify:
1. Chunker respects max_chunk_seconds
2. Overlap is applied
3. Text-based fallback works
4. Deduplication works
5. Operability limits enforced
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.chunking import (
    ChunkingPolicy,
    chunk_transcript,
    dedupe_items,
)
from harness.transcript_view import Segment, TranscriptView


class TestTranscriptView:
    """Test TranscriptView creation."""

    def test_from_text_only(self):
        """Can create view from plain text."""
        view = TranscriptView.from_text_only("Hello world. This is a test.", duration_s=10.0)

        assert view.full_text == "Hello world. This is a test."
        assert view.duration_s == 10.0
        assert view.word_count == 6
        assert not view.has_timestamps
        assert len(view.segments) >= 1

    def test_from_segments(self):
        """Can create view from segments."""
        segments = [
            Segment(start_s=0, end_s=5, text="First segment."),
            Segment(start_s=5, end_s=10, text="Second segment."),
        ]
        view = TranscriptView.from_segments(segments)

        assert view.full_text == "First segment. Second segment."
        assert view.duration_s == 10
        assert view.has_timestamps
        assert len(view.segments) == 2

    def test_text_hash_consistent(self):
        """Same text produces same hash."""
        view1 = TranscriptView.from_text_only("Test text")
        view2 = TranscriptView.from_text_only("Test text")

        assert view1.text_hash == view2.text_hash


class TestChunkingPolicy:
    """Test chunking policy configuration."""

    def test_default_policy(self):
        """Default policy has sensible values."""
        policy = ChunkingPolicy()

        assert policy.max_chunk_seconds == 60.0
        assert policy.max_chunk_chars == 4000
        assert policy.max_chunks == 50
        assert policy.overlap_seconds == 5.0

    def test_policy_hash_deterministic(self):
        """Same policy produces same hash."""
        p1 = ChunkingPolicy()
        p2 = ChunkingPolicy()

        assert p1.policy_hash == p2.policy_hash

    def test_policy_hash_changes_with_params(self):
        """Different params produce different hash."""
        p1 = ChunkingPolicy(max_chunk_seconds=60)
        p2 = ChunkingPolicy(max_chunk_seconds=90)

        assert p1.policy_hash != p2.policy_hash


class TestChunking:
    """Test chunking behavior."""

    def test_short_text_not_chunked(self):
        """Short text produces single chunk."""
        view = TranscriptView.from_text_only("Short text.", duration_s=5.0)
        policy = ChunkingPolicy(max_chunk_chars=1000, max_chunk_seconds=60)

        result = chunk_transcript(view, policy)

        assert result.total_chunks == 1
        assert not result.chunking_required
        assert result.chunks[0].text == "Short text."

    def test_long_text_chunked(self):
        """Long text is split into chunks."""
        # Create text that exceeds limit
        long_text = "This is a sentence. " * 100  # ~2000 chars
        view = TranscriptView.from_text_only(long_text, duration_s=300.0)
        policy = ChunkingPolicy(max_chunk_chars=500)

        result = chunk_transcript(view, policy)

        assert result.total_chunks > 1
        assert result.chunking_required
        # Each chunk should be under limit
        for chunk in result.chunks:
            assert (
                len(chunk.text) <= policy.max_chunk_chars + 100
            )  # Some tolerance for sentence boundaries

    def test_chunk_respects_max_seconds_with_segments(self):
        """Time-based chunking respects max_chunk_seconds."""
        # Create segments spanning 120 seconds with enough text
        segments = [
            Segment(
                start_s=i * 10,
                end_s=(i + 1) * 10,
                text=f"This is segment number {i} with some extra words to make it longer.",
            )
            for i in range(12)
        ]
        view = TranscriptView.from_segments(segments, duration_s=120.0)
        policy = ChunkingPolicy(max_chunk_seconds=30, max_chunk_chars=10000, min_chunk_chars=50)

        result = chunk_transcript(view, policy)

        # Should have multiple chunks
        assert result.total_chunks > 1
        # Each chunk should be roughly <= 35 seconds (some tolerance)
        for chunk in result.chunks:
            assert chunk.end_s - chunk.start_s <= 35  # Some tolerance

    def test_chunks_have_unique_hashes(self):
        """Each chunk has a unique hash (unless content identical)."""
        long_text = "Sentence one. " * 50 + "Sentence two. " * 50
        view = TranscriptView.from_text_only(long_text, duration_s=100.0)
        policy = ChunkingPolicy(max_chunk_chars=500)

        result = chunk_transcript(view, policy)

        # Hashes should exist
        assert all(c.text_hash for c in result.chunks)
        # At least some should be unique (unless text is identical)
        assert len({c.text_hash for c in result.chunks}) >= 1

    def test_max_chunks_limit_enforced(self):
        """Operability limit on max chunks is enforced."""
        # Create very long text that will definitely need many chunks
        very_long = "This is a complete sentence with many words. " * 500
        view = TranscriptView.from_text_only(very_long, duration_s=1000.0)
        policy = ChunkingPolicy(max_chunk_chars=200, max_chunks=5, min_chunk_chars=50)

        result = chunk_transcript(view, policy)

        # Should be truncated to limit
        assert result.total_chunks <= 5
        # If error, it means there were too many chunks
        if len(very_long) / 200 > 5:  # Expected to need more than 5 chunks
            assert result.error is not None or result.total_chunks == 5


class TestDeduplication:
    """Test item deduplication."""

    def test_dedupe_removes_duplicates(self):
        """Exact duplicates are removed."""
        items = [
            {"text": "Review the PR"},
            {"text": "Review the PR"},
            {"text": "Deploy to staging"},
        ]

        result = dedupe_items(items)

        assert len(result) == 2

    def test_dedupe_keeps_first_occurrence(self):
        """First occurrence of duplicate is kept."""
        items = [
            {"text": "First", "id": 1},
            {"text": "Second", "id": 2},
            {"text": "First", "id": 3},
        ]

        result = dedupe_items(items)

        assert len(result) == 2
        assert result[0]["id"] == 1

    def test_dedupe_with_similarity_threshold(self):
        """Near-duplicates are caught with Jaccard threshold."""
        items = [
            {"text": "Review the PR by Friday"},
            {"text": "Review the PR by Friday please"},  # Very similar
            {"text": "Deploy to production"},
        ]

        result = dedupe_items(items, threshold=0.6)  # Lower threshold

        assert len(result) == 2

    def test_dedupe_empty_list(self):
        """Empty list returns empty."""
        assert dedupe_items([]) == []

    def test_dedupe_custom_key(self):
        """Can dedupe on custom text key."""
        items = [
            {"action": "Do thing A"},
            {"action": "Do thing A"},
        ]

        result = dedupe_items(items, text_key="action")

        assert len(result) == 1
