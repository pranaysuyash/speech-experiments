"""
Tests for NLP schema and summarize runner.

Tests verify:
1. NLP artifact has required provenance fields
2. Parent linkage is enforced
3. Summary output is constrained (no freeform)
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness.nlp_schema import (
    ActionItem,
    Entity,
    NLPArtifact,
    NLPInputs,
    NLPProvenance,
    NLPRunContext,
    SummaryOutput,
    compute_text_hash,
    validate_nlp_artifact,
)


class TestNLPSchemaDataclasses:
    """Test NLP schema dataclasses."""

    def test_summary_output_to_dict(self):
        """SummaryOutput serializes correctly."""
        output = SummaryOutput(
            sentences=["First point.", "Second point."],
            total_sentences=2,
            source_word_count=100,
            compression_ratio=10.0,
        )
        d = output.to_dict()

        assert d["sentences"] == ["First point.", "Second point."]
        assert d["total_sentences"] == 2
        assert d["compression_ratio"] == 10.0

    def test_action_item_to_dict(self):
        """ActionItem serializes correctly."""
        item = ActionItem(
            text="Review the PR",
            assignee="Alice",
            due="2024-01-15",
            priority="high",
        )
        d = item.to_dict()

        assert d["text"] == "Review the PR"
        assert d["assignee"] == "Alice"
        assert d["due"] == "2024-01-15"
        assert d["priority"] == "high"

    def test_entity_to_dict(self):
        """Entity serializes correctly."""
        entity = Entity(
            text="Acme Corp",
            type="ORG",
            count=3,
            source_chunk_ids=[0, 1],
            confidence=0.85,
        )
        d = entity.to_dict()

        assert d["text"] == "Acme Corp"
        assert d["type"] == "ORG"
        assert d["count"] == 3
        assert d["confidence"] == 0.85


class TestNLPProvenanceLinkage:
    """Test that NLP artifacts properly link to parent."""

    @pytest.fixture
    def valid_artifact(self):
        """Create a valid NLP artifact."""
        run_context = NLPRunContext(
            task="summarize",
            nlp_model_id="gemini-2.0-flash",
            timestamp=datetime.now().isoformat(),
        )

        inputs = NLPInputs(
            parent_artifact_path="runs/test/asr/adhoc_123.json",
            parent_artifact_hash="abc123def456",
            asr_model_id="faster_whisper",
            asr_text_hash="text123hash456",
            transcript_word_count=500,
            audio_duration_s=60.0,
        )

        provenance = NLPProvenance(
            prompt_template="Summarize this: {transcript}",
            prompt_hash="prompt123hash",
            has_ground_truth=False,
            metrics_valid=True,
        )

        return NLPArtifact(
            run_context=run_context,
            inputs=inputs,
            provenance=provenance,
            output={"sentences": ["Test summary."]},
            metrics_structural={"latency_ms": 100},
        )

    def test_valid_artifact_passes_validation(self, valid_artifact):
        """Valid artifact passes validation."""
        validate_nlp_artifact(valid_artifact)  # Should not raise

    def test_missing_parent_hash_fails(self, valid_artifact):
        """Missing parent_artifact_hash fails validation."""
        valid_artifact.inputs.parent_artifact_hash = ""

        with pytest.raises(ValueError) as exc_info:
            validate_nlp_artifact(valid_artifact)

        assert "parent_artifact_hash" in str(exc_info.value)

    def test_missing_asr_text_hash_fails(self, valid_artifact):
        """Missing asr_text_hash fails validation."""
        valid_artifact.inputs.asr_text_hash = ""

        with pytest.raises(ValueError) as exc_info:
            validate_nlp_artifact(valid_artifact)

        assert "asr_text_hash" in str(exc_info.value)

    def test_missing_prompt_hash_fails(self, valid_artifact):
        """Missing prompt_hash fails validation."""
        valid_artifact.provenance.prompt_hash = ""

        with pytest.raises(ValueError) as exc_info:
            validate_nlp_artifact(valid_artifact)

        assert "prompt_hash" in str(exc_info.value)


class TestTextHash:
    """Test text hashing for provenance."""

    def test_hash_deterministic(self):
        """Same text produces same hash."""
        text = "Hello, world!"
        assert compute_text_hash(text) == compute_text_hash(text)

    def test_different_text_different_hash(self):
        """Different text produces different hash."""
        assert compute_text_hash("Hello") != compute_text_hash("Goodbye")

    def test_hash_length(self):
        """Hash has expected length."""
        h = compute_text_hash("test", length=16)
        assert len(h) == 16


class TestSummaryOutputConstraints:
    """Test that summary output is constrained."""

    def test_compression_ratio_calculated(self):
        """Compression ratio is properly calculated."""
        output = SummaryOutput(
            sentences=["One sentence summary."],
            total_sentences=1,
            source_word_count=100,
            compression_ratio=33.33,  # 100 / 3 words
        )

        d = output.to_dict()
        assert d["compression_ratio"] == 33.33

    def test_sentences_are_list(self):
        """Sentences must be a list."""
        output = SummaryOutput(
            sentences=["Point 1", "Point 2"],
            total_sentences=2,
            source_word_count=50,
            compression_ratio=12.5,
        )

        assert isinstance(output.sentences, list)
        assert len(output.sentences) == 2
