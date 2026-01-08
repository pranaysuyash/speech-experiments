"""
Text normalization utilities for consistent preprocessing.
Ensures text is consistently formatted across models.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Normalize text for ASR/TTS evaluation."""

    # Common contractions and their expansions
    CONTRACTIONS = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'re": " are",
        "'m": " am",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "here's": "here is",
        "what's": "what is",
        "where's": "where is",
        "who's": "who is",
        "how's": "how is",
        "let's": "let us",
        "y'all": "you all",
    }

    @staticmethod
    def normalize(text: str,
                  lowercase: bool = True,
                  remove_punctuation: bool = False,
                  expand_contractions: bool = False,
                  normalize_whitespace: bool = True) -> str:
        """
        Normalize text with configurable options.

        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            expand_contractions: Expand contractions (don't → do not)
            normalize_whitespace: Normalize whitespace to single spaces

        Returns:
            Normalized text
        """
        normalized = text

        # Expand contractions
        if expand_contractions:
            normalized = TextNormalizer._expand_contractions(normalized)

        # Remove punctuation
        if remove_punctuation:
            normalized = TextNormalizer._remove_punctuation(normalized)

        # Normalize whitespace
        if normalize_whitespace:
            normalized = TextNormalizer._normalize_whitespace(normalized)

        # Lowercase
        if lowercase:
            normalized = normalized.lower()

        return normalized.strip()

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """Expand contractions."""
        for contraction, expansion in TextNormalizer.CONTRACTIONS.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation but keep alphanumeric and spaces."""
        return re.sub(r'[^\w\s]', '', text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces and trim."""
        return ' '.join(text.split())

    @staticmethod
    def normalize_for_asr(text: str) -> str:
        """Normalize text for ASR evaluation."""
        return TextNormalizer.normalize(
            text,
            lowercase=True,
            remove_punctuation=False,  # Keep punctuation for accuracy
            expand_contractions=True,
            normalize_whitespace=True
        )

    @staticmethod
    def normalize_for_wer(text: str) -> str:
        """Normalize text specifically for WER calculation."""
        return TextNormalizer.normalize(
            text,
            lowercase=True,
            remove_punctuation=True,  # Remove punctuation for word-level comparison
            expand_contractions=True,
            normalize_whitespace=True
        )


class AudioDescriptionNormalizer:
    """Normalize audio descriptions and metadata."""

    @staticmethod
    def normalize_metadata(metadata: Dict[str, any]) -> Dict[str, str]:
        """Normalize metadata fields."""
        normalized = {}

        for key, value in metadata.items():
            if isinstance(value, str):
                normalized[key] = TextNormalizer.normalize_for_asr(value)
            else:
                normalized[key] = value

        return normalized

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class ComparisonNormalizer:
    """Normalize text for fair comparison across models."""

    @staticmethod
    def normalize_for_comparison(texts: Dict[str, str]) -> Dict[str, str]:
        """
        Normalize multiple texts for comparison.

        Args:
            texts: Dictionary of model_name -> transcription

        Returns:
            Dictionary with normalized texts
        """
        return {
            model_name: TextNormalizer.normalize_for_wer(text)
            for model_name, text in texts.items()
        }

    @staticmethod
    def extract_tokens(text: str) -> List[str]:
        """Extract tokens from normalized text."""
        normalized = TextNormalizer.normalize_for_wer(text)
        return normalized.split()

    @staticmethod
    def calculate_vocabulary_diversity(texts: List[str]) -> Dict[str, any]:
        """Calculate vocabulary statistics across texts."""
        all_tokens = []

        for text in texts:
            tokens = ComparisonNormalizer.extract_tokens(text)
            all_tokens.extend(tokens)

        unique_tokens = set(all_tokens)

        return {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(unique_tokens),
            'vocabulary_richness': len(unique_tokens) / max(1, len(all_tokens)),
            'avg_token_length': sum(len(token) for token in all_tokens) / max(1, len(all_tokens))
        }