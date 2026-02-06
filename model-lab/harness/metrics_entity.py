"""
Entity Error Rate metrics for production-focused ASR evaluation.
Captures errors that matter most: names, dates, numbers, amounts.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntityErrorResult:
    """Container for entity error analysis."""

    entity_error_rate: float
    total_entities: int
    correct_entities: int
    entity_types: dict[str, dict[str, int]]
    errors: list[dict[str, str]]


class EntityMetrics:
    """Calculate entity-level error metrics for production-focused evaluation."""

    # Entity patterns (basic - can be extended)
    PATTERNS = {
        "dates": [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY, DD-MM-YY
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",  # YYYY-MM-DD
        ],
        "numbers": [
            r"\b\d+(?:\.\d+)?\b",  # Decimal numbers
            r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",  # Numbers with commas
        ],
        "money": [
            r"\$\d+(?:\.\d{2})?\b",  # $10.50
            r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b",  # $1,000.00
        ],
        "times": [r"\b\d{1,2}:\d{2}(?::\d{2})?(?:AM|PM|am|pm)?\b", r"\b\d{1,2}(?:AM|PM|am|pm)\b"],
        "emails": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
        "urls": [r'\bhttps?://[^\s<>\[\]("`\']+\b'],
        "phones": [r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b"],
    }

    @staticmethod
    def extract_entities(text: str) -> dict[str, list[str]]:
        """Extract entities by category from text."""
        entities: dict[str, list[str]] = {category: [] for category in EntityMetrics.PATTERNS}

        for category, patterns in EntityMetrics.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities[category].extend(matches)

        return entities

    @staticmethod
    def compare_entities(
        reference_entities: dict[str, list[str]], hypothesis_entities: dict[str, list[str]]
    ) -> tuple[int, int, list[dict]]:
        """
        Compare entities between reference and hypothesis.

        Returns:
            (correct_count, total_count, error_list)
        """
        correct = 0
        total = 0
        errors = []

        for category in reference_entities:
            ref_set = set(reference_entities[category])
            hyp_set = set(hypothesis_entities.get(category, []))

            # Count entities in reference
            total += len(ref_set)

            # Count matches
            matches = ref_set & hyp_set
            correct += len(matches)

            # Record errors
            missed = ref_set - hyp_set
            extra = hyp_set - ref_set

            for entity in missed:
                errors.append({"type": "missed", "category": category, "entity": entity})

            for entity in extra:
                errors.append({"type": "extra", "category": category, "entity": entity})

        return correct, total, errors

    @staticmethod
    def calculate_eer(reference: str, hypothesis: str) -> EntityErrorResult:
        """
        Calculate Entity Error Rate (EER).

        EER = (missed_entities + extra_entities) / total_reference_entities
        """
        ref_entities = EntityMetrics.extract_entities(reference)
        hyp_entities = EntityMetrics.extract_entities(hypothesis)

        correct, total, errors = EntityMetrics.compare_entities(ref_entities, hyp_entities)

        entity_error_rate = 1.0 - (correct / max(1, total))

        # Calculate breakdown by entity type
        entity_types = {}
        for category in ref_entities:
            ref_count = len(ref_entities[category])
            hyp_count = len(hyp_entities[category])

            entity_types[category] = {
                "reference_count": ref_count,
                "hypothesis_count": hyp_count,
                "correct": len(set(ref_entities[category]) & set(hyp_entities[category])),
            }

        result = EntityErrorResult(
            entity_error_rate=entity_error_rate,
            total_entities=total,
            correct_entities=correct,
            entity_types=entity_types,
            errors=errors[:10],  # Limit errors for display
        )

        logger.info(f"Entity Error Rate: {entity_error_rate:.3f} ({correct}/{total} entities)")
        return result


class EntitySpecificMetrics:
    """Calculate metrics for specific entity types."""

    @staticmethod
    def calculate_number_accuracy(reference: str, hypothesis: str) -> float:
        """Calculate accuracy specifically for numerical entities."""
        ref_nums = EntityMetrics.extract_entities(reference)["numbers"]
        hyp_nums = EntityMetrics.extract_entities(hypothesis)["numbers"]

        if not ref_nums:
            return 1.0  # Perfect if no numbers to detect

        correct = len(set(ref_nums) & set(hyp_nums))
        return correct / len(ref_nums)

    @staticmethod
    def calculate_date_accuracy(reference: str, hypothesis: str) -> float:
        """Calculate accuracy specifically for date entities."""
        ref_dates = EntityMetrics.extract_entities(reference)["dates"]
        hyp_dates = EntityMetrics.extract_entities(hypothesis)["dates"]

        if not ref_dates:
            return 1.0  # Perfect if no dates to detect

        correct = len(set(ref_dates) & set(hyp_dates))
        return correct / len(ref_dates)

    @staticmethod
    def calculate_money_accuracy(reference: str, hypothesis: str) -> float:
        """Calculate accuracy specifically for money entities."""
        ref_money = EntityMetrics.extract_entities(reference)["money"]
        hyp_money = EntityMetrics.extract_entities(hypothesis)["money"]

        if not ref_money:
            return 1.0  # Perfect if no money to detect

        correct = len(set(ref_money) & set(hyp_money))
        return correct / len(ref_money)


def format_entity_result(result: EntityErrorResult) -> str:
    """Format entity error result for logging."""
    type_summary = ", ".join(
        [
            f"{cat}: {info['correct']}/{info['reference_count']}"
            for cat, info in result.entity_types.items()
            if info["reference_count"] > 0
        ]
    )

    return (
        f"EER: {result.entity_error_rate:.3f} "
        f"({result.correct_entities}/{result.total_entities} entities) "
        f"[{type_summary}]"
    )
