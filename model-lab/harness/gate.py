"""
Production gate: automated go/no-go decision for model deployment.
Encodes promotion criteria to prevent shipping bad models.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .results import ResultsManager

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


@dataclass
class PromotionCriteria:
    """Criteria that must be met for promotion to production."""

    # ASR requirements
    max_wer: float = 0.10  # 10% WER threshold
    max_cer: float = 0.05  # 5% CER threshold
    max_entity_error_rate: float = 0.15  # 15% EER threshold

    # Performance requirements
    max_latency_ms: float = 2000  # 2 second max latency
    max_rtf: float = 1.0  # Must be real-time or faster
    max_memory_mb: float = 4000  # 4GB max memory

    # Stability requirements
    min_success_rate: float = 0.95  # 95% of tests must pass
    min_test_cases: int = 10  # Minimum number of test cases

    # Comparison requirements
    must_beat_baseline: bool = True  # Must not regress from baseline
    baseline_model_id: str | None = None  # Model to compare against

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_wer": self.max_wer,
            "max_cer": self.max_cer,
            "max_entity_error_rate": self.max_entity_error_rate,
            "max_latency_ms": self.max_latency_ms,
            "max_rtf": self.max_rtf,
            "max_memory_mb": self.max_memory_mb,
            "min_success_rate": self.min_success_rate,
            "min_test_cases": self.min_test_cases,
            "must_beat_baseline": self.must_beat_baseline,
            "baseline_model_id": self.baseline_model_id,
        }


@dataclass
class GateCheck:
    """Result of a single gate check."""

    name: str
    passed: bool
    required_value: Any
    actual_value: Any
    message: str


@dataclass
class GateResult:
    """Result of running the production gate."""

    model_id: str
    model_version: str
    timestamp: str
    decision: str  # "PROMOTE", "REJECT", "NEEDS_REVIEW"
    current_status: str
    recommended_status: str
    checks: list[GateCheck]
    summary: dict[str, Any]
    blocking_issues: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return self.decision == "PROMOTE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "current_status": self.current_status,
            "recommended_status": self.recommended_status,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "required": c.required_value,
                    "actual": c.actual_value,
                    "message": c.message,
                }
                for c in self.checks
            ],
            "summary": self.summary,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
        }


class ProductionGate:
    """
    Automated production gate for model promotion decisions.

    Usage:
        gate = ProductionGate(runs_dir, criteria)
        result = gate.evaluate(model_id, model_version)

        if result.passed:
            print("✅ Model approved for production")
        else:
            print("❌ Model rejected:", result.blocking_issues)
    """

    def __init__(
        self,
        runs_dir: Path,
        criteria: PromotionCriteria | None = None,
        registry_path: Path | None = None,
    ):
        self.runs_dir = Path(runs_dir)
        self.criteria = criteria or PromotionCriteria()
        self.results_manager = ResultsManager(runs_dir)
        self.registry_path = registry_path or (runs_dir.parent / "model_registry.json")
        self._load_registry()

    def _load_registry(self):
        """Load or create model registry."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": {}}

    def _save_registry(self):
        """Save model registry."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def evaluate(
        self, model_id: str, model_version: str = "latest", task_type: str = "asr"
    ) -> GateResult:
        """
        Evaluate a model against production criteria.

        Args:
            model_id: Model identifier
            model_version: Version to evaluate
            task_type: Task type (asr, tts, chat)

        Returns:
            GateResult with promotion decision
        """
        checks = []
        blocking_issues = []
        warnings = []

        # Get latest batch results
        batch = self.results_manager.get_latest_batch(model_id, task_type)
        if not batch:
            return GateResult(
                model_id=model_id,
                model_version=model_version,
                timestamp=datetime.now().isoformat(),
                decision="REJECT",
                current_status=self._get_model_status(model_id),
                recommended_status=ModelStatus.EXPERIMENTAL.value,
                checks=[],
                summary={},
                blocking_issues=["No evaluation results found. Run golden tests first."],
                warnings=[],
            )

        summary = batch.summary

        # Check minimum test cases
        checks.append(self._check_min_tests(summary))

        # Check success rate
        checks.append(self._check_success_rate(summary))

        # Check WER (if available)
        if "wer_mean" in summary:
            checks.append(self._check_wer(summary))

        # Check latency
        if "latency_mean_ms" in summary:
            checks.append(self._check_latency(summary))

        # Check against baseline if required
        if self.criteria.must_beat_baseline and self.criteria.baseline_model_id:
            checks.append(self._check_baseline(model_id, task_type))

        # Collect blocking issues and warnings
        for check in checks:
            if not check.passed:
                if "warning" in check.name.lower():
                    warnings.append(check.message)
                else:
                    blocking_issues.append(check.message)

        # Make decision
        all_passed = all(c.passed for c in checks)
        has_blocking = len(blocking_issues) > 0

        if all_passed and not has_blocking:
            decision = "PROMOTE"
            recommended_status = ModelStatus.PRODUCTION.value
        elif has_blocking:
            decision = "REJECT"
            recommended_status = ModelStatus.EXPERIMENTAL.value
        else:
            decision = "NEEDS_REVIEW"
            recommended_status = ModelStatus.CANDIDATE.value

        result = GateResult(
            model_id=model_id,
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
            decision=decision,
            current_status=self._get_model_status(model_id),
            recommended_status=recommended_status,
            checks=checks,
            summary=summary,
            blocking_issues=blocking_issues,
            warnings=warnings,
        )

        self._log_result(result)
        return result

    def _check_min_tests(self, summary: dict[str, Any]) -> GateCheck:
        """Check minimum test case requirement."""
        total = summary.get("total_runs", 0)
        passed = total >= self.criteria.min_test_cases

        return GateCheck(
            name="min_test_cases",
            passed=passed,
            required_value=self.criteria.min_test_cases,
            actual_value=total,
            message=f"Need at least {self.criteria.min_test_cases} test cases, got {total}"
            if not passed
            else f"✓ {total} test cases",
        )

    def _check_success_rate(self, summary: dict[str, Any]) -> GateCheck:
        """Check success rate requirement."""
        rate = summary.get("success_rate", 0)
        passed = rate >= self.criteria.min_success_rate

        return GateCheck(
            name="success_rate",
            passed=passed,
            required_value=self.criteria.min_success_rate,
            actual_value=rate,
            message=f"Success rate {rate:.1%} below threshold {self.criteria.min_success_rate:.1%}"
            if not passed
            else f"✓ {rate:.1%} success rate",
        )

    def _check_wer(self, summary: dict[str, Any]) -> GateCheck:
        """Check WER requirement."""
        wer = summary.get("wer_mean", 1.0)
        passed = wer <= self.criteria.max_wer

        return GateCheck(
            name="wer",
            passed=passed,
            required_value=self.criteria.max_wer,
            actual_value=wer,
            message=f"WER {wer:.3f} exceeds threshold {self.criteria.max_wer:.3f}"
            if not passed
            else f"✓ WER {wer:.3f}",
        )

    def _check_latency(self, summary: dict[str, Any]) -> GateCheck:
        """Check latency requirement."""
        latency = summary.get("latency_mean_ms", float("inf"))
        passed = latency <= self.criteria.max_latency_ms

        return GateCheck(
            name="latency",
            passed=passed,
            required_value=self.criteria.max_latency_ms,
            actual_value=latency,
            message=f"Latency {latency:.0f}ms exceeds threshold {self.criteria.max_latency_ms:.0f}ms"
            if not passed
            else f"✓ Latency {latency:.0f}ms",
        )

    def _check_baseline(self, model_id: str, task_type: str) -> GateCheck:
        """Check if model beats baseline."""
        baseline_id = self.criteria.baseline_model_id
        if not baseline_id:
            return GateCheck(
                name="baseline_comparison",
                passed=True,  # Pass if no baseline configured
                required_value="beat_baseline",
                actual_value="no_baseline_configured",
                message="⚠️ No baseline model configured, skipping comparison",
            )

        baseline_batch = self.results_manager.get_latest_batch(baseline_id, task_type)
        current_batch = self.results_manager.get_latest_batch(model_id, task_type)

        if not baseline_batch:
            return GateCheck(
                name="baseline_comparison",
                passed=True,  # Pass if no baseline exists
                required_value="beat_baseline",
                actual_value="no_baseline",
                message="⚠️ No baseline results found, skipping comparison",
            )

        if not current_batch:
            return GateCheck(
                name="baseline_comparison",
                passed=False,
                required_value="beat_baseline",
                actual_value="no_results",
                message="No current results to compare",
            )

        # Compare WER
        current_wer = current_batch.summary.get("wer_mean", 1.0)
        baseline_wer = baseline_batch.summary.get("wer_mean", 1.0)

        passed = current_wer <= baseline_wer

        return GateCheck(
            name="baseline_comparison",
            passed=passed,
            required_value=f"<= {baseline_wer:.3f} (baseline)",
            actual_value=current_wer,
            message=f"WER {current_wer:.3f} worse than baseline {baseline_wer:.3f}"
            if not passed
            else f"✓ Beats baseline ({current_wer:.3f} <= {baseline_wer:.3f})",
        )

    def _get_model_status(self, model_id: str) -> str:
        """Get current model status from registry."""
        return (
            self.registry.get("models", {})
            .get(model_id, {})
            .get("status", ModelStatus.EXPERIMENTAL.value)
        )

    def promote(self, model_id: str, model_version: str, status: ModelStatus):
        """Update model status in registry."""
        if model_id not in self.registry["models"]:
            self.registry["models"][model_id] = {}

        self.registry["models"][model_id].update(
            {
                "status": status.value,
                "version": model_version,
                "promoted_at": datetime.now().isoformat(),
            }
        )

        self._save_registry()
        logger.info(f"Model {model_id} promoted to {status.value}")

    def _log_result(self, result: GateResult):
        """Log gate result."""
        icon = {"PROMOTE": "✅", "REJECT": "❌", "NEEDS_REVIEW": "⚠️"}[result.decision]

        logger.info(f"\n{'=' * 50}")
        logger.info(f"PRODUCTION GATE: {result.model_id}")
        logger.info(f"{'=' * 50}")
        logger.info(f"Decision: {icon} {result.decision}")
        logger.info(f"Current Status: {result.current_status}")
        logger.info(f"Recommended: {result.recommended_status}")
        logger.info("\nChecks:")

        for check in result.checks:
            icon = "✓" if check.passed else "✗"
            logger.info(f"  {icon} {check.name}: {check.message}")

        if result.blocking_issues:
            logger.info("\n⛔ Blocking Issues:")
            for issue in result.blocking_issues:
                logger.info(f"  - {issue}")

        if result.warnings:
            logger.info("\n⚠️ Warnings:")
            for warning in result.warnings:
                logger.info(f"  - {warning}")

        logger.info(f"{'=' * 50}\n")


def create_default_criteria() -> PromotionCriteria:
    """Create default production criteria."""
    return PromotionCriteria()


def create_strict_criteria() -> PromotionCriteria:
    """Create strict criteria for high-stakes applications."""
    return PromotionCriteria(
        max_wer=0.05,
        max_cer=0.02,
        max_entity_error_rate=0.05,
        max_latency_ms=1000,
        max_rtf=0.5,
        min_success_rate=0.99,
        min_test_cases=50,
    )
