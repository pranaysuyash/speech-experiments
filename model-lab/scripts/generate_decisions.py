#!/usr/bin/env python3
"""
Arsenal Decision Engine - Severity-Based Reasoning (v2.0)
Implements Decision Semantics Contract.

Core Principle: Evidence must be strict. Decisions must be tolerant.
"""

import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Add harness to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.model_card import EvidenceGrade, ModelCard, TaskType
from harness.operability import evaluate_operability, get_task_thresholds
from harness.registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("decisions")


class Severity(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL_FATAL = "fail_fatal"


class Outcome(Enum):
    RECOMMENDED = "recommended"
    ACCEPTABLE = "acceptable"
    REJECTED = "rejected"


@dataclass
class GateEvaluation:
    gate_name: str
    severity: Severity
    reason: str


@dataclass
class DecisionResult:
    model_id: str
    outcome: Outcome
    score: float
    pass_gates: list[str]
    warn_gates: list[str]
    fatal_gates: list[str]
    explanation: str


@dataclass
class PipelineComponent:
    """A single component in a recommended pipeline."""

    task: str
    model_id: str
    evidence_grade: str
    key_metric: float
    metric_name: str


@dataclass
class PipelineResult:
    """Result of pipeline evaluation - best model per required task."""

    pipeline: dict[str, str]  # task -> model_id
    components: list[PipelineComponent]
    outcome: Outcome
    fatal_reasons: list[str]
    warn_reasons: list[str]
    explanation: str


def load_use_cases() -> list[dict[str, Any]]:
    path = Path("docs/use_cases.yaml")
    if not path.exists():
        raise FileNotFoundError("docs/use_cases.yaml not found")
    with open(path) as f:
        return yaml.safe_load(f).get("use_cases", [])


def evaluate_gate(gate_name: str, evidence: Any, use_case: dict[str, Any]) -> GateEvaluation:
    """
    Evaluate a gate's severity based on use case context.

    Returns GateEvaluation with severity and reason.
    """
    fatal_gates = set(use_case.get("fatal_gates", []))
    warning_gates = set(use_case.get("warning_gates", []))

    # Map gate names to evidence signals
    # This is where we interpret the evidence

    # Check for hallucination (always fatal per Decision Semantics Section 8)
    if gate_name == "is_hallucinating":
        # Check if any evidence has hallucination flag
        for ev in evidence:
            if ev.gates.get("is_hallucinating") == "❌ Hallucinating":
                return GateEvaluation(
                    gate_name="is_hallucinating",
                    severity=Severity.FAIL_FATAL,
                    reason="Hallucination detected (universally fatal)",
                )
        return GateEvaluation(gate_name, Severity.PASS, "No hallucination")

    # Check WER validity
    if gate_name == "wer_valid":
        has_invalid_wer = any("❌" in str(ev.gates.get("wer_valid", "")) for ev in evidence)
        if has_invalid_wer:
            if gate_name in fatal_gates:
                return GateEvaluation(
                    gate_name, Severity.FAIL_FATAL, "WER outside acceptable bounds"
                )
            elif gate_name in warning_gates:
                return GateEvaluation(
                    gate_name, Severity.WARN, "WER degraded (conversational tolerance)"
                )
            else:
                return GateEvaluation(gate_name, Severity.WARN, "WER informational")
        return GateEvaluation(gate_name, Severity.PASS, "WER within bounds")

    # Check truncation
    if gate_name == "is_truncated":
        has_truncation = any(ev.gates.get("is_truncated") == "❌ Truncated" for ev in evidence)
        if has_truncation:
            if gate_name in fatal_gates:
                return GateEvaluation(gate_name, Severity.FAIL_FATAL, "Output truncated")
            else:
                return GateEvaluation(
                    gate_name, Severity.WARN, "Output truncated (acceptable for conversational)"
                )
        return GateEvaluation(gate_name, Severity.PASS, "Output complete")

    # Check coverage
    if gate_name == "coverage_low":
        has_low_coverage = any(ev.metrics.get("coverage_ratio", 1.0) < 0.6 for ev in evidence)
        if has_low_coverage:
            return GateEvaluation(gate_name, Severity.FAIL_FATAL, "Coverage < 60%")
        return GateEvaluation(gate_name, Severity.PASS, "Coverage adequate")

    # Check latency
    if gate_name == "latency_high":
        max_latency = use_case.get("constraints", {}).get("max_latency_ms")
        if max_latency:
            has_high_latency = any(
                ev.metrics.get("latency_ms_p50", 0) > max_latency for ev in evidence
            )
            if has_high_latency:
                if gate_name in fatal_gates:
                    return GateEvaluation(
                        gate_name, Severity.FAIL_FATAL, f"Latency > {max_latency}ms"
                    )
                else:
                    return GateEvaluation(gate_name, Severity.WARN, "Latency high (non-critical)")
        return GateEvaluation(gate_name, Severity.PASS, "Latency acceptable")

    # Check audio empty (V2V)
    if gate_name == "audio_empty":
        has_empty_audio = any(ev.metrics.get("audio_duration_s", 1.0) == 0 for ev in evidence)
        if has_empty_audio:
            return GateEvaluation(gate_name, Severity.FAIL_FATAL, "No audio generated")
        return GateEvaluation(gate_name, Severity.PASS, "Audio generated")

    # Check speaker error (Diarization)
    if gate_name == "speaker_error_high":
        has_speaker_error = any(
            abs(ev.metrics.get("speaker_count_error", 0)) > 1 for ev in evidence
        )
        if has_speaker_error:
            return GateEvaluation(gate_name, Severity.FAIL_FATAL, "Speaker count error > 1")
        return GateEvaluation(gate_name, Severity.PASS, "Speaker detection accurate")

    # Default: gate not recognized, treat as PASS
    return GateEvaluation(gate_name, Severity.PASS, f"Gate {gate_name} not evaluated")


def get_best_evidence_grade(card: ModelCard, task: str) -> EvidenceGrade:
    """
    Get best valid evidence grade for a task.

    IMPORTANT: Adhoc runs are EXCLUDED from decision evidence.
    They produce artifacts for debugging/output but don't affect recommendations.
    This is per the adhoc contract: grade=adhoc means "execution only, no evidence".
    """
    ev_list = [e for e in card.evidence if e.task == task]
    if not ev_list:
        return None

    # Filter to valid evidence only
    valid_ev = [e for e in ev_list if e.valid]
    if not valid_ev:
        return None

    # EXPLICITLY EXCLUDE ADHOC from decision evidence
    # Adhoc runs are for execution output, not for model evaluation/ranking
    non_adhoc_ev = [e for e in valid_ev if e.evidence_grade != EvidenceGrade.ADHOC]
    if not non_adhoc_ev:
        return None  # No non-adhoc evidence = no evidence for decisions

    # Rank: Golden > Smoke (adhoc excluded above)
    grades = [e.evidence_grade for e in non_adhoc_ev]
    if EvidenceGrade.GOLDEN_BATCH in grades:
        return EvidenceGrade.GOLDEN_BATCH
    if EvidenceGrade.SMOKE in grades:
        return EvidenceGrade.SMOKE
    return None  # Should not reach here after filtering


def evaluate_model_for_use_case(card: ModelCard, use_case: dict[str, Any]) -> DecisionResult:
    """
    Evaluate model against use case using Decision Semantics v2.0.

    Returns DecisionResult with outcome and explanation.
    """
    pass_gates = []
    warn_gates = []
    fatal_gates = []
    score = 0.0

    # 1. Check primary capabilities (hard requirement)
    primary_reqs = use_case["requirements"].get("primary", [])
    for req in primary_reqs:
        task = req["task"]
        min_grade_str = req.get("min_grade", "smoke")  # Default to smoke if not specified

        # Check if declared
        declared = any(d.task == task for d in card.declared_capabilities)
        if not declared:
            fatal_gates.append(f"Missing primary capability: {task}")
            continue

        # Check evidence exists
        grade = get_best_evidence_grade(card, task)
        if not grade:
            fatal_gates.append(f"No valid evidence for: {task}")
            continue

        # NEW: Check evidence grade meets minimum requirement
        try:
            min_grade = EvidenceGrade(min_grade_str)
            grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}

            if grade_rank.get(grade.value, 0) < grade_rank.get(min_grade.value, 0):
                # Evidence exists but is insufficient quality
                fatal_gates.append(
                    f"{task}: evidence grade {grade.value} < required {min_grade.value}"
                )
                continue
        except (ValueError, KeyError):
            # Invalid min_grade in config, skip check
            pass

        # Has capability with sufficient evidence
        score += 1.0
        if grade == EvidenceGrade.GOLDEN_BATCH:
            score += 0.5
        pass_gates.append(f"Has {task} ({grade.name})")

    # 2. Evaluate gates based on severity
    all_gates = set(use_case.get("fatal_gates", []) + use_case.get("warning_gates", []))

    # Get evidence for primary tasks
    primary_tasks = [req["task"] for req in primary_reqs]
    evidence = [e for e in card.evidence if e.task in primary_tasks]

    for gate_name in all_gates:
        gate_eval = evaluate_gate(gate_name, evidence, use_case)

        if gate_eval.severity == Severity.FAIL_FATAL:
            fatal_gates.append(gate_eval.reason)
        elif gate_eval.severity == Severity.WARN:
            warn_gates.append(gate_eval.reason)
            score -= 0.2  # Penalty for warnings
        else:  # PASS
            pass_gates.append(gate_eval.reason)

    # 3. Check deployment constraints
    constraints = use_case.get("constraints", {})
    if constraints.get("offline_capable"):
        if not card.deployment.offline_capable:
            fatal_gates.append("Requires offline capability")

    # 4. Check min_runs requirements
    min_runs_config = use_case.get("min_runs", {})
    min_runs_required = min_runs_config.get("required", {})
    insufficient_evidence = []

    for req in primary_reqs:
        task = req["task"]
        min_grade_str = req.get("min_grade", "smoke")
        min_needed = min_runs_required.get(task, 1)

        # Count valid runs for this task at required grade
        run_count = _count_valid_runs_for_model(card, task, min_grade_str)
        if run_count < min_needed:
            insufficient_evidence.append(
                f"INSUFFICIENT_EVIDENCE(task={task}, have={run_count}, need={min_needed})"
            )

    # 5. Determine outcome
    if fatal_gates:
        outcome = Outcome.REJECTED
    elif insufficient_evidence:
        # Insufficient evidence = ACCEPTABLE at best, never RECOMMENDED
        outcome = Outcome.ACCEPTABLE
        warn_gates.extend(insufficient_evidence)
    elif len(warn_gates) <= 1:
        outcome = Outcome.RECOMMENDED
    else:
        outcome = Outcome.ACCEPTABLE

    # 6. Generate explanation
    explanation = _generate_explanation(outcome, pass_gates, warn_gates, fatal_gates)

    return DecisionResult(
        model_id=card.model_id,
        outcome=outcome,
        score=score,
        pass_gates=pass_gates,
        warn_gates=warn_gates,
        fatal_gates=fatal_gates,
        explanation=explanation,
    )


def _count_valid_runs_for_model(card: ModelCard, task: str, min_grade: str) -> int:
    """Count valid runs for a model and task."""
    grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
    min_grade_rank = grade_rank.get(min_grade, 1)

    count = 0
    for evidence in card.evidence:
        if evidence.task != task:
            continue
        if not evidence.valid:
            continue

        ev_grade = (
            evidence.evidence_grade.value
            if hasattr(evidence.evidence_grade, "value")
            else str(evidence.evidence_grade)
        )
        if grade_rank.get(ev_grade, 0) < min_grade_rank:
            continue

        count += 1

    return count


def _generate_explanation(
    outcome: Outcome, pass_gates: list[str], warn_gates: list[str], fatal_gates: list[str]
) -> str:
    """Generate human-readable explanation per Decision Semantics Section 5."""
    lines = []

    if outcome == Outcome.RECOMMENDED:
        lines.append("✅ Best choice for this use case")
    elif outcome == Outcome.ACCEPTABLE:
        lines.append("⚠️ Usable with trade-offs")
    else:
        lines.append("❌ Not suitable for this use case")

    if pass_gates:
        lines.append("\nWhat worked:")
        for gate in pass_gates[:3]:  # Top 3
            lines.append(f"  • {gate}")

    if warn_gates and outcome != Outcome.REJECTED:
        lines.append("\nWhat you must accept:")
        for gate in warn_gates:
            lines.append(f"  • {gate}")

    if fatal_gates:
        lines.append("\nBlockers:")
        for gate in fatal_gates[:2]:  # Top 2
            lines.append(f"  • {gate}")
        if len(fatal_gates) > 2:
            lines.append(f"  • ... and {len(fatal_gates) - 2} more")

    return "\n".join(lines)


def generate_decisions_doc(results: dict[str, list[DecisionResult]]):
    """Generate DECISIONS.md with three-tier outcomes."""
    lines = [
        "# Arsenal Decision Matrix",
        "",
        "> **Core Principle**: Decisions = Observed Evidence + Declared Intent.",
        "> **Decision Semantics**: v2.0 (Graduated Outcomes)",
        "",
        "## Summary",
        "",
    ]

    for use_case_name, decisions in results.items():
        recommended = [d for d in decisions if d.outcome == Outcome.RECOMMENDED]
        acceptable = [d for d in decisions if d.outcome == Outcome.ACCEPTABLE]
        rejected = [d for d in decisions if d.outcome == Outcome.REJECTED]

        lines.append(f"### {use_case_name}")

        if recommended:
            lines.append(f"**✅ Recommended:** `{recommended[0].model_id}`")
        elif acceptable:
            lines.append(f"**⚠️ Acceptable:** `{acceptable[0].model_id}` (with trade-offs)")
        else:
            lines.append("**❌ No viable models found**")

        lines.append("")

        # Table
        lines.append("| Model | Outcome | Score | Details |")
        lines.append("|-------|---------|-------|---------|")

        # Sort: RECOMMENDED > ACCEPTABLE > REJECTED, then by score
        sorted_decs = sorted(
            decisions,
            key=lambda x: (
                x.outcome == Outcome.RECOMMENDED,
                x.outcome == Outcome.ACCEPTABLE,
                x.score,
            ),
            reverse=True,
        )

        for d in sorted_decs:
            icon = (
                "✅"
                if d.outcome == Outcome.RECOMMENDED
                else "⚠️"
                if d.outcome == Outcome.ACCEPTABLE
                else "❌"
            )
            outcome_str = d.outcome.name

            # Compress explanation for table
            if d.outcome == Outcome.REJECTED:
                detail = "<br>".join(d.fatal_gates[:2])
                if len(d.fatal_gates) > 2:
                    detail += f"<br>... +{len(d.fatal_gates) - 2} more"
            else:
                detail = f"{len(d.pass_gates)} strengths, {len(d.warn_gates)} warnings"

            lines.append(f"| **{d.model_id}** | {icon} {outcome_str} | {d.score:.1f} | {detail} |")

        lines.append("")

        # Detailed explanations for non-rejected
        if recommended or acceptable:
            lines.append("<details>")
            lines.append("<summary>Detailed Explanations</summary>")
            lines.append("")

            for d in recommended + acceptable:
                lines.append(f"#### {d.model_id}")
                lines.append("```")
                lines.append(d.explanation)
                lines.append("```")
                lines.append("")

            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open("docs/DECISIONS.md", "w") as f:
        f.write("\n".join(lines))
    print("✅ Generated docs/DECISIONS.md")


def select_best_model_for_task(
    cards: list[ModelCard], task: str, min_grade: str, thresholds: dict[str, Any] = None
) -> tuple:
    """
    Select the best model for a specific task based on evidence grade and metrics.

    Args:
        cards: List of model cards
        task: Task name (e.g., "asr", "v2v")
        min_grade: Minimum evidence grade required
        thresholds: Operability thresholds for this task (smoke-only)

    Returns:
        (model_id, evidence_grade, key_metric, metric_name, operability_rejections)
        or (None, None, None, None, rejections) if no viable model.
    """
    grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
    min_grade_rank = grade_rank.get(min_grade, 1)

    candidates = []
    all_operability_rejections = []  # Track operability failures

    for card in cards:
        # Check if model has this capability declared
        if not any(d.task == task for d in card.declared_capabilities):
            continue

        # Get best evidence for this task
        grade = get_best_evidence_grade(card, task)
        if not grade:
            continue

        # Check grade meets minimum
        if grade_rank.get(grade.value, 0) < min_grade_rank:
            continue

        # Get relevant metric based on task
        evidence = [e for e in card.evidence if e.task == task and e.valid]
        if not evidence:
            continue

        # Filter by operability thresholds (smoke evidence only)
        viable_evidence = []
        operability_rejections = []
        for e in evidence:
            if e.evidence_grade == EvidenceGrade.SMOKE and thresholds:
                passed, failures = evaluate_operability(task, e.metrics, thresholds)
                if not passed:
                    operability_rejections.append(f"{card.model_id}: {'; '.join(failures)}")
                    continue
            viable_evidence.append(e)

        evidence = viable_evidence
        if not evidence:
            # Track that this model was rejected for operability
            all_operability_rejections.extend(operability_rejections)
            continue

        # Task-specific metric selection
        key_metric = None  # None means "n/a"
        metric_name = "available"

        if task == "asr":
            # Check if any evidence has actual WER (with ground truth)
            wers = [e.metrics.get("wer") for e in evidence if e.metrics.get("wer") is not None]
            if wers:
                wer = min(wers)
                key_metric = 1.0 - wer  # Higher = better
                metric_name = "1-WER"
            else:
                # No WER available (no ground truth) - structural only
                key_metric = None
                metric_name = "n/a"
        elif task == "diarization":
            # Check if actual speaker error metric exists
            spk_errs = [
                abs(e.metrics.get("speaker_count_error", 0))
                for e in evidence
                if "speaker_count_error" in e.metrics
            ]
            if spk_errs:
                spk_err = min(spk_errs)
                key_metric = 1.0 / (1 + spk_err)
                metric_name = "speaker_accuracy"
            else:
                key_metric = None
                metric_name = "n/a"
        elif task == "vad":
            # Use speech_ratio as structural metric (can regress)
            speech_ratios = [
                e.metrics.get("speech_ratio")
                for e in evidence
                if e.metrics.get("speech_ratio") is not None
            ]
            if speech_ratios:
                ratio = sum(speech_ratios) / len(speech_ratios)
                key_metric = ratio
                metric_name = "speech_ratio"
            else:
                # Fallback to rtf if available
                rtfs = [e.metrics.get("rtf") for e in evidence if e.metrics.get("rtf") is not None]
                if rtfs:
                    key_metric = 1.0 / (1 + min(rtfs))  # Faster = higher score
                    metric_name = "1/rtf"
                else:
                    key_metric = None
                    metric_name = "n/a"
        elif task == "v2v":
            # Use rtf_like as primary metric (normalized by audio duration)
            rtf_likes = []
            for e in evidence:
                rtf = e.metrics.get("rtf_like")
                if rtf is not None:
                    rtf_likes.append(rtf)

            if rtf_likes:
                rtf_like = min(rtf_likes)  # Lower is better
                key_metric = 1.0 / (1 + rtf_like)  # Higher score = faster
                metric_name = f"rtf_like={rtf_like:.2f}"
            else:
                # Fallback to latency if rtf_like not available
                latencies = [
                    e.metrics.get("latency_ms")
                    for e in evidence
                    if e.metrics.get("latency_ms") is not None
                ]
                if latencies:
                    latency = min(latencies)
                    key_metric = 10000.0 / (1 + latency)
                    metric_name = f"latency_ms={latency:.0f}"
                else:
                    key_metric = None
                    metric_name = "n/a"
        elif task == "tts":
            key_metric = 1.0  # Structural existence
            metric_name = "available"
        else:
            key_metric = 1.0
            metric_name = "available"

        # Use 0.0 for sorting when metric is n/a
        sort_metric = key_metric if key_metric is not None else 0.0

        candidates.append(
            (
                card.model_id,
                grade.value,
                key_metric,  # Can be None (n/a)
                metric_name,
                grade_rank.get(grade.value, 0),
                sort_metric,  # For sorting only
            )
        )

    if not candidates:
        return None, None, None, None, all_operability_rejections

    # Sort by grade (desc) then sort_metric (desc)
    candidates.sort(key=lambda x: (x[4], x[5]), reverse=True)
    best = candidates[0]
    return best[0], best[1], best[2], best[3], []  # No rejections for successful selection


def evaluate_pipeline_for_use_case(
    cards: list[ModelCard], use_case: dict[str, Any]
) -> PipelineResult:
    """
    Evaluate and assemble the best pipeline for a use case.

    For each required task, select the best model independently.
    Returns a PipelineResult with the assembled stack.
    """
    pipeline = {}
    components = []
    fatal_reasons = []
    warn_reasons = []
    insufficient_evidence = []  # Track tasks with insufficient runs

    # Get operability thresholds for this use case
    operability_config = use_case.get("operability", {})

    # Get min_runs requirements
    min_runs_config = use_case.get("min_runs", {})
    min_runs_required = min_runs_config.get("required", {})
    min_runs_secondary = min_runs_config.get("secondary", {})

    # Get all required tasks
    primary_reqs = use_case["requirements"].get("primary", [])
    secondary_reqs = use_case["requirements"].get("secondary", [])

    # Select best model for each primary task
    for req in primary_reqs:
        task = req["task"]
        min_grade = req.get("min_grade", "smoke")
        thresholds = get_task_thresholds(operability_config, task)

        model_id, grade, metric, metric_name, op_rejections = select_best_model_for_task(
            cards, task, min_grade, thresholds
        )

        if model_id is None:
            if op_rejections:
                # Operability-specific failure
                fatal_reasons.append(
                    f"Primary task {task}: operability failed - {'; '.join(op_rejections[:2])}"
                )
            else:
                fatal_reasons.append(
                    f"No viable model for primary task: {task} (min_grade: {min_grade})"
                )
        else:
            pipeline[task] = model_id
            components.append(
                PipelineComponent(
                    task=task,
                    model_id=model_id,
                    evidence_grade=grade,
                    key_metric=metric,
                    metric_name=metric_name,
                )
            )

            # Check min_runs for primary task
            min_needed = min_runs_required.get(task, 1)
            run_count = _count_valid_runs_for_task(cards, model_id, task, min_grade)
            if run_count < min_needed:
                insufficient_evidence.append(
                    f"INSUFFICIENT_EVIDENCE(task={task}, have={run_count}, need={min_needed})"
                )

            # Check min_distinct_datasets for primary task (prevents same-dataset gaming)
            min_distinct_config = use_case.get("min_distinct_datasets", {})
            min_distinct_required = min_distinct_config.get("required", {})
            min_distinct_needed = min_distinct_required.get(task, 1)
            distinct_count = _count_distinct_datasets_for_task(cards, model_id, task, min_grade)
            if distinct_count < min_distinct_needed:
                insufficient_evidence.append(
                    f"INSUFFICIENT_DISTINCT_DATASETS(task={task}, have={distinct_count}, need={min_distinct_needed})"
                )

    # Select best model for secondary tasks (failures are warnings, not fatal)
    for req in secondary_reqs:
        task = req["task"]
        min_grade = req.get("min_grade", "smoke")
        thresholds = get_task_thresholds(operability_config, task)

        model_id, grade, metric, metric_name, op_rejections = select_best_model_for_task(
            cards, task, min_grade, thresholds
        )

        if model_id is None:
            if op_rejections:
                warn_reasons.append(f"Secondary task {task}: operability failed")
            else:
                warn_reasons.append(f"No model for secondary task: {task}")
        else:
            pipeline[task] = model_id
            components.append(
                PipelineComponent(
                    task=task,
                    model_id=model_id,
                    evidence_grade=grade,
                    key_metric=metric,
                    metric_name=metric_name,
                )
            )

    # Determine outcome
    if fatal_reasons:
        outcome = Outcome.REJECTED
    elif insufficient_evidence:
        # Insufficient evidence means ACCEPTABLE at best, never RECOMMENDED
        outcome = Outcome.ACCEPTABLE
        warn_reasons.extend(insufficient_evidence)
    elif warn_reasons:
        outcome = Outcome.ACCEPTABLE
    else:
        outcome = Outcome.RECOMMENDED

    # Generate explanation
    explanation = _generate_pipeline_explanation(outcome, components, fatal_reasons, warn_reasons)

    return PipelineResult(
        pipeline=pipeline,
        components=components,
        outcome=outcome,
        fatal_reasons=fatal_reasons,
        warn_reasons=warn_reasons,
        explanation=explanation,
    )


def _count_valid_runs_for_task(
    cards: list[ModelCard], model_id: str, task: str, min_grade: str
) -> int:
    """Count valid runs for a specific model and task."""
    grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
    min_grade_rank = grade_rank.get(min_grade, 1)

    for card in cards:
        if card.model_id != model_id:
            continue

        count = 0
        for evidence in card.evidence:
            if evidence.task != task:
                continue
            if not evidence.valid:
                continue

            # Check grade meets minimum
            ev_grade = (
                evidence.evidence_grade.value
                if hasattr(evidence.evidence_grade, "value")
                else str(evidence.evidence_grade)
            )
            if grade_rank.get(ev_grade, 0) < min_grade_rank:
                continue

            count += 1

        return count

    return 0


def _count_distinct_datasets_for_task(
    cards: list[ModelCard], model_id: str, task: str, min_grade: str
) -> int:
    """Count distinct dataset_ids among valid runs for a specific model and task."""
    grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
    min_grade_rank = grade_rank.get(min_grade, 1)

    for card in cards:
        if card.model_id != model_id:
            continue

        dataset_ids = set()
        for evidence in card.evidence:
            if evidence.task != task:
                continue
            if not evidence.valid:
                continue

            # Check grade meets minimum
            ev_grade = (
                evidence.evidence_grade.value
                if hasattr(evidence.evidence_grade, "value")
                else str(evidence.evidence_grade)
            )
            if grade_rank.get(ev_grade, 0) < min_grade_rank:
                continue

            # Track distinct dataset_ids
            dataset_id = getattr(evidence, "dataset_id", None)
            if dataset_id:
                dataset_ids.add(dataset_id)

        return len(dataset_ids)

    return 0


def _generate_pipeline_explanation(
    outcome: Outcome,
    components: list[PipelineComponent],
    fatal_reasons: list[str],
    warn_reasons: list[str],
) -> str:
    """Generate explanation for pipeline decision."""
    lines = []

    if outcome == Outcome.RECOMMENDED:
        lines.append("✅ Complete pipeline available")
    elif outcome == Outcome.ACCEPTABLE:
        lines.append("⚠️ Partial pipeline (missing secondary tasks)")
    else:
        lines.append("❌ Cannot assemble viable pipeline")

    if components:
        lines.append("\nPipeline components:")
        for c in components:
            metric_str = f"{c.key_metric:.2f}" if c.key_metric is not None else "n/a"
            lines.append(
                f"  • {c.task}: {c.model_id} ({c.evidence_grade}, {c.metric_name}={metric_str})"
            )

    if warn_reasons:
        lines.append("\nMissing (non-fatal):")
        for r in warn_reasons:
            lines.append(f"  • {r}")

    if fatal_reasons:
        lines.append("\nBlockers:")
        for r in fatal_reasons:
            lines.append(f"  • {r}")

    return "\n".join(lines)


def generate_pipeline_decisions_doc(
    results: dict[str, PipelineResult], single_model_results: dict[str, list[DecisionResult]]
):
    """Generate DECISIONS.md with both single-model and pipeline results."""
    lines = [
        "# Arsenal Decision Matrix",
        "",
        "> **Core Principle**: Decisions = Observed Evidence + Declared Intent.",
        "> **Decision Semantics**: v2.0 (Graduated Outcomes + Pipeline Support)",
        "",
        "## Summary",
        "",
    ]

    # Single-model use cases first
    for use_case_name, decisions in single_model_results.items():
        recommended = [d for d in decisions if d.outcome == Outcome.RECOMMENDED]
        acceptable = [d for d in decisions if d.outcome == Outcome.ACCEPTABLE]

        lines.append(f"### {use_case_name}")
        lines.append("*Evaluation mode: single_model*")
        lines.append("")

        if recommended:
            lines.append(f"**✅ Recommended:** `{recommended[0].model_id}`")
        elif acceptable:
            lines.append(f"**⚠️ Acceptable:** `{acceptable[0].model_id}` (with trade-offs)")
        else:
            lines.append("**❌ No viable models found**")

        lines.append("")

        # Table
        lines.append("| Model | Outcome | Score | Details |")
        lines.append("|-------|---------|-------|---------|")

        sorted_decs = sorted(
            decisions,
            key=lambda x: (
                x.outcome == Outcome.RECOMMENDED,
                x.outcome == Outcome.ACCEPTABLE,
                x.score,
            ),
            reverse=True,
        )

        for d in sorted_decs:
            icon = (
                "✅"
                if d.outcome == Outcome.RECOMMENDED
                else "⚠️"
                if d.outcome == Outcome.ACCEPTABLE
                else "❌"
            )
            if d.outcome == Outcome.REJECTED:
                detail = "<br>".join(d.fatal_gates[:2])
            else:
                detail = f"{len(d.pass_gates)} strengths, {len(d.warn_gates)} warnings"
            lines.append(
                f"| **{d.model_id}** | {icon} {d.outcome.name} | {d.score:.1f} | {detail} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

    # Pipeline use cases
    for use_case_name, result in results.items():
        lines.append(f"### {use_case_name}")
        lines.append("*Evaluation mode: pipeline*")
        lines.append("")

        icon = (
            "✅"
            if result.outcome == Outcome.RECOMMENDED
            else "⚠️"
            if result.outcome == Outcome.ACCEPTABLE
            else "❌"
        )
        lines.append(f"**{icon} {result.outcome.name}**")
        lines.append("")

        if result.pipeline:
            lines.append("**Recommended Pipeline:**")
            lines.append("```")
            for task, model_id in result.pipeline.items():
                lines.append(f"  {task}: {model_id}")
            lines.append("```")
            lines.append("")

            lines.append("| Task | Model | Grade | Metric |")
            lines.append("|------|-------|-------|--------|")
            for c in result.components:
                # metric_name already includes value (e.g., "rtf_like=0.56")
                lines.append(f"| {c.task} | {c.model_id} | {c.evidence_grade} | {c.metric_name} |")
            lines.append("")

        if result.fatal_reasons:
            lines.append("**Blockers:**")
            for r in result.fatal_reasons:
                lines.append(f"- {r}")
            lines.append("")

        if result.warn_reasons:
            lines.append("**Warnings:**")
            for r in result.warn_reasons:
                lines.append(f"- {r}")
            lines.append("")

        lines.append("---")
        lines.append("")

    with open("docs/DECISIONS.md", "w") as f:
        f.write("\n".join(lines))
    print("✅ Generated docs/DECISIONS.md (with pipeline support)")


def main():
    use_cases = load_use_cases()

    # Load Registry Metadata
    registry_meta = {}
    for model_id in ModelRegistry.list_models():
        registry_meta[model_id] = ModelRegistry.get_model_metadata(model_id)

    # Load all models
    models_dir = Path("models")
    cards = []

    for config_path in models_dir.glob("*/config.yaml"):
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            model_id = config.get("model_type")

            if not model_id or model_id not in registry_meta:
                logger.warning(f"Skipping {config_path}: unknown model_type {model_id}")
                continue

            # Load evidence from ALL task subdirectories
            evidence_data = []
            runs_dir = Path(f"runs/{model_id}")
            if runs_dir.exists():
                for run_file in runs_dir.glob("**/*.json"):
                    try:
                        with open(run_file) as rf:
                            run = json.load(rf)
                            if "evidence" not in run:
                                continue

                            ev_raw = run["evidence"]
                            ev = {}

                            # FIX 1: Handle both schemas
                            # ASR runs: {"capability": "asr", "evidence": {...}}
                            # Alignment runs: {"meta": {"task": "alignment"}, "evidence": {...}}
                            if "meta" in run and "task" in run["meta"]:
                                task_str = run["meta"]["task"]
                            elif "capability" in run:
                                task_str = run["capability"]
                            else:
                                # Skip runs without task/capability info
                                continue

                            # FIX 2: Convert task string to TaskType enum
                            try:
                                # TaskType enum values are lowercase
                                ev["task"] = TaskType(task_str.lower())
                            except ValueError:
                                # Skip evidence with unknown task types
                                continue

                            ev["dataset_id"] = ev_raw.get("dataset_id", "unknown")
                            ev["evidence_grade"] = ev_raw.get("grade")
                            ev["metrics"] = run.get("metrics", {})
                            ev["valid"] = ev_raw.get("wer_valid", True)
                            ev["invalid_reasons"] = []
                            ev["gates"] = ev_raw.get("sanity_gates", {})
                            ev["device"] = run["system"].get("device")

                            # Handle timestamp from either manifest or meta
                            if "manifest" in run and "timestamp" in run["manifest"]:
                                ev["run_date"] = run["manifest"]["timestamp"]
                                ev["verified_at"] = run["manifest"]["timestamp"]
                            elif "meta" in run and "timestamp" in run["meta"]:
                                ev["run_date"] = run["meta"]["timestamp"]
                                ev["verified_at"] = run["meta"]["timestamp"]
                            else:
                                ev["run_date"] = None
                                ev["verified_at"] = None

                            evidence_data.append(ev)
                    except Exception:
                        # Silently skip malformed runs
                        pass

            card = ModelCard.from_sources(
                model_id=model_id,
                registry_meta=registry_meta[model_id],
                config=config,
                evidence_data=evidence_data,
            )
            cards.append(card)
        except Exception as e:
            logger.warning(f"Skipping {config_path}: {e}")

    # Evaluate - route by evaluation_mode
    single_model_results = {}
    pipeline_results = {}

    for uc in use_cases:
        evaluation_mode = uc.get("evaluation_mode", "single_model")

        if evaluation_mode == "pipeline":
            # Pipeline evaluation: best model per task
            result = evaluate_pipeline_for_use_case(cards, uc)
            pipeline_results[uc["name"]] = result
        else:
            # Single-model evaluation (default)
            uc_results = []
            for card in cards:
                res = evaluate_model_for_use_case(card, uc)
                uc_results.append(res)
            single_model_results[uc["name"]] = uc_results

    # Generate combined DECISIONS.md
    generate_pipeline_decisions_doc(pipeline_results, single_model_results)

    # Generate machine-readable decisions.json for wrapper consumption
    generate_decisions_json(pipeline_results, single_model_results, cards)


def generate_decisions_json(
    pipeline_results: dict[str, PipelineResult],
    single_model_results: dict[str, list[DecisionResult]],
    cards: list[ModelCard],
):
    """
    Generate machine-readable decisions.json for wrapper consumption.

    Structure:
    {
        "generated_at": "...",
        "use_cases": {
            "<use_case_name>": {
                "evaluation_mode": "single_model|pipeline",
                "recommended": [...],  # model_ids with RECOMMENDED outcome
                "acceptable": [...],   # model_ids with ACCEPTABLE outcome
                "rejected": [...],     # model_ids with REJECTED outcome
            }
        },
        "tasks": {
            "<task>": {
                "best_by_outcome": "<model_id>",  # RECOMMENDED > ACCEPTABLE > REJECTED
                "best_by_grade": "<model_id>",   # For BEST_EFFORT fallback only
                "models": [
                    {"model_id": "...", "outcome": "...", "evidence_grade": "...", "key_metric": ...}
                ]
            }
        }
    }
    """
    from datetime import datetime

    decisions = {
        "generated_at": datetime.now().isoformat(),
        "schema_version": "1.0",
        "use_cases": {},
        "tasks": {},
    }

    # Use case decisions
    for uc_name, results in single_model_results.items():
        recommended = [d.model_id for d in results if d.outcome == Outcome.RECOMMENDED]
        acceptable = [d.model_id for d in results if d.outcome == Outcome.ACCEPTABLE]
        rejected = [d.model_id for d in results if d.outcome == Outcome.REJECTED]

        # Sort by score within each category
        results_sorted = sorted(results, key=lambda x: x.score, reverse=True)

        decisions["use_cases"][uc_name] = {
            "evaluation_mode": "single_model",
            "recommended": recommended,
            "acceptable": acceptable,
            "rejected": rejected,
            "best_model": recommended[0]
            if recommended
            else (acceptable[0] if acceptable else None),
            "models": [
                {
                    "model_id": d.model_id,
                    "outcome": d.outcome.value,
                    "score": d.score,
                    "fatal_reasons": d.fatal_gates[:2] if d.fatal_gates else [],
                }
                for d in results_sorted
            ],
        }

    for uc_name, result in pipeline_results.items():
        decisions["use_cases"][uc_name] = {
            "evaluation_mode": "pipeline",
            "outcome": result.outcome.value,
            "pipeline": result.pipeline,
            "components": [
                {
                    "task": c.task,
                    "model_id": c.model_id,
                    "evidence_grade": c.evidence_grade,
                    "key_metric": c.key_metric,
                }
                for c in result.components
            ],
            "fatal_reasons": result.fatal_reasons,
            "warn_reasons": result.warn_reasons,
        }

    # Task-level decisions (for simple --task queries)
    # Aggregate across all use cases to find best model per task
    task_models = {}  # task -> [(model_id, best_outcome, grade, metric)]

    grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
    outcome_rank = {"recommended": 3, "acceptable": 2, "rejected": 1}

    for card in cards:
        for ev in card.evidence:
            task = ev.task if isinstance(ev.task, str) else ev.task.value
            if task not in task_models:
                task_models[task] = []

            # Determine outcome for this model/task from use cases
            # Find any use case where this model has the task as primary
            best_outcome = "rejected"  # default
            for uc_name, results in single_model_results.items():
                for d in results:
                    if d.model_id == card.model_id and d.outcome != Outcome.REJECTED:
                        if outcome_rank.get(d.outcome.value, 0) > outcome_rank.get(best_outcome, 0):
                            best_outcome = d.outcome.value

            grade = (
                ev.evidence_grade if isinstance(ev.evidence_grade, str) else ev.evidence_grade.value
            )
            metric = ev.metrics.get("wer", ev.metrics.get("latency_ms_p50", 0))

            task_models[task].append(
                {
                    "model_id": card.model_id,
                    "outcome": best_outcome,
                    "evidence_grade": grade,
                    "key_metric": 1.0 - metric if "wer" in str(metric) else metric,
                    "valid": ev.valid,
                }
            )

    for task, models in task_models.items():
        # Sort by outcome (desc), then grade (desc), then metric
        models_sorted = sorted(
            models,
            key=lambda x: (
                outcome_rank.get(x["outcome"], 0),
                grade_rank.get(x["evidence_grade"], 0),
                x["key_metric"] if x["key_metric"] is not None and x["valid"] else -999,
            ),
            reverse=True,
        )

        # Deduplicate by model_id (keep best)
        seen = set()
        unique_models = []
        for m in models_sorted:
            if m["model_id"] not in seen:
                seen.add(m["model_id"])
                unique_models.append(m)

        best_by_outcome = unique_models[0] if unique_models else None

        # Best by grade only (for BEST_EFFORT fallback)
        by_grade = sorted(
            unique_models, key=lambda x: grade_rank.get(x["evidence_grade"], 0), reverse=True
        )
        best_by_grade = by_grade[0] if by_grade else None

        decisions["tasks"][task] = {
            "best_by_outcome": best_by_outcome["model_id"] if best_by_outcome else None,
            "best_by_grade": best_by_grade["model_id"] if best_by_grade else None,
            "models": unique_models,
        }

    with open("docs/decisions.json", "w") as f:
        json.dump(decisions, f, indent=2)
    print("✅ Generated docs/decisions.json (machine-readable)")


if __name__ == "__main__":
    main()
