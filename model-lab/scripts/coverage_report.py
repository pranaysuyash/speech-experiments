#!/usr/bin/env python3
"""
Coverage Report Generator.

Generates the models × tasks × evidence matrix that reveals:
- What each model claims
- What's actually tested
- What grade the evidence is
- Operability pass/fail
- Use case eligibility

This is the "north star" that shows what's real vs vibes.

Outputs:
- docs/COVERAGE.md (human-readable)
- docs/coverage.json (machine-readable)
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.model_card import ModelCard
from harness.registry import ModelRegistry


@dataclass
class TaskEvidence:
    """Evidence summary for a single task."""

    task: str
    declared: bool = False
    has_evidence: bool = False
    best_grade: str | None = None
    last_run: str | None = None
    run_count: int = 0
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    has_provenance: bool = False
    has_run_context: bool = False
    operability_status: dict[str, str] = field(default_factory=dict)  # use_case -> pass/fail


@dataclass
class ModelCoverage:
    """Coverage summary for a single model."""

    model_id: str
    version: str
    model_type: str
    status: str
    devices_supported: list[str]
    declared_capabilities: list[str]
    tasks: dict[str, TaskEvidence] = field(default_factory=dict)
    use_case_eligibility: dict[str, str] = field(
        default_factory=dict
    )  # use_case -> eligible/blocked/reason


def load_use_cases() -> list[dict[str, Any]]:
    """Load use cases from yaml."""
    use_cases_path = Path(__file__).parent.parent / "docs" / "use_cases.yaml"
    if not use_cases_path.exists():
        return []
    with open(use_cases_path) as f:
        data = yaml.safe_load(f)
    return data.get("use_cases", [])


def load_decisions() -> dict[str, Any]:
    """Load generated decisions."""
    decisions_path = Path(__file__).parent.parent / "docs" / "decisions.json"
    if not decisions_path.exists():
        return {}
    with open(decisions_path) as f:
        return json.load(f)


def get_model_coverage(card: ModelCard, use_cases: list[dict], decisions: dict) -> ModelCoverage:
    """Build coverage report for a single model."""

    # Basic info
    coverage = ModelCoverage(
        model_id=card.model_id,
        version=card.version or "unknown",
        model_type=card.model_type or "unknown",
        status=card.status or "unknown",
        devices_supported=card.devices_supported or [],
        declared_capabilities=[cap.task for cap in card.declared_capabilities],
    )

    # Track all known tasks
    all_tasks = {"asr", "vad", "diarization", "v2v", "tts", "alignment", "chat"}

    # Initialize task evidence
    for task in all_tasks:
        te = TaskEvidence(task=task)
        te.declared = task in coverage.declared_capabilities
        coverage.tasks[task] = te

    # Analyze evidence
    for evidence in card.evidence:
        task = evidence.task
        if task not in coverage.tasks:
            coverage.tasks[task] = TaskEvidence(task=task)

        te = coverage.tasks[task]
        te.has_evidence = True
        te.run_count += 1

        # Track best grade
        grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
        current_rank = grade_rank.get(te.best_grade, 0)
        new_rank = grade_rank.get(
            evidence.evidence_grade.value
            if hasattr(evidence.evidence_grade, "value")
            else str(evidence.evidence_grade),
            0,
        )
        if new_rank > current_rank:
            te.best_grade = (
                evidence.evidence_grade.value
                if hasattr(evidence.evidence_grade, "value")
                else str(evidence.evidence_grade)
            )

        # Track last run
        if evidence.run_date:
            if not te.last_run or evidence.run_date > te.last_run:
                te.last_run = evidence.run_date

        # Metrics summary (aggregate)
        for k, v in evidence.metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                if k not in te.metrics_summary:
                    te.metrics_summary[k] = v
                else:
                    # Keep best (lower for error metrics, higher otherwise)
                    if "error" in k.lower() or "wer" in k.lower() or "latency" in k.lower():
                        te.metrics_summary[k] = min(te.metrics_summary[k], v)
                    else:
                        te.metrics_summary[k] = max(te.metrics_summary[k], v)

    # Check use case eligibility
    decisions_data = decisions.get("decisions", {})
    for uc in use_cases:
        uc_id = uc["id"]
        uc_decision = decisions_data.get(uc_id, {})

        pipeline = uc_decision.get("pipeline", {})
        outcome = uc_decision.get("outcome", "unknown")

        # Check if this model is in the pipeline
        if card.model_id in pipeline.values():
            coverage.use_case_eligibility[uc_id] = f"✅ selected ({outcome})"
        else:
            # Check why not selected
            required_tasks = [r["task"] for r in uc.get("requirements", {}).get("primary", [])]
            model_tasks = coverage.declared_capabilities

            missing = [t for t in required_tasks if t not in model_tasks]
            if missing:
                coverage.use_case_eligibility[uc_id] = f"❌ missing capabilities: {missing}"
            else:
                # Has capability but not selected - probably evidence issue
                coverage.use_case_eligibility[uc_id] = (
                    "⚠️ not selected (evidence insufficient or competitor better)"
                )

    return coverage


def generate_coverage_report() -> list[ModelCoverage]:
    """Generate coverage report for all models."""
    registry = ModelRegistry()
    use_cases = load_use_cases()
    decisions = load_decisions()

    # Get registry metadata
    registry_meta = {}
    for model_id in registry.list_models():
        registry_meta[model_id] = registry.get_model_metadata(model_id)

    # Load model cards same way as generate_decisions.py
    models_dir = Path(__file__).parent.parent / "models"
    coverages = []

    for config_path in models_dir.glob("*/config.yaml"):
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            model_id = config.get("model_type")

            if not model_id or model_id not in registry_meta:
                continue

            # Load evidence from runs
            evidence_data = []
            runs_dir = Path(__file__).parent.parent / f"runs/{model_id}"
            if runs_dir.exists():
                for run_file in runs_dir.glob("**/*.json"):
                    if run_file.name == "summary.json":
                        continue
                    try:
                        with open(run_file) as rf:
                            run = json.load(rf)
                            if "evidence" not in run:
                                continue
                            evidence_data.append(run)
                    except Exception:
                        continue

            # Build coverage entry
            meta = registry_meta.get(model_id, {})
            coverage = ModelCoverage(
                model_id=model_id,
                version=meta.get("version", "unknown"),
                model_type=model_id,
                status=meta.get("status", "unknown"),
                devices_supported=meta.get("hardware", []),
                declared_capabilities=meta.get("capabilities", []),
            )

            # Analyze evidence
            for task in ["asr", "vad", "diarization", "v2v", "tts", "alignment", "chat"]:
                coverage.tasks[task] = TaskEvidence(task=task)
                coverage.tasks[task].declared = task in coverage.declared_capabilities

            for run in evidence_data:
                ev = run.get("evidence", {})
                meta_section = run.get("meta", {})

                # Get task from capability or meta.task
                if "capability" in run:
                    task = run["capability"]
                elif "task" in meta_section:
                    task = meta_section["task"]
                else:
                    continue

                if task not in coverage.tasks:
                    coverage.tasks[task] = TaskEvidence(task=task)

                te = coverage.tasks[task]
                te.has_evidence = True
                te.run_count += 1

                # Track grade
                grade = ev.get("grade", "")
                grade_rank = {"golden_batch": 3, "smoke": 2, "adhoc": 1}
                current_rank = grade_rank.get(te.best_grade, 0)
                if grade_rank.get(grade, 0) > current_rank:
                    te.best_grade = grade

                # Track metrics
                metrics = run.get("metrics", {})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and v is not None:
                        te.metrics_summary[k] = v

                # Check provenance/run_context
                te.has_provenance = "provenance" in run
                te.has_run_context = "run_context" in run

            # Check use case eligibility
            for uc in use_cases:
                uc_id = uc["id"]
                uc_decision = decisions.get("decisions", {}).get(uc_id, {})
                pipeline = uc_decision.get("pipeline", {})

                if model_id in pipeline.values():
                    coverage.use_case_eligibility[uc_id] = "✅ selected"
                else:
                    required_tasks = [
                        r["task"] for r in uc.get("requirements", {}).get("primary", [])
                    ]
                    missing = [t for t in required_tasks if t not in coverage.declared_capabilities]
                    if missing:
                        coverage.use_case_eligibility[uc_id] = f"❌ missing: {missing}"
                    else:
                        coverage.use_case_eligibility[uc_id] = "⚠️ not selected"

            coverages.append(coverage)

        except Exception as e:
            print(f"Warning: Failed to process {config_path}: {e}")
            continue

    return coverages


def write_markdown_report(coverages: list[ModelCoverage], output_path: Path):
    """Write human-readable coverage report."""
    lines = [
        "# Model Coverage Report",
        "",
        f"*Generated: {datetime.now().isoformat()}*",
        "",
        "This report shows what's actually tested vs what's claimed.",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.extend(
        [
            "## Summary",
            "",
            "| Model | Type | Capabilities | Evidence Tasks | Best Grade |",
            "|-------|------|--------------|----------------|------------|",
        ]
    )

    for cov in coverages:
        declared = ", ".join(cov.declared_capabilities) or "none"
        evidence_tasks = [t for t, te in cov.tasks.items() if te.has_evidence]
        evidence_str = ", ".join(evidence_tasks) or "none"

        grades = [te.best_grade for te in cov.tasks.values() if te.best_grade]
        best = (
            "golden_batch" if "golden_batch" in grades else "smoke" if "smoke" in grades else "none"
        )

        lines.append(
            f"| {cov.model_id} | {cov.model_type} | {declared} | {evidence_str} | {best} |"
        )

    lines.append("")

    # Detailed per-model sections
    lines.append("## Details")
    lines.append("")

    for cov in coverages:
        lines.append(f"### {cov.model_id}")
        lines.append("")
        lines.append(f"- **Version:** {cov.version}")
        lines.append(f"- **Type:** {cov.model_type}")
        lines.append(f"- **Status:** {cov.status}")
        lines.append(f"- **Devices:** {', '.join(cov.devices_supported) or 'unspecified'}")
        lines.append("")

        # Task evidence table
        lines.append("#### Task Evidence")
        lines.append("")
        lines.append("| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |")
        lines.append("|------|----------|--------------|-------|------|-------------|")

        for task in ["asr", "vad", "diarization", "v2v", "tts"]:
            te = cov.tasks.get(task, TaskEvidence(task=task))
            declared = "✅" if te.declared else "—"
            has_ev = "✅" if te.has_evidence else "❌" if te.declared else "—"
            grade = te.best_grade or "—"
            runs = str(te.run_count) if te.run_count else "—"

            # Format key metrics
            metrics_parts = []
            for k, v in list(te.metrics_summary.items())[:3]:
                if isinstance(v, float):
                    metrics_parts.append(f"{k}={v:.2f}")
                else:
                    metrics_parts.append(f"{k}={v}")
            metrics = ", ".join(metrics_parts) or "—"

            lines.append(f"| {task} | {declared} | {has_ev} | {grade} | {runs} | {metrics} |")

        lines.append("")

        # Use case eligibility
        if cov.use_case_eligibility:
            lines.append("#### Use Case Eligibility")
            lines.append("")
            for uc_id, status in cov.use_case_eligibility.items():
                lines.append(f"- **{uc_id}:** {status}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Coverage gaps
    lines.append("## Coverage Gaps")
    lines.append("")
    lines.append("Tasks declared but not tested:")
    lines.append("")

    gaps = []
    for cov in coverages:
        for task, te in cov.tasks.items():
            if te.declared and not te.has_evidence:
                gaps.append(f"- `{cov.model_id}` declares `{task}` but has no evidence")

    if gaps:
        lines.extend(gaps)
    else:
        lines.append("*None - all declared capabilities have evidence.*")

    lines.append("")

    output_path.write_text("\n".join(lines))


def write_json_report(coverages: list[ModelCoverage], output_path: Path):
    """Write machine-readable coverage report."""
    data = {"generated_at": datetime.now().isoformat(), "models": {}}

    for cov in coverages:
        model_data = {
            "model_id": cov.model_id,
            "version": cov.version,
            "model_type": cov.model_type,
            "status": cov.status,
            "devices_supported": cov.devices_supported,
            "declared_capabilities": cov.declared_capabilities,
            "tasks": {},
            "use_case_eligibility": cov.use_case_eligibility,
        }

        for task, te in cov.tasks.items():
            if te.declared or te.has_evidence:
                model_data["tasks"][task] = {
                    "declared": te.declared,
                    "has_evidence": te.has_evidence,
                    "best_grade": te.best_grade,
                    "run_count": te.run_count,
                    "last_run": te.last_run,
                    "metrics_summary": te.metrics_summary,
                }

        data["models"][cov.model_id] = model_data

    output_path.write_text(json.dumps(data, indent=2))


def main():
    print("Generating coverage report...")

    coverages = generate_coverage_report()

    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Write outputs
    write_markdown_report(coverages, docs_dir / "COVERAGE.md")
    print("✅ Generated docs/COVERAGE.md")

    write_json_report(coverages, docs_dir / "coverage.json")
    print("✅ Generated docs/coverage.json")

    # Print summary
    print("\nCoverage Summary:")
    print("-" * 50)

    for cov in coverages:
        declared = len(cov.declared_capabilities)
        tested = sum(1 for te in cov.tasks.values() if te.has_evidence)
        grades = [te.best_grade for te in cov.tasks.values() if te.best_grade]
        has_golden = "golden_batch" in grades

        status = "🟢" if tested >= declared and has_golden else "🟡" if tested > 0 else "🔴"
        print(f"{status} {cov.model_id}: {tested}/{declared} tasks tested, golden={has_golden}")

    print()


if __name__ == "__main__":
    main()
