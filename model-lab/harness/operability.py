"""
Operability Threshold Evaluation.

Operability thresholds apply to smoke evidence only.
A run that fails operability is not viable evidence for that use case.

Threshold operations:
- *_max: metric <= threshold
- *_min: metric >= threshold

If metric missing or None: treat as fail.
"""

from typing import Any


def evaluate_operability(
    task: str,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Evaluate operability thresholds for a task's metrics.

    Args:
        task: Task name (e.g., "vad", "v2v")
        metrics: Run metrics dict (e.g., {"rtf_like": 0.56, "speech_ratio": 0.44})
        thresholds: Threshold dict for this task (e.g., {"rtf_like_max": 1.2})

    Returns:
        (passed: bool, failures: list[str])

    Examples:
        >>> evaluate_operability("v2v", {"rtf_like": 0.56}, {"rtf_like_max": 1.2})
        (True, [])

        >>> evaluate_operability("v2v", {"rtf_like": 1.5}, {"rtf_like_max": 1.2})
        (False, ["rtf_like 1.50 > 1.2 (max)"])

        >>> evaluate_operability("v2v", {}, {"rtf_like_max": 1.2})
        (False, ["missing metric: rtf_like"])
    """
    if not thresholds:
        return True, []

    failures = []

    for threshold_key, threshold_value in thresholds.items():
        # Parse threshold type: metric_name + "_max" or "_min"
        if threshold_key.endswith("_max"):
            metric_name = threshold_key[:-4]  # Remove "_max"
            op = "max"
        elif threshold_key.endswith("_min"):
            metric_name = threshold_key[:-4]  # Remove "_min"
            op = "min"
        else:
            # Unknown threshold format, skip
            continue

        metric_value = metrics.get(metric_name)

        # Missing metric is a failure
        if metric_value is None:
            failures.append(f"missing metric: {metric_name}")
            continue

        # Evaluate threshold
        try:
            if op == "max" and metric_value > threshold_value:
                failures.append(f"{metric_name} {metric_value:.2f} > {threshold_value} (max)")
            elif op == "min" and metric_value < threshold_value:
                failures.append(f"{metric_name} {metric_value:.2f} < {threshold_value} (min)")
        except (TypeError, ValueError):
            failures.append(f"invalid metric value: {metric_name}={metric_value}")

    return len(failures) == 0, failures


def get_task_thresholds(
    operability_config: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """
    Get thresholds for a specific task from operability config.

    Args:
        operability_config: The 'operability' section from a use case
        task: Task name (e.g., "v2v", "vad")

    Returns:
        Threshold dict for the task, or empty dict if none defined
    """
    if not operability_config:
        return {}
    return operability_config.get(task, {})


def filter_viable_evidence(
    evidence_list: list[Any],
    task: str,
    thresholds: dict[str, Any],
    grade_filter: str | None = "smoke",
) -> tuple[list[Any], list[str]]:
    """
    Filter evidence list to only viable runs (pass operability for smoke grade).

    Args:
        evidence_list: List of evidence objects with .metrics and .grade
        task: Task name
        thresholds: Threshold dict for this task
        grade_filter: Only apply thresholds to this grade (default: "smoke")

    Returns:
        (viable_evidence, rejection_reasons)
    """
    viable = []
    rejections = []

    for evidence in evidence_list:
        # Only apply operability to specified grade
        if grade_filter and str(evidence.grade).lower() != grade_filter.lower():
            # Golden batch and other grades bypass operability
            viable.append(evidence)
            continue

        passed, failures = evaluate_operability(task, evidence.metrics, thresholds)

        if passed:
            viable.append(evidence)
        else:
            rejections.append(f"{evidence.model_id}: {'; '.join(failures)}")

    return viable, rejections
