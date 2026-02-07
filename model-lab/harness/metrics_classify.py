"""
Classification metrics for audio classification models.

LCS-02: CI-safe, dataset-free metrics for classify surface.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def accuracy_top1(y_true: list[Any], y_pred: list[Any]) -> float:
    """
    Compute top-1 accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy as float in [0, 1]
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision_recall_f1(
    y_true: list[Any],
    y_pred: list[Any],
    average: str = "macro",
    labels: list[Any] | None = None,
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: "macro", "micro", or "weighted"
        labels: Optional list of labels to include
    
    Returns:
        Dict with precision, recall, f1 keys
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    if average == "micro":
        # Micro: aggregate TP, FP, FN across all classes
        tp_total = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        fp_total = len(y_pred) - tp_total
        fn_total = len(y_true) - tp_total
        
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    # Macro or weighted: compute per-class then aggregate
    precisions = []
    recalls = []
    f1s = []
    weights = []
    
    label_counts = Counter(y_true)
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        weights.append(label_counts.get(label, 0))
    
    if average == "weighted":
        total_weight = sum(weights)
        if total_weight == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        precision = sum(p * w for p, w in zip(precisions, weights)) / total_weight
        recall = sum(r * w for r, w in zip(recalls, weights)) / total_weight
        f1 = sum(f * w for f, w in zip(f1s, weights)) / total_weight
    else:  # macro
        n = len(labels)
        precision = sum(precisions) / n if n > 0 else 0.0
        recall = sum(recalls) / n if n > 0 else 0.0
        f1 = sum(f1s) / n if n > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}


def confusion_matrix(
    y_true: list[Any],
    y_pred: list[Any],
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional ordered list of labels
    
    Returns:
        Dict with 'matrix' (2D list), 'labels' (label order)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1
    
    return {"matrix": matrix, "labels": labels}


def per_class_metrics(
    y_true: list[Any],
    y_pred: list[Any],
    labels: list[Any] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute per-class precision, recall, F1.
    
    Returns:
        Dict mapping label -> {precision, recall, f1, support}
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    label_counts = Counter(y_true)
    result = {}
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        result[label] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": label_counts.get(label, 0),
        }
    
    return result
