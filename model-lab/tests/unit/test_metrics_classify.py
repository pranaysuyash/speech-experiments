"""
Tests for classification metrics (LCS-02).

CI-safe, dataset-free unit tests with synthetic labels.
"""

import pytest

from harness.metrics_classify import (
    accuracy_top1,
    precision_recall_f1,
    confusion_matrix,
    per_class_metrics,
)


class TestAccuracyTop1:
    """Tests for accuracy_top1."""

    def test_perfect_accuracy(self):
        y_true = ["cat", "dog", "bird"]
        y_pred = ["cat", "dog", "bird"]
        assert accuracy_top1(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        y_true = ["cat", "dog", "bird"]
        y_pred = ["dog", "bird", "cat"]
        assert accuracy_top1(y_true, y_pred) == 0.0

    def test_partial_accuracy(self):
        y_true = ["cat", "dog", "bird", "cat"]
        y_pred = ["cat", "dog", "cat", "cat"]  # 3/4 correct
        assert accuracy_top1(y_true, y_pred) == 0.75

    def test_empty_raises_zero(self):
        assert accuracy_top1([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            accuracy_top1([1, 2, 3], [1, 2])


class TestPrecisionRecallF1:
    """Tests for precision_recall_f1."""

    def test_perfect_scores(self):
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "A", "B"]
        result = precision_recall_f1(y_true, y_pred)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_macro_vs_micro_imbalanced(self):
        # Imbalanced: 4 A's, 1 B
        # Predict all A -> precision for A=4/5, B=0. recall A=1, B=0
        y_true = ["A", "A", "A", "A", "B"]
        y_pred = ["A", "A", "A", "A", "A"]

        macro = precision_recall_f1(y_true, y_pred, average="macro")
        micro = precision_recall_f1(y_true, y_pred, average="micro")

        # Micro accuracy = 4/5 = 0.8
        assert micro["precision"] == 0.8
        assert micro["recall"] == 0.8

        # Macro: average of per-class
        # A: precision=4/5=0.8, recall=4/4=1.0, F1=0.888...
        # B: precision=0/0=0, recall=0/1=0, F1=0
        # Macro precision = (0.8 + 0) / 2 = 0.4
        assert macro["precision"] == 0.4
        assert macro["recall"] == 0.5  # (1.0 + 0) / 2

    def test_weighted_average(self):
        y_true = ["A", "A", "A", "A", "B"]
        y_pred = ["A", "A", "A", "A", "A"]
        
        weighted = precision_recall_f1(y_true, y_pred, average="weighted")
        # Weighted by support: A has 4, B has 1
        # Weighted precision = (0.8*4 + 0*1) / 5 = 0.64
        assert weighted["precision"] == pytest.approx(0.64)

    def test_empty_returns_zeros(self):
        result = precision_recall_f1([], [])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


class TestConfusionMatrix:
    """Tests for confusion_matrix."""

    def test_binary_confusion(self):
        y_true = ["pos", "pos", "neg", "neg", "neg"]
        y_pred = ["pos", "neg", "neg", "neg", "pos"]
        
        result = confusion_matrix(y_true, y_pred, labels=["neg", "pos"])
        matrix = result["matrix"]
        labels = result["labels"]
        
        assert labels == ["neg", "pos"]
        # Row = true, Col = pred
        # neg->neg: 2, neg->pos: 1
        # pos->neg: 1, pos->pos: 1
        assert matrix[0] == [2, 1]  # true neg
        assert matrix[1] == [1, 1]  # true pos

    def test_multiclass_shape(self):
        y_true = ["A", "B", "C", "A", "B", "C"]
        y_pred = ["A", "B", "C", "B", "C", "A"]
        
        result = confusion_matrix(y_true, y_pred)
        matrix = result["matrix"]
        
        # 3x3 matrix
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)

    def test_label_ordering_preserved(self):
        y_true = ["B", "A"]
        y_pred = ["B", "A"]
        
        # Explicitly order labels as ["A", "B"]
        result = confusion_matrix(y_true, y_pred, labels=["A", "B"])
        assert result["labels"] == ["A", "B"]


class TestPerClassMetrics:
    """Tests for per_class_metrics."""

    def test_per_class_structure(self):
        y_true = ["A", "A", "B", "B", "C"]
        y_pred = ["A", "A", "B", "C", "C"]
        
        result = per_class_metrics(y_true, y_pred)
        
        assert "A" in result
        assert "B" in result
        assert "C" in result
        
        # A: 2 correct out of 2
        assert result["A"]["precision"] == 1.0
        assert result["A"]["recall"] == 1.0
        assert result["A"]["support"] == 2
        
        # B: predicted 1, true 2, TP=1
        assert result["B"]["precision"] == 1.0  # 1/1
        assert result["B"]["recall"] == 0.5  # 1/2

    def test_handles_missing_predictions(self):
        y_true = ["A", "A", "B"]
        y_pred = ["A", "A", "A"]  # Never predict B
        
        result = per_class_metrics(y_true, y_pred)
        
        assert result["B"]["precision"] == 0.0
        assert result["B"]["recall"] == 0.0
        assert result["B"]["support"] == 1
