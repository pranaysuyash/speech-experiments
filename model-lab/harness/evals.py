"""
Clean evaluation utilities for model testing.
Handles metrics, comparisons, and result validation.
No model logic, just measurement and analysis.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class EvaluationResult:
    """Clean evaluation result container."""

    metric_name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence_interval: tuple | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata,
            "confidence_interval": self.confidence_interval,
        }


class AudioMetrics:
    """Audio-specific evaluation metrics."""

    @staticmethod
    def signal_to_noise_ratio(clean_signal: np.ndarray, noisy_signal: np.ndarray) -> float:
        """Calculate SNR between clean and noisy signals."""
        if clean_signal.shape != noisy_signal.shape:
            raise ValueError("Signals must have same shape")

        noise = noisy_signal - clean_signal
        signal_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise**2)

        if noise_power == 0:
            return float("inf")

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    @staticmethod
    def mean_squared_error(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate MSE between two signals."""
        if signal1.shape != signal2.shape:
            raise ValueError("Signals must have same shape")

        mse = np.mean((signal1 - signal2) ** 2)
        return float(mse)

    @staticmethod
    def cross_correlation(signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate normalized cross-correlation between signals."""
        if signal1.shape != signal2.shape:
            raise ValueError("Signals must have same shape")

        # Normalize signals
        sig1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
        sig2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)

        correlation = np.corrcoef(sig1_norm.flatten(), sig2_norm.flatten())[0, 1]
        return float(correlation)

    @staticmethod
    def spectral_distance(signal1: np.ndarray, signal2: np.ndarray, sample_rate: int) -> float:
        """Calculate spectral distance between signals."""
        if signal1.shape != signal2.shape:
            raise ValueError("Signals must have same shape")

        # Compute spectrograms
        spec1 = np.abs(np.fft.fft(signal1))
        spec2 = np.abs(np.fft.fft(signal2))

        # Calculate log-spectral distance
        log_spec1 = np.log(spec1 + 1e-10)
        log_spec2 = np.log(spec2 + 1e-10)

        distance = np.mean((log_spec1 - log_spec2) ** 2)
        return float(distance)


class TextMetrics:
    """Text-specific evaluation metrics."""

    @staticmethod
    def exact_match(predicted: str, reference: str) -> float:
        """Exact string match (0 or 1)."""
        return float(predicted.strip() == reference.strip())

    @staticmethod
    def character_error_rate(predicted: str, reference: str) -> float:
        """Character Error Rate (CER) for transcription."""
        pred = predicted.lower().strip()
        ref = reference.lower().strip()

        # Simple Levenshtein distance implementation
        m, n = len(pred), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        cer = dp[m][n] / max(len(ref), 1)
        return float(cer)

    @staticmethod
    def word_error_rate(predicted: str, reference: str) -> float:
        """Word Error Rate (WER) for transcription."""
        pred_words = predicted.lower().strip().split()
        ref_words = reference.lower().strip().split()

        # Simple WER calculation
        m, n = len(pred_words), len(ref_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i - 1] == ref_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        wer = dp[m][n] / max(len(ref_words), 1)
        return float(wer)

    @staticmethod
    def rouge_l(predicted: str, reference: str) -> float:
        """ROUGE-L score for summarization."""
        pred_tokens = predicted.lower().strip().split()
        ref_tokens = reference.lower().strip().split()

        # Simple LCS calculation
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]

        if len(ref_tokens) == 0:
            return 0.0

        rouge_l = lcs_length / len(ref_tokens)
        return float(rouge_l)


class EvaluationSuite:
    """Suite of evaluation metrics for systematic testing."""

    def __init__(self, name: str):
        self.name = name
        self.metrics: dict[str, Callable] = {}
        self.results: list[EvaluationResult] = []

    def add_metric(self, name: str, metric_func: Callable):
        """Add a metric function."""
        self.metrics[name] = metric_func

    def evaluate(
        self, prediction: Any, reference: Any, metadata: dict[str, Any] | None = None
    ) -> dict[str, EvaluationResult]:
        """Evaluate prediction against reference."""
        metadata = metadata or {}
        results = {}

        for metric_name, metric_func in self.metrics.items():
            try:
                value = metric_func(prediction, reference)
                result = EvaluationResult(metric_name=metric_name, value=value, metadata=metadata)
                results[metric_name] = result
                self.results.append(result)
            except Exception as e:
                # Store error as metadata
                result = EvaluationResult(
                    metric_name=metric_name,
                    value=float("nan"),
                    metadata={**metadata, "error": str(e)},
                )
                results[metric_name] = result
                self.results.append(result)

        return results

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all evaluations."""
        if not self.results:
            return {"error": "No evaluations performed"}

        summary = {"suite_name": self.name, "total_evaluations": len(self.results), "metrics": {}}

        # Group by metric name
        metric_groups = {}
        for result in self.results:
            if result.metric_name not in metric_groups:
                metric_groups[result.metric_name] = []
            metric_groups[result.metric_name].append(result.value)

        # Calculate statistics for each metric
        for metric_name, values in metric_groups.items():
            valid_values = [v for v in values if not np.isnan(v) and np.isfinite(v)]

            if valid_values:
                summary["metrics"][metric_name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "count": len(valid_values),
                    "total_count": len(values),
                }

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()

        data = []
        for result in self.results:
            row = {"metric_name": result.metric_name, "value": result.value, **result.metadata}
            data.append(row)

        return pd.DataFrame(data)

    def plot_distribution(self, metric_name: str, save_path: str | None = None):
        """Plot distribution of metric values."""
        values = [r.value for r in self.results if r.metric_name == metric_name]
        valid_values = [v for v in values if not np.isnan(v) and np.isfinite(v)]

        if not valid_values:
            print(f"No valid values for metric: {metric_name}")
            return

        plt.figure(figsize=(10, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(valid_values, bins=30, alpha=0.7, edgecolor="black")
        plt.title(f"Distribution of {metric_name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(valid_values)
        plt.title(f"Box Plot of {metric_name}")
        plt.ylabel("Value")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


# Pre-built evaluation suites
AUDIO_EVALUATION_SUITE = EvaluationSuite("audio_quality")
AUDIO_EVALUATION_SUITE.add_metric("mse", AudioMetrics.mean_squared_error)
AUDIO_EVALUATION_SUITE.add_metric("snr", AudioMetrics.signal_to_noise_ratio)
AUDIO_EVALUATION_SUITE.add_metric("correlation", AudioMetrics.cross_correlation)

TEXT_EVALUATION_SUITE = EvaluationSuite("text_quality")
TEXT_EVALUATION_SUITE.add_metric("exact_match", TextMetrics.exact_match)
TEXT_EVALUATION_SUITE.add_metric("cer", TextMetrics.character_error_rate)
TEXT_EVALUATION_SUITE.add_metric("wer", TextMetrics.word_error_rate)
TEXT_EVALUATION_SUITE.add_metric("rouge_l", TextMetrics.rouge_l)


class ModelComparator:
    """Compare multiple models using statistical tests."""

    def __init__(self):
        self.comparisons = []

    def compare_models(
        self,
        model_results: dict[str, list[float]],
        metric_name: str,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """
        Compare multiple models statistically.

        Args:
            model_results: Dict of model_name -> list of metric values
            metric_name: Name of the metric being compared
            confidence_level: Confidence level for intervals

        Returns:
            Comparison results with statistical tests
        """
        if len(model_results) < 2:
            raise ValueError("Need at least 2 models for comparison")

        results = {
            "metric_name": metric_name,
            "models": {},
            "statistical_tests": {},
            "rankings": {},
        }

        # Calculate statistics for each model
        for model_name, values in model_results.items():
            valid_values = [v for v in values if not np.isnan(v) and np.isfinite(v)]

            if valid_values:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                n = len(valid_values)

                # Confidence interval
                alpha = 1 - confidence_level
                t_val = stats.t.ppf(1 - alpha / 2, n - 1)
                margin = t_val * (std_val / np.sqrt(n))
                ci = (mean_val - margin, mean_val + margin)

                results["models"][model_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n": n,
                    "confidence_interval": ci,
                }

        # Perform pairwise t-tests
        model_names = list(results["models"].keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                values1 = model_results[model1]
                values2 = model_results[model2]

                valid1 = [v for v in values1 if not np.isnan(v) and np.isfinite(v)]
                valid2 = [v for v in values2 if not np.isnan(v) and np.isfinite(v)]

                if valid1 and valid2:
                    t_stat, p_value = stats.ttest_ind(valid1, valid2)
                    test_key = f"{model1}_vs_{model2}"
                    results["statistical_tests"][test_key] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }

        # Rank models by mean performance
        model_means = [(name, data["mean"]) for name, data in results["models"].items()]
        model_means.sort(key=lambda x: x[1], reverse=True)  # Higher is better
        results["rankings"]["by_mean"] = model_means

        return results

    def plot_comparison(
        self,
        model_results: dict[str, list[float]],
        metric_name: str,
        save_path: str | None = None,
    ):
        """Plot comparison of model results."""
        plt.figure(figsize=(12, 8))

        # Box plot
        plt.subplot(1, 2, 1)
        data = [values for values in model_results.values()]
        labels = list(model_results.keys())
        plt.boxplot(data, labels=labels)
        plt.title(f"{metric_name} Distribution by Model")
        plt.xticks(rotation=45)

        # Mean comparison
        plt.subplot(1, 2, 2)
        means = [np.mean(values) for values in model_results.values()]
        stds = [np.std(values) for values in model_results.values()]
        x_pos = range(len(labels))

        plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xticks(x_pos, labels, rotation=45)
        plt.title(f"Mean {metric_name} by Model")
        plt.ylabel(metric_name)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


def create_audio_suite() -> EvaluationSuite:
    """Create standard audio evaluation suite."""
    return AUDIO_EVALUATION_SUITE


def create_text_suite() -> EvaluationSuite:
    """Create standard text evaluation suite."""
    return TEXT_EVALUATION_SUITE
