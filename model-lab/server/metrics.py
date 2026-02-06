"""
Lightweight Prometheus-style metrics for production API.

Provides:
- Counter: Request totals by type
- Histogram: Request duration
- Gauge: Cache and memory stats
"""

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Counter:
    """Thread-safe counter metric."""

    name: str
    help_text: str
    labels: list[str] = field(default_factory=list)
    _values: dict = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, value: float = 1.0, **label_values) -> None:
        """Increment counter."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            self._values[key] += value

    def get(self, **label_values) -> float:
        """Get counter value."""
        key = tuple(sorted(label_values.items()))
        return self._values.get(key, 0.0)

    def collect(self) -> list[tuple[dict, float]]:
        """Collect all values for export."""
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


@dataclass
class Gauge:
    """Thread-safe gauge metric."""

    name: str
    help_text: str
    labels: list[str] = field(default_factory=list)
    _values: dict = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, **label_values) -> None:
        """Set gauge value."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **label_values) -> None:
        """Increment gauge."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, **label_values) -> None:
        """Decrement gauge."""
        self.inc(-value, **label_values)

    def get(self, **label_values) -> float:
        """Get gauge value."""
        key = tuple(sorted(label_values.items()))
        return self._values.get(key, 0.0)

    def collect(self) -> list[tuple[dict, float]]:
        """Collect all values for export."""
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


@dataclass
class Histogram:
    """Thread-safe histogram metric with predefined buckets."""

    name: str
    help_text: str
    labels: list[str] = field(default_factory=list)
    buckets: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    _bucket_counts: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _sums: dict = field(default_factory=lambda: defaultdict(float))
    _counts: dict = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, **label_values) -> None:
        """Record an observation."""
        key = tuple(sorted(label_values.items()))
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1

    def time(self, **label_values) -> Callable:
        """Context manager to time operations."""
        start = time.perf_counter()

        class Timer:
            def __init__(self, histogram, lv):
                self.histogram = histogram
                self.label_values = lv

            def __enter__(self):
                return self

            def __exit__(self, *args):
                duration = time.perf_counter() - start
                self.histogram.observe(duration, **self.label_values)

        return Timer(self, label_values)

    def collect(self) -> list[tuple[dict, dict]]:
        """Collect all histogram data for export."""
        with self._lock:
            result = []
            for key in self._counts:
                labels = dict(key)
                data = {
                    "count": self._counts[key],
                    "sum": self._sums[key],
                    "buckets": dict(self._bucket_counts[key]),
                }
                result.append((labels, data))
            return result


class MetricsRegistry:
    """Registry for all metrics."""

    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help_text: str, labels: list[str] = None) -> Counter:
        """Create or get a counter."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, help_text, labels or [])
            return self._metrics[name]

    def gauge(self, name: str, help_text: str, labels: list[str] = None) -> Gauge:
        """Create or get a gauge."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, help_text, labels or [])
            return self._metrics[name]

    def histogram(
        self, name: str, help_text: str, labels: list[str] = None, buckets: tuple = None
    ) -> Histogram:
        """Create or get a histogram."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(
                    name, help_text, labels or [], buckets or Histogram.buckets
                )
            return self._metrics[name]

    def export_text(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        for name, metric in self._metrics.items():
            lines.append(f"# HELP {name} {metric.help_text}")

            if isinstance(metric, Counter):
                lines.append(f"# TYPE {name} counter")
                for labels, value in metric.collect():
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    if label_str:
                        lines.append(f"{name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {name} gauge")
                for labels, value in metric.collect():
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    if label_str:
                        lines.append(f"{name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {name} histogram")
                for labels, data in metric.collect():
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    base_labels = f"{{{label_str}}}" if label_str else ""

                    # Bucket values
                    for bucket, count in sorted(data["buckets"].items()):
                        bucket_labels = (
                            f'{{{label_str},le="{bucket}"}}' if label_str else f'{{le="{bucket}"}}'
                        )
                        lines.append(f"{name}_bucket{bucket_labels} {count}")

                    # +Inf bucket (total count)
                    inf_labels = f'{{{label_str},le="+Inf"}}' if label_str else '{le="+Inf"}'
                    lines.append(f"{name}_bucket{inf_labels} {data['count']}")

                    # Sum and count
                    lines.append(f"{name}_sum{base_labels} {data['sum']}")
                    lines.append(f"{name}_count{base_labels} {data['count']}")

        return "\n".join(lines)


# Global registry instance
REGISTRY = MetricsRegistry()

# Pre-defined metrics for production API
REQUEST_TOTAL = REGISTRY.counter(
    "http_requests_total", "Total HTTP requests", labels=["method", "endpoint", "status"]
)

REQUEST_DURATION = REGISTRY.histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labels=["method", "endpoint"],
)

ASR_REQUESTS = REGISTRY.counter("asr_requests_total", "Total ASR transcription requests")

TTS_REQUESTS = REGISTRY.counter("tts_requests_total", "Total TTS synthesis requests")

ERRORS_TOTAL = REGISTRY.counter("errors_total", "Total errors", labels=["type", "endpoint"])

MODELS_LOADED = REGISTRY.gauge("models_loaded", "Number of models currently in cache")

CACHE_MEMORY_BYTES = REGISTRY.gauge("cache_memory_bytes", "Memory used by model cache in bytes")
