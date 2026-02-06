"""
Performance timing and monitoring utilities.
Ensures consistent performance measurement across models.
"""

import gc
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Container for timing results."""

    elapsed_time_s: float
    elapsed_time_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cpu_percent_before: float
    cpu_percent_after: float
    metadata: dict[str, Any]


class PerformanceTimer:
    """High-resolution performance timer with resource monitoring."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1e6

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    @contextmanager
    def time_operation(
        self, operation_name: str = "operation", collect_gc: bool = True
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager for timing operations with resource monitoring.

        Args:
            operation_name: Name of operation being timed
            collect_gc: Whether to run garbage collection before timing

        Yields:
            TimingResult with performance metrics
        """
        if collect_gc:
            gc.collect()

        # Pre-operation metrics
        memory_before = self.get_memory_mb()
        cpu_before = self.get_cpu_percent()

        start_time = time.perf_counter()

        # Create a mutable container for the result
        result_container: dict[str, Any] = {"result": None}

        try:
            yield result_container
        finally:
            # Post-operation metrics
            end_time = time.perf_counter()
            memory_after = self.get_memory_mb()
            cpu_after = self.get_cpu_percent()

            elapsed_time = end_time - start_time

            result = TimingResult(
                elapsed_time_s=elapsed_time,
                elapsed_time_ms=elapsed_time * 1000,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_after - memory_before,
                cpu_percent_before=cpu_before,
                cpu_percent_after=cpu_after,
                metadata={"operation_name": operation_name},
            )

            result_container["result"] = result

            logger.debug(f"⏱️  {operation_name}: {elapsed_time * 1000:.1f}ms")
            logger.debug(
                f"  Memory: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{memory_after - memory_before:.1f}MB)"
            )

    def measure_inference(self, func, *args, **kwargs) -> tuple[Any, TimingResult]:
        """
        Measure inference function performance.

        Returns:
            Tuple of (function_result, timing_result)
        """
        with self.time_operation(func.__name__) as timer:
            result = func(*args, **kwargs)

        return result, timer["result"]


class LatencyProfiler:
    """Profile latency distribution across multiple runs."""

    def __init__(self, num_runs: int = 10):
        self.num_runs = num_runs
        self.timer = PerformanceTimer()
        self.latencies_ms: list[float] = []

    @contextmanager
    def profile_run(self, run_index: int):
        """Profile a single run."""
        with self.timer.time_operation(f"run_{run_index}") as timing:  # type: Dict[str, Any]
            yield timing
        self.latencies_ms.append(timing["result"].elapsed_time_ms)

    def get_statistics(self) -> dict[str, float]:
        """Calculate latency statistics."""
        if not self.latencies_ms:
            return {}

        latencies = np.array(self.latencies_ms)
        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "num_runs": len(latencies),
        }


class MemoryProfiler:
    """Profile memory usage patterns."""

    @staticmethod
    def get_system_memory_info() -> dict[str, float]:
        """Get system-wide memory information."""
        vm = psutil.virtual_memory()
        return {
            "total_gb": vm.total / 1e9,
            "available_gb": vm.available / 1e9,
            "used_gb": vm.used / 1e9,
            "free_gb": vm.free / 1e9,
            "percent_used": vm.percent,
        }

    @staticmethod
    def check_memory_constraints(required_mb: float, safety_margin_gb: float = 2.0) -> bool:
        """
        Check if enough memory is available for operation.

        Args:
            required_mb: Required memory in MB
            safety_margin_gb: Safety margin in GB

        Returns:
            True if enough memory available
        """
        mem_info = MemoryProfiler.get_system_memory_info()
        available_mb = mem_info["available_gb"] * 1024

        required_with_margin = required_mb + (safety_margin_gb * 1024)

        if available_mb < required_with_margin:
            logger.warning(
                f"⚠️  Low memory: {available_mb:.0f}MB available, "
                f"{required_with_margin:.0f}MB required"
            )
            return False

        return True


def format_timing_result(result: TimingResult) -> str:
    """Format timing result for logging."""
    return (
        f"{result.elapsed_time_ms:.1f}ms, "
        f"Memory: {result.memory_before_mb:.1f}MB → {result.memory_after_mb:.1f}MB "
        f"(Δ{result.memory_delta_mb:+.1f}MB)"
    )
