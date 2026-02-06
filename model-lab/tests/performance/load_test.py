#!/usr/bin/env python3
"""
Load testing script for Model Lab production API.

Usage:
    # Basic health endpoint test
    uv run python tests/performance/load_test.py --target http://localhost:8000 --endpoint /health

    # Concurrent requests test
    uv run python tests/performance/load_test.py --target http://localhost:8000 --concurrent 10 --requests 100

    # Sustained load test
    uv run python tests/performance/load_test.py --target http://localhost:8000 --duration 60 --rate 10

Output:
    - Total requests, successes, failures
    - Latency percentiles (p50, p95, p99)
    - Requests per second
    - Error breakdown
"""

import argparse
import asyncio
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

try:
    import httpx
except ImportError:
    print("httpx required: uv pip install httpx")
    raise


@dataclass
class LoadTestResult:
    """Results from a load test run."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: Counter = field(default_factory=Counter)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def requests_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.total_requests / self.duration_seconds
        return 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return (self.successful / self.total_requests) * 100
        return 0.0

    def latency_percentile(self, p: int) -> float:
        """Get latency percentile in ms."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "LOAD TEST RESULTS",
            "=" * 60,
            f"Duration:          {self.duration_seconds:.2f}s",
            f"Total Requests:    {self.total_requests}",
            f"Successful:        {self.successful} ({self.success_rate:.1f}%)",
            f"Failed:            {self.failed}",
            f"Requests/sec:      {self.requests_per_second:.2f}",
            "",
            "LATENCY (ms)",
            f"  Min:    {min(self.latencies_ms) if self.latencies_ms else 0:.2f}",
            f"  Max:    {max(self.latencies_ms) if self.latencies_ms else 0:.2f}",
            f"  Mean:   {statistics.mean(self.latencies_ms) if self.latencies_ms else 0:.2f}",
            f"  p50:    {self.latency_percentile(50):.2f}",
            f"  p95:    {self.latency_percentile(95):.2f}",
            f"  p99:    {self.latency_percentile(99):.2f}",
        ]

        if self.errors:
            lines.append("")
            lines.append("ERRORS")
            for error, count in self.errors.most_common(10):
                lines.append(f"  {error}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    method: str = "GET",
    data: Optional[dict] = None,
) -> tuple[bool, float, Optional[str]]:
    """Make a single request and return (success, latency_ms, error)."""
    start = time.perf_counter()
    try:
        if method.upper() == "GET":
            response = await client.get(url, timeout=30.0)
        else:
            response = await client.post(url, json=data, timeout=30.0)

        latency = (time.perf_counter() - start) * 1000  # Convert to ms

        if response.status_code < 400:
            return True, latency, None
        else:
            return False, latency, f"HTTP {response.status_code}"

    except httpx.TimeoutException:
        latency = (time.perf_counter() - start) * 1000
        return False, latency, "Timeout"
    except httpx.ConnectError:
        latency = (time.perf_counter() - start) * 1000
        return False, latency, "ConnectionError"
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return False, latency, str(type(e).__name__)


async def run_concurrent_test(
    target: str,
    endpoint: str,
    concurrent: int,
    total_requests: int,
    method: str = "GET",
    data: Optional[dict] = None,
) -> LoadTestResult:
    """Run load test with N concurrent workers making M total requests."""
    url = f"{target.rstrip('/')}{endpoint}"
    result = LoadTestResult()
    result.start_time = time.time()

    semaphore = asyncio.Semaphore(concurrent)
    completed = 0

    async def worker():
        nonlocal completed
        async with httpx.AsyncClient() as client:
            while True:
                async with semaphore:
                    if completed >= total_requests:
                        return

                    completed += 1
                    success, latency, error = await make_request(client, url, method, data)

                    result.total_requests += 1
                    result.latencies_ms.append(latency)

                    if success:
                        result.successful += 1
                    else:
                        result.failed += 1
                        if error:
                            result.errors[error] += 1

    # Create worker tasks
    tasks = [asyncio.create_task(worker()) for _ in range(concurrent)]
    await asyncio.gather(*tasks)

    result.end_time = time.time()
    return result


async def run_sustained_test(
    target: str,
    endpoint: str,
    duration_seconds: int,
    rate_per_second: float,
    method: str = "GET",
    data: Optional[dict] = None,
) -> LoadTestResult:
    """Run sustained load test at a fixed rate for a duration."""
    url = f"{target.rstrip('/')}{endpoint}"
    result = LoadTestResult()
    result.start_time = time.time()

    interval = 1.0 / rate_per_second
    end_time = result.start_time + duration_seconds

    async with httpx.AsyncClient() as client:
        while time.time() < end_time:
            request_start = time.perf_counter()

            success, latency, error = await make_request(client, url, method, data)

            result.total_requests += 1
            result.latencies_ms.append(latency)

            if success:
                result.successful += 1
            else:
                result.failed += 1
                if error:
                    result.errors[error] += 1

            # Wait to maintain target rate
            elapsed = time.perf_counter() - request_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    result.end_time = time.time()
    return result


def main():
    parser = argparse.ArgumentParser(description="Load test Model Lab API")
    parser.add_argument("--target", required=True, help="Target URL (e.g., http://localhost:8000)")
    parser.add_argument("--endpoint", default="/health", help="Endpoint to test")
    parser.add_argument("--method", default="GET", choices=["GET", "POST"], help="HTTP method")

    # Concurrent test mode
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent workers")
    parser.add_argument("--requests", type=int, help="Total requests (concurrent mode)")

    # Sustained test mode
    parser.add_argument("--duration", type=int, help="Duration in seconds (sustained mode)")
    parser.add_argument("--rate", type=float, help="Requests per second (sustained mode)")

    args = parser.parse_args()

    # Validate arguments
    if args.duration and args.rate:
        # Sustained mode
        print(f"Running sustained load test:")
        print(f"  Target:    {args.target}")
        print(f"  Endpoint:  {args.endpoint}")
        print(f"  Duration:  {args.duration}s")
        print(f"  Rate:      {args.rate} req/s")
        print()

        result = asyncio.run(
            run_sustained_test(
                args.target,
                args.endpoint,
                args.duration,
                args.rate,
                args.method,
            )
        )
    else:
        # Concurrent mode
        total = args.requests or 100
        print(f"Running concurrent load test:")
        print(f"  Target:     {args.target}")
        print(f"  Endpoint:   {args.endpoint}")
        print(f"  Concurrent: {args.concurrent}")
        print(f"  Requests:   {total}")
        print()

        result = asyncio.run(
            run_concurrent_test(
                args.target,
                args.endpoint,
                args.concurrent,
                total,
                args.method,
            )
        )

    print(result.summary())

    # Exit with error if too many failures
    if result.success_rate < 99.0:
        print("\n⚠️  WARNING: Success rate below 99%")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
