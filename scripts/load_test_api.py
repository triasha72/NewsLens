#!/usr/bin/env python3
"""Run a bounded concurrent load test against an artifact-backed NewsLens API."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestResult:
    latency_ms: float
    status_code: int
    error: str | None = None


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sample.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _request(url: str, payload: bytes, timeout: float) -> RequestResult:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/recommend",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status_code = response.status
        error = None
    except urllib.error.HTTPError as exc:
        exc.read()
        status_code = exc.code
        error = f"HTTP {exc.code}"
    except (OSError, TimeoutError) as exc:
        status_code = 0
        error = type(exc).__name__
    return RequestResult((time.perf_counter() - started) * 1_000, status_code, error)


def summarize(results: list[RequestResult], elapsed_seconds: float) -> dict[str, Any]:
    """Summarize observed client-side load-test results without claiming an SLO."""
    if not results or elapsed_seconds <= 0:
        raise ValueError("Results must be non-empty and elapsed_seconds must be positive.")
    successful = [result for result in results if 200 <= result.status_code < 300]
    latencies = [result.latency_ms for result in successful]
    errors: dict[str, int] = {}
    for result in results:
        if result.error is not None:
            errors[result.error] = errors.get(result.error, 0) + 1
    summary: dict[str, Any] = {
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "success_rate": len(successful) / len(results),
        "elapsed_seconds": elapsed_seconds,
        "throughput_requests_per_second": len(results) / elapsed_seconds,
        "errors": errors,
    }
    if latencies:
        summary["successful_latency_ms"] = {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("reports/load-test.json"))
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.timeout <= 0:
        parser.error("requests, concurrency, and timeout must be positive")

    payload_object = json.loads(args.payload.read_text(encoding="utf-8"))
    payload = json.dumps(payload_object, separators=(",", ":")).encode("utf-8")
    readiness_url = f"{args.url.rstrip('/')}/ready"
    with urllib.request.urlopen(readiness_url, timeout=args.timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Readiness check returned HTTP {response.status}.")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_request, args.url, payload, args.timeout) for _ in range(args.requests)]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "1.0",
        "target_url": args.url,
        "concurrency": args.concurrency,
        "request_payload": payload_object,
        "summary": summarize(results, elapsed),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

