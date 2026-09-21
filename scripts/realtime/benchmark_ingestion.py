#!/usr/bin/env python3
"""Measure publish throughput and produced-to-searchable freshness."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def request_json(url: str, *, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        response = error.read().decode(errors="replace")
        try:
            return error.code, json.loads(response)
        except json.JSONDecodeError:
            return error.code, {"error": response}


def publish(url: str, payload: dict[str, Any]) -> tuple[int, float]:
    started = time.perf_counter()
    status, _ = request_json(f"{url}/events", payload=payload)
    return status, (time.perf_counter() - started) * 1_000


def wait_until_searchable(search_url: str, token: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    encoded = urllib.parse.urlencode({"q": token, "top_k": 1})
    while time.monotonic() < deadline:
        status, body = request_json(f"{search_url}/search?{encoded}")
        if status == 200 and body["returned_count"]:
            return body["results"][0]
        time.sleep(0.05)
    raise TimeoutError(f"article with token {token!r} did not become searchable")


def metric_value(url: str, name: str) -> int:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        for line in response.read().decode().splitlines():
            if line.startswith(f"{name} "):
                return int(float(line.split()[1]))
    raise ValueError(f"metric {name!r} was not found at {url}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--freshness-samples", type=int, default=25)
    parser.add_argument("--duplicate-probes", type=int, default=25)
    parser.add_argument("--ingestion-url", default="http://127.0.0.1:8080")
    parser.add_argument("--search-url", default="http://127.0.0.1:8000")
    parser.add_argument("--execution-host-role", choices=["application", "state"], default="application")
    parser.add_argument(
        "--consumer-url",
        action="append",
        default=["http://127.0.0.1:8081", "http://127.0.0.1:8082"],
    )
    parser.add_argument("--output", type=Path, default=Path("reports/realtime_load_v0_1.json"))
    args = parser.parse_args()
    if (
        args.events < 1
        or args.concurrency < 1
        or args.freshness_samples < 1
        or args.duplicate_probes < 0
    ):
        parser.error("event, concurrency, and sample counts must be positive")

    run_id = uuid.uuid4().hex[:10]
    published_at = datetime.now(UTC).isoformat()
    payloads = []
    for index in range(args.events):
        token = f"Loadtest{run_id}{index}"
        payloads.append(
            {
                "event_id": f"load-{run_id}-{index}",
                "article_id": f"load-{run_id}-{index}",
                "title": f"{token} technology update",
                "category": "technology",
                "published_at": published_at,
                "body": "A deterministic article used to measure ingestion behavior.",
                "produced_at": datetime.now(UTC).isoformat(),
            }
        )

    started = time.perf_counter()
    latencies: list[float] = []
    statuses: list[int] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(publish, args.ingestion_url, payload) for payload in payloads]
        for future in as_completed(futures):
            status, latency = future.result()
            statuses.append(status)
            latencies.append(latency)
    duration = time.perf_counter() - started

    sample_count = min(args.freshness_samples, len(payloads))
    freshness: list[float] = []
    search_failures = 0
    for payload in payloads[:sample_count]:
        token = payload["title"].split()[0]
        try:
            result = wait_until_searchable(args.search_url, token, timeout=30.0)
            freshness.append(float(result["index_freshness_ms"]))
        except TimeoutError:
            search_failures += 1

    duplicate_count = min(args.duplicate_probes, len(payloads))
    duplicate_metric = "newslens_ingestion_duplicates_total"
    duplicates_before = sum(
        metric_value(url, duplicate_metric) for url in args.consumer_url
    )
    duplicate_statuses = [
        publish(args.ingestion_url, payload)[0] for payload in payloads[:duplicate_count]
    ]
    duplicate_deadline = time.monotonic() + 30
    duplicates_observed = 0
    while time.monotonic() < duplicate_deadline:
        duplicates_after = sum(
            metric_value(url, duplicate_metric) for url in args.consumer_url
        )
        duplicates_observed = duplicates_after - duplicates_before
        if duplicates_observed >= duplicate_count:
            break
        time.sleep(0.1)

    accepted = sum(status == 202 for status in statuses)
    report = {
        "schema_version": "newslens.realtime-load.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": "local Docker Compose",
        "execution_host_role": args.execution_host_role,
        "endpoints": {"ingestion_url": args.ingestion_url, "search_url": args.search_url, "consumer_urls": args.consumer_url},
        "configuration": {
            "events": args.events,
            "concurrency": args.concurrency,
            "freshness_samples": sample_count,
            "duplicate_probes": duplicate_count,
        },
        "publish": {
            "accepted": accepted,
            "failed": len(statuses) - accepted,
            "failure_rate": (len(statuses) - accepted) / len(statuses),
            "throughput_events_per_second": accepted / duration,
            "latency_ms": {
                "mean": statistics.fmean(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
        },
        "produced_to_searchable_ms": {
            "successful_samples": len(freshness),
            "failed_samples": search_failures,
            "p50": percentile(freshness, 0.50),
            "p95": percentile(freshness, 0.95),
            "p99": percentile(freshness, 0.99),
        },
        "idempotency": {
            "duplicate_probes_accepted": sum(status == 202 for status in duplicate_statuses),
            "duplicates_observed_by_consumers": duplicates_observed,
        },
        "limitations": [
            "This is a single-machine Docker Compose result, not a production capacity claim.",
            "Freshness is sampled and includes client polling time only through the stored produced-to-indexed timestamp.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    successful = (
        accepted == args.events
        and search_failures == 0
        and all(status == 202 for status in duplicate_statuses)
        and duplicates_observed >= duplicate_count
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
