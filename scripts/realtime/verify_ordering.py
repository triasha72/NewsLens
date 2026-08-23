#!/usr/bin/env python3
"""Verify that an older article event cannot overwrite a newer version."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode() if payload else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def metric_value(url: str, name: str) -> int:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        for line in response.read().decode().splitlines():
            if line.startswith(f"{name} "):
                return int(float(line.split()[1]))
    raise ValueError(f"metric {name!r} was not found at {url}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingestion-url", default="http://127.0.0.1:8080")
    parser.add_argument("--search-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--consumer-url",
        action="append",
        default=["http://127.0.0.1:8081", "http://127.0.0.1:8082"],
    )
    parser.add_argument("--output", type=Path, default=Path("reports/realtime_ordering_v0_1.json"))
    args = parser.parse_args()

    identifier = uuid.uuid4().hex
    token = f"Ordering{identifier}"
    article_id = f"ordering-{identifier}"
    now = datetime.now(UTC)
    base = {
        "article_id": article_id,
        "category": "technology",
        "published_at": now.isoformat(),
        "body": "Event ordering probe.",
    }
    current = {
        **base,
        "event_id": f"ordering-current-{identifier}",
        "title": f"{token} Current",
        "produced_at": now.isoformat(),
    }
    stale = {
        **base,
        "event_id": f"ordering-stale-{identifier}",
        "title": f"{token} Stale",
        "produced_at": (now - timedelta(days=1)).isoformat(),
    }
    processed_metric = "newslens_ingestion_processed_total"
    processed_before = sum(
        metric_value(url, processed_metric) for url in args.consumer_url
    )
    request_json(f"{args.ingestion_url}/events", current)
    request_json(f"{args.ingestion_url}/events", stale)

    deadline = time.monotonic() + 30
    processed_delta = 0
    while time.monotonic() < deadline:
        processed_delta = (
            sum(metric_value(url, processed_metric) for url in args.consumer_url)
            - processed_before
        )
        if processed_delta >= 2:
            break
        time.sleep(0.1)

    query = urllib.parse.urlencode({"q": token, "top_k": 1})
    search = request_json(f"{args.search_url}/search?{query}")
    title = search["results"][0]["title"] if search["results"] else None
    current_preserved = title == current["title"]
    report = {
        "schema_version": "newslens.realtime-ordering.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": "local Docker Compose",
        "distinct_events_processed": processed_delta,
        "newer_version_preserved": current_preserved,
        "observed_title": title,
        "limitations": ["Producer clocks must be trustworthy because ordering uses produced_at."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if processed_delta >= 2 and current_preserved else 1


if __name__ == "__main__":
    raise SystemExit(main())
