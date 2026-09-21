#!/usr/bin/env python3
"""Measure consumer failover and backlog recovery in the local Compose stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def compose(file: Path, *arguments: str) -> None:
    subprocess.run(["docker", "compose", "-f", str(file), *arguments], check=True)


def consumer_partitions(file: Path, client_id: str) -> set[int]:
    command = [
        "docker",
        "compose",
        "-f",
        str(file),
        "exec",
        "-T",
        "kafka",
        "/opt/kafka/bin/kafka-consumer-groups.sh",
        "--bootstrap-server",
        "kafka:9092",
        "--group",
        "newslens-indexers",
        "--describe",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    partitions = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[1] == "news-events" and fields[-1] == client_id:
            partitions.add(int(fields[2]))
    if not partitions:
        raise RuntimeError(f"no partition assignment found for {client_id}")
    return partitions


def fnv_partition(key: str, count: int = 3) -> int:
    value = 2_166_136_261
    for byte in key.encode():
        value ^= byte
        value = (value * 16_777_619) & 0xFFFFFFFF
    signed = value if value < 0x80000000 else value - 0x100000000
    return abs(signed) % count


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


def event(label: str, target_partition: int | None = None) -> tuple[dict[str, Any], str]:
    identifier = uuid.uuid4().hex
    while target_partition is not None and fnv_partition(f"{label}-{identifier}") != target_partition:
        identifier = uuid.uuid4().hex
    token = f"Recovery{identifier}"
    now = datetime.now(UTC).isoformat()
    return (
        {
            "event_id": f"{label}-{identifier}",
            "article_id": f"{label}-{identifier}",
            "title": f"{token} technology update",
            "category": "technology",
            "published_at": now,
            "produced_at": now,
            "body": "Failure-recovery probe.",
        },
        token,
    )


def wait_search(search_url: str, token: str, timeout: float = 60.0) -> float:
    started = time.perf_counter()
    query = urllib.parse.urlencode({"q": token, "top_k": 1})
    while time.perf_counter() - started < timeout:
        body = request_json(f"{search_url}/search?{query}")
        if body["returned_count"]:
            return (time.perf_counter() - started) * 1_000
        time.sleep(0.1)
    raise TimeoutError(f"{token} did not become searchable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-file", type=Path, default=Path("deploy/realtime/compose.yaml"))
    parser.add_argument("--ingestion-url", default="http://127.0.0.1:8080")
    parser.add_argument("--search-url", default="http://127.0.0.1:8000")
    parser.add_argument("--execution-host-role", choices=["application", "state"], default="state")
    parser.add_argument("--output", type=Path, default=Path("reports/realtime_recovery_v0_1.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {"schema_version": "newslens.realtime-recovery.v1"}
    try:
        target_partition = min(
            consumer_partitions(args.compose_file, "newslens-consumer-1")
        )
        compose(args.compose_file, "stop", "consumer-1")
        failover_event, token = event("failover", target_partition)
        request_json(f"{args.ingestion_url}/events", failover_event)
        report["stopped_consumer_partition"] = target_partition
        report["one_consumer_stopped_searchable_ms"] = wait_search(args.search_url, token)

        compose(args.compose_file, "stop", "consumer-2")
        backlog_event, token = event("backlog")
        request_json(f"{args.ingestion_url}/events", backlog_event)
        time.sleep(2)
        recovery_started = time.perf_counter()
        compose(args.compose_file, "start", "consumer-1")
        wait_search(args.search_url, token)
        report["full_consumer_outage_recovery_ms"] = (
            time.perf_counter() - recovery_started
        ) * 1_000
    finally:
        compose(args.compose_file, "start", "consumer-1", "consumer-2")

    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "environment": "local Docker Compose",
            "execution_host_role": args.execution_host_role,
            "endpoints": {"ingestion_url": args.ingestion_url, "search_url": args.search_url, "compose_file": str(args.compose_file)},
            "limitations": ["Single-host process failure exercise; broker and host failures are out of scope."],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
