#!/usr/bin/env python3
"""Inject malformed Kafka bytes and verify the dead-letter contract."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def metric_value(url: str, name: str) -> int:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        for line in response.read().decode().splitlines():
            if line.startswith(f"{name} "):
                return int(float(line.split()[1]))
    raise ValueError(f"metric {name!r} was not found at {url}")


def compose_command(file: Path, *arguments: str) -> list[str]:
    return ["docker", "compose", "-f", str(file), *arguments]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-file", type=Path, default=Path("deploy/realtime/compose.yaml"))
    parser.add_argument(
        "--consumer-url",
        action="append",
        default=["http://127.0.0.1:8081", "http://127.0.0.1:8082"],
    )
    parser.add_argument("--execution-host-role", choices=["application", "state"], default="state")
    parser.add_argument("--output", type=Path, default=Path("reports/realtime_dlq_v0_1.json"))
    args = parser.parse_args()

    invalid_metric = "newslens_ingestion_invalid_total"
    dlq_metric = "newslens_ingestion_dead_letter_total"
    invalid_before = sum(metric_value(url, invalid_metric) for url in args.consumer_url)
    dlq_before = sum(metric_value(url, dlq_metric) for url in args.consumer_url)

    malformed = b"not-json"
    subprocess.run(
        compose_command(
            args.compose_file,
            "exec",
            "-T",
            "kafka",
            "/opt/kafka/bin/kafka-console-producer.sh",
            "--bootstrap-server",
            "kafka:9092",
            "--topic",
            "news-events",
        ),
        input=malformed + b"\n",
        check=True,
    )

    deadline = time.monotonic() + 30
    invalid_delta = 0
    dlq_delta = 0
    while time.monotonic() < deadline:
        invalid_delta = (
            sum(metric_value(url, invalid_metric) for url in args.consumer_url)
            - invalid_before
        )
        dlq_delta = sum(metric_value(url, dlq_metric) for url in args.consumer_url) - dlq_before
        if invalid_delta >= 1 and dlq_delta >= 1:
            break
        time.sleep(0.1)

    consumed = subprocess.run(
        compose_command(
            args.compose_file,
            "exec",
            "-T",
            "kafka",
            "/opt/kafka/bin/kafka-console-consumer.sh",
            "--bootstrap-server",
            "kafka:9092",
            "--topic",
            "news-events-dlq",
            "--from-beginning",
            "--max-messages",
            "1",
            "--timeout-ms",
            "10000",
        ),
        capture_output=True,
        check=True,
        text=True,
    )
    envelope = json.loads(consumed.stdout.strip().splitlines()[0])
    expected_payload = base64.b64encode(malformed).decode()
    envelope_valid = (
        envelope.get("source") == "news-events"
        and envelope.get("original_payload_base64") == expected_payload
        and bool(envelope.get("reason"))
    )
    report = {
        "schema_version": "newslens.realtime-dlq.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": "local Docker Compose",
        "execution_host_role": args.execution_host_role,
        "endpoints": {"consumer_urls": args.consumer_url, "compose_file": str(args.compose_file)},
        "invalid_events_observed": invalid_delta,
        "dead_letter_events_observed": dlq_delta,
        "dead_letter_envelope_valid": envelope_valid,
        "limitations": ["This probes malformed JSON; database-exhaustion DLQ behavior is unit tested."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if invalid_delta >= 1 and dlq_delta >= 1 and envelope_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
