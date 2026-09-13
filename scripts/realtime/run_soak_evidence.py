"""Run a bounded soak, recovery, DLQ, and readiness evidence workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def aggregate_load(
    reports: list[dict],
    duration_minutes: int,
    *,
    environment: str = "local Docker Compose",
    application_hosts: int = 1,
    stateful_services_highly_available: bool = False,
) -> dict:
    accepted = sum(item["publish"]["accepted"] for item in reports)
    failed = sum(item["publish"]["failed"] for item in reports)
    result = {
        "schema_version": "newslens.realtime-load.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": environment,
        "configuration": {
            "events": accepted + failed,
            "duration_minutes": duration_minutes,
            "bursts": len(reports),
            "events_per_burst": (accepted + failed) // len(reports),
        },
        "topology": {
            "application_hosts": application_hosts,
            "stateful_services_highly_available": stateful_services_highly_available,
        },
        "publish": {
            "accepted": accepted,
            "failed": failed,
            "failure_rate": failed / (accepted + failed) if accepted + failed else 1.0,
        },
        "produced_to_searchable_ms": {
            "p95": max(item["produced_to_searchable_ms"]["p95"] for item in reports),
            "failed_samples": sum(item["produced_to_searchable_ms"]["failed_samples"] for item in reports),
        },
        "burst_reports": len(reports),
        "limitations": [
            "The aggregate freshness value is the worst per-burst p95, not a pooled percentile.",
        ],
    }
    if application_hosts < 2:
        result["limitations"].insert(
            0, "This is a single-host soak, not a multi-host application result."
        )
    if not stateful_services_highly_available:
        result["limitations"].append(
            "Kafka and PostgreSQL are not highly available in this topology."
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-minutes", type=int, default=60)
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--environment", default="local Docker Compose")
    parser.add_argument("--application-hosts", type=int, default=1)
    parser.add_argument("--stateful-services-highly-available", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    args = parser.parse_args()
    if args.duration_minutes < 1 or args.events < args.duration_minutes:
        parser.error("events must be at least duration-minutes so every minute has a burst")
    if args.application_hosts < 1:
        parser.error("application-hosts must be at least 1")

    root = Path(__file__).resolve().parents[2]
    benchmark = root / "scripts/realtime/benchmark_ingestion.py"
    recovery = root / "scripts/realtime/failure_recovery.py"
    dlq = root / "scripts/realtime/verify_dlq.py"
    assess = root / "scripts/realtime/assess_readiness.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_per_burst, remainder = divmod(args.events, args.duration_minutes)
    reports: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="newslens-soak-") as temporary:
        temp = Path(temporary)
        for burst in range(args.duration_minutes):
            scheduled = time.monotonic() + 60
            burst_events = events_per_burst + int(burst < remainder)
            report = temp / f"burst-{burst:03d}.json"
            run([
                sys.executable, str(benchmark), "--events", str(burst_events),
                "--concurrency", str(args.concurrency), "--output", str(report),
            ])
            reports.append(load(report))
            remaining = scheduled - time.monotonic()
            if burst + 1 < args.duration_minutes and remaining > 0:
                time.sleep(remaining)

    load_report = aggregate_load(
        reports,
        args.duration_minutes,
        environment=args.environment,
        application_hosts=args.application_hosts,
        stateful_services_highly_available=args.stateful_services_highly_available,
    )
    load_path = args.output_dir / "realtime_soak_load_v0_1.json"
    load_path.write_text(json.dumps(load_report, indent=2) + "\n", encoding="utf-8")
    recovery_path = args.output_dir / "realtime_soak_recovery_v0_1.json"
    if args.skip_recovery:
        recovery_path.write_text(
            json.dumps({"full_consumer_outage_recovery_ms": 1_000_000_000}) + "\n",
            encoding="utf-8",
        )
    else:
        run([sys.executable, str(recovery), "--output", str(recovery_path)])
    dlq_path = args.output_dir / "realtime_soak_dlq_v0_1.json"
    run([sys.executable, str(dlq), "--output", str(dlq_path)])
    readiness_path = args.output_dir / "realtime_soak_readiness_v0_1.json"
    run([
        sys.executable, str(assess), "--load", str(load_path), "--recovery", str(recovery_path),
        "--dlq", str(dlq_path), "--output", str(readiness_path),
    ])
    print(f"readiness receipt: {readiness_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
