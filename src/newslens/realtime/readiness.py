"""Production-readiness checks for the real-time recommendation path."""

from __future__ import annotations

from typing import Any


def assess_realtime_readiness(
    load: dict[str, Any], recovery: dict[str, Any], dlq: dict[str, Any]
) -> dict[str, Any]:
    configuration = load["configuration"]
    duration = configuration.get("duration_minutes")
    checks = {
        "sustained_soak_duration": {
            "value": duration, "minimum_minutes": 60,
            "passed": duration is not None and duration >= 60,
        },
        "minimum_events": {
            "value": configuration["events"], "minimum": 100_000,
            "passed": configuration["events"] >= 100_000,
        },
        "publish_failure_rate": {
            "value": load["publish"]["failure_rate"], "maximum": 0.001,
            "passed": load["publish"]["failure_rate"] <= 0.001,
        },
        "freshness_p95_ms": {
            "value": load["produced_to_searchable_ms"]["p95"], "maximum": 1_000,
            "passed": load["produced_to_searchable_ms"]["p95"] <= 1_000,
        },
        "full_outage_recovery_ms": {
            "value": recovery["full_consumer_outage_recovery_ms"], "maximum": 30_000,
            "passed": recovery["full_consumer_outage_recovery_ms"] <= 30_000,
        },
        "dead_letter_envelope": {
            "value": dlq["dead_letter_envelope_valid"], "required": True,
            "passed": dlq["dead_letter_envelope_valid"] is True,
        },
        "multi_host_application_tier": {
            "value": {
                "environment": load.get("environment"),
                "application_hosts": load.get("topology", {}).get("application_hosts", 1),
            },
            "minimum_hosts": 2,
            "passed": (
                "multi-host" in load.get("environment", "").lower()
                and load.get("topology", {}).get("application_hosts", 1) >= 2
            ),
        },
        "stateful_services_high_availability": {
            "value": load.get("topology", {}).get("stateful_services_highly_available", False),
            "required": True,
            "passed": load.get("topology", {}).get("stateful_services_highly_available", False)
            is True,
        },
    }
    return {
        "schema_version": "newslens.realtime-readiness.v1",
        "decision": "ready" if all(check["passed"] for check in checks.values()) else "blocked",
        "checks": checks,
        "product_impact": "not measured; requires a separately approved online experiment",
    }
