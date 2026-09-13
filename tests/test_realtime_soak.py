from scripts.realtime.run_soak_evidence import aggregate_load


def _burst(accepted: int, failed: int, freshness: float) -> dict:
    return {
        "publish": {"accepted": accepted, "failed": failed},
        "produced_to_searchable_ms": {"p95": freshness, "failed_samples": 0},
    }


def test_aggregate_soak_load_sums_events_and_keeps_worst_freshness() -> None:
    result = aggregate_load([_burst(10, 0, 11.0), _burst(9, 1, 17.0)], 2)
    assert result["configuration"]["events"] == 20
    assert result["configuration"]["duration_minutes"] == 2
    assert result["publish"]["failure_rate"] == 0.05
    assert result["produced_to_searchable_ms"]["p95"] == 17.0
    assert result["topology"]["application_hosts"] == 1
    assert not result["topology"]["stateful_services_highly_available"]


def test_aggregate_soak_can_describe_a_multi_host_application_tier() -> None:
    result = aggregate_load(
        [_burst(10, 0, 11.0)],
        1,
        environment="AWS EC2 multi-host application tier",
        application_hosts=2,
    )
    assert result["topology"]["application_hosts"] == 2
    assert "single-host soak" not in " ".join(result["limitations"])
    assert "Kafka and PostgreSQL" in " ".join(result["limitations"])
