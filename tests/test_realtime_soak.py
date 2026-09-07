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
