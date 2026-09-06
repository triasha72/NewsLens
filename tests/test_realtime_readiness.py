import json
from pathlib import Path

from newslens.realtime.readiness import assess_realtime_readiness


def test_local_evidence_is_blocked_as_production_evidence():
    root = Path(__file__).parents[1] / "reports"
    result = assess_realtime_readiness(*[
        json.loads((root / name).read_text())
        for name in (
            "realtime_load_v0_1.json", "realtime_recovery_v0_1.json", "realtime_dlq_v0_1.json"
        )
    ])
    assert result["decision"] == "blocked"
    assert result["checks"]["publish_failure_rate"]["passed"]
    assert not result["checks"]["sustained_soak_duration"]["passed"]
    assert not result["checks"]["multi_host_environment"]["passed"]
