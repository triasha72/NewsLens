from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "load_test_api.py"
SPEC = importlib.util.spec_from_file_location("load_test_api", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates() -> None:
    assert MODULE.percentile([40.0, 10.0, 30.0, 20.0], 0.5) == pytest.approx(25.0)


def test_summarize_counts_successes_and_errors() -> None:
    results = [
        MODULE.RequestResult(10.0, 200),
        MODULE.RequestResult(20.0, 200),
        MODULE.RequestResult(30.0, 503, "HTTP 503"),
    ]
    summary = MODULE.summarize(results, 0.5)
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["throughput_requests_per_second"] == pytest.approx(6.0)
    assert summary["successful_latency_ms"]["p50"] == pytest.approx(15.0)
    assert summary["errors"] == {"HTTP 503": 1}
