import subprocess
import sys

import pytest

from scripts.realtime.build_multihost_receipt import build_receipt, verify_checksums, write_checksums


def test_receipt_rejects_high_availability_claim() -> None:
    with pytest.raises(ValueError, match="not highly available"):
        build_receipt({}, {}, {}, {"application_hosts": 2, "stateful_services_highly_available": True})


def test_checksums_detect_changed_artifact(tmp_path) -> None:
    artifact = tmp_path / "load.json"
    artifact.write_text("original")
    manifest = write_checksums([artifact], tmp_path)
    artifact.write_text("changed")
    assert not verify_checksums(manifest)


def test_checksums_verify_original_artifact(tmp_path) -> None:
    artifact = tmp_path / "load.json"
    artifact.write_text("original")
    manifest = write_checksums([artifact], tmp_path)
    assert verify_checksums(manifest)


def test_checksum_verification_cli(tmp_path) -> None:
    artifact = tmp_path / "load.json"
    artifact.write_text("original")
    manifest = write_checksums([artifact], tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/realtime/build_multihost_receipt.py", "--verify", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "checksums verified" in result.stdout


def test_receipt_requires_three_host_local_reports() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        build_receipt({}, {}, {}, {"application_hosts": 2, "stateful_services_highly_available": False})


def test_receipt_rejects_reports_that_contain_raw_endpoint_urls() -> None:
    reports = [
        {"execution_host_role": "application", "endpoints": {"ingestion_url": "http://10.0.0.1"}},
        {"execution_host_role": "state"},
        {"execution_host_role": "state"},
    ]
    with pytest.raises(ValueError, match="endpoint URLs"):
        build_receipt(*reports, {"application_hosts": 2, "stateful_services_highly_available": False})
