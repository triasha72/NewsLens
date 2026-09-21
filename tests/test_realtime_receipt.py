import pytest

from scripts.realtime.build_multihost_receipt import build_receipt, write_checksums, verify_checksums


def test_receipt_rejects_high_availability_claim() -> None:
    with pytest.raises(ValueError, match="not highly available"):
        build_receipt({}, {}, {}, {"application_hosts": 2, "stateful_services_highly_available": True})


def test_checksums_detect_changed_artifact(tmp_path) -> None:
    artifact = tmp_path / "load.json"
    artifact.write_text("original")
    manifest = write_checksums([artifact], tmp_path)
    artifact.write_text("changed")
    assert not verify_checksums(manifest)


def test_receipt_requires_three_host_local_reports() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        build_receipt({}, {}, {}, {"application_hosts": 2, "stateful_services_highly_available": False})
