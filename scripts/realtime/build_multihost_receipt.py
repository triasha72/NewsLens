"""Build an honest receipt from host-local NewsLens evidence stages."""

import hashlib
import json
from pathlib import Path


def build_receipt(load: dict, recovery: dict, dlq: dict, topology: dict) -> dict:
    """Return a receipt only for the supported non-HA topology."""
    if topology.get("stateful_services_highly_available"):
        raise ValueError("stateful services are not highly available in this topology")
    return {
        "schema_version": "newslens.multihost-receipt.v1",
        "topology": topology,
        "stages": {"load": load, "recovery": recovery, "dlq": dlq},
        "limitations": ["Kafka and PostgreSQL are single-instance stateful services."],
    }


def write_checksums(paths: list[Path], output_dir: Path) -> Path:
    manifest = output_dir / "multihost_checksums_v0_1.json"
    manifest.write_text(json.dumps({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))
    return manifest


def verify_checksums(manifest: Path) -> bool:
    return all((manifest.parent / name).exists() and hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest() == digest for name, digest in json.loads(manifest.read_text()).items())
