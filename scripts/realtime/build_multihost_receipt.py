"""Build an honest receipt from host-local NewsLens evidence stages."""

import argparse
import hashlib
import json
from pathlib import Path


def build_receipt(load: dict, recovery: dict, dlq: dict, topology: dict) -> dict:
    """Return a receipt only for the supported non-HA topology."""
    if topology.get("stateful_services_highly_available"):
        raise ValueError("stateful services are not highly available in this topology")
    for name, report, role in (("load", load, "application"), ("recovery", recovery, "state"), ("dlq", dlq, "state")):
        if report.get("execution_host_role") != role:
            raise ValueError(f"missing required field execution_host_role={role} in {name} report")
        if "endpoints" in report:
            raise ValueError(f"{name} report contains endpoint URLs; use endpoint_roles instead")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=Path)
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--dlq", type=Path)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path, help="Verify an existing checksum manifest.")
    args = parser.parse_args()
    if args.verify:
        if any((args.load, args.recovery, args.dlq, args.topology, args.output_dir)):
            parser.error("--verify cannot be combined with build arguments")
        if verify_checksums(args.verify):
            print("checksums verified")
            return 0
        print("checksum verification failed")
        return 1
    if not all((args.load, args.recovery, args.dlq, args.topology, args.output_dir)):
        parser.error("--load, --recovery, --dlq, --topology, and --output-dir are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(*(json.loads(path.read_text()) for path in (args.load, args.recovery, args.dlq, args.topology)))
    output = args.output_dir / "multihost_receipt_v0_1.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    write_checksums([args.load, args.recovery, args.dlq, args.topology, output], args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
