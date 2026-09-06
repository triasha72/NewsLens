#!/usr/bin/env python3
"""Combine load, recovery, and DLQ evidence into one truthful decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from newslens.realtime.readiness import assess_realtime_readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--dlq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_realtime_readiness(*[
        json.loads(path.read_text()) for path in (args.load, args.recovery, args.dlq)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"decision={result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
