#!/usr/bin/env python3
"""Train an immutable snapshot, promote/rollback, or resolve a serving artifact."""

import argparse
import json
from pathlib import Path

from newslens.operations.lifecycle import release, serving_path, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--snapshot", type=Path, required=True)
    t.add_argument("--cutoff", required=True)
    t.add_argument("--minimum-ndcg", type=float, required=True)
    t.add_argument("--minimum-validation", type=int, default=100)
    promote = sub.add_parser("promote")
    promote.add_argument("run_id")
    sub.add_parser("rollback")
    sub.add_parser("serving-path")
    a = p.parse_args()
    if a.command == "train":
        result = train(
            a.snapshot,
            a.root,
            cutoff=a.cutoff,
            minimum_ndcg=a.minimum_ndcg,
            minimum_validation=a.minimum_validation,
        )
    elif a.command == "serving-path":
        print(serving_path(a.root))
        return
    else:
        result = release(a.root, getattr(a, "run_id", None), rollback=a.command == "rollback")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
