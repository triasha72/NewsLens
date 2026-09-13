"""Rehearse verified release transitions in an isolated local registry."""

import argparse
import json
import time
from pathlib import Path

from newslens.operations.lifecycle import release, serving_path, train


def rehearse(snapshot, root, cutoffs, minimum_ndcg, minimum_validation):
    if root.exists():
        raise ValueError("use a new isolated registry; never rehearse on a live release root")
    if len(cutoffs) != 2 or cutoffs[0] == cutoffs[1]:
        raise ValueError("two distinct snapshot cutoffs required")
    events = []
    candidates = []
    for cutoff in cutoffs:
        started = time.perf_counter()
        args = {
            "cutoff": cutoff,
            "minimum_ndcg": minimum_ndcg,
            "minimum_validation": minimum_validation,
        }
        receipt = train(snapshot, root, **args)
        if receipt["status"] != "candidate":
            raise ValueError("candidate rejected; do not lower gates after inspecting results")
        assert train(snapshot, root, **args) == receipt
        candidates.append(receipt["run_id"])
        events.append(
            {
                "step": "train_and_verified_retry",
                "run_id": receipt["run_id"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    for candidate in candidates:
        started = time.perf_counter()
        release(root, candidate)
        path = serving_path(root)
        events.append(
            {
                "step": "promote_and_resolve",
                "run_id": candidate,
                "artifact_path": str(path),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    started = time.perf_counter()
    state = release(root, rollback=True)
    assert state["current"] == candidates[0]
    serving_path(root)
    events.append(
        {
            "step": "rollback_and_resolve",
            "run_id": state["current"],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    report = {
        "status": "completed",
        "scope": "local registry drill; no HTTP uptime or user lift claim",
        "events": events,
    }
    (root / "rehearsal.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cutoffs", nargs=2, required=True)
    p.add_argument("--minimum-ndcg", type=float, required=True)
    p.add_argument("--minimum-validation", type=int, default=100)
    a = p.parse_args()
    print(
        json.dumps(
            rehearse(a.snapshot, a.root, a.cutoffs, a.minimum_ndcg, a.minimum_validation), indent=2
        )
    )


if __name__ == "__main__":
    main()
