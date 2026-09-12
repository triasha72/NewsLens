"""Content-addressed training runs and atomic local release pointers (POSIX)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from newslens.artifacts import export_fallback_artifact, load_artifact
from newslens.data import audit_dataset, load_behaviors, load_news
from newslens.evaluation.fallback import evaluate_fallback_baseline


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lifecycle.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def train(
    snapshot: Path,
    root: Path,
    *,
    cutoff: str,
    minimum_ndcg: float,
    minimum_validation: int = 100,
    max_features: int = 50000,
) -> dict:
    """Evaluate the recipe chronologically, then refit on the available snapshot.

    The gate evaluates the training recipe, not independent quality of the refit.
    This snapshot must exclude the official final holdout. Nothing is auto-promoted.
    """
    if not math.isfinite(minimum_ndcg) or not 0 <= minimum_ndcg <= 1:
        raise ValueError("minimum_ndcg must be finite and between zero and one")
    if minimum_validation < 1:
        raise ValueError("minimum_validation must be positive")
    cutoff_time = pd.Timestamp(cutoff)
    if pd.isna(cutoff_time):
        raise ValueError("cutoff required")
    sources = {name: digest(snapshot / name) for name in ("news.tsv", "behaviors.tsv")}
    specification = {
        "sources": sources,
        "cutoff": cutoff_time.isoformat(),
        "minimum_ndcg": minimum_ndcg,
        "minimum_validation": minimum_validation,
        "max_features": max_features,
        "pipeline_version": 1,
    }
    run_id = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:24]
    directory = root / "runs" / run_id
    with locked(root):
        receipt_path = directory / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["status"] == "candidate":
                load_artifact(directory / "artifact")
                if digest(directory / "artifact/manifest.json") != receipt["manifest_sha256"]:
                    raise ValueError("candidate manifest changed")
            return receipt
        news = load_news(snapshot / "news.tsv")
        behaviors = load_behaviors(snapshot / "behaviors.tsv")
        behaviors = behaviors.loc[behaviors.timestamp <= cutoff_time].copy()
        if behaviors.empty:
            raise ValueError("no behaviors at or before cutoff")
        audit = audit_dataset(news, behaviors, "training-snapshot").to_dict()
        if audit["referenced_news_missing_metadata"] or audit["missing_titles"]:
            raise ValueError("snapshot failed catalog validation")
        report = evaluate_fallback_baseline(
            news, behaviors, max_features=max_features, bootstrap_samples=100
        )
        metrics = report.metrics.to_dict()
        passed = (
            metrics["evaluated_impressions"] >= minimum_validation
            and metrics["ndcg_at_k"] >= minimum_ndcg
            and metrics["empty_ranking_impressions"] == 0
        )
        # Detect a concurrently replaced source before publishing any candidate.
        if sources != {name: digest(snapshot / name) for name in sources}:
            raise ValueError("snapshot changed during training")
        receipt = {
            "run_id": run_id,
            "specification": specification,
            "audit": audit,
            "validation": metrics,
            "status": "candidate" if passed else "rejected",
            "evaluation_scope": "chronological recipe validation; refit is not final-test evidence",
        }
        directory.mkdir(parents=True, exist_ok=True)
        if passed:
            artifact = directory / "artifact"
            # An interrupted export can leave a complete artifact before its receipt.
            if artifact.exists():
                load_artifact(artifact)
            else:
                export_fallback_artifact(news, behaviors, artifact, max_features=max_features)
            receipt["manifest_sha256"] = digest(artifact / "manifest.json")
        atomic_json(receipt_path, receipt)
        return receipt


def checked_candidate(root: Path, run_id: str) -> Path:
    if len(run_id) != 24 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("invalid run ID")
    directory = root / "runs" / run_id
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt.get("status") != "candidate" or receipt.get("run_id") != run_id:
        raise ValueError("only a passing candidate can be released")
    artifact = directory / "artifact"
    if digest(artifact / "manifest.json") != receipt.get("manifest_sha256"):
        raise ValueError("candidate manifest changed")
    load_artifact(artifact)
    return artifact.resolve()


def release(root: Path, run_id: str | None = None, *, rollback: bool = False) -> dict:
    with locked(root):
        pointer = root / "release.json"
        state = json.loads(pointer.read_text()) if pointer.exists() else {}
        selected = state.get("previous") if rollback else run_id
        if not selected:
            raise ValueError("no candidate/previous release selected")
        checked_candidate(root, selected)
        if state.get("current") == selected:
            return state
        result = {"current": selected, "previous": state.get("current")}
        atomic_json(pointer, result)
        return result


def serving_path(root: Path) -> Path:
    state = json.loads((root / "release.json").read_text())
    return checked_candidate(root, state["current"])
