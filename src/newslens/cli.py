"""Command-line interface for NewsLens."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .data import audit_dataset, load_behaviors, load_news
from .evaluation import (
    evaluate_content_baseline,
    evaluate_fallback_baseline,
    evaluate_popularity_baseline,
)


def _run_data_audit(
    data_dir: Path,
    split: str,
    output: Path | None,
) -> None:
    split_directory = {
        "train": "MINDsmall_train",
        "dev": "MINDsmall_dev",
    }[split]
    split_path = data_dir / split_directory

    news = load_news(split_path / "news.tsv")
    behaviors = load_behaviors(split_path / "behaviors.tsv")
    audit = audit_dataset(
        news,
        behaviors,
        split,
    )

    serialized = json.dumps(
        audit.to_dict(),
        indent=2,
        sort_keys=True,
    )
    print(serialized)

    if output is not None:
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            f"{serialized}\n",
            encoding="utf-8",
        )
        print(f"Audit report written to {output}")


def _run_popularity_evaluation(
    data_dir: Path,
    output: Path,
    k: int,
    validation_fraction: float,
) -> None:
    split_path = data_dir / "MINDsmall_train"

    news = load_news(split_path / "news.tsv")
    behaviors = load_behaviors(split_path / "behaviors.tsv")

    report = evaluate_popularity_baseline(
        behaviors,
        news["news_id"],
        validation_fraction=validation_fraction,
        k=k,
    )

    serialized = json.dumps(
        report.to_dict(),
        indent=2,
        sort_keys=True,
    )
    print(serialized)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        f"{serialized}\n",
        encoding="utf-8",
    )
    print(f"Popularity evaluation report written to {output}")


def _run_content_evaluation(
    data_dir: Path,
    output: Path,
    k: int,
    validation_fraction: float,
    max_features: int,
) -> None:
    split_path = data_dir / "MINDsmall_train"

    news = load_news(split_path / "news.tsv")
    behaviors = load_behaviors(split_path / "behaviors.tsv")

    report = evaluate_content_baseline(
        news,
        behaviors,
        validation_fraction=validation_fraction,
        k=k,
        max_features=max_features,
    )

    serialized = json.dumps(
        report.to_dict(),
        indent=2,
        sort_keys=True,
    )
    print(serialized)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        f"{serialized}\n",
        encoding="utf-8",
    )
    print(f"Content evaluation report written to {output}")


def _run_fallback_evaluation(
    data_dir: Path,
    output: Path,
    k: int,
    validation_fraction: float,
    max_features: int,
) -> None:
    split_path = data_dir / "MINDsmall_train"

    news = load_news(split_path / "news.tsv")
    behaviors = load_behaviors(split_path / "behaviors.tsv")

    report = evaluate_fallback_baseline(
        news,
        behaviors,
        validation_fraction=validation_fraction,
        k=k,
        max_features=max_features,
    )

    serialized = json.dumps(
        report.to_dict(),
        indent=2,
        sort_keys=True,
    )
    print(serialized)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        f"{serialized}\n",
        encoding="utf-8",
    )
    print(f"Fallback evaluation report written to {output}")


def build_parser() -> argparse.ArgumentParser:
    """Create the NewsLens command-line parser."""

    parser = argparse.ArgumentParser(prog="newslens")
    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser(
        "audit-data",
        help=("Validate and summarize a local MIND-small split."),
    )
    audit_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=("Directory containing the extracted MIND-small folders."),
    )
    audit_parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="train",
    )
    audit_parser.add_argument(
        "--output",
        type=Path,
    )

    popularity_parser = subparsers.add_parser(
        "evaluate-popularity",
        help=("Evaluate training-only popularity on a chronological validation split."),
    )
    popularity_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=("Directory containing the extracted MIND-small folders."),
    )
    popularity_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/popularity_metrics.json"),
    )
    popularity_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cutoff used for all metrics.",
    )
    popularity_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=("Chronological fraction reserved for validation."),
    )

    content_parser = subparsers.add_parser(
        "evaluate-content",
        help=("Evaluate TF-IDF history recommendations on a chronological validation split."),
    )
    content_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=("Directory containing the extracted MIND-small folders."),
    )
    content_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/content_metrics.json"),
    )
    content_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cutoff used for all metrics.",
    )
    content_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=("Chronological fraction reserved for validation."),
    )
    content_parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum TF-IDF vocabulary size.",
    )

    fallback_parser = subparsers.add_parser(
        "evaluate-fallback",
        help=("Evaluate TF-IDF history recommendations with training-only popularity fallback."),
    )
    fallback_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=("Directory containing the extracted MIND-small folders."),
    )
    fallback_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/fallback_metrics.json"),
    )
    fallback_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Ranking cutoff used for all metrics.",
    )
    fallback_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=("Chronological fraction reserved for validation."),
    )
    fallback_parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum TF-IDF vocabulary size.",
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the NewsLens command-line interface."""

    args = build_parser().parse_args(argv)

    if args.command == "audit-data":
        _run_data_audit(
            args.data_dir,
            args.split,
            args.output,
        )
        return

    if args.command == "evaluate-popularity":
        _run_popularity_evaluation(
            args.data_dir,
            args.output,
            args.k,
            args.validation_fraction,
        )
        return

    if args.command == "evaluate-content":
        _run_content_evaluation(
            args.data_dir,
            args.output,
            args.k,
            args.validation_fraction,
            args.max_features,
        )
        return

    if args.command == "evaluate-fallback":
        _run_fallback_evaluation(
            args.data_dir,
            args.output,
            args.k,
            args.validation_fraction,
            args.max_features,
        )
        return

    print(f"NewsLens {__version__} is ready. Use --help to view available commands.")
