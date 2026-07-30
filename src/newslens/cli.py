"""Command-line interface for NewsLens."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .data import audit_dataset, load_behaviors, load_news


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
    audit = audit_dataset(news, behaviors, split)

    serialized = json.dumps(
        audit.to_dict(),
        indent=2,
        sort_keys=True,
    )
    print(serialized)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{serialized}\n", encoding="utf-8")
        print(f"Audit report written to {output}")


def build_parser() -> argparse.ArgumentParser:
    """Create the NewsLens command-line parser."""

    parser = argparse.ArgumentParser(prog="newslens")
    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser(
        "audit-data",
        help="Validate and summarize a local MIND-small split.",
    )
    audit_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the extracted MIND-small folders.",
    )
    audit_parser.add_argument(
        "--split",
        choices=("train", "dev"),
        default="train",
    )
    audit_parser.add_argument("--output", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the NewsLens command-line interface."""

    args = build_parser().parse_args(argv)

    if args.command == "audit-data":
        _run_data_audit(args.data_dir, args.split, args.output)
        return

    print(
        f"NewsLens {__version__} starter is ready. "
        "Next milestone: implement validated MIND data ingestion."
    )
