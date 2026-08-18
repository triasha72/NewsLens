from __future__ import annotations

import argparse
import json
from pathlib import Path

from newslens.evaluation.collaborative import (
    evaluate_collaborative_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--database", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--max-negatives-per-positive",
        type=int,
        default=1,
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    report, _ = evaluate_collaborative_model(
        Path(args.database),
        cutoff_timestamp=args.cutoff,
        k=args.k,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_negatives_per_positive=(
            args.max_negatives_per_positive
        ),
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = report.to_dict()

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nWrote evaluation report to {output}")


if __name__ == "__main__":
    main()
