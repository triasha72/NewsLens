from __future__ import annotations

import argparse
from pathlib import Path

from newslens.data.recsys_training import load_bpr_triples
from newslens.models.collaborative import CollaborativeRecommender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-negatives-per-positive", type=int, default=1)
    args = parser.parse_args()

    triples = load_bpr_triples(
        Path(args.database),
        cutoff_timestamp=args.cutoff,
        max_negatives_per_positive=args.max_negatives_per_positive,
    )
    model = CollaborativeRecommender(
        embedding_dim=args.embedding_dim,
    ).fit(triples, epochs=args.epochs)

    print(
        {
            "triples": len(triples),
            "users": len(model.user_to_index),
            "items": len(model.item_to_index),
            "embedding_dim": model.embedding_dim,
        }
    )


if __name__ == "__main__":
    main()
