"""Build deterministic retrieval queries from chronological histories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from newslens.evaluation.content import (
    _parse_history,
)
from newslens.models.two_tower import (
    TwoTowerNetwork,
)

from .base import RetrievalError
from .catalog import RetrievalCatalog


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One frozen user retrieval query."""

    impression_id: str
    vector: np.ndarray
    exclude_news_ids: tuple[str, ...]


def build_validation_queries(
    validation: pd.DataFrame,
    *,
    catalog: RetrievalCatalog,
    network: TwoTowerNetwork,
    max_history_length: int,
    query_count: int,
    seed: int,
) -> tuple[RetrievalQuery, ...]:
    """Build a deterministic sample of nonempty validation user vectors."""

    if (
        isinstance(max_history_length, bool)
        or not isinstance(
            max_history_length,
            int,
        )
        or max_history_length <= 0
    ):
        raise RetrievalError(
            "max_history_length must be a positive integer."
        )

    if (
        isinstance(query_count, bool)
        or not isinstance(
            query_count,
            int,
        )
        or query_count <= 0
    ):
        raise RetrievalError(
            "query_count must be a positive integer."
        )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise RetrievalError(
            "seed must be a non-negative integer."
        )

    article_positions = (
        catalog.id_to_position
    )

    eligible: list[
        tuple[
            str,
            tuple[str, ...],
        ]
    ] = []

    for row in validation.itertuples(
        index=False
    ):
        history_ids = _parse_history(
            row.history
        )

        usable = tuple(
            news_id
            for news_id in history_ids
            if news_id in article_positions
        )

        if not usable:
            continue

        eligible.append(
            (
                str(
                    row.impression_id
                ),
                usable,
            )
        )

    if not eligible:
        raise RetrievalError(
            "No validation histories can produce retrieval queries."
        )

    sample_size = min(
        query_count,
        len(eligible),
    )

    rng = np.random.default_rng(
        seed
    )

    chosen_indices = rng.choice(
        len(eligible),
        size=sample_size,
        replace=False,
    )

    selected = [
        eligible[
            int(index)
        ]
        for index in sorted(
            chosen_indices.tolist()
        )
    ]

    network.to(
        "cpu"
    )

    network.eval()

    queries: list[
        RetrievalQuery
    ] = []

    with torch.no_grad():
        for (
            impression_id,
            full_history,
        ) in selected:
            encoded_history = (
                full_history[
                    -max_history_length:
                ]
            )

            positions = [
                article_positions[
                    news_id
                ]
                for news_id
                in encoded_history
            ]

            history_embeddings = (
                torch.from_numpy(
                    catalog.vectors[
                        positions
                    ]
                )
                .unsqueeze(0)
            )

            history_mask = torch.ones(
                (
                    1,
                    len(positions),
                ),
                dtype=torch.bool,
            )

            user_embedding = (
                network.user_tower(
                    history_embeddings,
                    history_mask,
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            queries.append(
                RetrievalQuery(
                    impression_id=(
                        impression_id
                    ),
                    vector=(
                        user_embedding
                    ),
                    exclude_news_ids=(
                        full_history
                    ),
                )
            )

    return tuple(
        queries
    )
