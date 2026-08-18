import numpy as np
import torch

from newslens.data.two_tower_training import (
    TwoTowerTrainingExample,
)
from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerNetwork,
)
from newslens.training.two_tower import (
    TwoTowerTrainingConfig,
    train_two_tower,
)


def _features() -> dict[str, np.ndarray]:
    return {
        "h1": np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "h2": np.array(
            [0.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "h3": np.array(
            [0.0, 0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
        "p1": np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "p2": np.array(
            [0.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "p3": np.array(
            [0.0, 0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
    }


def _examples() -> tuple[
    TwoTowerTrainingExample,
    ...
]:
    return (
        TwoTowerTrainingExample(
            impression_id="i1",
            history_news_ids=("h1",),
            positive_news_id="p1",
        ),
        TwoTowerTrainingExample(
            impression_id="i2",
            history_news_ids=("h2",),
            positive_news_id="p2",
        ),
        TwoTowerTrainingExample(
            impression_id="i3",
            history_news_ids=("h3",),
            positive_news_id="p3",
        ),
        TwoTowerTrainingExample(
            impression_id="i4",
            history_news_ids=("h1",),
            positive_news_id="p1",
        ),
        TwoTowerTrainingExample(
            impression_id="i5",
            history_news_ids=("h2",),
            positive_news_id="p2",
        ),
        TwoTowerTrainingExample(
            impression_id="i6",
            history_news_ids=("h3",),
            positive_news_id="p3",
        ),
    )


def _network() -> TwoTowerNetwork:
    torch.manual_seed(42)

    return TwoTowerNetwork(
        TwoTowerConfig(
            input_dim=4,
            hidden_dim=8,
            embedding_dim=3,
            dropout=0.0,
            temperature=0.1,
        )
    )


def test_training_updates_parameters() -> None:
    network = _network()

    before = (
        network.article_tower
        .network[0]
        .weight
        .detach()
        .clone()
    )

    result = train_two_tower(
        network,
        _examples(),
        _features(),
        config=TwoTowerTrainingConfig(
            epochs=2,
            batch_size=6,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=42,
            device="cpu",
        ),
    )

    after = (
        network.article_tower
        .network[0]
        .weight
        .detach()
        .clone()
    )

    assert not torch.allclose(
        before,
        after,
    )

    assert len(result.epochs) == 2

    assert all(
        np.isfinite(
            epoch.average_loss
        )
        for epoch in result.epochs
    )


def test_duplicate_positive_articles_are_supported() -> None:
    result = train_two_tower(
        _network(),
        _examples(),
        _features(),
        config=TwoTowerTrainingConfig(
            epochs=1,
            batch_size=6,
            learning_rate=1e-2,
            weight_decay=0.0,
            seed=42,
            device="cpu",
        ),
    )

    assert (
        result.epochs[0].examples_seen
        == 6
    )

    assert (
        result.epochs[0]
        .skipped_no_negative_examples
        == 0
    )
