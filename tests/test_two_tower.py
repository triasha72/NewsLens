import torch

from newslens.models.two_tower import (
    TwoTowerConfig,
    TwoTowerModelError,
    TwoTowerNetwork,
    TwoTowerRecommender,
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


def test_article_and_user_embeddings_are_normalized() -> None:
    network = _network()
    network.eval()

    history = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    mask = torch.ones(
        (2, 2),
        dtype=torch.bool,
    )

    articles = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    user_embeddings = (
        network.encode_users(
            history,
            mask,
        )
    )

    article_embeddings = (
        network.encode_articles(
            articles
        )
    )

    assert torch.allclose(
        torch.linalg.vector_norm(
            user_embeddings,
            dim=1,
        ),
        torch.ones(2),
        atol=1e-6,
    )

    assert torch.allclose(
        torch.linalg.vector_norm(
            article_embeddings,
            dim=1,
        ),
        torch.ones(2),
        atol=1e-6,
    )


def test_masked_history_ignores_padding() -> None:
    network = _network()
    network.eval()

    single = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    single_mask = torch.tensor(
        [[True]]
    )

    padded = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [100.0, 100.0, 100.0, 100.0],
            ]
        ]
    )

    padded_mask = torch.tensor(
        [[True, False]]
    )

    first = network.encode_users(
        single,
        single_mask,
    )

    second = network.encode_users(
        padded,
        padded_mask,
    )

    assert torch.allclose(
        first,
        second,
        atol=1e-6,
    )


def test_in_batch_logits_have_expected_shape() -> None:
    network = _network()

    history = torch.randn(
        5,
        3,
        4,
    )

    mask = torch.ones(
        (5, 3),
        dtype=torch.bool,
    )

    positives = torch.randn(
        5,
        4,
    )

    logits = network.in_batch_logits(
        history,
        mask,
        positives,
    )

    assert logits.shape == (
        5,
        5,
    )


def test_recommender_scores_content_represented_new_article() -> None:
    network = _network()

    model = TwoTowerRecommender(
        network,
        {
            "history": [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "old": [
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            "new_article": [
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        },
    )

    recommendations = model.recommend(
        ["history"],
        candidate_news_ids=[
            "old",
            "new_article",
        ],
        top_k=2,
    )

    assert {
        recommendation.news_id
        for recommendation
        in recommendations
    } == {
        "old",
        "new_article",
    }


def test_unknown_user_history_abstains() -> None:
    network = _network()

    model = TwoTowerRecommender(
        network,
        {
            "candidate": [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        },
    )

    assert (
        model.recommend(
            ["unknown_history"],
            candidate_news_ids=[
                "candidate"
            ],
            top_k=1,
        )
        == []
    )


def test_recommendation_is_deterministic() -> None:
    network = _network()

    model = TwoTowerRecommender(
        network,
        {
            "history": [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "a": [
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            "b": [
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        },
    )

    first = model.recommend(
        ["history"],
        candidate_news_ids=[
            "b",
            "a",
        ],
        top_k=2,
    )

    second = model.recommend(
        ["history"],
        candidate_news_ids=[
            "a",
            "b",
        ],
        top_k=2,
    )

    assert first == second


def test_feature_dimension_mismatch_is_rejected() -> None:
    network = _network()

    try:
        TwoTowerRecommender(
            network,
            {
                "bad": [
                    1.0,
                    2.0,
                ]
            },
        )
    except TwoTowerModelError as error:
        assert (
            "feature dimension"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected feature mismatch error."
        )
