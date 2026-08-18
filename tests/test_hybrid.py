from collections.abc import Iterable

import pytest

from newslens.models.base import Recommendation
from newslens.models.content import (
    ColdStartUserError,
    ContentRecommendation,
)
from newslens.models.hybrid import (
    HybridModelError,
    HybridRecommender,
)


class _ContentStub:
    def __init__(
        self,
        scores: dict[str, float],
        *,
        cold_start: bool = False,
    ) -> None:
        self.scores = scores
        self.cold_start = cold_start
        self.is_fitted = True

    def recommend(
        self,
        history_news_ids: Iterable[str],
        *,
        candidate_news_ids: Iterable[str] | None = None,
        top_k: int = 10,
        exclude_history: bool = True,
    ) -> list[ContentRecommendation]:
        del history_news_ids
        del exclude_history

        if self.cold_start:
            raise ColdStartUserError(
                "Synthetic cold-start user."
            )

        candidates = set(
            candidate_news_ids or ()
        )

        ranked = sorted(
            (
                (news_id, score)
                for news_id, score
                in self.scores.items()
                if news_id in candidates
            ),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[:top_k]

        return [
            ContentRecommendation(
                news_id=news_id,
                title=news_id,
                category="test",
                score=score,
            )
            for news_id, score in ranked
        ]


class _CollaborativeStub:
    def __init__(
        self,
        scores: dict[str, float],
    ) -> None:
        self.scores = scores
        self.is_fitted = True

        self.user_to_index = {
            "u1": 0,
        }

        self.item_to_index = {
            news_id: index
            for index, news_id
            in enumerate(
                sorted(scores)
            )
        }

    def recommend_for_user(
        self,
        user_id: str,
        *,
        candidate_news_ids: Iterable[str],
        top_k: int = 10,
    ) -> list[Recommendation]:
        if user_id not in self.user_to_index:
            return []

        candidates = set(
            candidate_news_ids
        )

        ranked = sorted(
            (
                (news_id, score)
                for news_id, score
                in self.scores.items()
                if news_id in candidates
            ),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[:top_k]

        return [
            Recommendation(
                news_id=news_id,
                score=score,
                source="collaborative",
            )
            for news_id, score in ranked
        ]


class _PopularityStub:
    def __init__(
        self,
        scores: dict[str, float],
    ) -> None:
        self.scores = scores
        self.is_fitted = True

    def score(
        self,
        news_id: str,
    ) -> float:
        return self.scores.get(
            news_id,
            0.0,
        )

    def rank_candidates(
        self,
        candidate_news_ids: Iterable[str],
        *,
        top_k: int | None = None,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[str]:
        excluded = set(
            exclude_news_ids
        )

        ranked = sorted(
            {
                news_id
                for news_id
                in candidate_news_ids
                if news_id not in excluded
            },
            key=lambda news_id: (
                -self.score(news_id),
                news_id,
            ),
        )

        if top_k is None:
            return ranked

        return ranked[:top_k]


def _build_model(
    *,
    content_scores: dict[str, float],
    collaborative_scores: dict[str, float],
    popularity_scores: dict[str, float] | None = None,
    collaborative_weight: float = 0.10,
    minimum_supported_candidates: int = 0,
    cold_start: bool = False,
) -> HybridRecommender:
    return HybridRecommender(
        _ContentStub(
            content_scores,
            cold_start=cold_start,
        ),
        _CollaborativeStub(
            collaborative_scores
        ),
        _PopularityStub(
            popularity_scores or {}
        ),
        collaborative_weight=(
            collaborative_weight
        ),
        minimum_supported_candidates=(
            minimum_supported_candidates
        ),
    )


def test_zero_weight_preserves_content_order() -> None:
    model = _build_model(
        content_scores={
            "a": 0.9,
            "b": 0.8,
            "c": 0.1,
        },
        collaborative_scores={
            "a": -2.0,
            "b": 5.0,
            "c": 10.0,
        },
        collaborative_weight=0.0,
    )

    result = model.recommend_for_user(
        "u1",
        ["history"],
        candidate_news_ids=[
            "a",
            "b",
            "c",
        ],
        top_k=3,
    )

    assert [
        row.news_id
        for row in result
    ] == [
        "a",
        "b",
        "c",
    ]


def test_supported_residual_can_change_ranking() -> None:
    model = _build_model(
        content_scores={
            "a": 0.9,
            "b": 0.8,
            "c": 0.1,
        },
        collaborative_scores={
            "a": 0.0,
            "b": 1.0,
        },
        collaborative_weight=0.5,
    )

    result = model.recommend_for_user(
        "u1",
        ["history"],
        candidate_news_ids=[
            "a",
            "b",
            "c",
        ],
        top_k=3,
    )

    assert result[0].news_id == "b"
    assert result[0].source == "hybrid"


def test_below_support_gate_preserves_content_ranking() -> None:
    model = _build_model(
        content_scores={
            "a": 0.9,
            "b": 0.8,
            "c": 0.1,
        },
        collaborative_scores={
            "a": 0.0,
            "b": 1.0,
        },
        collaborative_weight=0.5,
        minimum_supported_candidates=3,
    )

    assert (
        model.collaborative_gate_open(
            "u1",
            ["a", "b", "c"],
        )
        is False
    )

    result = model.recommend_for_user(
        "u1",
        ["history"],
        candidate_news_ids=[
            "a",
            "b",
            "c",
        ],
        top_k=3,
    )

    assert [
        row.news_id
        for row in result
    ] == [
        "a",
        "b",
        "c",
    ]

    assert all(
        row.source == "content"
        for row in result
    )


def test_gate_opens_at_exact_threshold() -> None:
    model = _build_model(
        content_scores={
            "a": 0.9,
            "b": 0.8,
            "c": 0.1,
        },
        collaborative_scores={
            "a": 0.0,
            "b": 1.0,
        },
        collaborative_weight=0.5,
        minimum_supported_candidates=2,
    )

    assert (
        model.supported_candidate_count(
            "u1",
            ["a", "b", "c"],
        )
        == 2
    )

    assert (
        model.collaborative_gate_open(
            "u1",
            ["a", "b", "c"],
        )
        is True
    )

    result = model.recommend_for_user(
        "u1",
        ["history"],
        candidate_news_ids=[
            "a",
            "b",
            "c",
        ],
        top_k=3,
    )

    assert result[0].news_id == "b"


def test_unknown_user_never_opens_gate() -> None:
    model = _build_model(
        content_scores={
            "a": 0.9,
            "b": 0.8,
        },
        collaborative_scores={
            "a": 0.0,
            "b": 1.0,
        },
        collaborative_weight=0.5,
        minimum_supported_candidates=1,
    )

    assert (
        model.collaborative_gate_open(
            "unknown",
            ["a", "b"],
        )
        is False
    )


def test_content_cold_start_uses_popularity() -> None:
    model = _build_model(
        content_scores={},
        collaborative_scores={
            "a": 100.0,
        },
        popularity_scores={
            "a": 10.0,
            "b": 50.0,
            "c": 100.0,
        },
        collaborative_weight=0.5,
        minimum_supported_candidates=1,
        cold_start=True,
    )

    result = model.recommend_for_user(
        "u1",
        [],
        candidate_news_ids=[
            "a",
            "b",
            "c",
        ],
        top_k=3,
    )

    assert [
        row.news_id
        for row in result
    ] == [
        "c",
        "b",
        "a",
    ]

    assert all(
        row.source == "popularity"
        for row in result
    )


def test_constant_collaborative_scores_are_neutral() -> None:
    model = _build_model(
        content_scores={
            "a": 0.9,
            "b": 0.8,
            "c": 0.1,
        },
        collaborative_scores={
            "a": 5.0,
            "b": 5.0,
        },
        collaborative_weight=0.5,
    )

    result = model.recommend_for_user(
        "u1",
        ["history"],
        candidate_news_ids=[
            "a",
            "b",
            "c",
        ],
        top_k=3,
    )

    assert [
        row.news_id
        for row in result
    ] == [
        "a",
        "b",
        "c",
    ]


def test_invalid_collaborative_weight_is_rejected() -> None:
    with pytest.raises(
        HybridModelError,
        match="collaborative_weight",
    ):
        _build_model(
            content_scores={
                "a": 1.0,
            },
            collaborative_scores={},
            collaborative_weight=1.1,
        )


def test_negative_support_threshold_is_rejected() -> None:
    with pytest.raises(
        HybridModelError,
        match="minimum_supported_candidates",
    ):
        _build_model(
            content_scores={
                "a": 1.0,
            },
            collaborative_scores={
                "a": 1.0,
            },
            minimum_supported_candidates=-1,
        )
