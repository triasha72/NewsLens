from __future__ import annotations

import pandas as pd
import pytest

from newslens.models import (
    PopularityModelError,
    PopularityModelNotFittedError,
    PopularityRecommender,
)


def make_training_behaviors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impressions": [
                "N1-1 N2-0 N3-0",
                "N1-0 N2-1",
                "N1-1 N2-0",
            ]
        }
    )


def test_fit_learns_clicks_and_exposures() -> None:
    model = PopularityRecommender().fit(make_training_behaviors())

    statistics = model.statistics("N1")

    assert statistics.exposures == 3
    assert statistics.clicks == 2
    assert statistics.click_through_rate == pytest.approx(2 / 3)


def test_recommend_ranks_articles_by_training_clicks() -> None:
    model = PopularityRecommender().fit(make_training_behaviors())

    recommendations = model.recommend(top_k=3)

    assert recommendations == ["N1", "N2", "N3"]


def test_rank_candidates_includes_unseen_articles_with_zero_score() -> None:
    model = PopularityRecommender().fit(make_training_behaviors())

    ranked = model.rank_candidates(
        ["N4", "N3", "N1"],
        top_k=3,
    )

    assert ranked == ["N1", "N3", "N4"]
    assert model.score("N4") == 0.0


def test_recommend_excludes_requested_articles() -> None:
    model = PopularityRecommender().fit(make_training_behaviors())

    recommendations = model.recommend(
        top_k=2,
        exclude_news_ids={"N1"},
    )

    assert recommendations == ["N2", "N3"]


def test_model_only_learns_from_data_passed_to_fit() -> None:
    training = pd.DataFrame({"impressions": ["N1-1 N2-0"]})
    validation = pd.DataFrame({"impressions": ["N2-1 N2-1 N2-1"]})

    model = PopularityRecommender().fit(training)

    assert model.score("N1") == 1.0
    assert model.score("N2") == 0.0
    assert validation.loc[0, "impressions"] == "N2-1 N2-1 N2-1"


def test_recommend_before_fit_raises_error() -> None:
    model = PopularityRecommender()

    with pytest.raises(
        PopularityModelNotFittedError,
        match="Fit the popularity model",
    ):
        model.recommend()


@pytest.mark.parametrize("top_k", [0, -1])
def test_recommend_rejects_invalid_top_k(top_k: int) -> None:
    model = PopularityRecommender().fit(make_training_behaviors())

    with pytest.raises(
        PopularityModelError,
        match="top_k must be greater than zero",
    ):
        model.recommend(top_k=top_k)


def test_fit_requires_impressions_column() -> None:
    model = PopularityRecommender()

    with pytest.raises(
        PopularityModelError,
        match="'impressions' column",
    ):
        model.fit(pd.DataFrame({"user_id": ["U1"]}))


def test_fit_rejects_empty_candidate_data() -> None:
    model = PopularityRecommender()

    with pytest.raises(
        PopularityModelError,
        match="At least one candidate impression",
    ):
        model.fit(pd.DataFrame({"impressions": []}))
