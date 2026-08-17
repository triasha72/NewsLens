from __future__ import annotations

from collections.abc import Iterable

from newslens.models.base import Recommendation, RecommendationModel


class DummyModel:
    def recommend(
        self,
        history_news_ids: Iterable[str],
        *,
        candidate_news_ids: Iterable[str],
        top_k: int = 10,
    ) -> list[Recommendation]:
        del history_news_ids
        return [
            Recommendation(news_id=item, score=float(top_k - i), source="dummy")
            for i, item in enumerate(list(candidate_news_ids)[:top_k])
        ]


def test_structural_protocol_accepts_valid_model() -> None:
    model = DummyModel()
    assert isinstance(model, RecommendationModel)
    assert model.recommend([], candidate_news_ids=["a", "b"], top_k=1)[0].news_id == "a"
