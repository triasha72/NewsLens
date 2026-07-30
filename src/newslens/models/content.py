"""Content-based recommendations from user reading histories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentModelError(ValueError):
    """Raised when content-model input is invalid."""


class ContentModelNotFittedError(RuntimeError):
    """Raised when recommendations are requested before fitting."""


class ColdStartUserError(RuntimeError):
    """Raised when a user history cannot produce a content profile."""


@dataclass(frozen=True)
class ContentRecommendation:
    """One ranked content-based recommendation."""

    news_id: str
    title: str
    category: str
    score: float


class ContentBasedRecommender:
    """Recommend articles similar to a user's reading history."""

    def __init__(self, *, max_features: int = 50_000) -> None:
        if max_features <= 0:
            raise ContentModelError("max_features must be greater than zero.")

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        self._news: pd.DataFrame | None = None
        self._matrix: csr_matrix | None = None
        self._news_index: dict[str, int] | None = None
        self._vocabulary_article_count = 0

    @property
    def is_fitted(self) -> bool:
        """Return whether the content model is fitted."""

        return self._news is not None and self._matrix is not None and self._news_index is not None

    @property
    def vocabulary_size(self) -> int:
        """Return the number of fitted TF-IDF terms."""

        self._require_fitted()
        return len(self._vectorizer.vocabulary_)

    @property
    def indexed_article_count(self) -> int:
        """Return the number of indexed articles."""

        self._require_fitted()
        assert self._news is not None
        return len(self._news)

    @property
    def vocabulary_article_count(self) -> int:
        """Return the number of articles used to fit the vocabulary."""

        self._require_fitted()
        return self._vocabulary_article_count

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ContentModelNotFittedError(
                "Fit the content model before requesting recommendations."
            )

    def fit(
        self,
        news: pd.DataFrame,
        *,
        vocabulary_news_ids: Iterable[str] | None = None,
    ) -> ContentBasedRecommender:
        """Fit TF-IDF using training articles and index all supplied articles.

        When ``vocabulary_news_ids`` is supplied, only those articles influence
        the learned vocabulary and inverse-document frequencies. All articles
        are subsequently transformed so later candidates can still be ranked.
        """

        required_columns = {"news_id", "title"}
        missing_columns = required_columns.difference(news.columns)

        if missing_columns:
            formatted = ", ".join(sorted(missing_columns))
            raise ContentModelError(f"Missing required article columns: {formatted}.")

        if news.empty:
            raise ContentModelError("At least one article is required.")

        indexed_news = news.copy(deep=True).reset_index(drop=True)

        for column in (
            "news_id",
            "title",
            "abstract",
            "category",
            "subcategory",
        ):
            if column not in indexed_news.columns:
                indexed_news[column] = ""

            indexed_news[column] = indexed_news[column].fillna("").astype(str).str.strip()

        if indexed_news["news_id"].eq("").any():
            raise ContentModelError("Article identifiers cannot be empty.")

        if indexed_news["title"].eq("").any():
            raise ContentModelError("Article titles cannot be empty.")

        duplicate_ids = indexed_news.loc[
            indexed_news["news_id"].duplicated(keep=False),
            "news_id",
        ].unique()

        if len(duplicate_ids) > 0:
            raise ContentModelError(
                f"Duplicate article identifiers found: {', '.join(duplicate_ids)}"
            )

        documents = (
            indexed_news["title"]
            + " "
            + indexed_news["abstract"]
            + " "
            + indexed_news["category"]
            + " "
            + indexed_news["subcategory"]
        ).str.strip()

        if vocabulary_news_ids is None:
            vocabulary_mask = pd.Series(
                True,
                index=indexed_news.index,
            )
        else:
            selected_ids = set(vocabulary_news_ids)
            vocabulary_mask = indexed_news["news_id"].isin(selected_ids)

        vocabulary_documents = documents.loc[vocabulary_mask]

        if vocabulary_documents.empty:
            raise ContentModelError("No vocabulary articles were found in the news catalog.")

        try:
            self._vectorizer.fit(vocabulary_documents)
        except ValueError as error:
            raise ContentModelError(
                "The training article text produced no TF-IDF vocabulary."
            ) from error

        self._news = indexed_news
        self._matrix = self._vectorizer.transform(documents).tocsr()
        self._news_index = {news_id: index for index, news_id in enumerate(indexed_news["news_id"])}
        self._vocabulary_article_count = len(vocabulary_documents)

        return self

    def recommend(
        self,
        history_news_ids: Iterable[str],
        *,
        candidate_news_ids: Iterable[str] | None = None,
        top_k: int = 10,
        exclude_history: bool = True,
    ) -> list[ContentRecommendation]:
        """Rank candidates by similarity to a user's history profile."""

        self._require_fitted()

        if top_k <= 0:
            raise ContentModelError("top_k must be greater than zero.")

        assert self._news is not None
        assert self._matrix is not None
        assert self._news_index is not None

        if isinstance(history_news_ids, str):
            history_ids = history_news_ids.split()
        else:
            history_ids = list(history_news_ids)

        known_history_indices = [
            self._news_index[news_id] for news_id in history_ids if news_id in self._news_index
        ]

        if not known_history_indices:
            raise ColdStartUserError("The user has no known articles in their history.")

        history_matrix = self._matrix[known_history_indices]
        profile_vector = csr_matrix(history_matrix.mean(axis=0))

        if profile_vector.nnz == 0:
            raise ColdStartUserError("The user history contains no usable training vocabulary.")

        if candidate_news_ids is None:
            candidates = list(self._news_index)
        elif isinstance(candidate_news_ids, str):
            candidates = candidate_news_ids.split()
        else:
            candidates = list(candidate_news_ids)

        history_set = set(history_ids)

        candidate_ids = sorted(
            {
                news_id
                for news_id in candidates
                if news_id in self._news_index
                and (not exclude_history or news_id not in history_set)
            }
        )

        if not candidate_ids:
            return []

        candidate_indices = [self._news_index[news_id] for news_id in candidate_ids]

        similarities = cosine_similarity(
            profile_vector,
            self._matrix[candidate_indices],
        ).ravel()

        scored_candidates = list(
            zip(
                candidate_ids,
                candidate_indices,
                similarities,
                strict=True,
            )
        )

        ranked = sorted(
            scored_candidates,
            key=lambda item: (
                -float(item[2]),
                item[0],
            ),
        )[:top_k]

        return [
            ContentRecommendation(
                news_id=news_id,
                title=self._news.loc[index, "title"],
                category=self._news.loc[index, "category"],
                score=float(score),
            )
            for news_id, index, score in ranked
        ]
