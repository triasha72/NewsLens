"""TF-IDF article-search baseline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ArticleSearchError(ValueError):
    """Raised when article-search input is invalid."""


class ArticleSearchNotFittedError(RuntimeError):
    """Raised when search is requested before fitting."""


@dataclass(frozen=True)
class ArticleSearchResult:
    """One ranked article-search result."""

    news_id: str
    title: str
    category: str
    score: float


class TfidfArticleSearch:
    """Search articles using TF-IDF text vectors and cosine similarity."""

    def __init__(self, *, max_features: int = 50_000) -> None:
        if max_features <= 0:
            raise ArticleSearchError("max_features must be greater than zero.")

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

    @property
    def is_fitted(self) -> bool:
        """Return whether the search index has been fitted."""

        return self._news is not None and self._matrix is not None

    @property
    def vocabulary_size(self) -> int:
        """Return the fitted TF-IDF vocabulary size."""

        self._require_fitted()
        return len(self._vectorizer.vocabulary_)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ArticleSearchNotFittedError("Fit the TF-IDF search model before searching.")

    def fit(self, news: pd.DataFrame) -> TfidfArticleSearch:
        """Build a TF-IDF index from article metadata."""

        required_columns = {"news_id", "title"}
        missing_columns = required_columns.difference(news.columns)

        if missing_columns:
            formatted = ", ".join(sorted(missing_columns))
            raise ArticleSearchError(f"Missing required article columns: {formatted}.")

        if news.empty:
            raise ArticleSearchError("At least one article is required.")

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
            raise ArticleSearchError("Article identifiers cannot be empty.")

        if indexed_news["title"].eq("").any():
            raise ArticleSearchError("Article titles cannot be empty.")

        duplicate_ids = indexed_news.loc[
            indexed_news["news_id"].duplicated(keep=False),
            "news_id",
        ].unique()

        if len(duplicate_ids) > 0:
            raise ArticleSearchError(
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

        try:
            matrix = self._vectorizer.fit_transform(documents)
        except ValueError as error:
            raise ArticleSearchError(
                "The article text did not produce a TF-IDF vocabulary."
            ) from error

        self._news = indexed_news
        self._matrix = matrix.tocsr()

        return self

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        exclude_news_ids: Iterable[str] = (),
    ) -> list[ArticleSearchResult]:
        """Return articles ranked by cosine similarity to a query."""

        self._require_fitted()

        if not query.strip():
            raise ArticleSearchError("The search query cannot be empty.")

        if top_k <= 0:
            raise ArticleSearchError("top_k must be greater than zero.")

        assert self._news is not None
        assert self._matrix is not None

        query_vector = self._vectorizer.transform([query])

        if query_vector.nnz == 0:
            return []

        similarities = cosine_similarity(
            query_vector,
            self._matrix,
        ).ravel()

        excluded = set(exclude_news_ids)

        candidate_indices = [
            index
            for index, score in enumerate(similarities)
            if score > 0.0 and self._news.loc[index, "news_id"] not in excluded
        ]

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: (
                -float(similarities[index]),
                self._news.loc[index, "news_id"],
            ),
        )[:top_k]

        return [
            ArticleSearchResult(
                news_id=self._news.loc[index, "news_id"],
                title=self._news.loc[index, "title"],
                category=self._news.loc[index, "category"],
                score=float(similarities[index]),
            )
            for index in ranked_indices
        ]
