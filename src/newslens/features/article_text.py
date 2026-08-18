"""Leakage-aware dense article text features for neural recommenders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class ArticleTextFeatureError(ValueError):
    """Raised when article feature generation cannot be completed."""


@dataclass(frozen=True, slots=True)
class ArticleFeatureBatch:
    """Dense article features aligned with news identifiers."""

    news_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ArticleTextFeatureError(
                "Article feature values must be a two-dimensional matrix."
            )

        if len(self.news_ids) != self.values.shape[0]:
            raise ArticleTextFeatureError(
                "news_ids and feature rows must have the same length."
            )

    @property
    def article_count(self) -> int:
        """Return the number of represented articles."""

        return len(self.news_ids)

    @property
    def feature_dim(self) -> int:
        """Return the dense feature dimension."""

        return int(self.values.shape[1])

    @property
    def nonzero_mask(self) -> np.ndarray:
        """Return whether each article has nonzero learned text features."""

        return np.linalg.norm(
            self.values,
            axis=1,
        ) > 0.0

    @property
    def nonzero_article_count(self) -> int:
        """Return the number of articles with usable text features."""

        return int(
            np.count_nonzero(
                self.nonzero_mask
            )
        )

    @property
    def zero_article_count(self) -> int:
        """Return the number of zero-feature articles."""

        return (
            self.article_count
            - self.nonzero_article_count
        )

    @property
    def nonzero_fraction(self) -> float:
        """Return the fraction with usable dense text features."""

        if self.article_count == 0:
            return 0.0

        return (
            self.nonzero_article_count
            / self.article_count
        )

    def as_mapping(
        self,
        *,
        include_zero: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return news ID to dense feature vector mapping."""

        mapping: dict[
            str,
            np.ndarray,
        ] = {}

        mask = self.nonzero_mask

        for index, news_id in enumerate(
            self.news_ids
        ):
            if (
                not include_zero
                and not bool(mask[index])
            ):
                continue

            mapping[news_id] = (
                self.values[index].copy()
            )

        return mapping


class ArticleTextFeatureEncoder:
    """Fit train-only TF-IDF/SVD and transform the full article catalog.

    Both the TF-IDF vocabulary/IDF statistics and the SVD projection are
    fitted only using ``fitting_news_ids``.

    Articles outside that set may subsequently be transformed. This allows
    later candidate articles to receive features without allowing their text
    to influence the learned feature basis.
    """

    def __init__(
        self,
        *,
        max_features: int = 50_000,
        svd_components: int = 256,
        seed: int = 42,
    ) -> None:
        if (
            isinstance(max_features, bool)
            or not isinstance(
                max_features,
                int,
            )
            or max_features <= 0
        ):
            raise ArticleTextFeatureError(
                "max_features must be a positive integer."
            )

        if (
            isinstance(svd_components, bool)
            or not isinstance(
                svd_components,
                int,
            )
            or svd_components <= 0
        ):
            raise ArticleTextFeatureError(
                "svd_components must be a positive integer."
            )

        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise ArticleTextFeatureError(
                "seed must be a non-negative integer."
            )

        self.max_features = max_features
        self.svd_components = (
            svd_components
        )
        self.seed = seed

        self._vectorizer = (
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_features=max_features,
                sublinear_tf=True,
                norm="l2",
            )
        )

        self._svd = TruncatedSVD(
            n_components=(
                svd_components
            ),
            algorithm="randomized",
            n_iter=7,
            random_state=seed,
        )

        self._is_fitted = False
        self._fit_article_count = 0

    @property
    def is_fitted(self) -> bool:
        """Return whether the encoder has been fitted."""

        return self._is_fitted

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ArticleTextFeatureError(
                "Fit the article text encoder before transforming articles."
            )

    @property
    def vocabulary_size(self) -> int:
        """Return the fitted TF-IDF vocabulary size."""

        self._require_fitted()

        return len(
            self._vectorizer.vocabulary_
        )

    @property
    def fit_article_count(self) -> int:
        """Return articles allowed to influence the feature basis."""

        self._require_fitted()

        return self._fit_article_count

    @property
    def output_dim(self) -> int:
        """Return the dense article feature dimension."""

        return self.svd_components

    @property
    def explained_variance_ratio_sum(
        self,
    ) -> float:
        """Return cumulative explained variance of the SVD projection."""

        self._require_fitted()

        return float(
            self._svd
            .explained_variance_ratio_
            .sum()
        )

    @staticmethod
    def _prepare_news(
        news: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
    ]:
        required_columns = {
            "news_id",
            "title",
        }

        missing_columns = (
            required_columns.difference(
                news.columns
            )
        )

        if missing_columns:
            formatted = ", ".join(
                sorted(
                    missing_columns
                )
            )

            raise ArticleTextFeatureError(
                "Missing required article columns: "
                f"{formatted}."
            )

        if news.empty:
            raise ArticleTextFeatureError(
                "At least one article is required."
            )

        prepared = (
            news.copy(
                deep=True
            )
            .reset_index(
                drop=True
            )
        )

        for column in (
            "news_id",
            "title",
            "abstract",
            "category",
            "subcategory",
        ):
            if column not in prepared.columns:
                prepared[column] = ""

            prepared[column] = (
                prepared[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        if prepared[
            "news_id"
        ].eq("").any():
            raise ArticleTextFeatureError(
                "Article identifiers cannot be empty."
            )

        if prepared[
            "title"
        ].eq("").any():
            raise ArticleTextFeatureError(
                "Article titles cannot be empty."
            )

        duplicate_ids = prepared.loc[
            prepared[
                "news_id"
            ].duplicated(
                keep=False
            ),
            "news_id",
        ].unique()

        if len(duplicate_ids) > 0:
            preview = ", ".join(
                sorted(
                    duplicate_ids
                )[:3]
            )

            raise ArticleTextFeatureError(
                "Duplicate article identifiers "
                f"found: {preview}."
            )

        documents = (
            prepared["title"]
            + " "
            + prepared["abstract"]
            + " "
            + prepared["category"]
            + " "
            + prepared["subcategory"]
        ).str.strip()

        return (
            prepared,
            documents,
        )

    def fit(
        self,
        news: pd.DataFrame,
        *,
        fitting_news_ids: Iterable[str],
    ) -> ArticleTextFeatureEncoder:
        """Fit TF-IDF and SVD using only permitted training articles."""

        prepared, documents = (
            self._prepare_news(
                news
            )
        )

        if isinstance(
            fitting_news_ids,
            str,
        ):
            fitting_ids = set(
                fitting_news_ids.split()
            )
        else:
            fitting_ids = {
                str(news_id)
                for news_id
                in fitting_news_ids
            }

        if not fitting_ids:
            raise ArticleTextFeatureError(
                "At least one fitting article ID is required."
            )

        catalog_ids = set(
            prepared[
                "news_id"
            ]
        )

        unknown_ids = (
            fitting_ids
            - catalog_ids
        )

        if unknown_ids:
            preview = ", ".join(
                sorted(
                    unknown_ids
                )[:3]
            )

            raise ArticleTextFeatureError(
                "Fitting article IDs are missing "
                f"from the catalog: {preview}."
            )

        fitting_mask = prepared[
            "news_id"
        ].isin(
            fitting_ids
        )

        fitting_documents = (
            documents.loc[
                fitting_mask
            ]
        )

        try:
            fitting_tfidf = (
                self._vectorizer
                .fit_transform(
                    fitting_documents
                )
            )
        except ValueError as error:
            raise ArticleTextFeatureError(
                "Training article text produced no TF-IDF vocabulary."
            ) from error

        minimum_dimension = min(
            fitting_tfidf.shape
        )

        if (
            self.svd_components
            >= minimum_dimension
        ):
            raise ArticleTextFeatureError(
                "svd_components must be smaller "
                "than both the number of fitting "
                "articles and the fitted TF-IDF "
                "vocabulary size. "
                f"Received {self.svd_components} "
                f"for matrix shape "
                f"{fitting_tfidf.shape}."
            )

        self._svd.fit(
            fitting_tfidf
        )

        self._fit_article_count = int(
            fitting_mask.sum()
        )

        self._is_fitted = True

        return self

    def transform(
        self,
        news: pd.DataFrame,
    ) -> ArticleFeatureBatch:
        """Transform supplied articles using the frozen train-fitted basis."""

        self._require_fitted()

        prepared, documents = (
            self._prepare_news(
                news
            )
        )

        tfidf = (
            self._vectorizer.transform(
                documents
            )
        )

        dense = self._svd.transform(
            tfidf
        )

        dense = normalize(
            dense,
            norm="l2",
            axis=1,
            copy=False,
        )

        values = np.asarray(
            dense,
            dtype=np.float32,
        )

        return ArticleFeatureBatch(
            news_ids=tuple(
                prepared[
                    "news_id"
                ].tolist()
            ),
            values=values,
        )

    def fit_transform(
        self,
        news: pd.DataFrame,
        *,
        fitting_news_ids: Iterable[str],
    ) -> ArticleFeatureBatch:
        """Fit on training IDs and transform the supplied catalog."""

        self.fit(
            news,
            fitting_news_ids=(
                fitting_news_ids
            ),
        )

        return self.transform(
            news
        )
