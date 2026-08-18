"""Audit leakage-safe two-tower article feature coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from newslens.data import (
    load_behaviors,
    load_news,
    parse_impressions,
)
from newslens.evaluation.content import (
    _parse_history,
    _prepare_catalog,
    _training_vocabulary_news_ids,
)
from newslens.evaluation.split import (
    chronological_train_validation_split,
)
from newslens.features import (
    ArticleTextFeatureEncoder,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cutoff",
        default=(
            "2019-11-13T20:36:26"
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--svd-components",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    split_path = (
        args.data_dir
        / "MINDsmall_train"
    )

    print("Loading MIND-small train...")

    news = load_news(
        split_path
        / "news.tsv"
    )

    behaviors = load_behaviors(
        split_path
        / "behaviors.tsv"
    )

    print(
        "Creating chronological split..."
    )

    split = (
        chronological_train_validation_split(
            behaviors,
            validation_fraction=(
                args.validation_fraction
            ),
        )
    )

    actual_cutoff = (
        split.cutoff.isoformat()
    )

    if actual_cutoff != args.cutoff:
        raise RuntimeError(
            "Chronological cutoff mismatch: "
            f"expected {args.cutoff}, "
            f"got {actual_cutoff}."
        )

    catalog = _prepare_catalog(
        news
    )

    fitting_news_ids = (
        _training_vocabulary_news_ids(
            split.train,
            catalog,
        )
    )

    print(
        "Fitting train-only TF-IDF + SVD..."
    )

    encoder = (
        ArticleTextFeatureEncoder(
            max_features=(
                args.max_features
            ),
            svd_components=(
                args.svd_components
            ),
            seed=args.seed,
        )
    )

    feature_batch = (
        encoder.fit_transform(
            news,
            fitting_news_ids=(
                fitting_news_ids
            ),
        )
    )

    norms = np.linalg.norm(
        feature_batch.values,
        axis=1,
    )

    supported_news_ids = {
        news_id
        for news_id, norm
        in zip(
            feature_batch.news_ids,
            norms,
            strict=True,
        )
        if norm > 0.0
    }

    fitting_news_ids = set(
        fitting_news_ids
    )

    candidate_occurrences = 0
    supported_candidate_occurrences = 0

    clicked_occurrences = 0
    supported_clicked_occurrences = 0

    not_fit_candidate_occurrences = 0
    supported_not_fit_candidate_occurrences = 0

    not_fit_clicked_occurrences = 0
    supported_not_fit_clicked_occurrences = 0

    history_occurrences = 0
    supported_history_occurrences = 0

    impressions_with_history = 0
    impressions_with_usable_history = 0

    impressions_with_any_candidate_support = 0
    impressions_with_all_candidate_support = 0
    impressions_with_at_least_k_supported = 0

    zero_supported_candidate_impressions = 0

    unique_validation_candidates: set[
        str
    ] = set()

    unique_not_fit_validation_candidates: set[
        str
    ] = set()

    for row in split.validation.itertuples(
        index=False
    ):
        history_ids = _parse_history(
            row.history
        )

        if history_ids:
            impressions_with_history += 1

        history_occurrences += len(
            history_ids
        )

        supported_history_count = sum(
            news_id
            in supported_news_ids
            for news_id
            in history_ids
        )

        supported_history_occurrences += (
            supported_history_count
        )

        if supported_history_count > 0:
            impressions_with_usable_history += 1

        parsed = parse_impressions(
            str(
                row.impressions
            )
        )

        candidate_ids = [
            news_id
            for news_id, _
            in parsed
        ]

        if len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise RuntimeError(
                "Validation impression "
                f"{row.impression_id} "
                "contains duplicate candidate IDs."
            )

        unique_validation_candidates.update(
            candidate_ids
        )

        candidate_occurrences += len(
            candidate_ids
        )

        supported_candidate_count = sum(
            news_id
            in supported_news_ids
            for news_id
            in candidate_ids
        )

        supported_candidate_occurrences += (
            supported_candidate_count
        )

        if supported_candidate_count == 0:
            zero_supported_candidate_impressions += 1
        else:
            impressions_with_any_candidate_support += 1

        if (
            supported_candidate_count
            == len(candidate_ids)
        ):
            impressions_with_all_candidate_support += 1

        if (
            supported_candidate_count
            >= min(
                args.k,
                len(candidate_ids),
            )
        ):
            impressions_with_at_least_k_supported += 1

        for news_id, label in parsed:
            is_supported = (
                news_id
                in supported_news_ids
            )

            is_fit_article = (
                news_id
                in fitting_news_ids
            )

            if not is_fit_article:
                unique_not_fit_validation_candidates.add(
                    news_id
                )

                not_fit_candidate_occurrences += 1

                if is_supported:
                    supported_not_fit_candidate_occurrences += 1

            if label == 1:
                clicked_occurrences += 1

                if is_supported:
                    supported_clicked_occurrences += 1

                if not is_fit_article:
                    not_fit_clicked_occurrences += 1

                    if is_supported:
                        supported_not_fit_clicked_occurrences += 1

    def fraction(
        numerator: int,
        denominator: int,
    ) -> float:
        if denominator == 0:
            return 0.0

        return (
            numerator
            / denominator
        )

    payload = {
        "experiment": (
            "phase03_two_tower_feature_coverage_audit"
        ),
        "protocol": {
            "dataset": (
                "MINDsmall_train"
            ),
            "cutoff_timestamp": (
                actual_cutoff
            ),
            "training_impressions": (
                len(split.train)
            ),
            "validation_impressions": (
                len(split.validation)
            ),
            "official_dev_used": False,
            "validation_fraction": (
                args.validation_fraction
            ),
            "k": args.k,
            "seed": args.seed,
        },
        "encoder": {
            "max_features": (
                args.max_features
            ),
            "svd_components": (
                args.svd_components
            ),
            "fit_article_count": (
                encoder.fit_article_count
            ),
            "indexed_article_count": (
                feature_batch.article_count
            ),
            "tfidf_vocabulary_size": (
                encoder.vocabulary_size
            ),
            "svd_explained_variance_ratio_sum": (
                encoder
                .explained_variance_ratio_sum
            ),
            "nonzero_article_count": (
                feature_batch
                .nonzero_article_count
            ),
            "zero_article_count": (
                feature_batch
                .zero_article_count
            ),
            "nonzero_article_fraction": (
                feature_batch
                .nonzero_fraction
            ),
        },
        "validation_candidate_coverage": {
            "candidate_occurrences": (
                candidate_occurrences
            ),
            "supported_candidate_occurrences": (
                supported_candidate_occurrences
            ),
            "supported_candidate_fraction": (
                fraction(
                    supported_candidate_occurrences,
                    candidate_occurrences,
                )
            ),
            "unique_validation_candidates": (
                len(
                    unique_validation_candidates
                )
            ),
            "impressions_with_any_candidate_support": (
                impressions_with_any_candidate_support
            ),
            "impressions_with_all_candidate_support": (
                impressions_with_all_candidate_support
            ),
            "impressions_with_at_least_k_supported": (
                impressions_with_at_least_k_supported
            ),
            "zero_supported_candidate_impressions": (
                zero_supported_candidate_impressions
            ),
        },
        "validation_click_coverage": {
            "clicked_occurrences": (
                clicked_occurrences
            ),
            "supported_clicked_occurrences": (
                supported_clicked_occurrences
            ),
            "supported_clicked_fraction": (
                fraction(
                    supported_clicked_occurrences,
                    clicked_occurrences,
                )
            ),
        },
        "non_fit_article_coverage": {
            "unique_not_fit_validation_candidates": (
                len(
                    unique_not_fit_validation_candidates
                )
            ),
            "candidate_occurrences": (
                not_fit_candidate_occurrences
            ),
            "supported_candidate_occurrences": (
                supported_not_fit_candidate_occurrences
            ),
            "supported_candidate_fraction": (
                fraction(
                    supported_not_fit_candidate_occurrences,
                    not_fit_candidate_occurrences,
                )
            ),
            "clicked_occurrences": (
                not_fit_clicked_occurrences
            ),
            "supported_clicked_occurrences": (
                supported_not_fit_clicked_occurrences
            ),
            "supported_clicked_fraction": (
                fraction(
                    supported_not_fit_clicked_occurrences,
                    not_fit_clicked_occurrences,
                )
            ),
        },
        "history_coverage": {
            "history_occurrences": (
                history_occurrences
            ),
            "supported_history_occurrences": (
                supported_history_occurrences
            ),
            "supported_history_fraction": (
                fraction(
                    supported_history_occurrences,
                    history_occurrences,
                )
            ),
            "impressions_with_history": (
                impressions_with_history
            ),
            "impressions_with_usable_history": (
                impressions_with_usable_history
            ),
            "usable_history_fraction_among_nonempty": (
                fraction(
                    impressions_with_usable_history,
                    impressions_with_history,
                )
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 80)
    print(
        "PHASE 03B FEATURE COVERAGE"
    )
    print("=" * 80)

    print(
        "Feature articles:",
        feature_batch.nonzero_article_count,
        "/",
        feature_batch.article_count,
        (
            f"({feature_batch.nonzero_fraction:.2%})"
        ),
    )

    print(
        "Validation candidate support:",
        (
            f"{fraction(
                supported_candidate_occurrences,
                candidate_occurrences,
            ):.2%}"
        ),
    )

    print(
        "Validation click support:",
        (
            f"{fraction(
                supported_clicked_occurrences,
                clicked_occurrences,
            ):.2%}"
        ),
    )

    print(
        "Non-fit candidate support:",
        (
            f"{fraction(
                supported_not_fit_candidate_occurrences,
                not_fit_candidate_occurrences,
            ):.2%}"
        ),
    )

    print(
        "Non-fit clicked-item support:",
        (
            f"{fraction(
                supported_not_fit_clicked_occurrences,
                not_fit_clicked_occurrences,
            ):.2%}"
        ),
    )

    print(
        "Usable nonempty histories:",
        (
            f"{fraction(
                impressions_with_usable_history,
                impressions_with_history,
            ):.2%}"
        ),
    )

    print(
        "Validation impressions with >=k "
        "supported candidates:",
        impressions_with_at_least_k_supported,
        "/",
        len(split.validation),
        (
            f"({fraction(
                impressions_with_at_least_k_supported,
                len(split.validation),
            ):.2%})"
        ),
    )

    print()
    print(
        f"Wrote report to {args.output}"
    )


if __name__ == "__main__":
    main()
