"""Smoke-test a running Phase-07 recommendation service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from newslens.data import load_behaviors
from newslens.retrieval.catalog import (
    RetrievalCatalog,
)


def _find_history(
    *,
    data_dir: Path,
    catalog: RetrievalCatalog,
) -> tuple[str, ...]:
    behaviors = load_behaviors(
        data_dir
        / "MINDsmall_train"
        / "behaviors.tsv"
    )

    known = set(
        catalog.news_ids
    )

    for row in behaviors.itertuples(
        index=False
    ):
        raw = str(
            row.history
        ).strip()

        if (
            not raw
            or raw.lower()
            == "nan"
        ):
            continue

        usable = tuple(
            news_id
            for news_id
            in raw.split()
            if news_id in known
        )

        if len(usable) >= 2:
            return usable[-10:]

    raise RuntimeError(
        "Could not find a usable smoke-test history."
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    catalog = RetrievalCatalog.load_npz(
        args.catalog
    )

    history = _find_history(
        data_dir=args.data_dir,
        catalog=catalog,
    )

    base_url = args.base_url.rstrip(
        "/"
    )

    with httpx.Client(
        base_url=base_url,
        timeout=30.0,
    ) as client:
        health = client.get(
            "/health"
        )

        ready = client.get(
            "/ready"
        )

        model_info = client.get(
            "/model-info"
        )

        recommendation = client.post(
            "/v1/recommend",
            headers={
                "X-Request-ID": (
                    "phase07-smoke"
                ),
            },
            json={
                "history_news_ids": (
                    list(history)
                ),
                "top_k": 10,
            },
        )

        fallback = client.post(
            "/v1/recommend",
            json={
                "history_news_ids": [
                    "__UNKNOWN_NEWS_ID__"
                ],
                "top_k": 10,
            },
        )

        metrics = client.get(
            "/metrics"
        )

    passed = (
        health.status_code == 200
        and ready.status_code == 200
        and model_info.status_code == 200
        and recommendation.status_code == 200
        and fallback.status_code == 200
        and metrics.status_code == 200
        and recommendation.json()[
            "fallback_used"
        ]
        is False
        and fallback.json()[
            "fallback_used"
        ]
        is True
        and recommendation.json()[
            "returned_count"
        ]
        == 10
        and (
            "newslens_model_ready"
            in metrics.text
        )
    )

    payload = {
        "health_status": (
            health.status_code
        ),
        "ready_status": (
            ready.status_code
        ),
        "model_info_status": (
            model_info.status_code
        ),
        "recommend_status": (
            recommendation.status_code
        ),
        "fallback_status": (
            fallback.status_code
        ),
        "metrics_status": (
            metrics.status_code
        ),
        "model_info": (
            model_info.json()
        ),
        "recommendation": {
            "fallback_used": (
                recommendation.json()[
                    "fallback_used"
                ]
            ),
            "returned_count": (
                recommendation.json()[
                    "returned_count"
                ]
            ),
            "timings": (
                recommendation.json()[
                    "timings"
                ]
            ),
        },
        "fallback": {
            "fallback_used": (
                fallback.json()[
                    "fallback_used"
                ]
            ),
            "returned_count": (
                fallback.json()[
                    "returned_count"
                ]
            ),
        },
        "passed": passed,
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

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )

    if not passed:
        raise RuntimeError(
            "Phase-07 service smoke test failed."
        )


if __name__ == "__main__":
    main()
