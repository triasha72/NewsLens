from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from newslens.cli import main
from newslens.data import (
    WAREHOUSE_SCHEMA_VERSION,
    WarehouseExistsError,
    WarehouseValidationError,
    article_training_features,
    build_mind_warehouse_from_paths,
    summarize_warehouse,
)


def _write_mind_fixture(data_dir: Path) -> tuple[Path, Path]:
    split_path = data_dir / "MINDsmall_train"
    split_path.mkdir(parents=True)

    news_path = split_path / "news.tsv"
    news_path.write_text(
        "N1\tnews\tlocal\tCity opens a park\tPublic space expands\t"
        "https://example.com/1\t[]\t[]\n"
        "N2\tsports\tfootball\tTeam wins final\tA close match\t"
        "https://example.com/2\t[]\t[]\n"
        "N3\tfinance\tmarkets\tMarkets close higher\tStocks rise\t"
        "https://example.com/3\t[]\t[]\n",
        encoding="utf-8",
    )

    behaviors_path = split_path / "behaviors.tsv"
    behaviors_path.write_text(
        "I1\tU1\t01/01/2020 12:00:00 AM\t\tN1-1 N2-0\n"
        "I2\tU2\t01/03/2020 12:00:00 AM\tN1 N2\tN1-0 N2-1 N3-0\n",
        encoding="utf-8",
    )
    return news_path, behaviors_path


def test_build_warehouse_normalizes_validated_mind_records(tmp_path: Path) -> None:
    news_path, behaviors_path = _write_mind_fixture(tmp_path / "data")
    database = tmp_path / "warehouses" / "mind.duckdb"

    result = build_mind_warehouse_from_paths(
        news_path,
        behaviors_path,
        database,
        split="train",
    )

    assert result.schema_version == WAREHOUSE_SCHEMA_VERSION
    assert result.articles == 3
    assert result.behavior_events == 2
    assert result.history_events == 2
    assert result.candidate_interactions == 5
    assert result.clicks == 2
    assert result.news_sha256 is not None
    assert len(result.news_sha256) == 64
    assert result.behaviors_sha256 is not None
    assert len(result.behaviors_sha256) == 64

    with duckdb.connect(str(database), read_only=True) as connection:
        history = connection.execute(
            """
            SELECT history_position, news_id
            FROM user_history
            WHERE impression_id = 'I2'
            ORDER BY history_position
            """
        ).fetchall()
        candidates = connection.execute(
            """
            SELECT candidate_position, news_id, clicked
            FROM candidate_interactions
            WHERE impression_id = 'I1'
            ORDER BY candidate_position
            """
        ).fetchall()
        engagement = connection.execute(
            """
            SELECT candidate_exposures, clicks, click_through_rate
            FROM article_engagement
            WHERE news_id = 'N2'
            """
        ).fetchone()

    assert history == [(1, "N1"), (2, "N2")]
    assert candidates == [(1, "N1", True), (2, "N2", False)]
    assert engagement == (2, 1, 0.5)


def test_warehouse_summary_is_derived_from_sql(tmp_path: Path) -> None:
    news_path, behaviors_path = _write_mind_fixture(tmp_path / "data")
    database = tmp_path / "mind.duckdb"
    build_mind_warehouse_from_paths(
        news_path,
        behaviors_path,
        database,
        split="train",
    )

    summary = summarize_warehouse(database)

    assert summary.schema_version == WAREHOUSE_SCHEMA_VERSION
    assert summary.split == "train"
    assert summary.articles == 3
    assert summary.behavior_events == 2
    assert summary.users == 2
    assert summary.history_events == 2
    assert summary.candidate_interactions == 5
    assert summary.clicks == 2
    assert summary.first_event_timestamp == "2020-01-01T00:00:00"
    assert summary.last_event_timestamp == "2020-01-03T00:00:00"


def test_training_features_exclude_events_at_or_after_cutoff(tmp_path: Path) -> None:
    news_path, behaviors_path = _write_mind_fixture(tmp_path / "data")
    database = tmp_path / "mind.duckdb"
    build_mind_warehouse_from_paths(
        news_path,
        behaviors_path,
        database,
        split="train",
    )

    features = article_training_features(database, "2020-01-02T00:00:00")
    indexed = features.set_index("news_id")

    assert indexed.loc["N1", "candidate_exposures"] == 1
    assert indexed.loc["N1", "clicks"] == 1
    assert indexed.loc["N1", "exposed_users"] == 1
    assert indexed.loc["N1", "click_through_rate"] == 1.0
    assert indexed.loc["N2", "candidate_exposures"] == 1
    assert indexed.loc["N2", "clicks"] == 0
    assert indexed.loc["N3", "candidate_exposures"] == 0
    assert indexed.loc["N3", "clicks"] == 0


def test_atomic_rebuild_preserves_existing_database_after_invalid_source(
    tmp_path: Path,
) -> None:
    news_path, behaviors_path = _write_mind_fixture(tmp_path / "data")
    database = tmp_path / "mind.duckdb"
    build_mind_warehouse_from_paths(
        news_path,
        behaviors_path,
        database,
        split="train",
    )

    with pytest.raises(WarehouseExistsError, match="already exists"):
        build_mind_warehouse_from_paths(
            news_path,
            behaviors_path,
            database,
            split="train",
        )

    behaviors_path.write_text(
        "I3\tU3\t01/04/2020 12:00:00 AM\tN1\tN999-1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        WarehouseValidationError,
        match="candidate articles: N999",
    ):
        build_mind_warehouse_from_paths(
            news_path,
            behaviors_path,
            database,
            split="train",
            overwrite=True,
        )

    assert summarize_warehouse(database).behavior_events == 2


def test_warehouse_cli_builds_summarizes_and_exports_features(
    tmp_path: Path,
    capsys: object,
) -> None:
    data_dir = tmp_path / "data"
    _write_mind_fixture(data_dir)
    database = tmp_path / "warehouses" / "mind.duckdb"
    features_path = tmp_path / "reports" / "features.csv"

    main(
        [
            "build-warehouse",
            "--data-dir",
            str(data_dir),
            "--split",
            "train",
            "--output",
            str(database),
        ]
    )
    build_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "DuckDB warehouse written" in build_output

    main(["warehouse-summary", "--database", str(database)])
    summary_output = capsys.readouterr().out  # type: ignore[attr-defined]
    summary = json.loads(summary_output)
    assert summary["articles"] == 3
    assert summary["candidate_interactions"] == 5

    main(
        [
            "export-training-features",
            "--database",
            str(database),
            "--cutoff-timestamp",
            "2020-01-02T00:00:00",
            "--output",
            str(features_path),
        ]
    )
    feature_output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "Leakage-safe SQL feature snapshot written" in feature_output
    assert features_path.is_file()
    assert "candidate_exposures" in features_path.read_text(encoding="utf-8")
