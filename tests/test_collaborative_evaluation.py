from pathlib import Path

import duckdb

from newslens.evaluation.collaborative import (
    evaluate_collaborative_model,
)


def _build_test_warehouse(path: Path) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE articles (
                news_id VARCHAR PRIMARY KEY
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE behavior_events (
                impression_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                event_timestamp TIMESTAMP NOT NULL,
                history_size UINTEGER NOT NULL,
                candidate_count UINTEGER NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE candidate_interactions (
                impression_id VARCHAR NOT NULL,
                candidate_position UINTEGER NOT NULL,
                news_id VARCHAR NOT NULL,
                clicked BOOLEAN NOT NULL,
                PRIMARY KEY (
                    impression_id,
                    candidate_position
                )
            )
            """
        )

        connection.executemany(
            "INSERT INTO articles VALUES (?)",
            [
                ("N1",),
                ("N2",),
                ("N3",),
                ("N4",),
                ("N5",),
                ("N6",),
            ],
        )

        connection.executemany(
            """
            INSERT INTO behavior_events
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    "I1",
                    "U1",
                    "2019-11-13 10:00:00",
                    2,
                    2,
                ),
                (
                    "I2",
                    "U2",
                    "2019-11-13 11:00:00",
                    2,
                    2,
                ),
                (
                    "I3",
                    "U1",
                    "2019-11-14 10:00:00",
                    2,
                    3,
                ),
                (
                    "I4",
                    "U3",
                    "2019-11-14 11:00:00",
                    0,
                    2,
                ),
            ],
        )

        connection.executemany(
            """
            INSERT INTO candidate_interactions
            VALUES (?, ?, ?, ?)
            """,
            [
                ("I1", 0, "N1", True),
                ("I1", 1, "N2", False),
                ("I2", 0, "N3", True),
                ("I2", 1, "N4", False),
                ("I3", 0, "N1", True),
                ("I3", 1, "N2", False),
                ("I3", 2, "N6", False),
                ("I4", 0, "N1", True),
                ("I4", 1, "N2", False),
            ],
        )


def test_collaborative_evaluation_tracks_cold_start(
    tmp_path: Path,
) -> None:
    database = tmp_path / "evaluation.duckdb"
    _build_test_warehouse(database)

    report, examples = evaluate_collaborative_model(
        database,
        cutoff_timestamp="2019-11-14T00:00:00",
        k=2,
        embedding_dim=8,
        epochs=5,
        batch_size=2,
        seed=42,
    )

    assert report.training_triples == 2
    assert report.training_users == 2
    assert report.training_items == 4

    assert report.validation_impressions == 2
    assert report.candidate_occurrences == 5

    assert report.model_known_candidate_occurrences == 4
    assert report.unknown_candidate_occurrences == 1

    assert report.known_user_impressions == 1
    assert report.cold_start_user_impressions == 1
    assert report.cold_start_user_fraction == 0.5

    assert report.clicked_item_occurrences == 2
    assert report.unknown_clicked_item_occurrences == 0

    assert report.metrics.total_impressions == 2
    assert report.metrics.empty_ranking_impressions == 1

    assert len(examples) == 2
    assert examples[1].ranked_items == ()
