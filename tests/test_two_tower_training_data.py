import pandas as pd

from newslens.data.two_tower_training import (
    build_two_tower_positive_examples,
)


def test_builds_one_example_per_usable_positive() -> None:
    behaviors = pd.DataFrame(
        [
            {
                "impression_id": "i1",
                "history": "h1 h2 h3",
                "impressions": "p1-1 p2-1 n1-0",
            },
            {
                "impression_id": "i2",
                "history": "",
                "impressions": "p3-1 n2-0",
            },
        ]
    )

    result = (
        build_two_tower_positive_examples(
            behaviors,
            available_news_ids={
                "h1",
                "h2",
                "h3",
                "p1",
                "p2",
                "p3",
                "n1",
                "n2",
            },
            max_history_length=2,
        )
    )

    assert result.usable_example_count == 2

    assert [
        example.positive_news_id
        for example in result.examples
    ] == [
        "p1",
        "p2",
    ]

    assert all(
        example.history_news_ids
        == (
            "h2",
            "h3",
        )
        for example in result.examples
    )

    assert (
        result.truncated_history_impressions
        == 1
    )

    assert (
        result.empty_history_impressions
        == 1
    )


def test_unavailable_history_is_not_trainable() -> None:
    behaviors = pd.DataFrame(
        [
            {
                "impression_id": "i1",
                "history": "unknown1 unknown2",
                "impressions": "p1-1 n1-0",
            }
        ]
    )

    try:
        build_two_tower_positive_examples(
            behaviors,
            available_news_ids={
                "p1",
                "n1",
            },
        )
    except ValueError as error:
        assert (
            "No usable two-tower training examples"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected no-usable-example error."
        )


def test_unavailable_positive_is_counted() -> None:
    behaviors = pd.DataFrame(
        [
            {
                "impression_id": "i1",
                "history": "h1",
                "impressions": "missing-1 p1-1 n1-0",
            }
        ]
    )

    result = (
        build_two_tower_positive_examples(
            behaviors,
            available_news_ids={
                "h1",
                "p1",
                "n1",
            },
        )
    )

    assert result.usable_example_count == 1
    assert result.positive_click_occurrences == 2
    assert result.positive_without_features == 1
    assert result.skipped_positive_occurrences == 1
