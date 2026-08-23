"""Tests for deterministic real-time query understanding."""

from __future__ import annotations

import pytest

from newslens.realtime import understand_query


def test_understand_query_extracts_freshness_category_and_entity() -> None:
    intent = understand_query("latest Apple earnings")

    assert intent.normalized_query == "latest Apple earnings"
    assert intent.category == "business"
    assert intent.entity == "Apple"
    assert intent.prefers_freshness is True


def test_understand_query_handles_plain_topic_search() -> None:
    intent = understand_query("mars rover mission")

    assert intent.category == "science"
    assert intent.entity is None
    assert intent.prefers_freshness is False


def test_understand_query_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty"):
        understand_query("   ")
