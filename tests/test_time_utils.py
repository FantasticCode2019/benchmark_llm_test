"""Smoke test for ``utc_now_naive``: must return a NAIVE datetime so the
JSON report's ``started_at`` / ``finished_at`` keep their ``…Z`` suffix
exactly the way ``datetime.utcnow()`` used to render them.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from llm_bench.utils.time_utils import utc_now_naive


def test_utc_now_naive_is_naive_and_close_to_utc() -> None:
    now_naive = utc_now_naive()
    assert isinstance(now_naive, datetime)
    # Naive => no tzinfo. Important: appending "Z" downstream would
    # otherwise produce a misleading "+00:00Z" double-suffix.
    assert now_naive.tzinfo is None

    # Sanity: drift vs aware UTC should be < 5s (smoke check; this test
    # mainly proves the function actually returns "now").
    aware_utc = datetime.now(UTC).replace(tzinfo=None)
    assert abs(aware_utc - now_naive) < timedelta(seconds=5)


def test_isoformat_round_trip_matches_legacy_shape() -> None:
    rendered = utc_now_naive().isoformat() + "Z"
    # Same shape as `datetime.utcnow().isoformat() + "Z"`:
    # "YYYY-MM-DDTHH:MM:SS[.ffffff]Z"
    assert rendered.endswith("Z")
    assert "+" not in rendered  # absence of "+00:00" is the whole point
    assert "T" in rendered
