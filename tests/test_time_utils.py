"""Smoke test for ``beijing_now_naive``: must return a NAIVE datetime
in Beijing time (UTC+8), so the JSON report's ``started_at`` /
``finished_at`` keep a stable string shape when we append the
``+08:00`` suffix downstream.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from llm_bench.utils.time_utils import (
    BEIJING_ISO_SUFFIX,
    BEIJING_TZ,
    beijing_now_naive,
)


def test_beijing_now_naive_is_naive_and_close_to_beijing() -> None:
    now_naive = beijing_now_naive()
    assert isinstance(now_naive, datetime)
    # Naive => no tzinfo. Important: appending "+08:00" downstream would
    # otherwise produce a misleading "+08:00+08:00" double-suffix.
    assert now_naive.tzinfo is None

    # Sanity: drift vs aware Beijing time should be < 5s.
    aware_beijing = datetime.now(BEIJING_TZ).replace(tzinfo=None)
    assert abs(aware_beijing - now_naive) < timedelta(seconds=5)


def test_beijing_now_naive_is_eight_hours_ahead_of_utc() -> None:
    now_beijing = beijing_now_naive()
    now_utc = datetime.now(UTC).replace(tzinfo=None)
    # Beijing should be ~8h ahead of UTC (allow a few seconds of jitter
    # from the two `now()` calls running back-to-back).
    delta = now_beijing - now_utc
    assert abs(delta - timedelta(hours=8)) < timedelta(seconds=5)


def test_isoformat_round_trip_has_beijing_offset() -> None:
    rendered = beijing_now_naive().isoformat() + BEIJING_ISO_SUFFIX
    # Shape: "YYYY-MM-DDTHH:MM:SS[.ffffff]+08:00"
    assert rendered.endswith("+08:00")
    assert "T" in rendered
    # Absence of trailing "Z" is the whole point — that suffix would
    # have implied UTC, which this project no longer emits.
    assert not rendered.endswith("Z")
