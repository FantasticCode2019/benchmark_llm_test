"""Smoke tests for ``llm_bench.utils.format`` — byte / duration humanizers.

Pure-function helpers; no I/O. Anchors:
  - ``human_bytes`` boundary behavior at 1 KiB / 1 MiB / 1 GiB
  - ``fmt_duration`` formatting transitions seconds → minutes → hours
"""
from __future__ import annotations

from llm_bench.utils.format import fmt_duration, human_bytes


class TestHumanBytes:
    def test_small_bytes_stay_in_bytes(self) -> None:
        assert human_bytes(0) == "0.0 B"
        assert human_bytes(512) == "512.0 B"

    def test_unit_transitions(self) -> None:
        assert human_bytes(1024) == "1.0 KiB"
        assert human_bytes(1024 * 1024) == "1.0 MiB"
        assert human_bytes(1024 * 1024 * 1024) == "1.0 GiB"

    def test_none_renders_as_question_mark(self) -> None:
        assert human_bytes(None) == "?"


class TestFmtDuration:
    def test_sub_minute_in_seconds(self) -> None:
        assert fmt_duration(0.0) == "0.0s"
        assert fmt_duration(5.4) == "5.4s"
        assert fmt_duration(59.9) == "59.9s"

    def test_minute_range(self) -> None:
        assert fmt_duration(60) == "1m 00s"
        assert fmt_duration(123.4) == "2m 03s"

    def test_hour_range(self) -> None:
        assert fmt_duration(3600) == "1h 00m"
        assert fmt_duration(3661) == "1h 01m"
