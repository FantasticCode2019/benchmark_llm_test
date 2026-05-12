"""Smoke tests for ``llm_bench.utils.tokens`` — fallback token estimation
and the lenient ``ms → s`` / ``→ float`` coercers used by the
OpenAI-compat benchmark path when the server omits ``usage`` / ``timings``.
"""
from __future__ import annotations

from llm_bench.utils.tokens import ms_to_seconds, rough_token_count, to_float


class TestRoughTokenCount:
    def test_empty_string_is_zero(self) -> None:
        assert rough_token_count("") == 0

    def test_cjk_chars_counted_individually(self) -> None:
        # 4 CJK chars + 0 latin runs.
        assert rough_token_count("人工智能") == 4

    def test_latin_grouped_as_runs(self) -> None:
        # "hello world" -> 2 runs (whitespace boundary), zero CJK.
        assert rough_token_count("hello world") == 2

    def test_mixed_cjk_and_latin(self) -> None:
        # 2 CJK ("你好") + 1 latin run ("world") -> 3.
        assert rough_token_count("你好 world") == 3


class TestCoercion:
    def test_ms_to_seconds_with_int(self) -> None:
        assert ms_to_seconds(1500) == 1.5

    def test_ms_to_seconds_with_none_and_garbage(self) -> None:
        assert ms_to_seconds(None) == 0.0
        assert ms_to_seconds("not a number") == 0.0

    def test_to_float_round_trip(self) -> None:
        assert to_float(3.14) == 3.14
        assert to_float("2.5") == 2.5
        assert to_float(None) == 0.0
        assert to_float("nan-ish") == 0.0
