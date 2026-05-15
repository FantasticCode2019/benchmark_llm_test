"""Tests for the chart-lifecycle helpers in
``llm_bench.core.lifecycle``.

The full state machine (``ensure_installed`` and friends) needs a
running olares-cli to exercise end-to-end, but the smaller pure
helpers — install-status JSON parsing, the default-env merge, and
the ``MarketInstallFailed`` plumbing — are easy to unit-test and pin
the contract callers (orchestrator + ollama_multi_bench) rely on.
"""
from __future__ import annotations

from llm_bench.core.lifecycle import (
    _DEFAULT_INSTALL_ENVS,
    _merge_install_envs,
    _parse_install_status,
)


class TestMergeInstallEnvs:
    """Pin the contract that:
      * ``OLARES_USER_HUGGINGFACE_SERVICE`` is always present by default.
      * Caller entries with the same KEY override the default value.
      * KEY=VALUE entries are deduplicated by key (last write wins).
      * Strings without an ``=`` pass through verbatim and are NOT
        deduped (callers occasionally use this for arbitrary --env
        flags).
      * Non-strings are silently skipped.
    """

    def test_default_huggingface_service_present_when_envs_none(self) -> None:
        merged = _merge_install_envs(None)
        assert ("OLARES_USER_HUGGINGFACE_SERVICE=https://huggingface.co"
                in merged)

    def test_default_huggingface_service_present_when_envs_empty(self) -> None:
        # The most common path: ollama_multi_bench passes [].
        merged = _merge_install_envs([])
        assert merged == [
            "OLARES_USER_HUGGINGFACE_SERVICE=https://huggingface.co"]

    def test_caller_can_override_default_value(self) -> None:
        # Operator points the chart at a mirror — caller value wins.
        merged = _merge_install_envs(
            ["OLARES_USER_HUGGINGFACE_SERVICE=https://hf-mirror.com"])
        assert merged == [
            "OLARES_USER_HUGGINGFACE_SERVICE=https://hf-mirror.com"]

    def test_caller_can_blank_default(self) -> None:
        # Forwarding an empty value lets a caller explicitly opt out
        # of the default URL without forking lifecycle.py.
        merged = _merge_install_envs(["OLARES_USER_HUGGINGFACE_SERVICE="])
        assert merged == ["OLARES_USER_HUGGINGFACE_SERVICE="]

    def test_caller_extras_appended_after_defaults(self) -> None:
        merged = _merge_install_envs(["FOO=bar", "BAZ=qux"])
        assert merged == [
            "OLARES_USER_HUGGINGFACE_SERVICE=https://huggingface.co",
            "FOO=bar",
            "BAZ=qux",
        ]

    def test_caller_dedupes_within_own_list(self) -> None:
        # Two values for the same KEY in the caller list -> last wins.
        merged = _merge_install_envs(["FOO=one", "FOO=two"])
        assert merged.count("FOO=one") == 0
        assert "FOO=two" in merged

    def test_passthrough_for_entries_without_equals(self) -> None:
        # Rare path: a caller forwards a flag that doesn't fit
        # KEY=VALUE. We don't try to parse it, just append it
        # verbatim after the keyed block.
        merged = _merge_install_envs(["--no-defaults"])
        assert merged == [
            "OLARES_USER_HUGGINGFACE_SERVICE=https://huggingface.co",
            "--no-defaults",
        ]

    def test_non_string_entries_silently_dropped(self) -> None:
        # Defensive: a caller mistakenly passing None / int / dict
        # in the list shouldn't crash market_install.
        merged = _merge_install_envs([None, 42, {"a": 1}, "OK=1"])
        assert merged == [
            "OLARES_USER_HUGGINGFACE_SERVICE=https://huggingface.co",
            "OK=1",
        ]

    def test_default_constant_is_immutable_tuple(self) -> None:
        # If somebody accidentally turns this into a list later,
        # callers in long-running processes could mutate the global
        # default. Tuple keeps that footgun closed.
        assert isinstance(_DEFAULT_INSTALL_ENVS, tuple)


class TestParseInstallStatus:
    """``market install -o json`` parsing: the dict-or-None contract
    `_install_one` (in ``ollama_multi_bench``) leans on for the
    ``model_exists`` tri-state.
    """

    def test_returns_dict_for_success_payload(self) -> None:
        raw = ('{"status":"success","finalState":"running",'
               '"finalOpType":"install"}')
        out = _parse_install_status(raw)
        assert out == {"status": "success", "finalState": "running",
                       "finalOpType": "install"}

    def test_unwraps_single_element_list(self) -> None:
        # Some CLI versions wrap the single-app result in a list.
        out = _parse_install_status('[{"status":"failed"}]')
        assert out == {"status": "failed"}

    def test_returns_none_for_empty(self) -> None:
        assert _parse_install_status("") is None
        assert _parse_install_status(None) is None
        assert _parse_install_status("   ") is None

    def test_returns_none_for_bad_json(self) -> None:
        assert _parse_install_status("<<not json>>") is None

    def test_returns_none_for_wrong_shape(self) -> None:
        # JSON-valid but not an object -> caller falls back to
        # other signals.
        assert _parse_install_status("42") is None
        assert _parse_install_status('"a string"') is None
        assert _parse_install_status('[]') is None
