"""Enum round-trip + JSON serialization tests.

The benchmark's JSON report is part of the public contract — downstream
consumers parse ``api_type == "ollama"`` and ``install_decision ==
"fresh"`` as bare strings. ``StrEnum`` is the only sane way to give these
fields types in Python AND keep the wire format identical, so this file
locks that property in.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from llm_bench.domain import (
    ApiType,
    InstallDecision,
    ModelResult,
    QuestionResult,
)


class TestApiTypeParse:
    """ApiType.parse must accept the same forms the JSON config used to."""

    def test_accepts_canonical_lowercase(self):
        assert ApiType.parse("ollama") is ApiType.OLLAMA
        assert ApiType.parse("openai") is ApiType.OPENAI

    def test_case_insensitive(self):
        # The pre-Phase-5 code did `.lower()`, so any mixed-case must work.
        assert ApiType.parse("OLLAMA") is ApiType.OLLAMA
        assert ApiType.parse("OpenAI") is ApiType.OPENAI

    def test_pass_through_enum(self):
        # Already an enum -> returned as-is (no roundtrip).
        assert ApiType.parse(ApiType.OPENAI) is ApiType.OPENAI

    def test_none_uses_default(self):
        assert ApiType.parse(None, default=ApiType.OLLAMA) is ApiType.OLLAMA

    def test_empty_string_uses_default(self):
        # Empty-string config values used to fall through to defaults; preserve.
        assert ApiType.parse("", default=ApiType.OPENAI) is ApiType.OPENAI

    def test_no_default_raises(self):
        with pytest.raises(ValueError):
            ApiType.parse(None)

    def test_unknown_value_lists_allowed(self):
        with pytest.raises(ValueError) as exc_info:
            ApiType.parse("anthropic")
        assert "ollama" in str(exc_info.value)
        assert "openai" in str(exc_info.value)


class TestStrEnumSerialization:
    """`asdict()` + `json.dumps()` must emit the enum's STRING VALUE,
    not the Python repr — this is what keeps the JSON wire format
    byte-identical with the pre-enum era.
    """

    def test_api_type_json_value(self):
        # StrEnum inherits from str so json.dumps emits the value as a plain string.
        assert json.dumps(ApiType.OLLAMA) == '"ollama"'
        assert json.dumps(ApiType.OPENAI) == '"openai"'

    def test_install_decision_unknown_renders_as_empty(self):
        # InstallDecision.UNKNOWN == "" preserves the v0.1 default-string field.
        assert json.dumps(InstallDecision.UNKNOWN) == '""'

    def test_install_decision_named_values(self):
        assert json.dumps(InstallDecision.FRESH) == '"fresh"'
        assert json.dumps(InstallDecision.REUSED) == '"reused"'
        assert json.dumps(InstallDecision.RECOVERED) == '"recovered"'

    def test_model_result_asdict_keeps_string_form(self):
        # The end-to-end shape the JSON report sees.
        mr = ModelResult(app_name="app", model="m",
                         api_type=ApiType.OPENAI,
                         install_decision=InstallDecision.RECOVERED)
        dumped = json.dumps(asdict(mr))
        # Field appears verbatim as JSON string, not as
        # `"ApiType.OPENAI"` or `<ApiType.OPENAI: 'openai'>`.
        assert '"api_type": "openai"' in dumped
        assert '"install_decision": "recovered"' in dumped


class TestModelResultBehavior:
    """Smoke-test the dataclass methods that didn't change but now sit
    behind a moved module."""

    def test_has_thinking_label_yes_when_first_ok_has_thinking(self):
        mr = ModelResult(app_name="a", model="m",
                         questions=[
                             QuestionResult(prompt="p1", ok=False),
                             QuestionResult(prompt="p2", ok=True,
                                            has_thinking=True),
                         ])
        assert mr.has_thinking_label() == "Yes"

    def test_has_thinking_label_no_when_no_ok_rows(self):
        mr = ModelResult(app_name="a", model="m",
                         questions=[QuestionResult(prompt="p", ok=False)])
        assert mr.has_thinking_label() == "No"

    def test_avg_ignores_failed_rows(self):
        mr = ModelResult(app_name="a", model="m",
                         questions=[
                             QuestionResult(prompt="p1", ok=True, tps=10.0),
                             QuestionResult(prompt="p2", ok=False, tps=999.0),
                             QuestionResult(prompt="p3", ok=True, tps=20.0),
                         ])
        assert mr.avg("tps") == pytest.approx(15.0)

    def test_avg_zero_when_no_ok_rows(self):
        mr = ModelResult(app_name="a", model="m",
                         questions=[QuestionResult(prompt="p", ok=False)])
        assert mr.avg("tps") == 0.0
