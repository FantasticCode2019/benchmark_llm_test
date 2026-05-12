"""Tests for the typed config layer (``AppConfig`` + ``ResolvedOptions``).

These exercise the Phase-3 dataclass migration. Anything that touches
filesystem / subprocess / SMTP is OUT of scope; we only verify the
in-memory dict-to-dataclass parsing + resolution logic.
"""
from __future__ import annotations

import json

import pytest

from llm_bench.data.config import load_config
from llm_bench.domain import (
    ApiType,
    AppConfig,
    GlobalDefaults,
    ModelSpec,
    ResolvedOptions,
)
from llm_bench.exceptions import ConfigError, ConfigValidationError


@pytest.fixture
def minimal_cfg_dict() -> dict:
    """The smallest config the validator accepts. Everything else
    falls back to GlobalDefaults / EmailConfig defaults so tests can
    focus on the variable surface."""
    return {
        "models": [
            {"app_name": "appA", "model_name": "modelA"},
        ],
        "questions": ["q1"],
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "username": "u",
            "password": "p",
            "from": "u@example.com",
            "to": "u@example.com",
        },
    }


# ---------------------------------------------------------------------------
# AppConfig.from_dict — required fields + happy path
# ---------------------------------------------------------------------------


class TestAppConfigRequired:

    def test_missing_models_raises(self, minimal_cfg_dict):
        minimal_cfg_dict.pop("models")
        with pytest.raises(ConfigValidationError) as ei:
            AppConfig.from_dict(minimal_cfg_dict)
        assert "models" in str(ei.value)

    def test_empty_models_raises(self, minimal_cfg_dict):
        minimal_cfg_dict["models"] = []
        with pytest.raises(ConfigValidationError):
            AppConfig.from_dict(minimal_cfg_dict)

    def test_missing_questions_raises(self, minimal_cfg_dict):
        minimal_cfg_dict.pop("questions")
        with pytest.raises(ConfigValidationError) as ei:
            AppConfig.from_dict(minimal_cfg_dict)
        assert "questions" in str(ei.value)

    def test_missing_email_raises(self, minimal_cfg_dict):
        minimal_cfg_dict.pop("email")
        with pytest.raises(ConfigValidationError) as ei:
            AppConfig.from_dict(minimal_cfg_dict)
        assert "email" in str(ei.value)

    def test_email_missing_required_field_raises(self, minimal_cfg_dict):
        del minimal_cfg_dict["email"]["smtp_host"]
        with pytest.raises(ConfigValidationError) as ei:
            AppConfig.from_dict(minimal_cfg_dict)
        # The error message lists the offending field name.
        assert "smtp_host" in str(ei.value)

    def test_model_missing_app_name_raises(self, minimal_cfg_dict):
        minimal_cfg_dict["models"][0].pop("app_name")
        with pytest.raises(ConfigValidationError) as ei:
            AppConfig.from_dict(minimal_cfg_dict)
        assert "app_name" in str(ei.value)


class TestAppConfigDefaults:

    def test_defaults_match_global_defaults(self, minimal_cfg_dict):
        cfg = AppConfig.from_dict(minimal_cfg_dict)
        defaults = GlobalDefaults()
        # When the JSON omits every override, the parsed AppConfig.defaults
        # must equal a freshly-constructed GlobalDefaults().
        assert cfg.defaults == defaults

    def test_cooldown_seconds_zero_is_honored(self, minimal_cfg_dict):
        """`cooldown_seconds: 0` must NOT be silently replaced by 30 —
        explicit zeros are valid (skip-cooldown sentinel). This is the
        regression check for the `_coerce_int(...) or 30` bug we
        intentionally fixed."""
        minimal_cfg_dict["cooldown_seconds"] = 0
        cfg = AppConfig.from_dict(minimal_cfg_dict)
        assert cfg.cooldown_seconds == 0

    def test_delete_data_false_is_honored(self, minimal_cfg_dict):
        """`delete_data: false` must NOT collapse to True — same bug
        class as cooldown_seconds=0, but for booleans."""
        minimal_cfg_dict["delete_data"] = False
        cfg = AppConfig.from_dict(minimal_cfg_dict)
        assert cfg.defaults.delete_data is False

    def test_unknown_root_key_logged_but_not_fatal(self, minimal_cfg_dict,
                                                   caplog):
        minimal_cfg_dict["completely_unknown_knob"] = 42
        cfg = AppConfig.from_dict(minimal_cfg_dict)  # must not raise
        assert cfg is not None
        assert any("completely_unknown_knob" in rec.message
                   for rec in caplog.records)


# ---------------------------------------------------------------------------
# ResolvedOptions — the consolidated `_opt` killer
# ---------------------------------------------------------------------------


class TestResolvedOptions:

    def test_spec_override_wins(self):
        defaults = GlobalDefaults(install_timeout_minutes=90)
        spec = ModelSpec(app_name="a", model_name="m",
                         install_timeout_minutes=240)
        opts = ResolvedOptions.for_model(spec, defaults)
        assert opts.install_minutes == 240

    def test_defaults_used_when_spec_silent(self):
        defaults = GlobalDefaults(install_timeout_minutes=90)
        spec = ModelSpec(app_name="a", model_name="m")
        opts = ResolvedOptions.for_model(spec, defaults)
        assert opts.install_minutes == 90

    def test_explicit_spec_false_beats_default_true(self):
        """The `False` override must be honored, not silently replaced by
        the default's `True`. This is the bool-falsy-trap regression
        check that motivated `_first_set`."""
        defaults = GlobalDefaults(delete_data=True)
        spec = ModelSpec(app_name="a", model_name="m", delete_data=False)
        opts = ResolvedOptions.for_model(spec, defaults)
        assert opts.delete_data is False

    def test_legacy_set_public_during_run_promotes_auto_open(self):
        """`set_public_during_run=true` is a legacy alias that flips
        auto_open on regardless of the primary key. Must remain True
        even when auto_open_internal_entrance is False."""
        defaults = GlobalDefaults(auto_open_internal_entrance=False,
                                  set_public_during_run=True)
        spec = ModelSpec(app_name="a", model_name="m")
        opts = ResolvedOptions.for_model(spec, defaults)
        assert opts.auto_open is True

    def test_auto_open_default_when_neither_set(self):
        defaults = GlobalDefaults()  # auto_open_internal_entrance=True
        spec = ModelSpec(app_name="a", model_name="m")
        opts = ResolvedOptions.for_model(spec, defaults)
        assert opts.auto_open is True

    def test_api_type_from_spec_overrides_default(self):
        defaults = GlobalDefaults(api_type=ApiType.OLLAMA)
        spec = ModelSpec(app_name="a", model_name="m",
                         api_type=ApiType.OPENAI)
        opts = ResolvedOptions.for_model(spec, defaults)
        assert opts.api_type is ApiType.OPENAI


# ---------------------------------------------------------------------------
# load_config — the filesystem-facing wrapper around AppConfig.from_dict
# ---------------------------------------------------------------------------


class TestLoadConfig:

    def test_file_not_found_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError) as ei:
            load_config(str(tmp_path / "does-not-exist.json"))
        assert "not found" in str(ei.value)

    def test_invalid_json_raises_config_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ this is not json")
        with pytest.raises(ConfigError) as ei:
            load_config(str(path))
        assert "invalid JSON" in str(ei.value)

    def test_happy_path_returns_typed_appconfig(self, tmp_path,
                                                minimal_cfg_dict):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(minimal_cfg_dict))
        cfg = load_config(str(path))
        assert isinstance(cfg, AppConfig)
        assert cfg.models[0].app_name == "appA"
        assert cfg.email.sender == "u@example.com"
