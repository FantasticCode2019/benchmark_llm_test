"""Typed config layer.

Maps the **on-disk JSON schema** (documented in ``config.example.json``)
into a tree of dataclasses so the rest of the codebase can stop juggling
``dict.get(key, default)`` calls and the ``_opt(spec, cfg, key, default)``
override helper that previously sprawled across ``orchestrator.py``.

The JSON wire format is unchanged: ``AppConfig.from_dict`` accepts the
exact same keys ``load_config`` used to consume, validates required
ones (``models[]`` / ``questions[]`` / ``email``), and raises
:class:`ConfigValidationError` for any structural problem. Unknown keys
are tolerated (forward compat) but logged at WARNING so typos surface.

Resolution semantics (per-model override → global default → hard-coded
default) live in :meth:`ResolvedOptions.for_model`, in one place, so
each field is resolved exactly once per model run instead of inline at
every consumer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llm_bench.constants import DEFAULT_EMAIL_SUBJECT, LOG_NAMESPACE
from llm_bench.domain.enums import ApiType
from llm_bench.exceptions import ConfigValidationError

log = logging.getLogger(LOG_NAMESPACE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_set(*values: Any) -> Any:
    """Return the first ``value`` that is not None.

    Used by the override resolver to honour ``False`` / ``0`` / ``""``
    as legitimate, explicit overrides (whereas ``or`` would skip them).
    Falls back to None if every input is None — callers add the final
    hard-coded fallback themselves so the type stays right.
    """
    for v in values:
        if v is not None:
            return v
    return None


def _coerce_bool(value: Any, *, field_name: str) -> bool | None:
    """Lenient bool coercion for JSON values. Accepts true booleans, the
    strings "true"/"false"/"yes"/"no"/"on"/"off"/"1"/"0", and ints. None
    passes through so callers can distinguish "unset" from "set false".
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "on", "1"}:
            return True
        if low in {"false", "no", "off", "0", ""}:
            return False
    raise ConfigValidationError(
        f"{field_name}: expected boolean, got {value!r}")


def _coerce_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):  # bool is subclass of int — reject
        raise ConfigValidationError(
            f"{field_name}: expected integer, got bool {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                f"{field_name}: expected integer, got {value!r}") from exc
    raise ConfigValidationError(
        f"{field_name}: expected integer, got {value!r}")


# ---------------------------------------------------------------------------
# Per-model OpenAI overrides (resolved at bench time)
# ---------------------------------------------------------------------------


@dataclass
class OpenAIConfig:
    """Per-model knobs for the openai-shape benchmark; resolved by
    :func:`llm_bench.core.benchmark.openai.openai_config_from`, which
    merges :attr:`ModelSpec.openai_overrides` over
    :attr:`AppConfig.openai_defaults` over these hard-coded fallbacks.
    """
    api_key: str = "EMPTY"
    endpoint: str = "chat"            # "chat" -> /v1/chat/completions
                                      # "completion" -> /v1/completions
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    measure_ttft_approx: bool = True


# ---------------------------------------------------------------------------
# Global defaults (top-level cfg that can be overridden per model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlobalDefaults:
    """Top-level config fields that ALSO accept per-model overrides.

    These are the keys ``_opt(spec, cfg, key, default)`` used to lookup —
    centralizing the defaults in one dataclass eliminates the 15-way
    duplication and (more importantly) gives newcomers ONE place to
    read what every knob means and what its baseline value is.

    The values here are the SAME hard-coded defaults the pre-Phase-3
    code used. Changing them ripples to every model unless the model
    spec overrides them.
    """
    install_timeout_minutes: int = 90
    uninstall_timeout_minutes: int = 30
    request_timeout_seconds: int = 1800
    # Happy-path poll cadence used by `_poll_ollama_progress` /
    # `_poll_ollama_health`. The vllm pollers prefer the server-supplied
    # `cfg.probeIntervalMs` and only fall back here when the server
    # leaves it unset. Failure-mode retry stays hard-coded at
    # READINESS_FAILURE_RETRY_SECONDS so a flapping endpoint can't
    # silently slow down a healthy steady-state poll.
    readiness_probe_interval_seconds: int = 2
    delete_data: bool = True
    auto_open_internal_entrance: bool = True
    # Legacy alias: when true, force auto_open True regardless of
    # auto_open_internal_entrance. Preserved for old config files.
    set_public_during_run: bool = False
    skip_install_if_running: bool = True
    preserve_if_existed: bool = False
    uninstall_after_run: bool = True
    thinking: bool = False
    save_pod_logs_on_failure: bool = True
    pod_logs_dir: str = "/tmp"
    api_type: ApiType = ApiType.OLLAMA

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GlobalDefaults:
        """Build defaults from the top-level config dict.

        Each field falls back to its dataclass default when the JSON
        omits the key. We instantiate a fresh ``cls()`` to grab those
        defaults once (rather than duplicating literals across this
        method and the field declarations) — note ``slots=True`` strips
        the class-attribute defaults, so reading them off ``cls`` would
        return slot descriptors, not values.

        ``_first_set`` is used for every field so a config that sets
        ``delete_data: false`` (or ``cooldown_seconds: 0``) survives —
        a naive ``x or default`` would silently restore the default
        for any falsy-but-valid input.
        """
        base = cls()
        return cls(
            install_timeout_minutes=_first_set(
                _coerce_int(raw.get("install_timeout_minutes"),
                            field_name="install_timeout_minutes"),
                base.install_timeout_minutes),
            uninstall_timeout_minutes=_first_set(
                _coerce_int(raw.get("uninstall_timeout_minutes"),
                            field_name="uninstall_timeout_minutes"),
                base.uninstall_timeout_minutes),
            request_timeout_seconds=_first_set(
                _coerce_int(raw.get("request_timeout_seconds"),
                            field_name="request_timeout_seconds"),
                base.request_timeout_seconds),
            readiness_probe_interval_seconds=_first_set(
                _coerce_int(raw.get("readiness_probe_interval_seconds"),
                            field_name="readiness_probe_interval_seconds"),
                base.readiness_probe_interval_seconds),
            delete_data=_first_set(
                _coerce_bool(raw.get("delete_data"),
                             field_name="delete_data"),
                base.delete_data),
            auto_open_internal_entrance=_first_set(
                _coerce_bool(raw.get("auto_open_internal_entrance"),
                             field_name="auto_open_internal_entrance"),
                base.auto_open_internal_entrance),
            set_public_during_run=_first_set(
                _coerce_bool(raw.get("set_public_during_run"),
                             field_name="set_public_during_run"),
                base.set_public_during_run),
            skip_install_if_running=_first_set(
                _coerce_bool(raw.get("skip_install_if_running"),
                             field_name="skip_install_if_running"),
                base.skip_install_if_running),
            preserve_if_existed=_first_set(
                _coerce_bool(raw.get("preserve_if_existed"),
                             field_name="preserve_if_existed"),
                base.preserve_if_existed),
            uninstall_after_run=_first_set(
                _coerce_bool(raw.get("uninstall_after_run"),
                             field_name="uninstall_after_run"),
                base.uninstall_after_run),
            thinking=_first_set(
                _coerce_bool(raw.get("thinking"),
                             field_name="thinking"),
                base.thinking),
            save_pod_logs_on_failure=_first_set(
                _coerce_bool(raw.get("save_pod_logs_on_failure"),
                             field_name="save_pod_logs_on_failure"),
                base.save_pod_logs_on_failure),
            pod_logs_dir=str(raw.get("pod_logs_dir") or base.pod_logs_dir),
            api_type=ApiType.parse(raw.get("api_type"),
                                   default=base.api_type),
        )


# ---------------------------------------------------------------------------
# Per-model specification
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """One entry in the ``models[]`` array.

    Only ``app_name`` and ``model_name`` are required; every other field
    is an optional override over :class:`GlobalDefaults`. ``None`` means
    "inherit from defaults" — the resolver short-circuits on the first
    explicit value, so ``False`` / ``0`` / ``""`` work as real overrides.

    The ``openai`` block stays a raw dict (passed through to
    :func:`openai_config_from`) because its shape is fluid (callers often
    add ad-hoc keys via ``extra_body``); a typed mirror would just churn.
    """
    app_name: str
    model_name: str
    api_type: ApiType | None = None
    entrance_name: str | None = None
    endpoint_url: str | None = None
    envs: list[str] = field(default_factory=list)
    install_timeout_minutes: int | None = None
    uninstall_timeout_minutes: int | None = None
    request_timeout_seconds: int | None = None
    readiness_probe_interval_seconds: int | None = None
    delete_data: bool | None = None
    auto_open_internal_entrance: bool | None = None
    set_public_during_run: bool | None = None
    skip_install_if_running: bool | None = None
    preserve_if_existed: bool | None = None
    uninstall_after_run: bool | None = None
    thinking: bool | None = None
    save_pod_logs_on_failure: bool | None = None
    pod_logs_dir: str | None = None
    openai_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelSpec:
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"models[]: each entry must be an object, got {type(raw).__name__}")
        app_name = raw.get("app_name")
        model_name = raw.get("model_name")
        if not app_name or not isinstance(app_name, str):
            raise ConfigValidationError(
                "models[].app_name is required and must be a non-empty string")
        if not model_name or not isinstance(model_name, str):
            raise ConfigValidationError(
                "models[].model_name is required and must be a non-empty string")
        envs = raw.get("envs") or []
        if not isinstance(envs, list):
            raise ConfigValidationError(
                f"models[{app_name}].envs must be a list, got {type(envs).__name__}")
        openai_block = raw.get("openai") or {}
        if not isinstance(openai_block, dict):
            raise ConfigValidationError(
                f"models[{app_name}].openai must be an object, "
                f"got {type(openai_block).__name__}")
        # api_type=None when absent, so resolution falls back to defaults.
        api_type_raw = raw.get("api_type")
        api_type = (ApiType.parse(api_type_raw, default=None)
                    if api_type_raw not in (None, "") else None)
        return cls(
            app_name=app_name,
            model_name=model_name,
            api_type=api_type,
            entrance_name=raw.get("entrance_name") or None,
            endpoint_url=raw.get("endpoint_url") or None,
            envs=[str(e) for e in envs],
            install_timeout_minutes=_coerce_int(
                raw.get("install_timeout_minutes"),
                field_name=f"models[{app_name}].install_timeout_minutes"),
            uninstall_timeout_minutes=_coerce_int(
                raw.get("uninstall_timeout_minutes"),
                field_name=f"models[{app_name}].uninstall_timeout_minutes"),
            request_timeout_seconds=_coerce_int(
                raw.get("request_timeout_seconds"),
                field_name=f"models[{app_name}].request_timeout_seconds"),
            readiness_probe_interval_seconds=_coerce_int(
                raw.get("readiness_probe_interval_seconds"),
                field_name=f"models[{app_name}].readiness_probe_interval_seconds"),
            delete_data=_coerce_bool(
                raw.get("delete_data"),
                field_name=f"models[{app_name}].delete_data"),
            auto_open_internal_entrance=_coerce_bool(
                raw.get("auto_open_internal_entrance"),
                field_name=f"models[{app_name}].auto_open_internal_entrance"),
            set_public_during_run=_coerce_bool(
                raw.get("set_public_during_run"),
                field_name=f"models[{app_name}].set_public_during_run"),
            skip_install_if_running=_coerce_bool(
                raw.get("skip_install_if_running"),
                field_name=f"models[{app_name}].skip_install_if_running"),
            preserve_if_existed=_coerce_bool(
                raw.get("preserve_if_existed"),
                field_name=f"models[{app_name}].preserve_if_existed"),
            uninstall_after_run=_coerce_bool(
                raw.get("uninstall_after_run"),
                field_name=f"models[{app_name}].uninstall_after_run"),
            thinking=_coerce_bool(
                raw.get("thinking"),
                field_name=f"models[{app_name}].thinking"),
            save_pod_logs_on_failure=_coerce_bool(
                raw.get("save_pod_logs_on_failure"),
                field_name=f"models[{app_name}].save_pod_logs_on_failure"),
            pod_logs_dir=(str(raw["pod_logs_dir"])
                          if raw.get("pod_logs_dir") else None),
            openai_overrides=dict(openai_block),
        )


# ---------------------------------------------------------------------------
# Resolved per-model options (spec > defaults > hard-coded)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedOptions:
    """Per-model option bundle produced by collapsing ModelSpec overrides
    onto GlobalDefaults. Eliminates the 15 ``_opt(...)`` lookups
    ``bench_model`` used to make inline.

    Built ONCE at the start of every model run. Frozen + slotted so the
    orchestrator can pass it around without worrying about accidental
    mutation mid-run.
    """
    install_minutes: int
    uninstall_minutes: int
    request_timeout: int
    readiness_probe_interval_seconds: int
    delete_data: bool
    auto_open: bool
    skip_if_running: bool
    preserve_if_existed: bool
    uninstall_after: bool
    thinking: bool
    save_pod_logs: bool
    pod_logs_dir: str
    api_type: ApiType

    @classmethod
    def for_model(cls, spec: ModelSpec,
                  defaults: GlobalDefaults) -> ResolvedOptions:
        """Resolve every option for ``spec`` against ``defaults``.

        Semantics: ``spec.<field> if spec.<field> is not None else
        defaults.<field>`` — i.e. an explicit per-model ``False`` or
        ``0`` wins over the global default, while leaving the key out
        of the JSON inherits the global default. Mirrors the historical
        ``_opt`` helper exactly.
        """
        # auto_open has a legacy alias: `set_public_during_run=true` flips
        # it on regardless of `auto_open_internal_entrance`. Resolve each
        # input independently, then OR them so the legacy switch never
        # *clears* an explicit true.
        auto_open_primary = _first_set(spec.auto_open_internal_entrance,
                                       defaults.auto_open_internal_entrance)
        legacy_public = _first_set(spec.set_public_during_run,
                                   defaults.set_public_during_run)
        auto_open = bool(auto_open_primary) or bool(legacy_public)

        return cls(
            install_minutes=_first_set(spec.install_timeout_minutes,
                                       defaults.install_timeout_minutes),
            uninstall_minutes=_first_set(spec.uninstall_timeout_minutes,
                                         defaults.uninstall_timeout_minutes),
            request_timeout=_first_set(spec.request_timeout_seconds,
                                       defaults.request_timeout_seconds),
            readiness_probe_interval_seconds=_first_set(
                spec.readiness_probe_interval_seconds,
                defaults.readiness_probe_interval_seconds),
            delete_data=_first_set(spec.delete_data, defaults.delete_data),
            auto_open=auto_open,
            skip_if_running=_first_set(spec.skip_install_if_running,
                                       defaults.skip_install_if_running),
            preserve_if_existed=_first_set(spec.preserve_if_existed,
                                           defaults.preserve_if_existed),
            uninstall_after=_first_set(spec.uninstall_after_run,
                                       defaults.uninstall_after_run),
            thinking=_first_set(spec.thinking, defaults.thinking),
            save_pod_logs=_first_set(spec.save_pod_logs_on_failure,
                                     defaults.save_pod_logs_on_failure),
            pod_logs_dir=_first_set(spec.pod_logs_dir, defaults.pod_logs_dir),
            api_type=_first_set(spec.api_type, defaults.api_type),
        )


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


@dataclass
class EmailConfig:
    """SMTP destination + transport knobs.

    ``use_ssl`` is intentionally ``Optional[bool]`` so the mailer can
    apply its port-based heuristic ("465 → implicit TLS, anything else
    → STARTTLS") when the user leaves the field unset. Set explicitly
    to override the heuristic.
    """
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    sender: str   # "from" in the JSON (reserved word in Python)
    to: str
    use_ssl: bool | None = None
    smtp_timeout: int = 120
    smtp_retries: int = 3
    smtp_retry_backoff: int = 5
    subject: str = DEFAULT_EMAIL_SUBJECT

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EmailConfig:
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                "email: must be an object")
        missing = [k for k in ("smtp_host", "smtp_port", "username",
                               "password", "from", "to")
                   if not raw.get(k)]
        if missing:
            raise ConfigValidationError(
                f"email: required field(s) missing/empty: "
                f"{', '.join(missing)}")
        port = _coerce_int(raw["smtp_port"], field_name="email.smtp_port")
        if port is None:
            raise ConfigValidationError(
                "email.smtp_port: required (got None after coercion)")
        return cls(
            smtp_host=str(raw["smtp_host"]),
            smtp_port=port,
            username=str(raw["username"]),
            password=str(raw["password"]),
            sender=str(raw["from"]),
            to=str(raw["to"]),
            use_ssl=_coerce_bool(raw.get("use_ssl"),
                                 field_name="email.use_ssl"),
            smtp_timeout=_first_set(
                _coerce_int(raw.get("smtp_timeout"),
                            field_name="email.smtp_timeout"),
                120),
            smtp_retries=_first_set(
                _coerce_int(raw.get("smtp_retries"),
                            field_name="email.smtp_retries"),
                3),
            smtp_retry_backoff=_first_set(
                _coerce_int(raw.get("smtp_retry_backoff"),
                            field_name="email.smtp_retry_backoff"),
                5),
            subject=str(raw.get("subject") or DEFAULT_EMAIL_SUBJECT),
        )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


_KNOWN_ROOT_KEYS = frozenset({
    # Global defaults — handled by GlobalDefaults.from_dict
    "install_timeout_minutes", "uninstall_timeout_minutes",
    "request_timeout_seconds", "readiness_probe_interval_seconds",
    "delete_data", "auto_open_internal_entrance",
    "set_public_during_run", "skip_install_if_running",
    "preserve_if_existed", "uninstall_after_run", "thinking",
    "save_pod_logs_on_failure", "pod_logs_dir", "api_type",
    # Root-only
    "cli_path", "cooldown_seconds", "output_dir", "sudo_password",
    "openai_defaults", "models", "questions", "email",
    # Legacy fields preserved for forward-compat (silently ignored by
    # the current implementation but accepted by older configs so a
    # stale .json doesn't trip the unknown-key warning).
    "api_ready_timeout_minutes", "api_ready_probe_interval_seconds",
    "warmup_retries", "warmup_retry_sleep_seconds",
    # Removed when the explicit `ollama /api/pull` escape hatch was
    # dropped — the chart launcher pulls every supported ollama* chart.
    "pull_model", "pull_timeout_seconds", "pull_max_attempts",
    "pull_retry_sleep_seconds",
})


@dataclass
class AppConfig:
    """Fully parsed config file. Returned by :func:`load_config`.

    Holds the structured pieces (defaults / models / email) plus the
    root-only knobs (``cli_path``, ``cooldown_seconds``, ``output_dir``,
    ``sudo_password``, ``openai_defaults``). The ``openai_defaults``
    dict is kept raw because it gets merged at bench time by
    :func:`openai_config_from`, which knows the full set of fields
    OpenAIConfig understands.
    """
    defaults: GlobalDefaults
    models: list[ModelSpec]
    questions: list[str]
    email: EmailConfig
    cli_path: str | None = None
    cooldown_seconds: int = 30
    output_dir: str | None = None
    sudo_password: str | None = None
    openai_defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AppConfig:
        """Validate + construct an :class:`AppConfig` from the parsed JSON.

        Raises :class:`ConfigValidationError` for missing/typed-wrong
        required pieces. Unknown root keys are logged at WARNING so
        typos surface during local iteration (but don't break the run).
        """
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                "config root: must be a JSON object at top level")

        for key in raw:
            if key not in _KNOWN_ROOT_KEYS:
                log.warning("config: unknown top-level key %r (ignored); "
                            "this may be a typo", key)

        models_raw = raw.get("models")
        if not isinstance(models_raw, list) or not models_raw:
            raise ConfigValidationError(
                "config: 'models' must be a non-empty list")

        questions_raw = raw.get("questions")
        if not isinstance(questions_raw, list) or not questions_raw:
            raise ConfigValidationError(
                "config: 'questions' must be a non-empty list")

        if "email" not in raw:
            raise ConfigValidationError(
                "config: 'email' section is required")

        openai_defaults = raw.get("openai_defaults") or {}
        if not isinstance(openai_defaults, dict):
            raise ConfigValidationError(
                "config.openai_defaults: must be an object")

        return cls(
            defaults=GlobalDefaults.from_dict(raw),
            models=[ModelSpec.from_dict(m) for m in models_raw],
            questions=[str(q) for q in questions_raw],
            email=EmailConfig.from_dict(raw["email"]),
            cli_path=(str(raw["cli_path"]) if raw.get("cli_path") else None),
            cooldown_seconds=_first_set(
                _coerce_int(raw.get("cooldown_seconds"),
                            field_name="cooldown_seconds"),
                30),
            output_dir=(str(raw["output_dir"])
                        if raw.get("output_dir") else None),
            sudo_password=(str(raw["sudo_password"])
                           if raw.get("sudo_password") else None),
            openai_defaults=dict(openai_defaults),
        )


__all__ = [
    "AppConfig",
    "EmailConfig",
    "GlobalDefaults",
    "ModelSpec",
    "OpenAIConfig",
    "ResolvedOptions",
]
