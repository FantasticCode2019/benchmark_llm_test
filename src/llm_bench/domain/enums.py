"""String-valued enums for fields that previously lived as bare ``str``.

Why ``StrEnum`` (3.11+)?

* ``StrEnum`` instances *are* ``str`` instances. That means
  ``json.dumps(InstallDecision.FRESH)`` emits ``"fresh"`` without any
  custom encoder, and ``dataclasses.asdict(model_result)`` preserves
  the string value end-to-end. This is essential because the JSON
  report is part of the project's "behavior contract" — downstream
  consumers parse ``api_type == "ollama"`` and ``install_decision ==
  "fresh"`` literally, so the wire format must not drift.

* The enum gives type-checkers (mypy / IDE) something to enforce; the
  set of valid values is documented in code, not in scattered string
  literals.

Adding a new value: pick a kebab-case-free string (snake_case for
multi-word states), add it here, and update any ``str``-typed call
sites that produce / consume the value.
"""
from __future__ import annotations

from enum import StrEnum


class ApiType(StrEnum):
    """Wire protocol used to drive the model under test.

    OLLAMA  -> ``/api/generate`` (stream=false), server-precise TTFT/TPS.
    OPENAI  -> ``/v1/chat/completions`` (stream=false), TTFT approximated
               via a max_tokens=1 round-trip or, when ``spec.thinking=True``,
               a streaming probe.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"

    @classmethod
    def parse(cls, value: str | ApiType | None,
              *, default: ApiType | None = None) -> ApiType:
        """Permissive constructor used by config loading. Accepts the
        enum itself, a case-insensitive name, or None / empty string
        (returns ``default`` or raises ``ValueError`` if no default was
        supplied).
        """
        if isinstance(value, cls):
            return value
        if value is None or value == "":
            if default is not None:
                return default
            raise ValueError("api_type is required and has no default")
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(sorted(m.value for m in cls))
            raise ValueError(
                f"unknown api_type {value!r}; expected one of: {allowed}"
            ) from exc


class InstallDecision(StrEnum):
    """How ``ensure_installed`` ended up satisfying the chart-present
    precondition.

    UNKNOWN     -> ensure_installed never ran (e.g. the bench bailed
                   earlier on a config error). Renders as ``""`` in the
                   JSON to stay byte-identical with the pre-enum era,
                   where the field defaulted to ``""``.
    FRESH       -> chart was not installed; we installed it from scratch.
    REUSED      -> chart was already running (and ``skip_install_if_running``
                   was true), so we left it alone.
    RECOVERED   -> chart was in a stuck / rolled-back state, we did
                   uninstall+reinstall (or installed-directly for rolled-
                   back releases) to recover.
    """

    UNKNOWN = ""
    FRESH = "fresh"
    REUSED = "reused"
    RECOVERED = "recovered"


__all__ = ["ApiType", "InstallDecision"]
